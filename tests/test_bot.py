"""Tests for the GitHub issue bot: intent parsing, delivery and failure paths.

Everything that talks to GitHub is injected: the PyGithub client is either a
small recording double or a ``unittest.mock.Mock`` (to assert the exact API
calls), and git is a fake. One real-git test exercises
:class:`SubprocessGitClient` against a bare repository on disk, so clone,
commit and push are covered without a network.
"""

from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from github import GithubException, UnknownObjectException

from moray_surgeon import bot as bot_module
from moray_surgeon.bot import (
    DEFAULT_FAILURE_LABEL,
    MAX_INTENT_CHARS,
    BotConfig,
    BotError,
    SubprocessGitClient,
    build_intent,
    build_pull_request_body,
    build_remote_url,
    run_github_bot,
)
from moray_surgeon.orchestrator import RefactorReport
from moray_surgeon.schema import AddTypeHints, RefactorPlan
from moray_surgeon.verify import VerifyResult

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "Moray Test",
    "GIT_AUTHOR_EMAIL": "moray@example.com",
    "GIT_COMMITTER_NAME": "Moray Test",
    "GIT_COMMITTER_EMAIL": "moray@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_COUNT": "2",
    "GIT_CONFIG_KEY_0": "commit.gpgsign",
    "GIT_CONFIG_VALUE_0": "false",
    "GIT_CONFIG_KEY_1": "core.autocrlf",
    "GIT_CONFIG_VALUE_1": "false",
}


def _config(**overrides: object) -> BotConfig:
    values: dict[str, object] = {"token": "test-token"}
    values.update(overrides)
    return BotConfig(**values)  # type: ignore[arg-type]


def _success_report(architecture_diff: str = "") -> RefactorReport:
    plan = RefactorPlan(
        instructions=[AddTypeHints(id="s1", target_file="app.py", function_name=None)],
        summary="Annotate the helpers.",
    )
    return RefactorReport(
        success=True,
        attempts=1,
        plan=plan,
        log=["[0] add_type_hints: 2 change(s) in app.py"],
        last_verify=VerifyResult(ok=True, stage="all"),
        architecture_diff=architecture_diff,
    )


def _failure_report(stage: str = "pytest", stdout: str = "", stderr: str = "") -> RefactorReport:
    return RefactorReport(
        success=False,
        attempts=3,
        plan=None,
        log=[],
        last_verify=VerifyResult(ok=False, stage=stage, stdout=stdout, stderr=stderr),
    )


class FakeIssue:
    """Recording stand-in for ``github.Issue.Issue``."""

    def __init__(
        self,
        number: int = 7,
        title: str | None = "Rename old_name",
        body: str | None = "Please use new_name.",
        comment_error: Exception | None = None,
    ) -> None:
        self.number = number
        self.title = title
        self.body = body
        self.comment_error = comment_error
        self.comments: list[str] = []
        self.labels: list[str] = []
        self.state = "open"

    def create_comment(self, body):
        if self.comment_error is not None:
            raise self.comment_error
        self.comments.append(body)

    def add_to_labels(self, *labels: str):
        self.labels.extend(labels)

    def edit(self, **kwargs: object):
        state = kwargs.get("state")
        if isinstance(state, str):
            self.state = state


class FakePullRequest:
    """Minimal pull request handle."""

    def __init__(self, url: str) -> None:
        self.html_url = url


class FakeRepo:
    """Recording stand-in for ``github.Repository.Repository``."""

    def __init__(
        self,
        issue: FakeIssue | None = None,
        *,
        default_branch: str = "main",
        labels: tuple[str, ...] = (DEFAULT_FAILURE_LABEL,),
    ) -> None:
        self.issue = issue or FakeIssue()
        self.default_branch = default_branch
        self.labels = set(labels)
        self.created_labels: list[tuple[str, str]] = []
        self.pull_requests: list[dict[str, object]] = []
        self.pull_error: Exception | None = None

    def get_issue(self, number):
        if number != self.issue.number:
            raise UnknownObjectException(404, {"message": "Not Found"})
        return self.issue

    def get_label(self, name):
        if name not in self.labels:
            raise UnknownObjectException(404, {"message": "Not Found"})
        return name

    def create_label(self, name, color):
        self.labels.add(name)
        self.created_labels.append((name, color))
        return name

    def create_pull(self, base, head, *, title, body):
        if self.pull_error is not None:
            raise self.pull_error
        request = FakePullRequest(
            f"https://github.com/acme/widgets/pull/{len(self.pull_requests) + 1}"
        )
        self.pull_requests.append(
            {"base": base, "head": head, "title": title, "body": body, "pr": request}
        )
        return request


class FakeGithub:
    """Stand-in for the PyGithub client."""

    def __init__(self, repo: FakeRepo) -> None:
        self.repo = repo
        self.requested: list[str] = []

    def get_repo(self, name):
        self.requested.append(name)
        return self.repo


class FailingGithub:
    """Client whose repository lookup always fails."""

    def get_repo(self, name):
        raise UnknownObjectException(404, {"message": "Not Found"})


class FakeGit:
    """Recording git client."""

    def __init__(
        self,
        *,
        ahead: int = 1,
        clone_error: Exception | None = None,
        push_error: Exception | None = None,
    ) -> None:
        self.ahead = ahead
        self.clone_error = clone_error
        self.push_error = push_error
        self.calls: list[tuple[str, str, str | None, str | None]] = []

    def clone(self, remote_url, destination, branch, base_branch):
        self.calls.append(("clone", remote_url, branch, base_branch))
        if self.clone_error is not None:
            raise self.clone_error
        destination.mkdir(parents=True, exist_ok=True)

    def commit_pending(self, workdir, message):
        self.calls.append(("commit", message, None, None))
        return False

    def commits_ahead(self, workdir, base_branch):
        self.calls.append(("ahead", base_branch, None, None))
        return self.ahead

    def push(self, workdir, branch):
        self.calls.append(("push", branch, None, None))
        if self.push_error is not None:
            raise self.push_error


class FakeOrchestrator:
    """Stand-in for the orchestrator; records the intent and clone path."""

    def __init__(
        self, report: RefactorReport | None = None, error: Exception | None = None
    ) -> None:
        self.report = report
        self.error = error
        self.received: tuple[str, Path] | None = None
        self.clone_existed = False

    def run(self, intent, repo_root, context="", target_files=None, dry_run=False):
        self.received = (intent, repo_root)
        self.clone_existed = Path(repo_root).is_dir()
        if self.error is not None:
            raise self.error
        return self.report


def _run(repo, report=None, *, git=None, config=None, orchestrator_error=None):
    orchestrator = FakeOrchestrator(report, orchestrator_error)
    result = run_github_bot(
        7,
        "acme/widgets",
        config=config or _config(),
        github=FakeGithub(repo),
        git=git or FakeGit(),
        orchestrator_factory=lambda root: orchestrator,
    )
    return result, orchestrator


# --------------------------------------------------------------------------- #
# Intent parsing
# --------------------------------------------------------------------------- #


def test_build_intent_combines_title_and_body():
    assert build_intent("  Rename old  ", "  Use new.\n") == "Rename old\n\nUse new."


def test_build_intent_accepts_a_missing_body():
    assert build_intent("Rename old", None) == "Rename old"
    assert build_intent("Rename old", "   ") == "Rename old"


def test_build_intent_accepts_a_missing_title():
    assert build_intent(None, "Use new.") == "Use new."


def test_build_intent_rejects_an_empty_issue():
    with pytest.raises(BotError, match="nothing to refactor"):
        build_intent("   ", None)


def test_build_intent_truncates_a_huge_body():
    intent = build_intent("Title", "x" * (MAX_INTENT_CHARS + 100))

    assert "[... issue truncated by the moray-surgeon bot ...]" in intent
    assert len(intent) < MAX_INTENT_CHARS + 100


# --------------------------------------------------------------------------- #
# Repository names, URLs and configuration
# --------------------------------------------------------------------------- #


def test_build_remote_url_uses_the_https_git_endpoint():
    assert (
        build_remote_url("https://github.com", "acme/widgets")
        == "https://github.com/acme/widgets.git"
    )


def test_build_remote_url_accepts_a_public_literal_address():
    """Public addresses are allowed; only local/private/reserved ones are not."""
    assert (
        build_remote_url("https://8.8.8.8", "acme/widgets")
        == "https://8.8.8.8/acme/widgets.git"
    )


@pytest.mark.parametrize(
    "server",
    [
        "ftp://github.com",
        "http://localhost:8080",
        "http://api.localhost",
        "http://127.0.0.1",
        "http://[::1]",
        "http://10.0.0.8",
        "http://192.168.1.5",
        "http://169.254.169.254",
        "http://0.0.0.0",
        "http://224.0.0.1",
        "https://user:pass@github.com",
        "https://github.com/?token=x",
        "https://github.com/#fragment",
        "https://github.com:notaport",
        "https://",
    ],
)
def test_build_remote_url_rejects_unsafe_servers(server):
    with pytest.raises(BotError):
        build_remote_url(server, "acme/widgets")


@pytest.mark.parametrize(
    "repo_name",
    [
        "",
        "acme",
        "acme/",
        "/widgets",
        "acme/widgets/extra",
        "acme/../widgets",
        "https://github.com/acme/widgets",
        "acme/widgets?x=1",
        "acme/widgets\nrm -rf /",
        "acme\\widgets",
    ],
)
def test_build_remote_url_rejects_malformed_repo_names(repo_name):
    with pytest.raises(BotError, match="owner/name"):
        build_remote_url("https://github.com", repo_name)


def test_config_from_env_reads_the_documented_variables(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_API_URL", "https://api.github.com")
    monkeypatch.setenv("MORAY_BOT_BASE_BRANCH", "release")
    monkeypatch.setenv("MORAY_BOT_FAILURE_LABEL", "bot-failed")
    monkeypatch.setenv("MORAY_BOT_CLOSE_ISSUE", "yes")
    monkeypatch.setenv("MORAY_BOT_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("MORAY_TIMEOUT", "30")
    monkeypatch.setenv("MORAY_MODEL", "claude-sonnet-4-5")
    monkeypatch.setenv("MORAY_API_TYPE", "anthropic")

    config = BotConfig.from_env()

    assert config.token == "tok"
    assert config.api_url == "https://api.github.com"
    assert config.base_branch == "release"
    assert config.failure_label == "bot-failed"
    assert config.close_on_failure is True
    assert config.max_attempts == 5
    assert config.model_timeout == 30.0
    assert config.model == "claude-sonnet-4-5"
    assert config.api_type == "anthropic"


def test_config_requires_a_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with pytest.raises(BotError, match="GITHUB_TOKEN is required"):
        BotConfig.from_env()


def test_config_rejects_an_unsafe_api_url(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_API_URL", "http://127.0.0.1:8000")

    with pytest.raises(BotError, match="GITHUB_API_URL"):
        BotConfig.from_env()


def test_config_rejects_a_bad_attempt_count(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("MORAY_BOT_MAX_ATTEMPTS", "many")

    with pytest.raises(BotError, match="MORAY_BOT_MAX_ATTEMPTS"):
        BotConfig.from_env()


def test_config_rejects_a_non_positive_attempt_count(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("MORAY_BOT_MAX_ATTEMPTS", "0")

    with pytest.raises(BotError, match="MORAY_BOT_MAX_ATTEMPTS must be >= 1"):
        BotConfig.from_env()


def test_config_rejects_a_bad_timeout(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    monkeypatch.setenv("MORAY_TIMEOUT", "slow")
    with pytest.raises(BotError, match="MORAY_TIMEOUT must be a number"):
        BotConfig.from_env()

    monkeypatch.setenv("MORAY_TIMEOUT", "0")
    with pytest.raises(BotError, match="MORAY_TIMEOUT must be > 0"):
        BotConfig.from_env()


def test_explicit_close_flag_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("MORAY_BOT_CLOSE_ISSUE", "true")

    assert BotConfig.from_env(close_on_failure=False).close_on_failure is False


def test_enterprise_api_url_is_passed_to_pygithub(monkeypatch):
    factory = Mock(return_value=Mock())
    monkeypatch.setattr(bot_module, "Github", factory)
    config = _config(api_url="https://ghe.example.com/api/v3")

    bot_module._connect_github(config)

    factory.assert_called_once_with(
        login_or_token="test-token", base_url="https://ghe.example.com/api/v3"
    )


def test_default_orchestrator_factory_wires_the_model_settings(monkeypatch):
    captured: dict[str, object] = {}

    class RecordingParser:
        def __init__(self, **kwargs: object) -> None:
            captured["parser"] = kwargs

    monkeypatch.setattr(bot_module, "IntentParser", RecordingParser)
    monkeypatch.setattr(bot_module, "Verifier", lambda root: ("verifier", root))
    config = _config(
        model="my-model",
        model_base_url="http://localhost:9000",
        api_key="key",
        api_type="openai",
        model_timeout=5.0,
        max_attempts=4,
    )

    from moray_surgeon.orchestrator import Orchestrator

    orchestrator = bot_module._orchestrator_factory(config)(Path("."))

    assert isinstance(orchestrator, Orchestrator)
    assert captured["parser"] == {
        "model": "my-model",
        "base_url": "http://localhost:9000",
        "timeout": 5.0,
        "api_key": "key",
        "api_type": "openai",
    }
    assert orchestrator._max_attempts == 4


# --------------------------------------------------------------------------- #
# Successful delivery
# --------------------------------------------------------------------------- #


def test_success_flow_clones_pushes_and_opens_a_pull_request():
    issue = FakeIssue(number=7, title="Rename old_name", body="Please use new_name.")
    repo = FakeRepo(issue)
    git = FakeGit()

    result, orchestrator = _run(repo, _success_report(), git=git)

    assert result.success is True
    assert result.branch == "moray-bot/issue-7"
    assert result.pull_request_url == "https://github.com/acme/widgets/pull/1"
    assert git.calls == [
        ("clone", "https://github.com/acme/widgets.git", "moray-bot/issue-7", "main"),
        ("commit", "moray-surgeon: resolve #7", None, None),
        ("ahead", "main", None, None),
        ("push", "moray-bot/issue-7", None, None),
    ]
    assert orchestrator.received is not None
    assert orchestrator.received[0] == "Rename old_name\n\nPlease use new_name."
    assert orchestrator.clone_existed is True
    assert issue.comments == []
    assert issue.labels == []

    request = repo.pull_requests[0]
    assert request["base"] == "main"
    assert request["head"] == "moray-bot/issue-7"
    assert "Closes #7" in request["body"]
    assert "Annotate the helpers." in request["body"]
    assert "add_type_hints" in request["body"]
    assert request["title"].startswith("moray-surgeon: resolve #7")


def test_default_branch_can_be_overridden():
    repo = FakeRepo(default_branch="main")
    git = FakeGit()

    _run(repo, _success_report(), git=git, config=_config(base_branch="release"))

    assert git.calls[0] == (
        "clone",
        "https://github.com/acme/widgets.git",
        "moray-bot/issue-7",
        "release",
    )
    assert repo.pull_requests[0]["base"] == "release"


def test_branch_prefix_can_be_overridden():
    repo = FakeRepo()
    git = FakeGit()

    result, _ = _run(repo, _success_report(), git=git, config=_config(branch_prefix="bot/x-"))

    assert result.branch == "bot/x-7"
    assert git.calls[3] == ("push", "bot/x-7", None, None)


def test_work_dir_keeps_the_clone_for_debugging(tmp_path):
    work_dir = tmp_path / "work"
    issue = FakeIssue()
    repo = FakeRepo(issue)
    orchestrator = FakeOrchestrator(_success_report())

    result = run_github_bot(
        7,
        "acme/widgets",
        config=_config(),
        github=FakeGithub(repo),  # type: ignore[arg-type]
        git=FakeGit(),
        work_dir=work_dir,
        orchestrator_factory=lambda root: orchestrator,
    )

    assert result.success is True
    assert (work_dir / "repo").is_dir()
    assert orchestrator.received is not None
    assert orchestrator.received[1] == (work_dir / "repo").resolve()


def test_pull_request_title_is_bounded():
    issue = FakeIssue(title="x" * 400)
    repo = FakeRepo(issue)

    _run(repo, _success_report())

    title = repo.pull_requests[0]["title"]
    assert isinstance(title, str)
    assert len(title) <= 250
    assert title.endswith("...")


def test_pull_request_body_carries_the_architecture_diff():
    repo = FakeRepo()

    _run(repo, _success_report(architecture_diff="+ class Splitter:"))

    body = repo.pull_requests[0]["body"]
    assert "### Architecture diff" in body
    assert "+ class Splitter:" in body


def test_custom_backend_client_is_used_when_none_is_injected(monkeypatch):
    """The PyGithub constructor and API calls are exactly what the bot expects."""
    issue = Mock()
    issue.number = 5
    issue.title = "Tidy helpers"
    issue.body = None
    repo = Mock()
    repo.default_branch = "main"
    repo.get_issue.return_value = issue
    repo.create_pull.return_value = Mock(html_url="https://github.com/o/r/pull/3")
    client = Mock()
    client.get_repo.return_value = repo
    factory = Mock(return_value=client)
    monkeypatch.setattr(bot_module, "Github", factory)

    result = run_github_bot(
        5,
        "o/r",
        config=BotConfig(token="tok"),
        git=FakeGit(),
        orchestrator_factory=lambda root: FakeOrchestrator(_success_report()),
    )

    factory.assert_called_once_with(login_or_token="tok")
    client.get_repo.assert_called_once_with("o/r")
    repo.get_issue.assert_called_once_with(5)
    kwargs = repo.create_pull.call_args.kwargs
    assert kwargs["base"] == "main"
    assert kwargs["head"] == "moray-bot/issue-5"
    assert "Closes #5" in kwargs["body"]
    assert result.pull_request_url == "https://github.com/o/r/pull/3"


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_failed_verification_comments_feedback_and_labels_the_issue():
    issue = FakeIssue()
    repo = FakeRepo(issue)
    git = FakeGit()

    result, _ = _run(
        repo,
        _failure_report(stage="pytest", stdout="FAILED tests/test_app.py::test_x"),
        git=git,
    )

    assert result.success is False
    assert result.message == "verification failed at pytest"
    assert len(issue.comments) == 1
    assert "[pytest failed]" in issue.comments[0]
    assert "FAILED tests/test_app.py::test_x" in issue.comments[0]
    assert "No branch was pushed" in issue.comments[0]
    assert issue.labels == [DEFAULT_FAILURE_LABEL]
    assert repo.pull_requests == []
    assert not any(call[0] in {"commit", "push"} for call in git.calls)


def test_failed_verification_prefers_stderr_over_stdout():
    repo = FakeRepo()

    _run(repo, _failure_report(stage="ruff", stdout="stdout noise", stderr="E501 boom"))

    assert "E501 boom" in repo.issue.comments[0]


def test_missing_failure_label_is_created_then_applied():
    repo = FakeRepo(labels=())

    _run(repo, _failure_report())

    assert repo.created_labels == [(DEFAULT_FAILURE_LABEL, bot_module.FAILURE_LABEL_COLOR)]
    assert repo.issue.labels == [DEFAULT_FAILURE_LABEL]


def test_issue_is_closed_when_configured():
    repo = FakeRepo()

    _run(repo, _failure_report(), config=_config(close_on_failure=True))

    assert repo.issue.state == "closed"


def test_issue_stays_open_by_default():
    repo = FakeRepo()

    _run(repo, _failure_report())

    assert repo.issue.state == "open"


def test_orchestrator_exception_is_reported_on_the_issue():
    issue = FakeIssue()
    repo = FakeRepo(issue)
    git = FakeGit()

    result, _ = _run(
        repo, git=git, orchestrator_error=RuntimeError("model endpoint unreachable")
    )

    assert result.success is False
    assert "RuntimeError" in result.message
    assert "model endpoint unreachable" in issue.comments[0]
    assert issue.labels == [DEFAULT_FAILURE_LABEL]
    assert not any(call[0] == "push" for call in git.calls)


def test_a_run_without_a_commit_is_not_delivered():
    repo = FakeRepo()
    git = FakeGit(ahead=0)

    result, _ = _run(repo, _success_report(), git=git)

    assert result.success is False
    assert result.message == "no commit produced"
    assert "nothing to review" in repo.issue.comments[0]
    assert not any(call[0] == "push" for call in git.calls)


def test_git_failures_are_redacted_before_they_reach_the_issue():
    issue = FakeIssue()
    repo = FakeRepo(issue)
    git = FakeGit(clone_error=BotError("fatal: auth failed for s3cret-token"))

    result, _ = _run(repo, _success_report(), git=git, config=_config(token="s3cret-token"))

    assert result.success is False
    assert "s3cret-token" not in issue.comments[0]
    assert "***" in issue.comments[0]
    assert issue.labels == [DEFAULT_FAILURE_LABEL]


def test_pull_request_failure_keeps_the_pushed_branch_and_comments():
    issue = FakeIssue()
    repo = FakeRepo(issue)
    repo.pull_error = GithubException(422, {"message": "Validation Failed"})

    result, _ = _run(repo, _success_report())

    assert result.success is False
    assert "Branch moray-bot/issue-7 was pushed" in issue.comments[0]
    assert "Validation Failed" in issue.comments[0]
    assert issue.labels == [DEFAULT_FAILURE_LABEL]


def test_a_failed_comment_becomes_a_bot_error():
    issue = FakeIssue(
        comment_error=GithubException(403, {"message": "Resource not accessible"})
    )
    repo = FakeRepo(issue)

    with pytest.raises(BotError, match="could not report the failure"):
        _run(repo, _failure_report())


def test_unreadable_repository_raises_before_any_work():
    with pytest.raises(BotError, match="cannot read repository"):
        run_github_bot(
            7,
            "acme/missing",
            config=_config(),
            github=FailingGithub(),  # type: ignore[arg-type]
            git=FakeGit(),
        )


def test_unreadable_issue_raises_before_the_clone():
    repo = FakeRepo(FakeIssue(number=7))
    git = FakeGit()

    with pytest.raises(BotError, match=r"cannot read issue #99"):
        run_github_bot(
            99,
            "acme/widgets",
            config=_config(),
            github=FakeGithub(repo),  # type: ignore[arg-type]
            git=git,
        )

    assert git.calls == []


def test_non_positive_issue_number_is_rejected():
    with pytest.raises(BotError, match="must be positive"):
        run_github_bot(0, "acme/widgets", config=_config())


def test_noop_report_is_reported_with_its_stage():
    repo = FakeRepo()

    result, _ = _run(repo, _failure_report(stage="noop"))

    assert result.message == "verification failed at noop"
    assert "noop failed" in repo.issue.comments[0]


def test_report_without_a_verify_result_explains_the_exhaustion():
    report = RefactorReport(
        success=False, attempts=3, plan=None, log=[], last_verify=None
    )
    repo = FakeRepo()

    result, _ = _run(repo, report)

    assert result.message == "no verified plan"
    assert "exhausted 3 attempt(s)" in repo.issue.comments[0]


def test_blank_error_text_has_a_placeholder_reason():
    assert bot_module._first_line("   ") == "unknown error"


def test_empty_server_url_is_rejected():
    with pytest.raises(BotError, match="must not be empty"):
        build_remote_url("   ", "acme/widgets")


# --------------------------------------------------------------------------- #
# Pull request body
# --------------------------------------------------------------------------- #


def test_pull_request_body_contains_plan_log_and_verification():
    body = build_pull_request_body(7, "Rename old", _success_report())

    assert "Closes #7" in body
    assert "**Issue:** Rename old" in body
    assert "Annotate the helpers." in body
    assert "`[0] add_type_hints: 2 change(s) in app.py`" in body
    assert "ruff" in body and "pytest" in body


def test_pull_request_body_works_without_a_report():
    body = build_pull_request_body(7, "Rename old", None)

    assert "Closes #7" in body
    assert "### Plan" not in body


# --------------------------------------------------------------------------- #
# SubprocessGitClient (real git, local bare remote)
# --------------------------------------------------------------------------- #


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
        env={**os.environ, **_GIT_ENV},
    )
    assert result.returncode == 0, result.stderr
    return result


def _seed_remote(tmp_path: Path) -> Path:
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "--initial-branch=main")
    (seed / "app.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "base")
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", "-q", "--initial-branch=main", str(remote))
    _git(seed, "push", "-q", str(remote), "HEAD:refs/heads/main")
    return remote


def test_subprocess_git_clone_commit_and_push_round_trip(tmp_path):
    remote = _seed_remote(tmp_path)
    client = SubprocessGitClient(git_name="Bot", git_email="bot@example.com")
    clone = tmp_path / "clone"

    client.clone(str(remote), clone, "moray-bot/issue-1", "main")

    assert _git(clone, "symbolic-ref", "--short", "HEAD").stdout.strip() == (
        "moray-bot/issue-1"
    )
    assert _git(clone, "config", "--get", "user.name").stdout.strip() == "Bot"
    assert client.commits_ahead(clone, "main") == 0
    assert client.commit_pending(clone, "nothing to do") is False

    (clone / "app.py").write_text("def f(x) -> None:\n    return x\n", encoding="utf-8")

    assert client.commit_pending(clone, "annotate") is True
    assert client.commits_ahead(clone, "main") == 1

    client.push(clone, "moray-bot/issue-1")

    head = _git(remote, "rev-parse", "--verify", "refs/heads/moray-bot/issue-1")
    assert len(head.stdout.strip()) == 40


def test_subprocess_git_reports_a_missing_remote(tmp_path):
    client = SubprocessGitClient(git_name="Bot", git_email="bot@example.com")

    with pytest.raises(BotError, match="git clone failed"):
        client.clone(
            str(tmp_path / "missing.git"),
            tmp_path / "clone",
            "moray-bot/issue-1",
            "main",
        )


class _LocalRemoteGit(SubprocessGitClient):
    """Subprocess client that clones a local bare repo instead of GitHub."""

    def __init__(self, local_url: Path) -> None:
        super().__init__(git_name="Bot", git_email="bot@example.com")
        self._local_url = local_url

    def clone(self, remote_url, destination, branch, base_branch):
        super().clone(str(self._local_url), destination, branch, base_branch)


class _StaticParser:
    """Minimal IntentParser double: always returns the same plan."""

    def __init__(self, plan: RefactorPlan) -> None:
        self.plan = plan
        self.intents: list[str] = []

    def parse(self, intent, repo_root, extra_context=""):
        self.intents.append(intent)
        return self.plan


class _GreenVerifier:
    """Minimal Verifier double: every gate passes."""

    def verify(self) -> VerifyResult:
        return VerifyResult(ok=True, stage="all")


def test_bot_end_to_end_with_the_real_orchestrator_and_git(tmp_path):
    """Clone -> plan -> apply -> verify -> commit -> push, with real git.

    Only the model, the verifier and the GitHub API are doubles; the AST engine,
    the transaction (which creates the commit) and the git client are real.
    """
    from moray_surgeon.orchestrator import Orchestrator

    remote = _seed_remote(tmp_path)
    issue = FakeIssue(number=11, title="Annotate f", body="Add a return annotation.")
    repo = FakeRepo(issue)
    plan = RefactorPlan(
        instructions=[AddTypeHints(id="s1", target_file="app.py", function_name="f")]
    )
    parser = _StaticParser(plan)

    result = run_github_bot(
        11,
        "acme/widgets",
        config=_config(),
        github=FakeGithub(repo),  # type: ignore[arg-type]
        git=_LocalRemoteGit(remote),
        orchestrator_factory=lambda root: Orchestrator(
            parser,
            _GreenVerifier(),  # type: ignore[arg-type]
            max_attempts=1,
            scope_guard=False,
            differential=False,
        ),
    )

    assert result.success is True
    assert parser.intents == ["Annotate f\n\nAdd a return annotation."]
    pushed = _git(remote, "show", "moray-bot/issue-11:app.py")
    assert "def f(x) -> None:" in pushed.stdout
    assert "Closes #11" in repo.pull_requests[0]["body"]
    assert issue.comments == []


def test_git_auth_travels_through_the_environment_and_is_redacted(
    monkeypatch, tmp_path
):
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs: object):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command, 1, "", "fatal: authentication failed for s3cret-token"
        )

    monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
    monkeypatch.setattr(bot_module.subprocess, "run", fake_run)
    client = SubprocessGitClient(token="s3cret-token", server_url="https://github.com")

    with pytest.raises(BotError) as excinfo:
        client.clone(
            "https://github.com/acme/widgets.git",
            tmp_path / "clone",
            "moray-bot/issue-1",
            "main",
        )

    message = str(excinfo.value)
    assert "s3cret-token" not in message
    assert "***" in message
    command = captured["command"]
    assert isinstance(command, list)
    assert all("s3cret-token" not in str(argument) for argument in command)
    env = captured["env"]
    assert isinstance(env, dict)
    # The caller's two GIT_CONFIG_* entries are preserved; ours is appended.
    assert env["GIT_CONFIG_COUNT"] == "3"
    assert env["GIT_CONFIG_KEY_2"] == "http.https://github.com/.extraheader"
    encoded = env["GIT_CONFIG_VALUE_2"].split()[-1]
    assert base64.b64decode(encoded) == b"x-access-token:s3cret-token"
