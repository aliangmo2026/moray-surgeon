"""Tests for historical rewrites: branch isolation, replay and failure paths.

Every test uses a real temporary git repository and the real orchestrator (with a
static plan and a green verifier), so the worktree, the transaction commit, the
rebase replay and the cleanup are exercised end to end. No model is called.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from moray_surgeon.exceptions import TimeTravelError
from moray_surgeon.orchestrator import Orchestrator
from moray_surgeon.schema import AddTypeHints, RefactorPlan
from moray_surgeon.time_travel import (
    SkipVerifier,
    TimeTravelResult,
    _github_slug,
    open_time_travel_pull_request,
    push_rewritten_branch,
    run_time_travel,
)
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


@pytest.fixture(autouse=True)
def _isolated_git_environment(monkeypatch):
    for name, value in _GIT_ENV.items():
        monkeypatch.setenv(name, value)


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result


def _rev(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", ref).stdout.strip()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _rev(repo, "HEAD")


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "--initial-branch=main")
    return repo


def _history_repo(tmp_path: Path) -> Path:
    """Three commits: base, an unrelated file, and an append that replays cleanly."""
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "def old_name():\n    return 1\n")
    _commit(repo, "base")
    _write(repo / "util.py", "VALUE = 2\n")
    _commit(repo, "add util")
    with (repo / "app.py").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n\ndef later():\n    return 3\n")
    _commit(repo, "append later")
    return repo


class _StaticParser:
    """IntentParser double returning one plan and recording the intent."""

    def __init__(self, plan: RefactorPlan) -> None:
        self.plan = plan
        self.intents: list[str] = []

    def parse(self, intent, repo_root, extra_context=""):
        self.intents.append(intent)
        return self.plan


class _GreenVerifier:
    """Verifier double: every gate passes without running a tool."""

    def verify(self) -> VerifyResult:
        return VerifyResult(ok=True, stage="all")


def _factory(plan: RefactorPlan):
    parser = _StaticParser(plan)

    def build(worktree: Path) -> Orchestrator:
        return Orchestrator(
            parser,
            _GreenVerifier(),  # type: ignore[arg-type]
            max_attempts=1,
            scope_guard=False,
            differential=False,
        )

    return build, parser


def _hints_plan(function_name: str = "old_name") -> RefactorPlan:
    return RefactorPlan(
        instructions=[
            AddTypeHints(id="s1", target_file="app.py", function_name=function_name)
        ]
    )


def test_skip_verifier_reports_green_without_running_tools():
    assert SkipVerifier().verify().ok is True


def test_rewrite_creates_a_new_branch_and_preserves_the_original(tmp_path):
    repo = _history_repo(tmp_path)
    original_tip = _rev(repo, "main")
    factory, parser = _factory(_hints_plan())

    result = run_time_travel(
        repo, "main", 2, "annotate old_name", orchestrator_factory=factory
    )

    assert isinstance(result, TimeTravelResult)
    assert result.branch == "main"
    assert result.new_branch == "moray-time-travel/main"
    assert result.base_commit == _rev(repo, "main~2")
    assert result.original_tip == original_tip
    assert result.requested_steps == 2
    assert result.replayed_commits == 2
    assert parser.intents == ["annotate old_name"]

    # The original branch and the user's working tree are untouched.
    assert _rev(repo, "main") == original_tip
    assert "-> None" not in (repo / "app.py").read_text(encoding="utf-8")
    assert _git(repo, "status", "--porcelain").stdout == ""

    # New history: base -> refactor -> add util -> append later.
    assert _rev(repo, result.new_branch) == result.rewritten_tip
    assert _rev(repo, f"{result.refactor_commit}^") == result.base_commit
    messages = _git(
        repo, "log", "--format=%s", "--reverse", f"{result.base_commit}..{result.new_branch}"
    ).stdout.splitlines()
    assert len(messages) == 3
    assert messages[0].startswith("moray-surgeon:")
    assert messages[1] == "add util"
    assert messages[2] == "append later"

    # The refactor is present at the tip and in the first replayed commit.
    tip_app = _git(repo, "show", f"{result.new_branch}:app.py").stdout
    assert "def old_name() -> None:" in tip_app
    assert "def later():" in tip_app
    first_replay = _rev(repo, f"{result.new_branch}~1")
    assert "-> None" in _git(repo, "show", f"{first_replay}:app.py").stdout

    # The temporary worktree is gone; only the user's checkout remains.
    worktrees = _git(repo, "worktree", "list", "--porcelain").stdout
    assert "moray-time-travel-" not in worktrees
    tracked = _git(repo, "ls-tree", "-r", "--name-only", result.new_branch).stdout
    assert ".moray-cache" not in tracked
    assert "moray-time-travel" not in _git(repo, "branch", "--list", "moray-tt-source-*").stdout


def test_a_conflict_aborts_and_leaves_the_original_history_untouched(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "def value():\n    return 1\n")
    _commit(repo, "base")
    _write(repo / "app.py", "def value():  # v2\n    return 2\n")
    _commit(repo, "rewrite the signature line")
    original_tip = _rev(repo, "main")
    factory, _ = _factory(_hints_plan("value"))

    with pytest.raises(TimeTravelError, match="conflict"):
        run_time_travel(repo, "main", 1, "annotate value", orchestrator_factory=factory)

    assert _rev(repo, "main") == original_tip
    assert _git(repo, "show", "main:app.py").stdout == "def value():  # v2\n    return 2\n"
    missing = _git(
        repo,
        "rev-parse",
        "--verify",
        "--quiet",
        "refs/heads/moray-time-travel/main",
        check=False,
    )
    assert missing.returncode != 0
    assert _git(repo, "branch", "--list", "moray-tt-source-*").stdout.strip() == ""
    worktrees = _git(repo, "worktree", "list", "--porcelain").stdout
    assert "moray-time-travel-" not in worktrees


def test_too_many_steps_is_rejected_before_any_work(tmp_path):
    repo = _history_repo(tmp_path)
    original_tip = _rev(repo, "main")
    factory, parser = _factory(_hints_plan())

    with pytest.raises(TimeTravelError, match="fewer than"):
        run_time_travel(repo, "main", 99, "annotate", orchestrator_factory=factory)

    assert parser.intents == []
    assert _rev(repo, "main") == original_tip
    assert _git(repo, "branch", "--list", "moray-time-travel/*").stdout.strip() == ""


def test_merge_commits_in_the_window_are_rejected(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "def f():\n    return 1\n")
    _commit(repo, "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo / "feature.py", "VALUE = 1\n")
    _commit(repo, "feature work")
    _git(repo, "checkout", "-q", "main")
    _write(repo / "main.py", "OTHER = 2\n")
    _commit(repo, "main work")
    _git(repo, "merge", "-q", "--no-ff", "feature", "-m", "merge feature")
    factory, parser = _factory(_hints_plan("f"))

    with pytest.raises(TimeTravelError, match="merge commits"):
        run_time_travel(repo, "main", 2, "annotate", orchestrator_factory=factory)

    assert parser.intents == []
    assert _git(repo, "branch", "--list", "moray-time-travel/*").stdout.strip() == ""


def test_an_existing_rewrite_branch_requires_force(tmp_path):
    repo = _history_repo(tmp_path)
    _git(repo, "branch", "moray-time-travel/main", _rev(repo, "main~2"))
    factory, parser = _factory(_hints_plan())

    with pytest.raises(TimeTravelError, match="already exists"):
        run_time_travel(repo, "main", 2, "annotate", orchestrator_factory=factory)

    assert parser.intents == []
    assert _rev(repo, "moray-time-travel/main") == _rev(repo, "main~2")

    result = run_time_travel(
        repo, "main", 2, "annotate", orchestrator_factory=factory, force=True
    )
    assert result.new_branch == "moray-time-travel/main"
    assert _rev(repo, result.new_branch) == result.rewritten_tip
    assert result.rewritten_tip != _rev(repo, "main~2")


def test_a_dirty_user_worktree_is_left_alone(tmp_path):
    repo = _history_repo(tmp_path)
    with (repo / "app.py").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n# local edit\n")
    factory, _ = _factory(_hints_plan())

    run_time_travel(repo, "main", 2, "annotate", orchestrator_factory=factory)

    content = (repo / "app.py").read_text(encoding="utf-8")
    assert "# local edit" in content
    assert "-> None" not in content
    assert " M app.py" in _git(repo, "status", "--porcelain").stdout


def test_repository_hooks_do_not_block_the_temporary_commit(tmp_path):
    repo = _history_repo(tmp_path)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_bytes(b"#!/bin/sh\nexit 1\n")
    os.chmod(hook, 0o755)
    factory, _ = _factory(_hints_plan())

    result = run_time_travel(
        repo, "main", 2, "annotate", orchestrator_factory=factory
    )

    assert result.replayed_commits == 2


def test_push_publishes_the_rewritten_branch(tmp_path):
    repo = _history_repo(tmp_path)
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", "-q", "--initial-branch=main", str(remote))
    _git(repo, "remote", "add", "origin", str(remote))
    factory, _ = _factory(_hints_plan())

    result = run_time_travel(repo, "main", 2, "annotate", orchestrator_factory=factory)
    push_rewritten_branch(repo, "origin", result.new_branch)

    assert _git(remote, "rev-parse", f"refs/heads/{result.new_branch}").stdout.strip() == (
        result.rewritten_tip
    )


def test_install_rejects_a_non_git_repository(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(TimeTravelError, match="git working tree"):
        run_time_travel(
            plain, "main", 1, "annotate", orchestrator_factory=_factory(_hints_plan())[0]
        )


def test_detached_head_requires_an_explicit_branch(tmp_path):
    repo = _history_repo(tmp_path)
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    factory, _ = _factory(_hints_plan())

    with pytest.raises(TimeTravelError, match="detached HEAD"):
        run_time_travel(repo, None, 1, "annotate", orchestrator_factory=factory)


def test_invalid_branch_name_is_rejected(tmp_path):
    repo = _history_repo(tmp_path)
    factory, _ = _factory(_hints_plan())

    with pytest.raises(TimeTravelError, match="invalid branch name"):
        run_time_travel(repo, "--upload-pack=evil", 1, "annotate", orchestrator_factory=factory)


# --------------------------------------------------------------------------- #
# Pull request helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url,slug",
    [
        ("https://github.com/acme/widgets.git", "acme/widgets"),
        ("https://github.com/acme/widgets", "acme/widgets"),
        ("git@github.com:acme/widgets.git", "acme/widgets"),
        ("ssh://git@github.com/acme/widgets.git", "acme/widgets"),
    ],
)
def test_github_slug_parses_supported_remotes(url, slug):
    assert _github_slug(url) == slug


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/acme/widgets.git",
        "https://user:secret@github.com/acme/widgets.git",
        "ftp://github.com/acme/widgets.git",
        "https://github.com/acme/widgets?ref=x",
        "https://github.com/only-one-segment",
        "git@gitlab.com:acme/widgets.git",
        "/tmp/local-remote.git",
    ],
)
def test_github_slug_rejects_unsupported_remotes(url):
    with pytest.raises(TimeTravelError):
        _github_slug(url)


class _FakePullRequest:
    html_url = "https://github.com/acme/widgets/pull/9"


class _FakeRepository:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def create_pull(self, base, head, *, title, body):
        self.calls.append({"base": base, "head": head, "title": title, "body": body})
        return _FakePullRequest()


class _FakeGithub:
    def __init__(self) -> None:
        self.slug: str | None = None
        self.repository = _FakeRepository()

    def get_repo(self, slug: str):
        self.slug = slug
        return self.repository


def _pr_result() -> TimeTravelResult:
    return TimeTravelResult(
        branch="main",
        new_branch="moray-time-travel/main",
        base_commit="a" * 40,
        original_tip="b" * 40,
        refactor_commit="c" * 40,
        rewritten_tip="d" * 40,
        requested_steps=2,
        replayed_commits=2,
    )


def test_pull_request_uses_the_github_remote_and_explains_the_force_push(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "VALUE = 1\n")
    _commit(repo, "base")
    _git(repo, "remote", "add", "origin", "git@github.com:acme/widgets.git")
    fake = _FakeGithub()

    url = open_time_travel_pull_request(
        repo,
        remote="origin",
        base_branch="main",
        head_branch="moray-time-travel/main",
        intent="annotate",
        result=_pr_result(),
        token="token",
        github_factory=lambda token: fake,  # type: ignore[arg-type]
    )

    assert fake.slug == "acme/widgets"
    assert url == "https://github.com/acme/widgets/pull/9"
    call = fake.repository.calls[0]
    assert call["base"] == "main"
    assert call["head"] == "moray-time-travel/main"
    assert "git push --force-with-lease origin moray-time-travel/main:main" in call["body"]
    assert "rewrite" in call["title"]


def test_pull_request_requires_a_token(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "VALUE = 1\n")
    _commit(repo, "base")
    _git(repo, "remote", "add", "origin", "git@github.com:acme/widgets.git")

    with pytest.raises(TimeTravelError, match="GITHUB_TOKEN"):
        open_time_travel_pull_request(
            repo,
            remote="origin",
            base_branch="main",
            head_branch="moray-time-travel/main",
            intent="annotate",
            result=_pr_result(),
            token="",
        )


def test_pull_request_rejects_a_non_github_remote(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "VALUE = 1\n")
    _commit(repo, "base")
    _git(repo, "remote", "add", "origin", "https://gitlab.com/acme/widgets.git")

    with pytest.raises(TimeTravelError, match=r"only github\.com"):
        open_time_travel_pull_request(
            repo,
            remote="origin",
            base_branch="main",
            head_branch="moray-time-travel/main",
            intent="annotate",
            result=_pr_result(),
            token="token",
            github_factory=lambda token: _FakeGithub(),  # type: ignore[arg-type]
        )
