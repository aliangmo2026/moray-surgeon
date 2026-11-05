"""Autonomous GitHub issue bot: issue text in, verified pull request out.

The bot is the cloud-side wrapper around the existing engine. It never invents
edits: the model still emits JSON instructions, the AST dispatcher applies them,
and ruff/mypy/pytest decide whether the branch is allowed to exist.

Data flow::

    Issue #N (title + body)                    GitHub REST (PyGithub)
        |
        v
    intent = title + "\\n\\n" + body
        |
        v
    git clone https://github.com/<owner>/<repo>.git   (token via git config env)
        |
        v
    git checkout -B moray-bot/issue-N origin/<default>
        |
        v
    Orchestrator.run(intent, clone)            plan -> apply -> verify -> commit
        |                     \\
        | success              \\ failure
        v                       v
    git commit (no-op if the   Issue comment with VerifyResult.feedback()
    orchestrator committed)    + "needs-human" label (+ optional close)
        |
        v
    git push --force-with-lease origin HEAD:refs/heads/<branch>
        |
        v
    repo.create_pull(base=<default>, head=<branch>)  -> PR URL

Security properties:

* ``GITHUB_TOKEN`` is read from the environment only. It is never written into a
  remote URL, a log line or an exception message: authentication travels through
  ``GIT_CONFIG_*`` environment variables, so it does not appear in ``ps``.
* ``repo_name`` must be exactly ``owner/name``. GitHub server/API URLs are
  validated before any request: http(s) only, no embedded credentials, and no
  localhost/loopback/private/link-local/reserved/multicast address.
* Every git call is a subprocess with a timeout (project rules R3 and R5).

The model endpoint keeps the rules it already has in
:mod:`moray_surgeon.intent.providers` (http/https, no credentials, local Ollama
allowed); the strict address policy above applies to the GitHub endpoints this
module introduces.
"""

from __future__ import annotations

import base64
import ipaddress
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit

from github import Github, GithubException, UnknownObjectException

from moray_surgeon.exceptions import MorayError
from moray_surgeon.intent import IntentParser
from moray_surgeon.orchestrator import Orchestrator, RefactorReport
from moray_surgeon.verify import Verifier

if TYPE_CHECKING:
    from github.Issue import Issue
    from github.Repository import Repository

logger = logging.getLogger(__name__)

TOKEN_ENV = "GITHUB_TOKEN"
SERVER_URL_ENV = "GITHUB_SERVER_URL"
API_URL_ENV = "GITHUB_API_URL"
DEFAULT_SERVER_URL = "https://github.com"
DEFAULT_BRANCH_PREFIX = "moray-bot/issue-"
DEFAULT_FAILURE_LABEL = "needs-human"
FAILURE_LABEL_COLOR = "d73a4a"
DEFAULT_GIT_NAME = "moray-surgeon[bot]"
DEFAULT_GIT_EMAIL = "moray-surgeon[bot]@users.noreply.github.com"
# GitHub issue bodies can be 64 KiB; the model prompt is not a place for that.
MAX_INTENT_CHARS = 12000
# Prefix length of a pull request title before truncation (GitHub allows 256).
_MAX_PR_TITLE_CHARS = 250
_GIT_TIMEOUT_SECONDS = 120
_LOCAL_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")
_REPO_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")

OrchestratorFactory = Callable[[Path], Orchestrator]


class BotError(MorayError):
    """The bot cannot start or complete its work.

    Covers configuration mistakes (missing token, malformed repository name,
    unsafe endpoint), GitHub API failures and rejected git operations. Failures
    *after* the issue was read are reported on the issue and returned as
    ``BotResult(success=False)`` instead of raising.
    """


@dataclass(frozen=True)
class BotConfig:
    """Everything a bot run needs, resolved from the environment.

    Attributes:
        token: GitHub token used for the API and for non-interactive git auth.
        server_url: GitHub web origin used to build the clone URL.
        api_url: Optional REST API root (GitHub Enterprise); None uses
            ``https://api.github.com``.
        base_branch: Pull request base; None uses the repository default branch.
        failure_label: Label applied when the refactor cannot be delivered.
        close_on_failure: Also close the issue instead of only labelling it.
        branch_prefix: Branch prefix; the default yields
            ``moray-bot/issue-<number>`` exactly.
        model: Model name passed to :class:`IntentParser`.
        model_base_url: Model API root (Ollama by default).
        api_key: Model credential, if the dialect needs one.
        api_type: ``ollama``, ``openai`` or ``anthropic``.
        model_timeout: Per-request model timeout in seconds.
        max_attempts: Plan/apply/verify rounds.
        git_name: Commit author/committer name written into the clone.
        git_email: Commit author/committer email written into the clone.
    """

    token: str
    server_url: str = DEFAULT_SERVER_URL
    api_url: str | None = None
    base_branch: str | None = None
    failure_label: str = DEFAULT_FAILURE_LABEL
    close_on_failure: bool = False
    branch_prefix: str = DEFAULT_BRANCH_PREFIX
    model: str = "qwen2.5-coder:7b"
    model_base_url: str = "http://localhost:11434"
    api_key: str | None = None
    api_type: str = "ollama"
    model_timeout: float = 120.0
    max_attempts: int = 3
    git_name: str = DEFAULT_GIT_NAME
    git_email: str = DEFAULT_GIT_EMAIL

    @classmethod
    def from_env(cls, *, close_on_failure: bool | None = None) -> BotConfig:
        """Build a configuration from the process environment.

        Args:
            close_on_failure: Explicit CLI override. None falls back to
                ``MORAY_BOT_CLOSE_ISSUE`` (truthy values: 1/true/yes/on).

        Returns:
            A validated configuration.

        Raises:
            BotError: ``GITHUB_TOKEN`` is missing, an endpoint is unsafe, or a
                numeric variable is malformed.
        """
        token = os.environ.get(TOKEN_ENV, "").strip()
        if not token:
            raise BotError(
                f"{TOKEN_ENV} is required: the bot reads the issue and opens the pull request"
            )
        server = (
            os.environ.get(SERVER_URL_ENV, "").strip() or DEFAULT_SERVER_URL
        )
        server = _validate_remote_url(server, what=SERVER_URL_ENV)
        api = os.environ.get(API_URL_ENV, "").strip() or None
        if api is not None:
            api = _validate_remote_url(api, what=API_URL_ENV)

        if close_on_failure is None:
            close_on_failure = _truthy(os.environ.get("MORAY_BOT_CLOSE_ISSUE", ""))

        return cls(
            token=token,
            server_url=server,
            api_url=api,
            base_branch=os.environ.get("MORAY_BOT_BASE_BRANCH", "").strip() or None,
            failure_label=(
                os.environ.get("MORAY_BOT_FAILURE_LABEL", "").strip()
                or DEFAULT_FAILURE_LABEL
            ),
            close_on_failure=close_on_failure,
            branch_prefix=(
                os.environ.get("MORAY_BOT_BRANCH_PREFIX", "").strip()
                or DEFAULT_BRANCH_PREFIX
            ),
            model=os.environ.get("MORAY_MODEL", "qwen2.5-coder:7b"),
            model_base_url=os.environ.get("MORAY_BASE_URL", "http://localhost:11434"),
            api_key=os.environ.get("MORAY_API_KEY") or None,
            api_type=os.environ.get("MORAY_API_TYPE", "ollama"),
            model_timeout=_env_float("MORAY_TIMEOUT", 120.0),
            max_attempts=_env_int("MORAY_BOT_MAX_ATTEMPTS", 3),
            git_name=os.environ.get("MORAY_BOT_GIT_NAME", "").strip() or DEFAULT_GIT_NAME,
            git_email=os.environ.get("MORAY_BOT_GIT_EMAIL", "").strip() or DEFAULT_GIT_EMAIL,
        )


@dataclass(frozen=True)
class BotResult:
    """Outcome of one bot run.

    Attributes:
        success: True only when the branch was pushed and a pull request exists.
        issue_number: The issue that was processed.
        branch: Branch name the bot used (created and reset from the base).
        pull_request_url: Web URL of the created pull request; empty on failure.
        message: Short human-readable reason on failure, empty on success.
    """

    success: bool
    issue_number: int
    branch: str
    pull_request_url: str = ""
    message: str = ""


class GitClient(Protocol):
    """The git operations the bot needs; injectable so tests stay offline."""

    def clone(
        self, remote_url: str, destination: Path, branch: str, base_branch: str
    ) -> None:
        """Clone ``remote_url`` and check out a fresh ``branch`` from ``base_branch``."""

    def push(self, workdir: Path, branch: str) -> None:
        """Publish ``branch`` to ``origin``, replacing a stale bot branch."""

    def commit_pending(self, workdir: Path, message: str) -> bool:
        """Commit any remaining worktree changes; return whether a commit was made."""

    def commits_ahead(self, workdir: Path, base_branch: str) -> int:
        """Return how many commits ``HEAD`` carries over ``origin/<base_branch>``."""


class SubprocessGitClient:
    """Real :class:`GitClient` built on ``git`` subprocesses.

    Authentication is passed through ``GIT_CONFIG_*`` environment variables
    (``http.<server>/.extraheader``) rather than the command line, so the token
    never shows up in ``ps`` output, in a remote URL, or in a git error message.
    The clone also receives a local ``user.name``/``user.email``: the orchestrator
    commits from its own subprocess, so an identity has to exist in the clone's
    config for CI runners that have no global one.
    """

    def __init__(
        self,
        token: str = "",
        server_url: str = DEFAULT_SERVER_URL,
        git_name: str = DEFAULT_GIT_NAME,
        git_email: str = DEFAULT_GIT_EMAIL,
        timeout: float = _GIT_TIMEOUT_SECONDS,
    ) -> None:
        """Bind authentication, the remote origin and the commit identity.

        Args:
            token: GitHub token; empty disables the authorization header.
            server_url: GitHub origin the token belongs to, used as the git
                config URL prefix for the header.
            git_name: Local ``user.name`` written into every clone.
            git_email: Local ``user.email`` written into every clone.
            timeout: Per-command timeout in seconds.
        """
        self._token = token
        self._server_url = server_url.rstrip("/")
        self._git_name = git_name
        self._git_email = git_email
        self._timeout = timeout

    def clone(
        self, remote_url: str, destination: Path, branch: str, base_branch: str
    ) -> None:
        """Clone and reset ``branch`` onto ``origin/<base_branch>``.

        Raises:
            BotError: git failed (bad URL, unreachable remote, missing base).
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._check(["clone", "--quiet", remote_url, str(destination)])
        self._check(["config", "user.name", self._git_name], cwd=destination)
        self._check(["config", "user.email", self._git_email], cwd=destination)
        self._check(
            ["checkout", "--quiet", "-B", branch, f"origin/{base_branch}"],
            cwd=destination,
        )

    def push(self, workdir: Path, branch: str) -> None:
        """Push ``HEAD`` to ``refs/heads/<branch>`` with a lease.

        A re-run of the same issue replaces the previous bot commit, but never a
        commit somebody pushed after this clone was taken.

        Raises:
            BotError: The remote rejected the push or is unreachable.
        """
        self._check(
            [
                "push",
                "--quiet",
                "--force-with-lease",
                "origin",
                f"HEAD:refs/heads/{branch}",
            ],
            cwd=workdir,
        )

    def commit_pending(self, workdir: Path, message: str) -> bool:
        """Commit everything staged-or-untracked when the clone is dirty.

        The orchestrator normally commits on success, so this is usually a
        no-op (returns False). It exists so a change produced outside the
        transaction still reaches the pull request.
        """
        self._check(["add", "-A"], cwd=workdir)
        diff = self._git(["diff", "--cached", "--quiet"], cwd=workdir)
        if diff.returncode == 0:
            return False
        if diff.returncode != 1:  # pragma: no cover - git only uses 0/1 here
            raise BotError(f"git diff --cached failed: {self._detail(diff)}")
        self._check(["commit", "--quiet", "-m", message], cwd=workdir)
        return True

    def commits_ahead(self, workdir: Path, base_branch: str) -> int:
        """Count the commits between ``origin/<base>`` and ``HEAD``."""
        result = self._check(
            ["rev-list", "--count", f"origin/{base_branch}..HEAD"], cwd=workdir
        )
        try:
            return int(result.stdout.strip())
        except ValueError as exc:  # pragma: no cover - git always prints a number
            raise BotError(f"unexpected rev-list output: {result.stdout!r}") from exc

    def _git(
        self, args: list[str], cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run one git command with the token in the environment, never argv."""
        environment = dict(os.environ)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        if self._token:
            encoded = base64.b64encode(
                f"x-access-token:{self._token}".encode()
            ).decode("ascii")
            # Preserve any GIT_CONFIG_* entries the caller already set (CI and
            # the test harness use them for signing/line-ending policy).
            index = _existing_git_config_count(environment)
            environment["GIT_CONFIG_COUNT"] = str(index + 1)
            environment[f"GIT_CONFIG_KEY_{index}"] = (
                f"http.{self._server_url}/.extraheader"
            )
            environment[f"GIT_CONFIG_VALUE_{index}"] = (
                f"AUTHORIZATION: basic {encoded}"
            )
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self._timeout,
            check=False,
        )

    def _check(
        self, args: list[str], cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run git and raise :class:`BotError` (token-redacted) on failure."""
        result = self._git(args, cwd)
        if result.returncode != 0:
            raise BotError(f"git {args[0]} failed: {self._detail(result)}")
        return result

    def _detail(self, result: subprocess.CompletedProcess[str]) -> str:
        """Return redacted stderr (or stdout) from a failed command."""
        detail = (result.stderr or result.stdout).strip()
        return _redact(detail, self._token) or f"git exited with {result.returncode}"


def run_github_bot(
    issue_number: int,
    repo_name: str,
    *,
    config: BotConfig | None = None,
    github: Github | None = None,
    git: GitClient | None = None,
    work_dir: Path | None = None,
    orchestrator_factory: OrchestratorFactory | None = None,
) -> BotResult:
    """Refactor one issue and deliver it as a pull request.

    Args:
        issue_number: Issue to read and resolve (must be >= 1).
        repo_name: ``owner/name`` of the repository. A full URL is rejected.
        config: Resolved configuration; None reads the environment through
            :meth:`BotConfig.from_env`.
        github: Authenticated PyGithub client; injected by tests.
        git: Git client; injected by tests. Defaults to
            :class:`SubprocessGitClient` with the configured token.
        work_dir: Keep the clone under this directory instead of a temp dir.
        orchestrator_factory: Builds the orchestrator for the clone; injected by
            tests. Defaults to :class:`IntentParser` + :class:`Verifier`.

    Returns:
        A result whose ``success`` is True only after the branch was pushed and
        the pull request was created. Failures after the issue was read also post
        the diagnostics as a comment and apply the failure label.

    Raises:
        BotError: Configuration or GitHub API failure before the work starts
            (missing token, invalid repo name, unreadable repository/issue).
    """
    if issue_number < 1:
        raise BotError(f"issue_number must be positive, got {issue_number}")
    resolved = config or BotConfig.from_env()
    client = github or _connect_github(resolved)
    repo = _get_repo(client, repo_name)
    issue = _get_issue(repo, issue_number)
    intent = build_intent(issue.title, issue.body)
    branch = f"{resolved.branch_prefix}{issue_number}"
    base_branch = resolved.base_branch or repo.default_branch
    remote_url = build_remote_url(resolved.server_url, repo_name)
    git_client = git or SubprocessGitClient(
        token=resolved.token,
        server_url=resolved.server_url,
        git_name=resolved.git_name,
        git_email=resolved.git_email,
    )

    failure: tuple[str, str] | None = None
    report: RefactorReport | None = None
    pushed = False
    try:
        with _workspace(work_dir) as workspace:
            clone_dir = workspace / "repo"
            git_client.clone(remote_url, clone_dir, branch, base_branch)
            orchestrator = (orchestrator_factory or _orchestrator_factory(resolved))(
                clone_dir
            )
            report = orchestrator.run(intent, clone_dir)
            if not report.success:
                failure = _report_failure(report)
            else:
                git_client.commit_pending(
                    clone_dir, f"moray-surgeon: resolve #{issue_number}"
                )
                if git_client.commits_ahead(clone_dir, base_branch) == 0:
                    failure = (
                        "no commit produced",
                        "The refactor verified but produced no commit, so there is "
                        "nothing to review. No branch was pushed.",
                    )
                else:
                    git_client.push(clone_dir, branch)
                    pushed = True
    except Exception as exc:
        # Log the redacted message, never the traceback: a third-party client
        # could put the token in an exception string, and CI logs are public.
        detail = _redact(f"{type(exc).__name__}: {exc}", resolved.token)
        logger.error("moray-surgeon bot run failed: %s", detail)
        failure = (_first_line(detail), detail)

    if failure is not None:
        return _report_on_issue(repo, issue, resolved, branch, *failure)

    try:
        pull_request = repo.create_pull(
            base=base_branch,
            head=branch,
            title=_pull_request_title(issue_number, issue.title),
            body=build_pull_request_body(issue_number, issue.title, report),
        )
    except GithubException as exc:
        detail = (
            f"Branch {branch} was pushed, but creating the pull request failed: "
            f"{_github_error(exc)}"
        )
        return _report_on_issue(
            repo,
            issue,
            resolved,
            branch,
            _first_line(detail),
            detail,
            pushed=pushed,
        )

    logger.info("opened %s for issue #%d", pull_request.html_url, issue_number)
    return BotResult(
        success=True,
        issue_number=issue_number,
        branch=branch,
        pull_request_url=pull_request.html_url,
    )


def build_intent(title: str | None, body: str | None) -> str:
    """Combine an issue title and body into the refactor intent.

    Args:
        title: Issue title (may be empty).
        body: Issue body (GitHub returns None for an empty body).

    Returns:
        ``"<title>\\n\\n<body>"``, the body alone when the title is empty, or the
        title alone when the body is empty. Text longer than
        :data:`MAX_INTENT_CHARS` is truncated with an explicit marker.

    Raises:
        BotError: Both fields are empty, so there is nothing to refactor.
    """
    title_text = (title or "").strip()
    body_text = (body or "").strip()
    if not title_text and not body_text:
        raise BotError("issue has an empty title and body; nothing to refactor")
    intent = (
        f"{title_text}\n\n{body_text}"
        if title_text and body_text
        else title_text or body_text
    )
    if len(intent) > MAX_INTENT_CHARS:
        intent = (
            intent[:MAX_INTENT_CHARS]
            + "\n\n[... issue truncated by the moray-surgeon bot ...]"
        )
    return intent


def build_remote_url(server_url: str, repo_name: str) -> str:
    """Build the https clone URL for ``owner/name`` on ``server_url``.

    Args:
        server_url: GitHub web origin, validated here.
        repo_name: ``owner/name``; anything that could smuggle a host, query or
            fragment is rejected.

    Returns:
        ``<server>/<owner>/<name>.git``.

    Raises:
        BotError: The server URL is unsafe or the repository name is malformed.
    """
    server = _validate_remote_url(server_url, what=SERVER_URL_ENV)
    return f"{server}/{_validate_repo_name(repo_name)}.git"


def build_pull_request_body(
    issue_number: int, issue_title: str, report: RefactorReport | None
) -> str:
    """Render the pull request description from the refactor report."""
    lines = [
        f"Closes #{issue_number}",
        "",
        "## moray-surgeon autonomous refactor",
        "",
        f"**Issue:** {issue_title.strip()}",
        "",
    ]
    if report is not None and report.plan is not None and report.plan.summary.strip():
        lines += ["### Plan", "", report.plan.summary.strip(), ""]
    if report is not None and report.log:
        lines += ["### Applied instructions", ""]
        lines += [f"- `{entry}`" for entry in report.log]
        lines.append("")
    lines += [
        "### Verification",
        "",
        "`ruff`, `mypy` and `pytest` passed in the cloned repository before this",
        "branch was pushed. The model produced JSON instructions only; every edit",
        "was applied and scope-checked by the AST engine.",
    ]
    if report is not None and report.architecture_diff.strip():
        lines += [
            "",
            "### Architecture diff",
            "",
            "````text",
            report.architecture_diff.strip(),
            "````",
        ]
    return "\n".join(lines).strip() + "\n"


def _connect_github(config: BotConfig) -> Github:
    """Create the PyGithub client; the API root must have been validated."""
    if config.api_url:
        return Github(login_or_token=config.token, base_url=config.api_url)
    return Github(login_or_token=config.token)


def _get_repo(client: Github, repo_name: str) -> Repository:
    """Fetch ``owner/name``, translating API failures into :class:`BotError`."""
    validated = _validate_repo_name(repo_name)
    try:
        return client.get_repo(validated)
    except GithubException as exc:
        raise BotError(f"cannot read repository {validated!r}: {_github_error(exc)}") from exc


def _get_issue(repo: Repository, issue_number: int) -> Issue:
    """Fetch one issue, translating API failures into :class:`BotError`."""
    try:
        return repo.get_issue(issue_number)
    except GithubException as exc:
        raise BotError(f"cannot read issue #{issue_number}: {_github_error(exc)}") from exc


def _report_failure(report: RefactorReport) -> tuple[str, str]:
    """Turn a failed refactor report into (short reason, comment detail)."""
    verify = report.last_verify
    if verify is None:
        return (
            "no verified plan",
            f"moray-surgeon exhausted {report.attempts} attempt(s) without producing "
            "a plan that passed verification.",
        )
    return (f"verification failed at {verify.stage}", verify.feedback())


def _report_on_issue(
    repo: Repository,
    issue: Issue,
    config: BotConfig,
    branch: str,
    reason: str,
    detail: str,
    *,
    pushed: bool = False,
) -> BotResult:
    """Comment the diagnostics, label the issue and optionally close it."""
    reason = _redact(reason, config.token)
    detail = _redact(detail, config.token)
    try:
        issue.create_comment(_failure_comment(reason, detail, branch, pushed))
        _ensure_label(repo, issue, config.failure_label)
        if config.close_on_failure:
            issue.edit(state="closed")
    except GithubException as exc:
        raise BotError(
            f"could not report the failure on issue #{issue.number}: {_github_error(exc)}"
        ) from exc
    return BotResult(
        success=False,
        issue_number=issue.number,
        branch=branch,
        message=reason[:200],
    )


def _failure_comment(reason: str, detail: str, branch: str, pushed: bool) -> str:
    """Render the issue comment posted when the refactor cannot be delivered."""
    outcome = (
        f"The branch `{branch}` was pushed, but the pull request could not be opened."
        if pushed
        else "No branch was pushed and no pull request was opened."
    )
    return (
        "**moray-surgeon could not complete this refactor.**\n\n"
        f"Reason: {reason}\n\n"
        f"{outcome}\n\n"
        "````text\n"
        f"{detail}\n"
        "````\n"
    )


def _ensure_label(repo: Repository, issue: Issue, label: str) -> None:
    """Create the failure label when missing, then attach it to the issue."""
    try:
        repo.get_label(label)
    except UnknownObjectException:
        repo.create_label(label, FAILURE_LABEL_COLOR)
    issue.add_to_labels(label)


def _pull_request_title(issue_number: int, issue_title: str) -> str:
    """Build a bounded PR title that links the issue."""
    title = f"moray-surgeon: resolve #{issue_number} - {issue_title.strip()}"
    if len(title) <= _MAX_PR_TITLE_CHARS:
        return title
    return title[: _MAX_PR_TITLE_CHARS - 3] + "..."


def _orchestrator_factory(config: BotConfig) -> OrchestratorFactory:
    """Bind the production parser/verifier pair to a clone directory."""

    def build(repo_root: Path) -> Orchestrator:
        parser = IntentParser(
            model=config.model,
            base_url=config.model_base_url,
            timeout=config.model_timeout,
            api_key=config.api_key,
            api_type=config.api_type,
        )
        return Orchestrator(parser, Verifier(repo_root), max_attempts=config.max_attempts)

    return build


@contextmanager
def _workspace(work_dir: Path | None) -> Iterator[Path]:
    """Yield a directory to clone into; a temp dir is removed afterwards."""
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="moray-bot-") as temporary:
            yield Path(temporary)
        return
    root = work_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    yield root


def _validate_repo_name(repo_name: str) -> str:
    """Accept only ``owner/name``; reject URLs, traversal and option injection."""
    if not repo_name:
        raise BotError("repository name is required (owner/name)")
    parts = repo_name.split("/")
    if (
        len(parts) != 2
        or any(part in {"", ".", ".."} for part in parts)
        or len(repo_name) > 200
        or _REPO_NAME_RE.fullmatch(repo_name) is None
    ):
        raise BotError(f"repository name must be 'owner/name', got {repo_name!r}")
    return repo_name


def _validate_remote_url(url: str, *, what: str) -> str:
    """Validate a GitHub endpoint before any request is made.

    Only http/https is accepted. Credentials, queries and fragments are
    rejected, as are local names and any literal IP address that is not global
    (loopback, private, link-local, reserved, multicast, unspecified). Host
    names are not resolved: the operator sets these variables, and the literal
    checks cover the addresses a hostile payload could spell directly.

    Args:
        url: Candidate endpoint.
        what: Environment variable name used in the error message.

    Returns:
        The URL without a trailing slash.

    Raises:
        BotError: The URL is missing, malformed or points somewhere unsafe.
    """
    candidate = (url or "").strip()
    if not candidate:
        raise BotError(f"{what} must not be empty")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise BotError(f"{what} must use http or https, got {candidate!r}")
    if parsed.username or parsed.password:
        raise BotError(f"{what} must not embed credentials")
    if parsed.query or parsed.fragment:
        raise BotError(f"{what} must not carry a query string or fragment")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise BotError(f"{what} has an invalid port: {candidate!r}") from exc
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise BotError(f"{what} must name a host")
    if host == "localhost" or host.endswith(_LOCAL_HOST_SUFFIXES):
        raise BotError(f"{what} must not point at a local host: {host!r}")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return candidate.rstrip("/")
    unsafe = (
        not address.is_global
        or address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )
    if unsafe:
        raise BotError(f"{what} must be a public address, got {host!r}")
    return candidate.rstrip("/")


def _github_error(exc: GithubException) -> str:
    """Render a PyGithub exception without leaking anything sensitive."""
    data = exc.data
    message = data.get("message", data) if isinstance(data, dict) else data
    return f"HTTP {exc.status}: {message}"


def _existing_git_config_count(environment: dict[str, str]) -> int:
    """Return the caller's ``GIT_CONFIG_COUNT`` so new entries can be appended."""
    try:
        return max(0, int(environment.get("GIT_CONFIG_COUNT", "0") or "0"))
    except ValueError:  # pragma: no cover - git itself would reject this
        return 0


def _redact(text: str, token: str) -> str:
    """Remove ``token`` from text that may be logged or posted publicly."""
    if not token:
        return text
    return text.replace(token, "***")


def _first_line(text: str) -> str:
    """Return a short single-line summary for the result and comment header."""
    stripped = text.strip()
    if not stripped:
        return "unknown error"
    return stripped.splitlines()[0][:200]


def _truthy(value: str) -> bool:
    """Interpret the usual environment-variable booleans."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read a positive integer variable, rejecting malformed values loudly."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise BotError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise BotError(f"{name} must be >= 1, got {value}")
    return value


def _env_float(name: str, default: float) -> float:
    """Read a positive float variable, rejecting malformed values loudly."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise BotError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise BotError(f"{name} must be > 0, got {value}")
    return value
