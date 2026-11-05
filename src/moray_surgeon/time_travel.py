"""Rewrite a window of historical commits on a new branch, safely.

History rewriting is the one operation that can destroy work, so this module is
built around a hard rule: **the original branch and the user's working tree are
never touched**. The rewrite happens in a temporary linked worktree on a new
branch; every failure path removes that worktree and the new branch.

The flow for ``moray-surgeon time-travel --branch main --steps N --intent ...``:

1. Resolve the branch tip and its ``N``-commit ancestor (the base). Refuse merge
   commits inside the window: replaying merges is a different, much riskier
   problem.
2. ``git worktree add -b moray-time-travel/<branch> <tmp> <base>``. The user's
   checkout and index are untouched.
3. Run the :class:`~moray_surgeon.orchestrator.Orchestrator` inside the worktree
   (plan → AST apply → verify → commit). Repository hooks are disabled for the
   orchestrator's own commits: the orchestrator's gates are the gate here, and a
   user's pre-commit hook would otherwise audit the temporary checkout too.
4. Replay the original ``base..tip`` commits on top of the refactor commit with a
   non-interactive ``git rebase --onto``. A conflict aborts the rebase, deletes
   the new branch and leaves the original history intact.
5. Move the new branch to the rewritten tip and remove the worktree.

Push and pull-request creation are separate, explicit steps:
:func:`push_rewritten_branch` and :func:`open_time_travel_pull_request`. A
rewritten history cannot be merged normally; the PR is for review, and a
maintainer force-pushes after approval (the PR body says so).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import uuid4

from github import Github, GithubException

from moray_surgeon.exceptions import MorayError, TimeTravelError
from moray_surgeon.verify import Verifier, VerifyResult

if TYPE_CHECKING:
    from moray_surgeon.orchestrator import Orchestrator

    OrchestratorFactory = Callable[[Path], Orchestrator]

_GIT_TIMEOUT_SECONDS = 180
_NEW_BRANCH_PREFIX = "moray-time-travel/"
_GITHUB_HOSTS = {"github.com", "www.github.com"}
_REMOTE_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
_SLUG_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TimeTravelResult:
    """Outcome of one successful history rewrite.

    Attributes:
        branch: Original branch whose history was rewritten.
        new_branch: Branch holding the rewritten history in the same repository.
        base_commit: Commit the refactor started from (``branch~steps``).
        original_tip: The original branch tip (never modified).
        refactor_commit: The commit produced by the orchestrator on top of base.
        rewritten_tip: Tip of ``new_branch`` after replaying the window.
        requested_steps: Number of commits the caller asked to rewrite.
        replayed_commits: Commits actually replayed (empty ones are dropped).
    """

    branch: str
    new_branch: str
    base_commit: str
    original_tip: str
    refactor_commit: str
    rewritten_tip: str
    requested_steps: int
    replayed_commits: int


class SkipVerifier(Verifier):
    """Verifier stand-in for ``time-travel --skip-verify``.

    Historical checkouts often fail today's ruff/mypy/pytest gates for reasons
    unrelated to the intent. This verifier reports a green result without running
    anything; the AST scope gate still applies, and the caller is told that the
    verification gate was skipped. It is never the default.
    """

    def __init__(self) -> None:
        """Bind to the current directory without touching the filesystem."""
        super().__init__(Path("."))

    def verify(self) -> VerifyResult:
        """Report every gate as passed without executing a single tool."""
        return VerifyResult(ok=True, stage="all")


def run_time_travel(
    repo_root: Path,
    branch: str | None,
    steps: int,
    intent: str,
    *,
    orchestrator_factory: OrchestratorFactory,
    force: bool = False,
) -> TimeTravelResult:
    """Apply ``intent`` to the base of the last ``steps`` commits and replay them.

    Args:
        repo_root: Any path inside the target repository.
        branch: Branch to rewrite; None uses the current branch (fails on a
            detached HEAD).
        steps: Number of most recent commits to rewrite; must be at least 1 and
            the branch must have a commit before the window.
        intent: Natural-language refactor request handed to the orchestrator.
        orchestrator_factory: Builds an orchestrator for the temporary worktree.
            Injected by tests; the CLI wires the real model parser and verifier.
        force: Replace an existing ``moray-time-travel/<branch>`` branch.

    Returns:
        A summary of the new branch and the rewritten range.

    Raises:
        TimeTravelError: Any precondition failed, the refactor did not verify, or
            replaying a commit conflicted. In every case the original branch and
            working tree are untouched and the temporary branch/worktree is gone.
    """
    if steps < 1:
        raise TimeTravelError(f"steps must be at least 1, got {steps}")
    if not intent.strip():
        raise TimeTravelError("intent must not be empty")
    root = _repo_root(repo_root)
    target_branch = branch if branch is not None else _current_branch(root)
    _validate_branch_name(target_branch)
    tip = _rev(root, target_branch, f"branch {target_branch!r} does not exist")
    base = _rev(
        root,
        f"{tip}~{steps}",
        f"branch {target_branch!r} has fewer than {steps} commit(s) before it",
    )
    merge_commits = _git(root, "rev-list", "--merges", f"{base}..{tip}").stdout.strip()
    if merge_commits:
        raise TimeTravelError(
            "the selected window contains merge commits; refusing to flatten "
            "history. Rewrite a linear range or rebase manually."
        )
    window = int(_git(root, "rev-list", "--count", f"{base}..{tip}").stdout.strip())
    if window != steps:  # pragma: no cover - rev-list is authoritative
        raise TimeTravelError(f"expected {steps} commit(s) in the window, found {window}")

    new_branch = f"{_NEW_BRANCH_PREFIX}{target_branch}"
    _validate_branch_name(new_branch)
    if _branch_exists(root, new_branch):
        if not force:
            raise TimeTravelError(
                f"branch {new_branch!r} already exists; re-run with --force to replace it"
            )
        if _branch_checked_out(root, new_branch):
            raise TimeTravelError(
                f"branch {new_branch!r} is checked out in a worktree; refusing to replace it"
            )
        _git(root, "branch", "-D", new_branch)

    workspace = Path(tempfile.mkdtemp(prefix="moray-time-travel-"))
    worktree = workspace / "repo"
    empty_hooks = workspace / "hooks-disabled"
    empty_hooks.mkdir()
    excludes_file = workspace / "git-excludes"
    # The code index is created before the transaction, and in a linked worktree
    # the indexer's exclude write lands in the per-worktree gitdir, which git does
    # not read for ignore rules. Ignore it explicitly for this worktree so the
    # transaction's `git stash -u` cannot capture the SQLite cache.
    excludes_file.write_text("/.moray-cache/\n", encoding="utf-8")
    worktree_added = False
    success = False
    temp_source: str | None = None
    try:
        _git(root, "worktree", "add", "--quiet", "-b", new_branch, str(worktree), base)
        worktree_added = True
        with _isolated_git_config(empty_hooks, excludes_file):
            orchestrator = orchestrator_factory(worktree)
            try:
                report = orchestrator.run(intent, worktree)
            except MorayError as exc:
                raise TimeTravelError(
                    f"the refactor failed; history left untouched: {exc}"
                ) from exc
            if not report.success:
                verify = report.last_verify
                detail = (
                    verify.feedback()
                    if verify is not None
                    else f"no verified plan after {report.attempts} attempt(s)"
                )
                raise TimeTravelError(
                    f"the refactor did not verify; history left untouched.\n{detail}"
                )
            refactor_commit = _rev(worktree, "HEAD", "the worktree has no HEAD")
            if refactor_commit == base:
                raise TimeTravelError(
                    "the orchestrator reported success but produced no commit; "
                    "history left untouched"
                )

            temp_source = f"moray-tt-source-{uuid4().hex[:12]}"
            _git(root, "branch", temp_source, tip)
            replay = _git(
                worktree,
                "rebase",
                "--onto",
                refactor_commit,
                base,
                temp_source,
                check=False,
            )
            if replay.returncode != 0:
                _git(worktree, "rebase", "--abort", check=False)
                detail = (replay.stderr or replay.stdout).strip()
                raise TimeTravelError(
                    "replaying the history stopped (conflict or empty commit); "
                    f"aborted and left {target_branch!r} untouched:\n{detail}"
                )

        rewritten_tip = _rev(worktree, "HEAD", "the replayed history has no HEAD")
        _git(root, "update-ref", f"refs/heads/{new_branch}", rewritten_tip)
        _git(worktree, "checkout", "--quiet", new_branch)
        _git(root, "branch", "-D", temp_source)
        temp_source = None
        replayed = int(
            _git(
                root, "rev-list", "--count", f"{refactor_commit}..{rewritten_tip}"
            ).stdout.strip()
        )
        success = True
        return TimeTravelResult(
            branch=target_branch,
            new_branch=new_branch,
            base_commit=base,
            original_tip=tip,
            refactor_commit=refactor_commit,
            rewritten_tip=rewritten_tip,
            requested_steps=steps,
            replayed_commits=replayed,
        )
    finally:
        # Remove the worktree first: the failed rebase leaves the temporary
        # source branch checked out there, and git refuses to delete a branch
        # that is checked out in any worktree.
        if worktree_added:
            _git(root, "worktree", "remove", "--force", str(worktree), check=False)
        shutil.rmtree(workspace, ignore_errors=True)
        _git(root, "worktree", "prune", check=False)
        if temp_source is not None:
            _git(root, "branch", "-D", temp_source, check=False)
        if not success:
            _git(root, "branch", "-D", new_branch, check=False)


def push_rewritten_branch(repo_root: Path, remote: str, branch: str) -> None:
    """Push a rewritten branch to ``remote`` without forcing.

    Args:
        repo_root: Any path inside the repository.
        remote: Remote name (``origin`` by default in the CLI).
        branch: Local branch created by :func:`run_time_travel`.

    Raises:
        TimeTravelError: The remote is invalid, the branch is missing, or the push
            was rejected (for example because the remote branch already exists
            with different history). The local branch is never deleted.
    """
    root = _repo_root(repo_root)
    _validate_remote_name(remote)
    _validate_branch_name(branch)
    _git(root, "push", "--quiet", remote, f"refs/heads/{branch}:refs/heads/{branch}")


def open_time_travel_pull_request(
    repo_root: Path,
    *,
    remote: str,
    base_branch: str,
    head_branch: str,
    intent: str,
    result: TimeTravelResult,
    token: str,
    github_factory: Callable[[str], Github] | None = None,
) -> str:
    """Open a review pull request for a rewritten branch.

    The PR cannot rewrite ``base_branch`` by itself; its body states that a
    maintainer must force-push after approval. Only ``github.com`` remotes are
    supported; the remote is parsed for ``owner/repo`` and never fetched.

    Args:
        repo_root: Any path inside the repository.
        remote: Remote name holding the GitHub URL.
        base_branch: Branch the PR targets (the original branch).
        head_branch: Rewritten branch, already pushed.
        intent: Original refactor intent, quoted in the PR body.
        result: Rewrite summary.
        token: GitHub token; callers read ``GITHUB_TOKEN``.
        github_factory: Builds the PyGithub client; injected by tests.

    Returns:
        The pull request's web URL.

    Raises:
        TimeTravelError: The token is missing, the remote is not a GitHub remote,
            or the API rejected the request.
    """
    if not token.strip():
        raise TimeTravelError(
            "GITHUB_TOKEN is required to open the review pull request"
        )
    root = _repo_root(repo_root)
    _validate_remote_name(remote)
    remote_url = _git(root, "remote", "get-url", remote).stdout.strip()
    slug = _github_slug(remote_url)
    client = (github_factory or Github)(token)
    try:
        repository = client.get_repo(slug)
        pull_request = repository.create_pull(
            base=base_branch,
            head=head_branch,
            title=(
                "moray-surgeon time-travel: rewrite "
                f"{result.replayed_commits} commit(s) of {base_branch}"
            ),
            body=_pull_request_body(remote, base_branch, head_branch, intent, result),
        )
    except GithubException as exc:
        raise TimeTravelError(f"could not open the pull request: {exc}") from exc
    return str(pull_request.html_url)


def _pull_request_body(
    remote: str,
    base_branch: str,
    head_branch: str,
    intent: str,
    result: TimeTravelResult,
) -> str:
    """Render the PR body, including the force-push that actually lands it."""
    return "\n".join(
        [
            f"## Rewritten history for `{base_branch}`",
            "",
            f"Intent: {intent.strip()}",
            "",
            f"- base commit: `{result.base_commit}`",
            f"- original tip: `{result.original_tip}`",
            f"- refactor commit: `{result.refactor_commit}`",
            f"- rewritten tip: `{result.rewritten_tip}`",
            f"- replayed commits: {result.replayed_commits} of {result.requested_steps}",
            "",
            "This branch exists for review. A normal merge cannot rewrite the base",
            "branch's history; after approval a maintainer force-pushes it:",
            "",
            "```bash",
            f"git push --force-with-lease {remote} {head_branch}:{base_branch}",
            "```",
        ]
    )


def _github_slug(remote_url: str) -> str:
    """Return ``owner/repo`` for a github.com remote; reject everything else."""
    raw = remote_url.strip()
    if not raw:
        raise TimeTravelError("the remote has no URL")
    if "://" in raw:
        parsed = urlsplit(raw)
        if parsed.scheme not in {"https", "ssh"}:
            raise TimeTravelError(
                f"unsupported remote scheme {parsed.scheme!r}; use https or ssh"
            )
        if parsed.scheme == "https" and (parsed.username or parsed.password):
            raise TimeTravelError("the remote URL must not embed credentials")
        if parsed.query or parsed.fragment:
            raise TimeTravelError("the remote URL must not carry a query or fragment")
        host = (parsed.hostname or "").lower()
        path = parsed.path
    else:
        host_part, separator, path = raw.partition(":")
        if not separator:
            raise TimeTravelError(f"cannot parse remote URL {raw!r}")
        host = host_part.rsplit("@", 1)[-1].lower()
    if host not in _GITHUB_HOSTS:
        raise TimeTravelError(
            f"only github.com remotes can open a pull request; {host!r} is not "
            "supported. Push the branch and open the PR manually."
        )
    slug = path.strip("/")
    if slug.endswith(".git"):
        slug = slug[:-4]
    if slug.count("/") != 1 or not _SLUG_RE.fullmatch(slug) or ".." in slug:
        raise TimeTravelError(f"cannot parse a repository from remote URL {raw!r}")
    return slug


def _repo_root(path: Path) -> Path:
    """Resolve the repository root from any path inside it."""
    candidate = path.expanduser().resolve()
    if not candidate.exists():
        raise TimeTravelError(f"{path} does not exist")
    result = _git(candidate, "rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        raise TimeTravelError(f"{path} is not inside a git working tree")
    return Path(result.stdout.strip()).resolve()


def _current_branch(root: Path) -> str:
    """Return the checked-out branch; a detached HEAD is an error."""
    result = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    branch = result.stdout.strip()
    if result.returncode != 0 or not branch:
        raise TimeTravelError(
            "the repository is on a detached HEAD; pass --branch explicitly"
        )
    return branch


def _validate_branch_name(branch: str) -> None:
    """Reject names Git could interpret as options or cannot store as a ref."""
    if not branch or branch.startswith("-"):
        raise TimeTravelError(f"invalid branch name {branch!r}")
    result = subprocess.run(
        ["git", "check-ref-format", f"refs/heads/{branch}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_GIT_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise TimeTravelError(f"invalid branch name {branch!r}")


def _validate_remote_name(remote: str) -> None:
    """Reject remote names that are not a simple config key segment."""
    if not _REMOTE_NAME_RE.fullmatch(remote) or remote.startswith("-"):
        raise TimeTravelError(f"invalid remote name {remote!r}")


def _branch_exists(root: Path, branch: str) -> bool:
    """Return True when the local branch ref exists."""
    result = _git(
        root, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False
    )
    return result.returncode == 0


def _branch_checked_out(root: Path, branch: str) -> bool:
    """Return True when a linked worktree has ``branch`` checked out."""
    listing = _git(root, "worktree", "list", "--porcelain").stdout
    return f"branch refs/heads/{branch}" in listing.splitlines()


def _rev(root: Path, ref: str, message: str) -> str:
    """Resolve ``ref`` to a commit SHA or raise with ``message``."""
    result = _git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        raise TimeTravelError(message)
    return value


def _git(
    cwd: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run one git command with a hard timeout; never use a shell."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeTravelError(
            f"git {' '.join(args)} timed out after {_GIT_TIMEOUT_SECONDS}s"
        ) from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
        raise TimeTravelError(f"git {' '.join(args)} failed: {detail}")
    return result


@contextmanager
def _isolated_git_config(empty_hooks: Path, excludes_file: Path) -> Iterator[None]:
    """Scope two git settings to the temporary worktree.

    * ``core.hooksPath`` points at an empty directory. The orchestrator commits
      inside the temporary worktree; without this, a repository's pre-commit hook
      (including the one installed by ``moray-surgeon install-hook``) would audit
      the temporary checkout and could block a commit the orchestrator already
      verified.
    * ``core.excludesFile`` hides ``.moray-cache`` so the transaction's
      ``git stash -u`` never captures the SQLite index. (In a linked worktree the
      indexer's own ``.git/info/exclude`` write lands in the per-worktree gitdir,
      which git does not consult for ignore rules.)

    The override travels through ``GIT_CONFIG_*`` so it never persists in any git
    config and is restored even when the block raises.
    """
    entries = (
        ("core.hooksPath", str(empty_hooks)),
        ("core.excludesFile", str(excludes_file)),
    )
    try:
        count = max(0, int(os.environ.get("GIT_CONFIG_COUNT", "0") or "0"))
    except ValueError:
        count = 0
    keys = ["GIT_CONFIG_COUNT"]
    for offset in range(len(entries)):
        keys.append(f"GIT_CONFIG_KEY_{count + offset}")
        keys.append(f"GIT_CONFIG_VALUE_{count + offset}")
    saved = {key: os.environ.get(key) for key in keys}
    os.environ["GIT_CONFIG_COUNT"] = str(count + len(entries))
    for offset, (config_key, config_value) in enumerate(entries):
        os.environ[f"GIT_CONFIG_KEY_{count + offset}"] = config_key
        os.environ[f"GIT_CONFIG_VALUE_{count + offset}"] = config_value
    try:
        yield
    finally:
        for saved_key, saved_value in saved.items():
            if saved_value is None:
                os.environ.pop(saved_key, None)
            else:
                os.environ[saved_key] = saved_value
