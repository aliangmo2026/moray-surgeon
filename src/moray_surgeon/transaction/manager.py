"""Repository-wide transaction semantics for one refactor run.

A transaction snapshots the user's uncommitted work with ``git stash`` before the
engine touches anything, then either commits the refactor or restores the
pre-refactor state. Only git's own plumbing is used: the working tree, index and
untracked files are the backup, and no library-level copy can reproduce those
three states faithfully.

Invariants:

* On a failed transaction the snapshot is popped back, so the user ends up in
  exactly the state they started in.
* A rollback failure is never swallowed: raising is the only honest signal when
  the safety net itself failed.
* Every git call is a single subprocess with a hard timeout.
"""

from __future__ import annotations

import subprocess
from collections.abc import Container
from pathlib import Path
from types import TracebackType

from moray_surgeon.exceptions import GitError

_STASH_MESSAGE = "moray-surgeon: pre-refactor snapshot"
_GIT_TIMEOUT_SECONDS = 30
# Git's "nothing to snapshot" chatter is not an error; anything else is.
_HARMLESS_STASH_MESSAGES = ("No local changes",)


class TransactionManager:
    """All-or-nothing wrapper around a repository refactor.

    Usage::

        with TransactionManager(repo_root, "refactor: rename parser"):
            execute_plan(plan, repo_root)

    Entering snapshots any uncommitted work, leaving commits the refactor, and an
    exception inside the block rolls the working tree back to HEAD and restores
    the snapshot.

    Attributes:
        _stashed: True only while a snapshot exists and still needs to be
            dropped (success) or popped (rollback).
    """

    def __init__(self, repo_root: Path, commit_message: str) -> None:
        """Bind the transaction to a repository and a commit subject line.

        Args:
            repo_root: Repository root. Every git command runs with this cwd.
            commit_message: Commit subject used when the block exits cleanly.
        """
        self._repo_root = repo_root.resolve()
        self._commit_message = commit_message
        self._stashed = False

    def __enter__(self) -> TransactionManager:
        """Verify the repo, snapshot uncommitted work and start the transaction.

        Returns:
            This manager, so ``with ... as tx`` can inspect state.

        Raises:
            GitError: ``repo_root`` is not a git working tree, the repository has
                no HEAD commit, or the snapshot could not be created.
        """
        self._verify_repository()
        self._verify_head_exists()
        self._stashed = False

        status = self._git("status", "--porcelain")
        if status.stdout.strip():
            snapshot = self._git(
                "stash",
                "push",
                "-u",
                "-m",
                _STASH_MESSAGE,
                harmless_output=_HARMLESS_STASH_MESSAGES,
            )
            # "No local changes to save" means git created no stash entry; a
            # later pop would then fail, so trust the message over porcelain.
            self._stashed = not _has_harmless_message(snapshot)
        return self

    def commit(self) -> None:
        """Commit the transaction, merging the pre-refactor snapshot back in.

        When a snapshot exists it is first re-applied to the worktree so the
        user's own changes are part of the same commit (the plan's success path
        must never silently discard them). ``apply`` rather than ``pop`` keeps
        the snapshot alive until the commit exists, so a commit that fails for
        any reason leaves the only copy of the user's work recoverable.

        Raises:
            GitError: git refused to apply the snapshot (overlapping edits in the
                same hunk), to stage, or to commit. The snapshot is retained.
        """
        if self._stashed:
            self._git("stash", "apply")
        self._git("add", "-A")
        self._git("commit", "-m", self._commit_message)
        if self._stashed:
            self._git("stash", "drop")
            self._stashed = False

    def rollback(self) -> None:
        """Restore the exact pre-transaction worktree, then restore the snapshot.

        ``reset --hard`` drops tracked edits, ``clean -fd`` removes files the
        transaction created, and the final ``stash pop`` puts the user's own
        uncommitted work back. A failure at any step propagates: swallowing it
        would hide the one case where the safety net did not hold.

        Raises:
            GitError: Any of the three git commands failed.
        """
        self._git("reset", "--hard", "HEAD")
        self._git("clean", "-fd")
        if self._stashed:
            self._git("stash", "pop")
            self._stashed = False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Commit on success, roll back on failure, and never suppress the error."""
        if exc_type is None:
            self.commit()
        else:
            self.rollback()

    def _verify_repository(self) -> None:
        """Fail before touching anything when ``repo_root`` is not a work tree."""
        if not self._repo_root.is_dir():
            raise GitError(f"repo_root {self._repo_root} is not a directory")
        result = self._git("rev-parse", "--is-inside-work-tree")
        if result.stdout.strip() != "true":
            raise GitError(f"not a git working tree: {self._repo_root}")

    def _verify_head_exists(self) -> None:
        """Refuse an empty repository: HEAD is the only rollback baseline."""
        result = self._git(
            "rev-parse",
            "--verify",
            "--quiet",
            "HEAD",
            ignore_returncodes=frozenset({1}),
        )
        if result.returncode != 0:
            raise GitError("空仓库无法使用事务机制 (no HEAD commit to roll back to)")

    def _git(
        self,
        *args: str,
        ignore_returncodes: Container[int] | None = None,
        harmless_output: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        """Run one git command in the repository and translate failures.

        Args:
            *args: Arguments after the executable, e.g. ``("commit", "-m", "x")``.
            ignore_returncodes: Return codes the caller handles itself (e.g. 1
                from ``rev-parse --verify --quiet`` for a missing HEAD).
            harmless_output: Output substrings that make a non-zero exit benign
                (used only for stash's "nothing to do" chatter).

        Returns:
            The completed process, already decoded as UTF-8.

        Raises:
            GitError: Non-zero exit outside the ignore/harmless sets, with git's
                stderr embedded in the message and kept on the exception.
        """
        result = subprocess.run(
            ["git", *args],
            cwd=self._repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
        if result.returncode != 0:
            if ignore_returncodes is not None and result.returncode in ignore_returncodes:
                return result
            if harmless_output and _contains_any(_combined_output(result), harmless_output):
                return result
            stderr = result.stderr.strip()
            raise GitError(f"git {' '.join(args)} failed:\n{stderr}", stderr)
        return result


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    """Merge both streams: git is inconsistent about which one carries warnings."""
    return f"{result.stdout}\n{result.stderr}"


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


def _has_harmless_message(result: subprocess.CompletedProcess[str]) -> bool:
    return _contains_any(_combined_output(result), _HARMLESS_STASH_MESSAGES)
