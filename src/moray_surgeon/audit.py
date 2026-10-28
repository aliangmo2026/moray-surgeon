"""Read-only CI audit: prove a change set is surgical before it is merged.

The engine edits a working tree inside a Git transaction and rolls back on a red
gate. CI has no transaction — the change already exists as a diff — so this module
is the read-only mirror of that contract. It never writes, commits, stashes or
restores anything.

* ``git diff --name-status <base>`` plus the untracked file list is the change
  set.
* Every changed Python file is fingerprinted twice: once from the base commit
  (``git show <base>:<path>``) and once from disk. The node-level delta is the
  structural evidence in the report.
* When a plan is supplied, the same :func:`~moray_surgeon.engine.scope_guard.verify_scope`
  used by the orchestrator decides whether each changed node is authorised. A
  created, deleted, renamed or unparseable file is a violation: no engine atom can
  produce any of those.
* The verification gates run unchanged (``ruff`` → ``mypy`` → ``pytest``), so
  "the diff is in scope" and "the project still works" are answered by the exact
  code that guards local refactors.

Without ``--plan`` the scope gate is informational: the report lists what changed
node by node, but a human PR that adds files is not treated as an engine
violation. With ``--plan`` the diff must be explainable by that plan.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import libcst as cst

from moray_surgeon.engine.dag import topological_sort
from moray_surgeon.engine.scope_guard import get_ast_fingerprint, verify_scope
from moray_surgeon.exceptions import GitError, ScopeViolationError
from moray_surgeon.schema import RefactorPlan
from moray_surgeon.verify import Verifier, VerifyResult

_GIT_TIMEOUT_SECONDS = 120
_OUTPUT_LIMIT = 4000

_STATUS_NAMES = {
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "added",
}


@dataclass(frozen=True)
class FileChange:
    """One entry of the audit's change set.

    Attributes:
        path: Repository-relative POSIX path of the file after the change.
        status: ``"added"``, ``"modified"``, ``"deleted"`` or ``"renamed"``.
        origin: Previous path for a rename, otherwise None.
        changed_nodes: Qualified AST node names whose fingerprint differs between
            the base commit and the working tree. Empty for non-Python files.
        parse_error: True when either side of a Python file is not valid Python.
    """

    path: str
    status: str
    origin: str | None = None
    changed_nodes: tuple[str, ...] = ()
    parse_error: bool = False

    def as_dict(self) -> dict[str, object]:
        """Serialize the change for the JSON report."""
        return {
            "path": self.path,
            "status": self.status,
            "origin": self.origin,
            "changed_nodes": list(self.changed_nodes),
            "parse_error": self.parse_error,
        }


@dataclass(frozen=True)
class AuditReport:
    """Outcome of one audit run.

    Attributes:
        ok: True only when the scope gate found no violation and every
            verification gate passed.
        repo_root: Absolute repository root that was audited.
        base_ref: The ref the caller asked to compare against.
        base_commit: The commit ``base_ref`` resolved to.
        plan_path: Path of the authorising plan, or None for an informational run.
        changes: Every file in the change set, sorted by path.
        scope_violations: Human-readable violations; empty when in scope.
        verify: The verification result, never None for a completed audit.
    """

    ok: bool
    repo_root: str
    base_ref: str
    base_commit: str
    plan_path: str | None
    changes: tuple[FileChange, ...]
    scope_violations: tuple[str, ...]
    verify: VerifyResult

    def as_dict(self) -> dict[str, object]:
        """Serialize the report; verification output is bounded for artefacts."""
        return {
            "ok": self.ok,
            "repo_root": self.repo_root,
            "base_ref": self.base_ref,
            "base_commit": self.base_commit,
            "plan": self.plan_path,
            "files": [change.as_dict() for change in self.changes],
            "scope_violations": list(self.scope_violations),
            "verification": {
                "ok": self.verify.ok,
                "stage": self.verify.stage,
                "stdout": self.verify.stdout[-_OUTPUT_LIMIT:],
                "stderr": self.verify.stderr[-_OUTPUT_LIMIT:],
            },
        }

    def to_json(self) -> str:
        """Render the report as pretty JSON, preserving non-ASCII paths."""
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False)


def run_audit(
    repo_root: Path,
    *,
    base_ref: str = "HEAD",
    plan: RefactorPlan | None = None,
    plan_path: str | None = None,
    verifier: Verifier | None = None,
) -> AuditReport:
    """Audit the working tree against ``base_ref`` without editing anything.

    Args:
        repo_root: Repository root to audit.
        base_ref: Git revision to compare the working tree against. In CI this is
            the pull request's base branch (``origin/main``); locally ``HEAD``
            audits uncommitted work.
        plan: Optional plan that authorises the diff. When given, every changed
            node and every created/deleted/renamed file must be explained by it.
        plan_path: Path recorded in the report for traceability.
        verifier: Verifier to reuse; a default one is built for ``repo_root``.

    Returns:
        A report; ``ok`` is False when the scope gate or any verification gate
        failed. Building the report never raises for a red gate.

    Raises:
        GitError: ``repo_root`` is not a Git work tree, or ``base_ref`` cannot be
            resolved.
        DependencyError: ``plan`` repeats an id, names a missing one, or is cyclic.
    """
    root = repo_root.resolve()
    if not (root / ".git").exists():
        raise GitError(f"{root} is not a git repository root (no .git)")
    if plan is not None:
        topological_sort(plan.instructions)
    base_commit = _resolve_base(root, base_ref)

    changes: list[FileChange] = []
    violations: list[str] = []
    for path, status, origin in _collect_changes(root, base_commit):
        change, file_violations = _audit_file(root, base_commit, path, status, origin, plan)
        changes.append(change)
        violations.extend(file_violations)

    verify = verifier.verify() if verifier is not None else Verifier(root).verify()
    return AuditReport(
        ok=not violations and verify.ok,
        repo_root=str(root),
        base_ref=base_ref,
        base_commit=base_commit,
        plan_path=plan_path,
        changes=tuple(sorted(changes, key=lambda change: change.path)),
        scope_violations=tuple(violations),
        verify=verify,
    )


def _audit_file(
    root: Path,
    base_commit: str,
    path: str,
    status: str,
    origin: str | None,
    plan: RefactorPlan | None,
) -> tuple[FileChange, list[str]]:
    """Fingerprint one changed file and return its report entry plus violations."""
    before_path = origin or path
    before = None if status == "added" else _read_blob(root, base_commit, before_path)
    after = None if status == "deleted" else (root / path).read_bytes().decode("utf-8", "replace")

    if not path.endswith(".py"):
        violations = _file_level_violations(path, status, origin, plan)
        return FileChange(path=path, status=status, origin=origin), violations

    before_nodes, before_error = _fingerprint(before)
    after_nodes, after_error = _fingerprint(after)
    changed = tuple(
        sorted(
            key
            for key in set(before_nodes) | set(after_nodes)
            if before_nodes.get(key) != after_nodes.get(key)
        )
    )
    parse_error = before_error or after_error
    change = FileChange(
        path=path,
        status=status,
        origin=origin,
        changed_nodes=changed,
        parse_error=parse_error,
    )

    violations = _file_level_violations(path, status, origin, plan)
    if plan is None:
        if parse_error:
            violations.append(f"{path}: does not parse as Python")
        return change, violations
    if status != "modified":
        # Created, deleted, renamed and unparseable files are already reported by
        # _file_level_violations: no plan can authorise them.
        return change, violations
    if parse_error:
        violations.append(f"{path}: does not parse as Python")
        return change, violations
    try:
        verify_scope(before_nodes, after or "", plan, target_file=path)
    except ScopeViolationError as exc:
        violations.extend(f"{path}: {node}" for node in exc.violations)
    return change, violations


def _file_level_violations(
    path: str,
    status: str,
    origin: str | None,
    plan: RefactorPlan | None,
) -> list[str]:
    """Return violations that depend only on the file's status, not its content."""
    if plan is None:
        # Informational mode: a human PR may add files. The gates still decide.
        return []
    if status == "added":
        return [f"{path}: created outside the plan (the engine never creates files)"]
    if status == "deleted":
        return [f"{path}: deleted outside the plan (the engine never deletes files)"]
    if status == "renamed":
        return [f"{path}: renamed from {origin} outside the plan (the engine never moves files)"]
    if not path.endswith(".py"):
        return [f"{path}: non-Python file changed outside the plan"]
    return []


def _fingerprint(source: str | None) -> tuple[dict[str, str], bool]:
    """Fingerprint one side of a file; ``(nodes, parse_error)``."""
    if source is None:
        return {}, False
    try:
        return get_ast_fingerprint(source), False
    except cst.ParserSyntaxError:
        return {}, True


def _collect_changes(root: Path, base_commit: str) -> list[tuple[str, str, str | None]]:
    """List ``(path, status, origin)`` for the diff versus ``base_commit``.

    ``git diff --name-status`` covers tracked files (staged and unstaged relative
    to the base commit); ``git ls-files --others`` adds untracked, non-ignored
    files, which a local audit must not silently skip.
    """
    changes: list[tuple[str, str, str | None]] = []
    for line in _git(root, "diff", "--name-status", "-M", base_commit).splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        code = fields[0][:1]
        status = _STATUS_NAMES.get(code, "modified")
        if code in ("R", "C") and len(fields) >= 3:
            changes.append((fields[2], status, fields[1]))
        else:
            changes.append((fields[1], status, None))
    for path in _git(root, "ls-files", "--others", "--exclude-standard").splitlines():
        if path:
            changes.append((path, "added", None))
    return changes


def _resolve_base(root: Path, base_ref: str) -> str:
    """Resolve ``base_ref`` to a commit id, reporting a clear setup error."""
    try:
        return _git(root, "rev-parse", "--verify", f"{base_ref}^{{commit}}").strip()
    except GitError as exc:
        raise GitError(
            f"cannot resolve base ref {base_ref!r} in {root}: {exc.stderr or exc}",
            exc.stderr,
        ) from exc


def _read_blob(root: Path, commit: str, path: str) -> str:
    """Read one file at ``commit`` exactly as committed (UTF-8, lossy decode)."""
    return _run_git(root, ("show", f"{commit}:{path}")).decode("utf-8", errors="replace")


def _git(root: Path, *args: str) -> str:
    """Run one git command and return decoded stdout."""
    return _run_git(root, args).decode("utf-8", errors="replace")


def _run_git(root: Path, args: Sequence[str]) -> bytes:
    """Run git with path quoting disabled; raise ``GitError`` on any failure."""
    command = ["git", "-c", "core.quotepath=false", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitError(f"git {args[0]} failed: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(f"git {' '.join(args)} failed: {stderr}", stderr)
    return completed.stdout
