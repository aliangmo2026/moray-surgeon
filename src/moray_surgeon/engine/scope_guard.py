"""Prove an attempt only changed the AST nodes its plan was allowed to touch.

The verification gates answer "does the project still work?". They cannot answer
"did the engine rewrite something I never asked about?" — a silent edit to an
unrelated function can pass ruff, mypy and pytest if its behaviour happens to
stay inside the tested surface. This module closes that gap with a second,
structural gate:

* Before the plan runs, every Python file in the repository is fingerprinted.
  A fingerprint maps a *qualified* node name (``Class.method``, ``outer.inner``,
  ``<module>`` for module-level statements) to a SHA-256 of that node's rendered
  source.
* After the plan runs, the same map is recomputed for every changed file.
  A node whose digest changed (or appeared/disappeared) is only accepted when the
  plan explains it: it is named by a file-scoped instruction, it is a reference
  site of a rename (its code contains the new name as an identifier), or the
  instruction is inherently file-global (``replace_call``,
  ``add_type_hints`` without a function name).
* Anything else raises :class:`ScopeViolationError`, the caller rolls the
  transaction back, and the offending node names travel to the next prompt.

Imports are deliberately excluded from ``<module>``: the controlled import atoms
edit the import block by design, and their effect is covered by the verification
gates. Everything else at module level — assignments, constants, ``if``/``try``
blocks — is fingerprinted.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import libcst as cst

from moray_surgeon.exceptions import ScopeViolationError
from moray_surgeon.schema.instructions import (
    AddImport,
    AddTypeHints,
    ExtractFunction,
    Noop,
    RefactorPlan,
    RemoveImport,
    RenameSymbol,
    ReplaceCall,
)

MODULE_KEY = "<module>"

# Build artefacts, virtualenvs and tool caches never contain plan targets and
# parsing them would only slow the guard down.
_IGNORED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "site-packages",
        "__pycache__",
        "build",
        "dist",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
    }
)

_DUPLICATE_SUFFIX = re.compile(r"#\d+$")


def get_ast_fingerprint(source: str) -> dict[str, str]:
    """Summarise every function and class in ``source`` as a SHA-256 digest.

    The ``<module>`` entry covers module-level statements excluding imports and
    definitions, so a changed constant or control-flow block is visible while an
    ``add_import``/``remove_import`` is not.

    Args:
        source: Python module source.

    Returns:
        A mapping from qualified node name to digest. Duplicate qualified names
        (conditional definitions) get a ``#2``/``#3`` suffix in source order.

    Raises:
        libcst.ParserSyntaxError: ``source`` is not valid Python.
    """
    fingerprints, _names = _index_module(source)
    return fingerprints


def verify_scope(
    original_fps: dict[str, str],
    new_source: str,
    plan: RefactorPlan,
    *,
    target_file: str | None = None,
) -> None:
    """Raise :class:`ScopeViolationError` when ``new_source`` exceeds the plan.

    Args:
        original_fps: Fingerprints captured before the plan ran, for the file
            ``new_source`` belongs to (see :func:`get_ast_fingerprint`).
        new_source: The file's content after the plan ran.
        plan: The plan that was executed.
        target_file: Repository-relative path of the file being checked. When
            given, file-scoped instructions only authorise changes in the file
            they name. When ``None``, the file identity is unknown and every
            instruction is treated as applying to it — permissive, so the
            orchestrator always passes the path.

    Raises:
        ScopeViolationError: At least one node changed without authorisation;
            ``violations`` lists the qualified names and the message names the
            file (when known).
        libcst.ParserSyntaxError: ``new_source`` is not valid Python.
    """
    new_fps, identifiers = _index_module(new_source)
    violations = tuple(
        key
        for key in sorted(set(original_fps) | set(new_fps))
        if original_fps.get(key) != new_fps.get(key)
        and not _is_allowed(key, identifiers.get(key, frozenset()), plan, target_file)
    )
    if violations:
        location = f"{target_file}: " if target_file else ""
        listed = ", ".join(violations)
        raise ScopeViolationError(
            f"{location}AST scope guard rejected {len(violations)} node(s) the plan "
            f"did not authorise: {listed}",
            violations,
        )


@dataclass(frozen=True)
class RepositorySnapshot:
    """Fingerprints and raw digests of every Python file before an attempt.

    Attributes:
        fingerprints: Repository-relative path -> node fingerprint map.
        digests: Repository-relative path -> SHA-256 of the raw file text, used
            to skip files the attempt did not change.
    """

    fingerprints: dict[str, dict[str, str]]
    digests: dict[str, str]


def snapshot_repository(repo_root: Path) -> RepositorySnapshot:
    """Fingerprint every Python file under ``repo_root``.

    Unparseable files are still tracked, as a single ``<raw>`` digest, so an
    attempt that touches them is caught instead of crashing the guard.

    Args:
        repo_root: Repository root to scan.

    Returns:
        The snapshot; ignored directories (VCS, virtualenvs, caches, build
        output) are skipped.

    Raises:
        OSError: ``repo_root`` cannot be walked or a file cannot be read.
    """
    root = repo_root.resolve()
    fingerprints: dict[str, dict[str, str]] = {}
    digests: dict[str, str] = {}
    for path in _python_files(root):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        source = raw.decode("utf-8", errors="replace")
        digests[relative] = _digest_bytes(raw)
        try:
            fingerprints[relative] = get_ast_fingerprint(source)
        except cst.ParserSyntaxError:
            fingerprints[relative] = {"<raw>": _digest(source)}
    return RepositorySnapshot(fingerprints=fingerprints, digests=digests)


def verify_repository_scope(
    snapshot: RepositorySnapshot,
    repo_root: Path,
    plan: RefactorPlan,
) -> None:
    """Re-scan the repository and verify every file the attempt changed.

    Unchanged files are skipped without parsing. A file that appeared or
    disappeared is itself a violation: no atom in the plan can do either, so it
    means something outside the plan touched the tree.

    Args:
        snapshot: The pre-attempt snapshot for the same repository.
        repo_root: Repository root to scan again.
        plan: The plan that was executed.

    Raises:
        ScopeViolationError: A file was created, deleted, or contains a node
            change the plan does not authorise.
        OSError: A changed file could not be read.
    """
    root = repo_root.resolve()
    current: dict[str, Path] = {}
    for path in _python_files(root):
        current[path.relative_to(root).as_posix()] = path

    created = sorted(set(current) - set(snapshot.fingerprints))
    if created:
        raise ScopeViolationError(
            f"AST scope guard: the plan created file(s) it does not target: "
            f"{', '.join(created)}",
            tuple(created),
        )
    deleted = sorted(set(snapshot.fingerprints) - set(current))
    if deleted:
        raise ScopeViolationError(
            f"AST scope guard: the plan deleted file(s) it does not target: "
            f"{', '.join(deleted)}",
            tuple(deleted),
        )

    for relative in sorted(current):
        raw = current[relative].read_bytes()
        if _digest_bytes(raw) == snapshot.digests[relative]:
            continue
        verify_scope(
            snapshot.fingerprints[relative],
            raw.decode("utf-8", errors="replace"),
            plan,
            target_file=relative,
        )


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #


def _index_module(source: str) -> tuple[dict[str, str], dict[str, frozenset[str]]]:
    """Build both the fingerprint map and the identifier sets in one parse."""
    module = cst.parse_module(source)
    renderer = cst.Module(body=())
    nodes = _collect_scoped_nodes(module)

    module_statements = [
        statement
        for statement in module.body
        if not isinstance(statement, (cst.Import, cst.ImportFrom, cst.ClassDef, cst.FunctionDef))
        and not _is_import_line(statement)
    ]
    module_code = cst.Module(body=module_statements).code if module_statements else ""

    fingerprints: dict[str, str] = {MODULE_KEY: _digest(module_code.strip())}
    identifiers: dict[str, frozenset[str]] = {MODULE_KEY: _identifiers(module_statements)}

    for key, node in nodes.items():
        code = renderer.code_for_node(node).strip()
        fingerprints[key] = _digest(code)
        identifiers[key] = _identifiers([node])
    return fingerprints, identifiers


class _ScopedNodeCollector(cst.CSTVisitor):
    """Collect every class/function definition under its qualified name."""

    def __init__(self) -> None:
        self._scope: list[str] = []
        self._seen: dict[str, int] = {}
        self.nodes: dict[str, cst.CSTNode] = {}

    # libcst dispatches by these exact CamelCase hook names.
    def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
        self._record(node.name.value, node)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
        self._scope.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
        self._record(node.name.value, node)

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
        self._scope.pop()

    def _record(self, name: str, node: cst.CSTNode) -> None:
        """Assign a unique qualified key and descend into the new scope."""
        base = ".".join((*self._scope, name))
        count = self._seen.get(base, 0) + 1
        self._seen[base] = count
        key = base if count == 1 else f"{base}#{count}"
        self.nodes[key] = node
        self._scope.append(name)


class _NameCollector(cst.CSTVisitor):
    """Collect every identifier spelled anywhere in a subtree."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: cst.Name) -> None:  # noqa: N802
        self.names.add(node.value)


def _collect_scoped_nodes(module: cst.Module) -> dict[str, cst.CSTNode]:
    """Return qualified name -> definition node for the whole module."""
    collector = _ScopedNodeCollector()
    module.visit(collector)
    return collector.nodes


def _identifiers(nodes: Sequence[cst.CSTNode]) -> frozenset[str]:
    """Union of every ``Name`` value in the given subtrees."""
    collector = _NameCollector()
    for node in nodes:
        node.visit(collector)
    return frozenset(collector.names)


def _is_import_line(statement: cst.BaseStatement) -> bool:
    """True for a top-level statement that is only imports (``;`` included)."""
    if not isinstance(statement, cst.SimpleStatementLine):
        return False
    return all(isinstance(item, (cst.Import, cst.ImportFrom)) for item in statement.body)


# --------------------------------------------------------------------------- #
# Authorisation
# --------------------------------------------------------------------------- #


def _is_allowed(
    key: str,
    identifiers: frozenset[str],
    plan: RefactorPlan,
    target_file: str | None,
) -> bool:
    """True when at least one instruction in ``plan`` explains this change."""
    base_key = _DUPLICATE_SUFFIX.sub("", key)
    module_level = key == MODULE_KEY
    for instruction in plan.instructions:
        if isinstance(instruction, (Noop, AddImport, RemoveImport)):
            # Noop edits nothing; import text is not part of any fingerprint.
            continue
        scoped = target_file is None or instruction.target_file == target_file
        if isinstance(instruction, RenameSymbol):
            if scoped and (
                module_level
                or _name_matches(base_key, instruction.old_name)
                or _name_matches(base_key, instruction.new_name)
            ):
                return True
            if instruction.old_name in identifiers or instruction.new_name in identifiers:
                # A reference site rewritten by Jedi in this file.
                return True
            continue
        if not scoped:
            continue
        if module_level:
            # A rename of a constant, a module-level call site or an extracted
            # block all rewrite a statement this fingerprint covers.
            return True
        if isinstance(instruction, ReplaceCall):
            return True
        if isinstance(instruction, AddTypeHints):
            if instruction.function_name is None:
                return True
            if _name_matches(base_key, instruction.function_name):
                return True
        elif isinstance(instruction, ExtractFunction):
            if _name_matches(base_key, instruction.function_name):
                return True
    return False


def _name_matches(key: str, name: str) -> bool:
    """True when ``key`` is ``name``, a member of it, or a dotted segment of it.

    ``Class.method`` and ``Class.method.inner`` belong to ``Class``: renaming a
    class necessarily re-keys every member even though their code is unchanged,
    so they must travel with the name. A member of an unrelated same-named node
    in another scope is also accepted — the guard is scoped per file and the
    verification gates still cover behaviour.
    """
    return (
        key == name
        or key.startswith(f"{name}.")
        or key.endswith(f".{name}")
        or f".{name}." in key
    )


# --------------------------------------------------------------------------- #
# File walking and hashing
# --------------------------------------------------------------------------- #


def _python_files(root: Path) -> Iterator[Path]:
    """Yield every .py file under ``root``, never following directory links."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in _IGNORED_DIR_NAMES)
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                yield Path(dirpath) / filename


def _digest_bytes(raw: bytes) -> str:
    """Stable SHA-256 of raw file bytes."""
    return hashlib.sha256(raw).hexdigest()


def _digest(text: str) -> str:
    """Stable SHA-256 of a text fragment."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
