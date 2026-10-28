"""Rename a symbol using Jedi-resolved references and libcst edits.

Jedi answers *where* the symbol is used (scope-aware and across the repository);
libcst applies the edit without disturbing comments, layout or string literals.
Only references whose module lives inside ``repo_root`` are rewritten, so a
rename can never touch the standard library or site-packages.

This is the one project-aware atom: a cross-file rename must read and rewrite
every affected file. Writes happen synchronously inside the caller's transaction,
so commit/rollback still owns the outcome as a unit.
"""

from __future__ import annotations

import importlib
import keyword
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import jedi
import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from moray_surgeon.engine.base import EditResult
from moray_surgeon.exceptions import PathTraversalError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Only a ceiling for the temporary budget below; a repository larger than this
# needs an out-of-process index rather than a single Jedi search.
_MAX_REFERENCE_FILES = 5000
_SKIP_DIR_NAMES = frozenset(
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


class _CoarseRename(cst.CSTTransformer):
    """Fallback: replace every ``Name`` with the old spelling in one file."""

    def __init__(self, old_name: str, new_name: str) -> None:
        self._old_name = old_name
        self._new_name = new_name
        self.changes = 0

    # libcst dispatches by this exact CamelCase hook name.
    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        """Swap the identifier spelling while keeping every surrounding token."""
        if original_node.value != self._old_name:
            return updated_node
        self.changes += 1
        return updated_node.with_changes(value=self._new_name)


class _PositionRename(cst.CSTTransformer):
    """Replace only the ``Name`` nodes at Jedi-reported coordinates."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, positions: set[tuple[int, int]], new_name: str) -> None:
        self._positions = positions
        self._new_name = new_name
        self.changes = 0

    # libcst dispatches by this exact CamelCase hook name.
    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        """Rename a node only when Jedi placed a reference at its start."""
        start = self.get_metadata(PositionProvider, original_node).start
        if (start.line, start.column) not in self._positions:
            return updated_node
        self.changes += 1
        return updated_node.with_changes(value=self._new_name)


def rename_symbol(
    repo_root: Path,
    target_file: str,
    old_name: str,
    new_name: str,
) -> EditResult:
    """Rename every project reference to ``old_name`` defined in ``target_file``.

    Jedi resolves references instead of matching text: renaming a class never
    touches ``obj.old_name`` attribute accesses or ``kwarg=old_name`` keyword
    arguments, and a local variable's rename never escapes its function. Every
    definition of ``old_name`` in the target file is used as a starting point;
    unrelated same-named symbols in other scopes are not rewritten.

    Affected files inside ``repo_root`` are rewritten in place (this atom is
    project-aware by design); the caller's transaction provides atomicity.

    Args:
        repo_root: Repository root; references outside it are never rewritten.
        target_file: Repository-relative path to the file that defines the symbol.
        old_name: Identifier to rename.
        new_name: Replacement identifier.

    Returns:
        An :class:`EditResult` for ``target_file``; ``changes`` counts every
        replaced reference across all rewritten files.

    Raises:
        ValueError: A name is empty, non-identifier, a keyword, or names are equal.
        PathTraversalError: ``target_file`` resolves outside ``repo_root``.
        FileNotFoundError: The target file does not exist.
        OSError: An affected file could not be read or written.
    """
    _validate_identifier("old_name", old_name)
    _validate_identifier("new_name", new_name)
    if old_name == new_name:
        raise ValueError("old_name and new_name must differ")

    root = repo_root.resolve()
    target = (root / target_file).resolve()
    if not target.is_relative_to(root):
        raise PathTraversalError(f"target_file {target_file!r} resolves outside the repository")
    if not target.is_file():
        raise FileNotFoundError(f"target_file {target_file!r} does not exist")
    source = target.read_text(encoding="utf-8")

    try:
        rewrites, changes = _collect_rewrites(root, target, source, old_name, new_name)
    except Exception as exc:
        # Jedi (or libcst on a referenced file) failed: keep the refactor useful
        # by falling back to the old single-file textual rename.
        logger.warning(
            "jedi analysis failed for %s (%s); falling back to text-level rename",
            target,
            exc,
            exc_info=True,
        )
        fallback = _rename_textually(source, old_name, new_name)
        if fallback.changes:
            target.write_text(fallback.new_source, encoding="utf-8")
        return fallback

    for path, new_source in rewrites.items():
        path.write_text(new_source, encoding="utf-8")

    return EditResult(new_source=rewrites.get(target, source), changes=changes)


def _collect_rewrites(
    root: Path,
    target: Path,
    source: str,
    old_name: str,
    new_name: str,
) -> tuple[dict[Path, str], int]:
    """Resolve references and rewrite every affected file in memory."""
    positions = _jedi_positions(root, target, old_name)
    rewrites: dict[Path, str] = {}
    changes = 0

    for path, file_positions in positions.items():
        original = source if path == target else path.read_text(encoding="utf-8")
        wrapper = MetadataWrapper(cst.parse_module(original))
        transformer = _PositionRename(file_positions, new_name)
        new_module = wrapper.visit(transformer)
        if transformer.changes:
            rewrites[path] = new_module.code
            changes += transformer.changes

    return rewrites, changes


def _jedi_positions(
    root: Path,
    target: Path,
    old_name: str,
) -> dict[Path, set[tuple[int, int]]]:
    """Map each in-repo file to the exact ``(line, column)`` positions to rename."""
    positions: dict[Path, set[tuple[int, int]]] = {}
    with _jedi_reference_budget(root):
        project = jedi.Project(path=str(root))
        script = jedi.Script(path=str(target), project=project)
        definitions = [
            name
            for name in script.get_names(all_scopes=True, definitions=True, references=False)
            if name.name == old_name
        ]

        for definition in definitions:
            references = script.get_references(
                definition.line, definition.column, scope="project"
            )
            for reference in references:
                if reference.name != old_name or reference.module_path is None:
                    continue
                path = Path(reference.module_path).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    continue
                positions.setdefault(path, set()).add((reference.line, reference.column))
    return positions


@contextmanager
def _jedi_reference_budget(repo_root: Path) -> Iterator[None]:
    """Temporarily raise Jedi's per-search file budget to cover the repository.

    Jedi hard-codes ``_PARSED_FILE_LIMIT = 30`` and ``_OPENED_FILE_LIMIT = 2000``
    for project-scope reference searches. Past either limit it silently returns
    *partial* results, so a rename in a 60-file project would rewrite half the
    call sites and leave dangling names behind. Raising the budget to the
    repository's own file count keeps the search complete; the values are
    restored afterwards so later callers see stock Jedi behaviour.

    The private names are probed defensively: if a future Jedi version renames
    or removes them, the search still runs (with Jedi's own limits) instead of
    crashing the engine.
    """
    try:
        references = importlib.import_module("jedi.inference.references")
    except ImportError:  # pragma: no cover - only a broken/very old Jedi install
        yield
        return

    needed = min(_MAX_REFERENCE_FILES, _python_file_count(repo_root) + 200)
    saved: list[tuple[str, int]] = []
    for attribute in ("_PARSED_FILE_LIMIT", "_OPENED_FILE_LIMIT"):
        current = getattr(references, attribute, None)
        if isinstance(current, int) and needed > current:
            saved.append((attribute, current))
            setattr(references, attribute, needed)
    try:
        yield
    finally:
        for attribute, current in saved:
            setattr(references, attribute, current)


def _python_file_count(root: Path) -> int:
    """Count .py files under ``root`` without following directory links."""
    total = 0
    for _dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIR_NAMES)
        total += sum(1 for name in filenames if name.endswith(".py"))
        if total >= _MAX_REFERENCE_FILES:
            return _MAX_REFERENCE_FILES
    return total


def _rename_textually(source: str, old_name: str, new_name: str) -> EditResult:
    """Coarse single-file rename used only when Jedi/libcst analysis fails."""
    transformer = _CoarseRename(old_name, new_name)
    new_module = cst.parse_module(source).visit(transformer)
    return EditResult(new_source=new_module.code, changes=transformer.changes)


def _validate_identifier(role: str, value: str) -> None:
    """Reject empty / non-identifier / keyword names before touching the CST."""
    if not value:
        raise ValueError(f"{role} must not be empty")
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(f"{role} must be a valid Python identifier, got {value!r}")
