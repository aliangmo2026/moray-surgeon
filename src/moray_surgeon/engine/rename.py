"""Rename a symbol using indexed candidates, Jedi verification and libcst edits.

Jedi answers *where* the symbol is used (scope-aware and across the repository);
libcst applies the edit without disturbing comments, layout or string literals.
Only references whose module lives inside ``repo_root`` are rewritten, so a
rename can never touch the standard library or site-packages.

This is the one project-aware atom: a cross-file rename must read and rewrite
every affected file. Writes happen synchronously inside the caller's transaction,
so commit/rollback still owns the outcome as a unit.

Since the code index exists, the search no longer starts by making Jedi parse the
whole repository. The index (see :mod:`moray_surgeon.engine.indexer`) names the
files that mention the identifier at all, libcst locates the exact positions in
those files, and Jedi verifies each position by resolving it to one of the
symbol's definitions. The rewrite set is therefore identical to what Jedi's
project-wide reference search reported, while the files that cannot mention the
symbol are never opened. When the database is unavailable the atom falls back to
that reference search, so the index can only make it faster, never weaker.
"""

from __future__ import annotations

import importlib
import keyword
import logging
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import jedi
import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from moray_surgeon.engine.base import EditResult
from moray_surgeon.engine.indexer import Indexer, get_indexer
from moray_surgeon.exceptions import CacheError, PathTraversalError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Only a ceiling for the temporary budget below; a repository larger than this
# needs an out-of-process index rather than a single Jedi search.
_MAX_REFERENCE_FILES = 5000
# Verifying a candidate costs one Jedi inference per occurrence, while a
# project-wide reference search shares that inference across every file. On a
# repository of small modules the indexed path wins at every density (30x at 5%
# of the files, still 2.6x at 90%); on a project whose modules carry rich import
# graphs it loses once the symbol is spread across most of them (3.8x slower at
# 53 of 59 files in the stress project). The cutoff is therefore conservative: the
# index is used when the symbol is concentrated in a minority of the files, and
# the proven single search runs otherwise.
_MAX_INDEXED_SHARE = 0.30
# Below this many candidate files the index path is cheap enough that the share
# test would only add noise.
_MIN_INDEXED_CANDIDATES = 4
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
    indexer: Indexer | None = None,
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
        indexer: Optional shared index handle. When omitted the process-wide one
            for ``repo_root`` is used, which lets a plan reuse a single refresh
            across all of its instructions.

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
        rewrites, changes = _collect_rewrites(
            root, target, source, old_name, new_name, indexer
        )
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
    _reindex(root, indexer, rewrites)

    return EditResult(new_source=rewrites.get(target, source), changes=changes)


def _collect_rewrites(
    root: Path,
    target: Path,
    source: str,
    old_name: str,
    new_name: str,
    indexer: Indexer | None = None,
) -> tuple[dict[Path, str], int]:
    """Resolve references and rewrite every affected file in memory.

    Each affected file is parsed once: the same :class:`MetadataWrapper` that
    yields the positions handed to Jedi is reused for the rewrite, so the
    position that was verified is by construction the position that is edited.
    """
    plan = _reference_plan(root, target, source, old_name, indexer)
    rewrites: dict[Path, str] = {}
    changes = 0

    for path, (wrapper, file_positions) in plan.items():
        transformer = _PositionRename(file_positions, new_name)
        new_module = wrapper.visit(transformer)
        if transformer.changes:
            rewrites[path] = new_module.code
            changes += transformer.changes

    return rewrites, changes


def _reference_plan(
    root: Path,
    target: Path,
    source: str,
    old_name: str,
    indexer: Indexer | None,
) -> dict[Path, tuple[MetadataWrapper, set[tuple[int, int]]]]:
    """Map every file to rewrite onto its parsed module and verified positions.

    The index decides *which* strategy to use as well as which files to look at:

    * a symbol concentrated in a few files is verified file by file (the index
      makes this cheap by never reading the rest of the repository);
    * a symbol spread across most of the repository is resolved by one
      project-wide Jedi search, which shares its inference instead of paying it
      once per occurrence.

    A cache that cannot be read is not an error for a refactor, so both paths
    degrade to the project-wide search that this atom used before the index
    existed.
    """
    try:
        handle = indexer or get_indexer(root)
        candidates = list(handle.files_with(old_name))
        if target not in candidates:
            # The target is always the starting point of the rename, even when the
            # index has no row for it (an unparsed file, a generated module).
            candidates.append(target)
        if _project_scan_is_cheaper(handle, candidates):
            logger.debug(
                "%s appears in %d of %d indexed files; using one project-wide search",
                old_name,
                len(candidates),
                handle.file_count(),
            )
            return _wrap_plan(_jedi_positions(root, target, old_name), target, source)
        return _indexed_plan(root, target, source, old_name, candidates)
    except (CacheError, OSError) as exc:
        logger.warning(
            "code index unavailable for %s (%s); scanning with Jedi instead", root, exc
        )
        return _wrap_plan(_jedi_positions(root, target, old_name), target, source)


def _project_scan_is_cheaper(handle: Indexer, candidates: list[Path]) -> bool:
    """True when one project-wide search beats per-file verification."""
    if len(candidates) <= _MIN_INDEXED_CANDIDATES:
        return False
    return len(candidates) > handle.file_count() * _MAX_INDEXED_SHARE


def _wrap_plan(
    positions: dict[Path, set[tuple[int, int]]],
    target: Path,
    source: str,
) -> dict[Path, tuple[MetadataWrapper, set[tuple[int, int]]]]:
    """Attach a parsed module to each file of a position map."""
    return {
        path: (
            _parse_module(source if path == target else path.read_text(encoding="utf-8")),
            file_positions,
        )
        for path, file_positions in positions.items()
    }


def _indexed_plan(
    root: Path,
    target: Path,
    source: str,
    old_name: str,
    candidates: list[Path],
) -> dict[Path, tuple[MetadataWrapper, set[tuple[int, int]]]]:
    """Position candidates with libcst and prove each one with Jedi."""
    plan: dict[Path, tuple[MetadataWrapper, set[tuple[int, int]]]] = {}
    with _jedi_reference_budget(root):
        project = jedi.Project(path=str(root))
        # One script for the target serves both the origin lookup and the
        # verification of the target's own mentions; Jedi's first script is by far
        # the most expensive call in this atom, so it is created once.
        target_script = jedi.Script(path=str(target), project=project)
        origins = _definition_origins(target_script, target, old_name)
        if not origins:
            return plan
        for path in candidates:
            wrapper = _parse_module(
                source if path == target else path.read_text(encoding="utf-8")
            )
            script = (
                target_script
                if path == target
                else jedi.Script(path=str(path), project=project)
            )
            verified = _verified_positions(script, wrapper, old_name, origins)
            if verified:
                plan[path] = (wrapper, verified)
    return plan


def _definition_origins(
    script: jedi.Script, target: Path, old_name: str
) -> set[tuple[str, int, int]]:
    """Identities of everything ``old_name`` in ``target`` stands for.

    Every definition Jedi finds in the target file counts, plus whatever those
    definitions ultimately resolve to. The second half is what keeps a rename
    started from an ``import`` binding working: the binding is the local
    definition, but the symbol it names — and every other module importing it —
    resolves to the original module, exactly as Jedi's project-wide reference
    search used to report.

    Returns:
        ``(normalised module path, line, column)`` triples. Coordinates are the
        ones Jedi itself reports, which is what makes them comparable with the
        results of :func:`_verified_positions`.
    """
    origins: set[tuple[str, int, int]] = set()
    for definition in script.get_names(all_scopes=True, definitions=True, references=False):
        if definition.name != old_name or definition.module_path is None:
            continue
        origins.add(_identity(definition.module_path, definition.line, definition.column))
        for resolved in script.goto(
            definition.line, definition.column, follow_imports=True
        ):
            if resolved.module_path is None:
                continue
            origins.add(_identity(resolved.module_path, resolved.line, resolved.column))
    return origins


def _verified_positions(
    script: jedi.Script,
    wrapper: MetadataWrapper,
    old_name: str,
    origins: set[tuple[str, int, int]],
) -> set[tuple[int, int]]:
    """Positions in ``script``'s file that resolve to one of ``origins``."""
    scanned = _scan_name_positions(wrapper, old_name)
    if not scanned:
        return set()
    verified: set[tuple[int, int]] = set()
    for line, column in scanned:
        for resolved in script.goto(line=line, column=column, follow_imports=True):
            if resolved.module_path is None:
                continue
            if _identity(resolved.module_path, resolved.line, resolved.column) in origins:
                verified.add((line, column))
                break
    return verified


class _NameScan(cst.CSTVisitor):
    """Every ``Name`` spelled ``name`` that a rename may rewrite.

    ``call(name=1)`` keywords are skipped on purpose: Jedi's project reference
    search does not report them as references to the parameter, and renaming a
    parameter while rewriting its keyword call sites would change the public call
    convention of every caller outside this repository.
    """

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, name: str) -> None:
        self._name = name
        self._keywords: set[int] = set()
        self.positions: set[tuple[int, int]] = set()

    # libcst dispatches by this exact CamelCase hook name.
    def visit_Arg(self, node: cst.Arg) -> None:  # noqa: N802
        """Remember keyword names so they can be filtered out below."""
        keyword = node.keyword
        if keyword is not None:
            self._keywords.add(id(keyword))

    # libcst dispatches by this exact CamelCase hook name.
    def visit_Name(self, node: cst.Name) -> None:  # noqa: N802
        """Record the position of a matching, non-keyword identifier."""
        if node.value != self._name or id(node) in self._keywords:
            return
        start = self.get_metadata(PositionProvider, node).start
        self.positions.add((start.line, start.column))


def _scan_name_positions(wrapper: MetadataWrapper, name: str) -> set[tuple[int, int]]:
    """Collect the candidate positions of ``name`` in a parsed module."""
    scan = _NameScan(name)
    wrapper.visit(scan)
    return scan.positions


def _parse_module(text: str) -> MetadataWrapper:
    """Parse source into a wrapper that can be visited more than once."""
    return MetadataWrapper(cst.parse_module(text))


def _identity(module_path: str, line: int, column: int) -> tuple[str, int, int]:
    """A comparable identity for a Jedi definition or reference.

    ``normcase`` keeps the comparison stable on Windows, where Jedi may report
    the same file with different letter case than the repository root.
    """
    return (os.path.normcase(str(Path(module_path).resolve())), line, column)


def _reindex(
    root: Path, indexer: Indexer | None, paths: Iterable[Path]
) -> None:
    """Feed the files just written back into the index.

    Without this the next instruction of the same plan would query rows that
    describe the pre-edit sources. A failure only costs a refresh later, so it is
    logged rather than raised.
    """
    written = list(paths)
    if not written:
        return
    try:
        handle = indexer or get_indexer(root)
        handle.reindex(written)
    except (CacheError, OSError) as exc:
        logger.warning("could not refresh the code index after writing files: %s", exc)


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
