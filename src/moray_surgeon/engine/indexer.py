"""Project-wide symbol index with an incremental SQLite cache.

Why this module exists
----------------------
``rename_symbol`` and ``move_symbol`` are the only project-aware atoms. Jedi's
``get_references(..., scope="project")`` answers them precisely, but it parses
*every* module in the repository to do so, and that bill is paid again for every
instruction of every attempt.

The index answers one much cheaper question — *which files contain this
identifier at all?* — from a local SQLite database, so Jedi only has to resolve
the handful of files that can possibly reference the symbol. Jedi still decides
every edit: the index proposes candidate positions and Jedi verifies each one by
resolving it. A stale or missing entry can therefore only cost a candidate, never
a wrong rewrite, and the atoms fall back to a full Jedi scan when the database
cannot be used.

Why ``ast`` and not libcst
--------------------------
libcst is mandatory for *rewriting* (R2): it is what keeps comments, quoting and
indentation byte-identical. This module only reads. Extracting a symbol table is
a pure name walk, where the standard library ``ast`` is far cheaper than a libcst
``MetadataWrapper``/``PositionProvider`` pass — measured on 88-line modules:
0.5 ms/file against 23.4 ms/file. On a 5,000-file repository that is the
difference between a 2.5 s cold build and a 2-minute one, which decides whether
the cache is worth having at all. Nothing here writes source code; every byte
that reaches a file still comes out of libcst.

What is stored
--------------
``symbols`` holds one row per identifier occurrence: its spelling, a descriptive
kind (``class``, ``method``, ``function``, ``param``, ``variable``, ``import``,
``attribute``, ``reference``), the dotted scope chain that encloses it
(``Widget.ping``) and its file/line. Columns deliberately do not live here:
positions are recomputed with libcst on the few candidate files, which keeps the
index free of the character/byte column traps and guarantees the positions handed
to Jedi are the same ones the rewriter uses.

Keyword arguments never appear: ``ast`` models ``call(name=1)`` as a ``keyword``
node with a plain string, and Jedi's project reference search does not report
keyword arguments as references to the parameter either. Leaving them out of the
index keeps the two paths agreeing on what a rename may touch.

Incremental refresh
-------------------
``files`` stores ``(mtime_ns, size, sha256)``. A refresh stats the tree and
re-parses only files whose mtime or size moved; a touch that leaves the content
alone is caught by the digest and costs no parse. Files that disappeared are
dropped with their symbols, and files the engine itself rewrote are re-indexed
immediately through :meth:`Indexer.invalidate`.

Every statement below is a literal with ``?`` placeholders: no SQL text is ever
built by concatenation, interpolation or formatting, and no query string is
passed in through a variable.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import sqlite3
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from moray_surgeon.exceptions import CacheError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

CACHE_DIRECTORY = ".moray-cache"
DATABASE_FILENAME = "index.sqlite3"
# Bumped whenever the table layout changes; a mismatch rebuilds the cache, which
# is always safe because every row can be re-derived from the sources.
SCHEMA_VERSION = 1

_SQLITE_TIMEOUT_SECONDS = 5.0
_SCHEMA_VERSION_KEY = "schema_version"
_GIT_EXCLUDE_MARKER = "# moray-surgeon: local code index (safe to delete)"
_SKIP_DIR_NAMES = frozenset(
    {
        CACHE_DIRECTORY,
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


@dataclass(frozen=True)
class SymbolHit:
    """One indexed identifier occurrence.

    Attributes:
        file: Repository-relative POSIX path of the file holding the symbol.
        name: The identifier exactly as spelled in the source.
        kind: Descriptive role, e.g. ``class``, ``method`` or ``param``.
        scope: Dotted chain of enclosing definitions, ``""`` at module level.
        line: 1-based line of the occurrence.
    """

    file: str
    name: str
    kind: str
    scope: str
    line: int


@dataclass(frozen=True)
class IndexStats:
    """Outcome of one :meth:`Indexer.refresh` call.

    Attributes:
        total_files: Python files covered by the index after the refresh.
        parsed_files: Files (re)parsed during this refresh.
        unchanged_files: Files skipped because mtime and size were unchanged.
        rehashed_files: Files whose mtime moved but whose content did not, so the
            digest matched and no parse was needed.
        removed_files: Entries dropped because the file no longer exists.
        failed_files: Files that could not be read or parsed; they are kept in the
            index without symbols so a broken file does not force a re-parse on
            every refresh.
        symbols: Total symbol rows in the database after the refresh.
        elapsed_seconds: Wall-clock duration of the refresh.
    """

    total_files: int
    parsed_files: int
    unchanged_files: int
    rehashed_files: int
    removed_files: int
    failed_files: int
    symbols: int
    elapsed_seconds: float


@dataclass(frozen=True)
class IndexSummary:
    """Read-only description of the on-disk database.

    Attributes:
        path: Location of the SQLite file.
        files: Indexed Python files.
        symbols: Indexed symbol rows.
        size_bytes: Size of the database file, 0 when it does not exist.
    """

    path: Path
    files: int
    symbols: int
    size_bytes: int


@dataclass(frozen=True)
class _RawSymbol:
    """A symbol extracted from one module, before it is written to SQLite."""

    name: str
    kind: str
    scope: str
    line: int


class _SymbolVisitor(ast.NodeVisitor):
    """Collect ``(name, kind, scope, line)`` for every identifier in a module.

    The kind is descriptive rather than normative: only the *name* is load-bearing
    for lookups, while ``kind`` and ``scope`` make the database readable for
    introspection and give the engine a way to explain a hit.
    """

    def __init__(self) -> None:
        self._scopes: list[tuple[str, str]] = []
        self.symbols: list[_RawSymbol] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record the class, then walk its body inside the class scope."""
        self._record(node.name, "class", node)
        self._scopes.append(("class", node.name))
        self.generic_visit(node)
        self._scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record a function or method together with its parameters."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """``async def`` is indexed exactly like ``def``."""
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Lambdas have parameters but no name of their own."""
        self._scopes.append(("function", "<lambda>"))
        for argument in _iter_arguments(node.args):
            self._record(argument.arg, "param", argument)
        self.generic_visit(node)
        self._scopes.pop()

    def visit_Import(self, node: ast.Import) -> None:
        """``import a.b`` binds ``a``; ``import a as b`` binds ``b``."""
        for alias in node.names:
            binding = alias.asname or alias.name.split(".")[0]
            self._record(binding, "import", alias)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Both the imported name and its local alias are mentionable names."""
        for alias in node.names:
            if alias.name != "*":
                self._record(alias.name, "import", alias)
            if alias.asname:
                self._record(alias.asname, "import", alias)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Attribute names are candidates too: ``mod.Symbol`` is a real usage."""
        self._record(node.attr, "attribute", node)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        """A ``global`` declaration names a module-level binding."""
        for name in node.names:
            self._record(name, "variable", node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """A ``nonlocal`` declaration names an enclosing binding."""
        for name in node.names:
            self._record(name, "variable", node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """``except E as err`` binds ``err``."""
        if node.name:
            self._record(node.name, "variable", node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """A store is a binding; a load is a reference."""
        kind = "variable" if isinstance(node.ctx, ast.Store) else "reference"
        self._record(node.id, kind, node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        kind = "method" if self._scopes and self._scopes[-1][0] == "class" else "function"
        # The function name belongs to the enclosing scope; its parameters and
        # body belong to the function's own scope, which is why the push happens
        # between the two.
        self._record(node.name, kind, node)
        self._scopes.append(("function", node.name))
        for argument in _iter_arguments(node.args):
            self._record(argument.arg, "param", argument)
        self.generic_visit(node)
        self._scopes.pop()

    def _record(self, name: str, kind: str, node: ast.AST) -> None:
        self.symbols.append(
            _RawSymbol(
                name=name,
                kind=kind,
                scope=".".join(scope_name for _, scope_name in self._scopes),
                line=getattr(node, "lineno", 0),
            )
        )


def _iter_arguments(arguments: ast.arguments) -> Iterator[ast.arg]:
    """Yield every declared parameter, including ``*args`` and ``**kwargs``."""
    yield from arguments.posonlyargs
    yield from arguments.args
    yield from arguments.kwonlyargs
    if arguments.vararg is not None:
        yield arguments.vararg
    if arguments.kwarg is not None:
        yield arguments.kwarg


def _extract_symbols(raw: bytes) -> tuple[list[_RawSymbol], bool]:
    """Parse one module and return its symbols plus whether parsing succeeded.

    ``ast.parse`` accepts bytes and honours PEP 263 coding cookies and a BOM, so
    files that are not UTF-8 are indexed by their declared encoding.
    """
    try:
        module = ast.parse(raw)
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return [], False
    visitor = _SymbolVisitor()
    visitor.visit(module)
    return visitor.symbols, True


def _iter_python_files(root: Path) -> Iterator[Path]:
    """Yield the Python files of ``root``, skipping dependency and cache folders.

    Directory symlinks are never followed, and a file symlink is skipped when it
    resolves outside ``root``, so the index can only describe the repository it
    was built for.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIR_NAMES)
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = Path(dirpath) / filename
            if path.is_symlink() and not path.resolve().is_relative_to(root):
                logger.warning("skipping %s: it links outside the repository", path)
                continue
            yield path


class Indexer:
    """Queryable symbol table for one repository, cached in SQLite.

    Instances hold configuration only, never an open connection: every operation
    opens and closes its own SQLite handle, so a long-lived object (the CLI, the
    MCP server) cannot leak one and two processes cannot fight over a stale
    connection.

    Usage::

        indexer = Indexer(repo_root)
        indexer.refresh()                     # stats, then parses what changed
        for path in indexer.files_with("User"):
            ...
    """

    def __init__(self, repo_root: Path, cache_dir: Path | str | None = None) -> None:
        """Bind the index to a repository.

        Args:
            repo_root: Repository root; everything indexed stays inside it.
            cache_dir: Cache location. A relative path is resolved against
                ``repo_root``, which is what makes ``--path some/repo`` behave as
                expected from any working directory. Defaults to
                ``<repo_root>/.moray-cache``.
        """
        self._repo_root = repo_root.resolve()
        raw_cache = (
            self._repo_root / CACHE_DIRECTORY if cache_dir is None else Path(cache_dir)
        )
        self._cache_dir = (
            raw_cache if raw_cache.is_absolute() else self._repo_root / raw_cache
        ).resolve()
        self._needs_full_refresh = True
        self._pending: set[Path] = set()
        self._suspended = False

    @property
    def repo_root(self) -> Path:
        """The resolved repository root this index describes."""
        return self._repo_root

    @property
    def cache_dir(self) -> Path:
        """Directory holding the SQLite database."""
        return self._cache_dir

    @property
    def db_path(self) -> Path:
        """Path of the SQLite database file."""
        return self._cache_dir / DATABASE_FILENAME

    def refresh(self, *, force: bool = False) -> IndexStats:
        """Bring the index in line with the working tree, incrementally.

        Only files whose mtime or size changed are read; a changed mtime with an
        unchanged digest updates the stat row and skips the parse. Deleting a file
        drops its rows.

        Args:
            force: Re-parse every file, ignoring both the stat fast path and the
                digest check. Used by tests and after an out-of-band rewrite.

        Returns:
            Counters describing what this refresh did.

        Raises:
            CacheError: The cache directory or database could not be written.
        """
        started = time.perf_counter()
        self._prepare_cache_directory()
        parsed = unchanged = rehashed = removed = failed = 0
        connection = self._connect()
        try:
            self._create_schema(connection)
            known = {row[0] for row in connection.execute("SELECT path FROM files")}
            seen: set[str] = set()
            for path in _iter_python_files(self._repo_root):
                relative = path.relative_to(self._repo_root).as_posix()
                seen.add(relative)
                status = self._index_file(connection, path, force=force)
                if status == "parsed":
                    parsed += 1
                elif status == "unchanged":
                    unchanged += 1
                elif status == "rehashed":
                    rehashed += 1
                elif status == "failed":
                    failed += 1
                elif status == "removed":
                    removed += 1
            for stale in sorted(known - seen):
                self._drop_file(connection, stale)
                removed += 1
            connection.commit()
            total_files = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            symbols = connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        except sqlite3.Error as exc:
            raise CacheError(f"could not refresh the code index: {exc}") from exc
        finally:
            connection.close()
        self._needs_full_refresh = False
        self._pending.clear()
        self._suspended = False
        return IndexStats(
            total_files=int(total_files),
            parsed_files=parsed,
            unchanged_files=unchanged,
            rehashed_files=rehashed,
            removed_files=removed,
            failed_files=failed,
            symbols=int(symbols),
            elapsed_seconds=time.perf_counter() - started,
        )

    def ensure_ready(self) -> None:
        """Make sure queries see the current tree before they run.

        A refresh is skipped when this handle already scanned the tree and no
        write was reported since, because the atoms call this on every lookup and
        a stat walk of a large repository is not free (~0.09 s for 5,000 files).

        Raises:
            CacheError: The database does not exist and could not be built, or the
                index is suspended for the current run.
        """
        if self._suspended:
            raise CacheError(
                "the code index is suspended for this run after an earlier failure"
            )
        if self._needs_full_refresh or not self.db_path.exists():
            self.refresh()
            return
        if self._pending:
            pending = sorted(self._pending)
            self._pending.clear()
            self._refresh_paths(pending)

    def suspend(self) -> None:
        """Stop using the database for the rest of the run.

        The orchestrator suspends the handle when a refresh failed just before a
        transaction. Rebuilding the cache during that transaction would recreate
        a directory the stash may be holding, and ``git stash apply`` would then
        refuse to restore the user's files; falling back to a full Jedi scan for
        one run is the safe trade. A later successful :meth:`refresh` clears the
        suspension.
        """
        self._suspended = True

    @property
    def suspended(self) -> bool:
        """True while the handle refuses queries after a failed refresh."""
        return self._suspended

    def files_with(self, name: str, *, limit: int | None = None) -> tuple[Path, ...]:
        """Return absolute in-repo paths of files that mention ``name``.

        Args:
            name: Identifier to look up. Matched exactly against the symbol table.
            limit: Optional cap on the number of files returned.

        Returns:
            Existing files, ordered by repository-relative path. Empty when the
            identifier never appears.

        Raises:
            CacheError: The database could not be read.
        """
        if not name:
            return ()
        self.ensure_ready()
        connection = self._connect()
        try:
            if limit is None:
                rows = connection.execute(
                    "SELECT DISTINCT file FROM symbols WHERE name = ? ORDER BY file",
                    (name,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT DISTINCT file FROM symbols WHERE name = ? ORDER BY file LIMIT ?",
                    (name, max(1, limit)),
                ).fetchall()
        except sqlite3.Error as exc:
            raise CacheError(f"could not query the code index: {exc}") from exc
        finally:
            connection.close()
        return tuple(
            path for path in (self._repo_root / row[0] for row in rows) if path.is_file()
        )

    def file_count(self) -> int:
        """Number of Python files the index covers.

        Used by the project-aware atoms to decide whether resolving the
        identifier file by file is cheaper than one project-wide search.

        Returns:
            The number of indexed files.

        Raises:
            CacheError: The database could not be read.
        """
        self.ensure_ready()
        connection = self._connect()
        try:
            return int(connection.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        except sqlite3.Error as exc:
            raise CacheError(f"could not query the code index: {exc}") from exc
        finally:
            connection.close()

    def symbols_named(self, name: str, *, limit: int = 200) -> tuple[SymbolHit, ...]:
        """Return the indexed occurrences of one identifier, for inspection.

        The ``symbols`` table is declared as ``(file, name, kind, scope, line)``,
        which is the column order mapped here; ``SCHEMA_VERSION`` guards the
        layout, so the mapping cannot drift silently.

        Args:
            name: Identifier to look up.
            limit: Maximum number of rows returned.

        Returns:
            Hits ordered by file and line.

        Raises:
            CacheError: The database could not be read.
        """
        if not name:
            return ()
        self.ensure_ready()
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM symbols WHERE name = ? ORDER BY file, line LIMIT ?",
                (name, max(1, limit)),
            ).fetchall()
        except sqlite3.Error as exc:
            raise CacheError(f"could not query the code index: {exc}") from exc
        finally:
            connection.close()
        return tuple(
            SymbolHit(file=row[0], name=row[1], kind=row[2], scope=row[3], line=row[4])
            for row in rows
        )

    def invalidate(self, paths: Iterable[Path]) -> None:
        """Report files the engine just wrote, so their rows are rebuilt.

        Paths outside the repository are ignored: the cache describes this
        repository and nothing else.
        """
        for path in paths:
            resolved = path.resolve()
            if resolved.is_relative_to(self._repo_root):
                self._pending.add(resolved)

    def reindex(self, paths: Iterable[Path]) -> None:
        """Rebuild the rows of files the engine has just written.

        This is :meth:`invalidate` followed by an immediate refresh of exactly
        those files, which keeps the next instruction of the same plan from
        querying rows that describe the pre-edit sources. It never walks the tree,
        so it costs one parse per written file.

        Args:
            paths: Files to re-read. Paths outside the repository are ignored.

        Raises:
            CacheError: The database could not be updated, or the index is
                suspended for the current run.
        """
        if self._suspended:
            raise CacheError(
                "the code index is suspended for this run after an earlier failure"
            )
        written = [
            path.resolve() for path in paths if path.resolve().is_relative_to(self._repo_root)
        ]
        if not written:
            return
        self._refresh_paths(sorted(set(written)))
        for path in written:
            self._pending.discard(path)

    def clear(self) -> bool:
        """Delete the on-disk index.

        Returns:
            True when a database existed and was removed, False when there was
            nothing to remove.

        Raises:
            CacheError: The database file exists but could not be deleted.
        """
        removed = False
        for suffix in ("", "-wal", "-shm"):
            target = self._cache_dir / f"{DATABASE_FILENAME}{suffix}"
            try:
                if target.exists():
                    target.unlink()
                    removed = True
            except OSError as exc:
                raise CacheError(f"could not delete {target}: {exc}") from exc
        if self._cache_dir.is_dir() and not any(self._cache_dir.iterdir()):
            self._cache_dir.rmdir()
        self._needs_full_refresh = True
        self._pending.clear()
        return removed

    def summary(self) -> IndexSummary:
        """Describe the database without touching the sources.

        Returns:
            Counters and the file size; zeros when no index has been built yet.

        Raises:
            CacheError: The database exists but could not be read.
        """
        if not self.db_path.exists():
            return IndexSummary(path=self.db_path, files=0, symbols=0, size_bytes=0)
        connection = self._connect()
        try:
            self._create_schema(connection)
            files = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            symbols = connection.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        except sqlite3.Error as exc:
            raise CacheError(f"could not read the code index: {exc}") from exc
        finally:
            connection.close()
        return IndexSummary(
            path=self.db_path,
            files=int(files),
            symbols=int(symbols),
            size_bytes=self.db_path.stat().st_size,
        )

    def _prepare_cache_directory(self) -> None:
        """Create the cache directory and keep it out of version control."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CacheError(
                f"could not create the cache directory {self._cache_dir}: {exc}"
            ) from exc
        _ignore_in_git(self._repo_root, self._cache_dir)

    def _connect(self) -> sqlite3.Connection:
        """Open a tuned connection, translating driver errors into ``CacheError``."""
        try:
            connection = sqlite3.connect(str(self.db_path), timeout=_SQLITE_TIMEOUT_SECONDS)
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.Error as exc:
            raise CacheError(f"could not open the code index at {self.db_path}: {exc}") from exc
        return connection

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        """Create the tables, or rebuild them when the layout version moved.

        The version lives in ``meta`` rather than ``PRAGMA user_version`` so it is
        read and written with bound parameters, like every other value here.
        """
        try:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = ?", (_SCHEMA_VERSION_KEY,)
            ).fetchone()
        except sqlite3.Error:
            row = None  # no meta table yet: an empty or pre-meta database
        if row is not None and row[0] == str(SCHEMA_VERSION):
            return
        connection.executescript(
            """
            DROP TABLE IF EXISTS symbols;
            DROP TABLE IF EXISTS files;
            DROP TABLE IF EXISTS meta;
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE files (
                path TEXT PRIMARY KEY,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                digest TEXT NOT NULL,
                symbol_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE symbols (
                file TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                scope TEXT NOT NULL,
                line INTEGER NOT NULL
            );
            CREATE INDEX symbols_name ON symbols (name);
            CREATE INDEX symbols_file ON symbols (file);
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (_SCHEMA_VERSION_KEY, str(SCHEMA_VERSION)),
        )

    def _index_file(self, connection: sqlite3.Connection, path: Path, *, force: bool) -> str:
        """(Re)index one file and return what happened.

        Returns one of ``parsed``, ``unchanged``, ``rehashed``, ``failed`` or
        ``removed`` so the caller can fill :class:`IndexStats` without duplicating
        the decision logic.
        """
        relative = path.relative_to(self._repo_root).as_posix()
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._drop_file(connection, relative)
            return "removed"
        except OSError as exc:
            logger.warning("cannot stat %s: %s", path, exc)
            return "failed"
        previous = connection.execute(
            "SELECT mtime_ns, size, digest FROM files WHERE path = ?", (relative,)
        ).fetchone()
        if (
            not force
            and previous is not None
            and previous[0] == stat.st_mtime_ns
            and previous[1] == stat.st_size
        ):
            return "unchanged"
        try:
            raw = path.read_bytes()
        except OSError as exc:
            logger.warning("cannot read %s: %s", path, exc)
            return "failed"
        digest = hashlib.sha256(raw).hexdigest()
        if not force and previous is not None and previous[2] == digest:
            # The file was touched (mtime moved) but its content is identical, so
            # the symbols are still correct and re-parsing would be wasted work.
            connection.execute(
                "UPDATE files SET mtime_ns = ?, size = ? WHERE path = ?",
                (stat.st_mtime_ns, stat.st_size, relative),
            )
            return "rehashed"
        symbols, parsed = _extract_symbols(raw)
        connection.execute("DELETE FROM symbols WHERE file = ?", (relative,))
        connection.executemany(
            "INSERT INTO symbols (file, name, kind, scope, line) VALUES (?, ?, ?, ?, ?)",
            [(relative, symbol.name, symbol.kind, symbol.scope, symbol.line) for symbol in symbols],
        )
        connection.execute(
            "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?)",
            (relative, stat.st_mtime_ns, stat.st_size, digest, len(symbols)),
        )
        if not parsed:
            # Keep the row so an unparseable file is not re-read on every refresh.
            logger.warning("could not parse %s; indexed without symbols", path)
            return "failed"
        return "parsed"

    def _drop_file(self, connection: sqlite3.Connection, relative: str) -> None:
        """Remove every row belonging to one file."""
        connection.execute("DELETE FROM symbols WHERE file = ?", (relative,))
        connection.execute("DELETE FROM files WHERE path = ?", (relative,))

    def _refresh_paths(self, paths: list[Path]) -> None:
        """Re-index the files the engine itself wrote since the last scan.

        Raises:
            CacheError: The database could not be opened or written.
        """
        self._prepare_cache_directory()
        connection = self._connect()
        try:
            self._create_schema(connection)
            for path in paths:
                if path.is_relative_to(self._repo_root):
                    self._index_file(connection, path, force=True)
            connection.commit()
        except sqlite3.Error as exc:
            raise CacheError(f"could not update the code index: {exc}") from exc
        finally:
            connection.close()


_INDEXERS: dict[tuple[str, str], Indexer] = {}


def get_indexer(repo_root: Path, cache_dir: Path | str | None = None) -> Indexer:
    """Return the process-wide :class:`Indexer` handle for a repository.

    Handles are stateless apart from their incremental-refresh bookkeeping, so
    sharing one per ``(repo_root, cache_dir)`` pair lets the orchestrator refresh
    once per attempt while every atom inside that attempt reuses the result
    instead of walking the tree again.

    Args:
        repo_root: Repository root.
        cache_dir: Optional cache location; part of the identity of the handle.

    Returns:
        The shared handle, created on first use.
    """
    key = (str(Path(repo_root).resolve()), "" if cache_dir is None else str(cache_dir))
    handle = _INDEXERS.get(key)
    if handle is None:
        handle = Indexer(repo_root, cache_dir)
        _INDEXERS[key] = handle
    return handle


def reset_indexers() -> None:
    """Forget every cached handle; the next lookup starts from a fresh scan.

    Used by tests and by long-lived hosts that want to release references to
    repositories they no longer serve.
    """
    _INDEXERS.clear()


def clear_cache(repo_root: Path, cache_dir: Path | str | None = None) -> bool:
    """Delete the on-disk index for a repository.

    Args:
        repo_root: Repository root.
        cache_dir: Optional cache location, mirroring :class:`Indexer`.

    Returns:
        True when a database existed and was removed.

    Raises:
        CacheError: The database exists but could not be deleted.
    """
    return Indexer(repo_root, cache_dir).clear()


def _git_directory(repo_root: Path) -> Path | None:
    """The repository's git directory, following a worktree's ``gitdir`` file."""
    candidate = repo_root / ".git"
    if candidate.is_dir():
        return candidate
    if not candidate.is_file():
        return None
    try:
        text = candidate.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    prefix = "gitdir:"
    if not text.lower().startswith(prefix):
        return None
    target = Path(text[len(prefix) :].strip())
    return target if target.is_absolute() else (repo_root / target)


def _ignore_in_git(repo_root: Path, cache_dir: Path) -> None:
    """Add the cache directory to ``.git/info/exclude`` once.

    ``TransactionManager`` stages with ``git add -A``, so an unignored cache would
    be committed together with the refactor and would also take part in the stash
    round-trip. ``.git/info/exclude`` is the local, never-published ignore file, so
    writing one line there keeps the user's ``.gitignore`` untouched. Failing to
    write it only means the cache may show up as untracked, which is why the error
    is logged instead of raised.
    """
    if not cache_dir.is_relative_to(repo_root):
        return
    relative = cache_dir.relative_to(repo_root).as_posix()
    if not relative or relative == ".":
        return
    git_dir = _git_directory(repo_root)
    if git_dir is None:
        return
    exclude = git_dir / "info" / "exclude"
    pattern = f"/{relative}/"
    try:
        existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        if pattern in existing.splitlines():
            return
        exclude.parent.mkdir(parents=True, exist_ok=True)
        separator = "" if not existing or existing.endswith("\n") else "\n"
        exclude.write_text(
            f"{existing}{separator}{_GIT_EXCLUDE_MARKER}\n{pattern}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("could not add %s to %s: %s", pattern, exclude, exc)
