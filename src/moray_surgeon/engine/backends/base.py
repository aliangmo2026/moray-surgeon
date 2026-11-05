"""Language-neutral contract for the surgical edit engine.

A :class:`LanguageBackend` owns every per-language operation the dispatcher
needs: parsing, syntax probing and one method per AST-surgery atom. The
dispatcher stays language-blind — it resolves a backend from the target file's
suffix and hands instructions over — while the schema instructions remain the
single language-agnostic wire protocol.

The reference implementation is
:class:`moray_surgeon.engine.backends.python_backend.PythonBackend` (libcst for
the AST surgery, Jedi for project-aware reference resolution). Adding a
language means subclassing this ABC, implementing every atom for that
language's toolchain and calling :func:`register_backend` with the suffixes it
claims; the dispatcher, the schema and the orchestrator are untouched.

Method signatures deliberately mirror the legacy atom functions one to one, so
a backend is a transparent seam: single-file atoms are ``source`` in /
:class:`EditResult` out; the project-aware atoms (``rename_symbol``,
``move_symbol``, ``inline_variable``, ``inline_function``, ``remove_dead_code``)
also receive repository context and may rewrite related files themselves, in
place, inside the caller's transaction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from moray_surgeon.exceptions import UnsupportedLanguageError
from moray_surgeon.schema.instructions import (
    AddImport,
    AddTypeHints,
    ConvertToMatchCase,
    ExtractFunction,
    ExtractVariable,
    InlineFunction,
    InlineVariable,
    Instruction,
    RemoveDeadCode,
    RemoveImport,
    RenameSymbol,
    ReplaceCall,
    SplitClass,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moray_surgeon.engine.base import EditResult
    from moray_surgeon.engine.indexer import Indexer

_BACKENDS: dict[str, LanguageBackend] = {}


def register_backend(backend: LanguageBackend) -> None:
    """Claim every suffix in ``backend.suffixes`` for ``backend``.

    Re-registering a suffix replaces the previous claim, which makes test
    doubles and third-party overrides possible. Registration is idempotent per
    suffix: the mapping is the single lookup the dispatcher performs.

    Args:
        backend: A fully constructible backend instance (backends are
            stateless; repository context travels through method arguments).
    """
    for suffix in backend.suffixes:
        _BACKENDS[suffix.lower()] = backend


def registered_suffixes() -> Iterator[str]:
    """Yield the suffixes that currently have a backend, sorted for stability."""
    return iter(sorted(_BACKENDS))


def backend_for(path: Path) -> LanguageBackend:
    """Resolve the backend that owns ``path``'s suffix.

    Args:
        path: A target (or source) file path; only its suffix is inspected.

    Returns:
        The registered backend instance for the suffix (case-insensitive).

    Raises:
        UnsupportedLanguageError: No backend registered for the suffix —
            including files without any suffix. Raised before any byte of the
            file is read, so an unsupported target can never be half-edited.
    """
    suffix = path.suffix.lower()
    backend = _BACKENDS.get(suffix)
    if backend is None:
        known = ", ".join(registered_suffixes())
        raise UnsupportedLanguageError(
            f"no LanguageBackend for {suffix!r} files (registered: {known})"
        )
    return backend


class LanguageBackend(ABC):
    """Per-language implementation of the AST surgery atoms.

    Attributes:
        language: Human-readable language name, for logs and error messages.
        suffixes: File suffixes this backend claims when registered. Case is
            irrelevant; lookups normalise to lowercase.
    """

    language: ClassVar[str]
    suffixes: ClassVar[tuple[str, ...]]

    @abstractmethod
    def parse(self, source: str) -> object:
        """Parse ``source`` into the backend's native AST handle.

        The handle is opaque to the rest of the engine: only the backend's own
        atoms interpret it. Implementations must preserve everything that is
        not the edit (comments, blank lines, quoting) — that is project rule
        R2, and it is why Python parses with libcst rather than ``ast``.
        """

    @abstractmethod
    def verify_syntax(self, source: str) -> bool:
        """Report whether ``source`` parses cleanly.

        Never raises: a syntax error is a ``False``, not an exception, so
        callers can use this as a cheap pre-flight gate.
        """

    def apply(
        self,
        instruction: Instruction,
        source: str,
        repo_root: Path,
        target: Path,
        indexer: Indexer | None,
    ) -> EditResult:
        """Route one non-move instruction to its atom method.

        Concrete on purpose: the instruction-type → method mapping is neutral
        protocol glue, identical for every backend, so a new language implements
        only the atoms below. ``MoveSymbol`` never reaches this method — the
        dispatcher routes it to :meth:`move_symbol` because a move writes
        several files itself.

        Args:
            instruction: A schema instruction (the language-agnostic protocol).
            source: Current text of ``target`` (read by the dispatcher).
            repo_root: Resolved repository root.
            target: Resolved, containment-checked path of ``target_file``.
            indexer: Code-index handle for the project-aware atoms.

        Returns:
            The atom's result; ``new_source`` is written back by the dispatcher.

        Raises:
            TypeError: The instruction type has no atom (schema/engine drift).
        """
        if isinstance(instruction, RenameSymbol):
            return self.rename_symbol(
                repo_root, instruction.target_file, instruction.old_name, instruction.new_name,
                indexer,
            )
        if isinstance(instruction, ExtractFunction):
            return self.extract_function(
                source, instruction.function_name, instruction.start_line, instruction.end_line
            )
        if isinstance(instruction, ConvertToMatchCase):
            return self.convert_to_match_case(source, instruction.start_line, instruction.end_line)
        if isinstance(instruction, ExtractVariable):
            return self.extract_variable(
                source,
                instruction.start_line,
                instruction.start_column,
                instruction.end_line,
                instruction.end_column,
                instruction.variable_name,
            )
        if isinstance(instruction, InlineVariable):
            return self.inline_variable(source, instruction.variable_name, repo_root, target)
        if isinstance(instruction, InlineFunction):
            return self.inline_function(source, instruction.function_name, repo_root, target)
        if isinstance(instruction, SplitClass):
            return self.split_class(
                source,
                instruction.class_name,
                instruction.methods_to_move,
                instruction.new_class_name,
            )
        if isinstance(instruction, RemoveDeadCode):
            return self.remove_dead_code(source, instruction.symbol_name, repo_root, target)
        if isinstance(instruction, ReplaceCall):
            return self.replace_call(source, instruction.old_call, instruction.new_call)
        if isinstance(instruction, AddTypeHints):
            return self.add_type_hints(source, instruction.function_name)
        if isinstance(instruction, AddImport):
            return self.add_import(source, instruction.module, instruction.name, instruction.alias)
        if isinstance(instruction, RemoveImport):
            return self.remove_import(source, instruction.module, instruction.name)
        raise TypeError(f"unsupported instruction type: {type(instruction).__name__}")

    @abstractmethod
    def rename_symbol(
        self,
        repo_root: Path,
        target_file: str,
        old_name: str,
        new_name: str,
        indexer: Indexer | None = None,
    ) -> EditResult:
        """Rename every project reference to ``old_name`` as defined in ``target_file``.

        Project-aware: reference resolution spans files. The implementation
        writes every affected file itself and returns the target file's final
        source (the dispatcher writes it back idempotently).
        """

    @abstractmethod
    def extract_function(
        self, source: str, function_name: str, start_line: int, end_line: int
    ) -> EditResult:
        """Move the statements fully inside ``[start_line, end_line]`` into a new function."""

    @abstractmethod
    def extract_variable(
        self,
        source: str,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
        variable_name: str,
    ) -> EditResult:
        """Extract the expression at ``[start, end]`` (absolute coordinates) into a variable."""

    @abstractmethod
    def inline_variable(
        self, source: str, variable_name: str, repo_root: Path, target_file: Path
    ) -> EditResult:
        """Inline the single assignment of ``variable_name`` into all of its reads."""

    @abstractmethod
    def inline_function(
        self, source: str, function_name: str, repo_root: Path, target_path: Path
    ) -> EditResult:
        """Inline the single-return ``function_name`` into all same-file call sites."""

    @abstractmethod
    def remove_dead_code(
        self, source: str, symbol_name: str, repo_root: Path, target_path: Path
    ) -> EditResult:
        """Delete the unreferenced top-level ``symbol_name`` from ``source``."""

    @abstractmethod
    def split_class(
        self,
        source: str,
        class_name: str,
        methods_to_move: Sequence[str],
        new_class_name: str,
    ) -> EditResult:
        """Move ``methods_to_move`` out of ``class_name`` into a new sibling class."""

    @abstractmethod
    def replace_call(self, source: str, old_call: str, new_call: str) -> EditResult:
        """Replace every call whose callee renders as ``old_call`` with ``new_call``."""

    @abstractmethod
    def add_type_hints(self, source: str, function_name: str | None = None) -> EditResult:
        """Add missing return annotations; ``function_name=None`` targets the whole file."""

    @abstractmethod
    def add_import(
        self, source: str, module: str, name: str | None = None, alias: str | None = None
    ) -> EditResult:
        """Add one module-level import, keeping the import block sorted."""

    @abstractmethod
    def remove_import(
        self, source: str, module: str, name: str | None = None
    ) -> EditResult:
        """Remove module-level imports of ``module``; ``name`` narrows ``from`` imports."""

    @abstractmethod
    def convert_to_match_case(
        self, source: str, start_line: int, end_line: int
    ) -> EditResult:
        """Rewrite the if/elif/else chain inside ``[start_line, end_line]`` into ``match``."""

    @abstractmethod
    def move_symbol(
        self,
        repo_root: Path,
        source_path: Path,
        target_path: Path,
        symbol_name: str,
        indexer: Indexer | None = None,
    ) -> EditResult:
        """Move the top-level ``symbol_name`` from ``source_path`` to ``target_path``.

        Project-aware like :meth:`rename_symbol`: importer files are rewritten
        too, all writes happen inside the method, and the returned
        :class:`EditResult` reports the total number of changed files.
        """
