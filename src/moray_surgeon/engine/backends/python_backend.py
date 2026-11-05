"""The Python :class:`LanguageBackend`: libcst for surgery, Jedi for references.

This module is the *seam*, not a rewrite: every atom method is a transparent
one-line delegate to the legacy atom function whose behavior 900 existing tests
pin down. libcst and Jedi become implementation details visible only here — the
dispatcher and the schema neither know nor care which toolchain runs beneath.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import libcst as cst

from moray_surgeon.engine.backends.base import LanguageBackend
from moray_surgeon.engine.convert_match import convert_to_match_case
from moray_surgeon.engine.extract import extract_function
from moray_surgeon.engine.extract_variable import extract_variable
from moray_surgeon.engine.import_manager import add_import, remove_import
from moray_surgeon.engine.inline_function import inline_function
from moray_surgeon.engine.inline_variable import inline_variable
from moray_surgeon.engine.move_symbol import move_symbol
from moray_surgeon.engine.remove_dead_code import remove_dead_code
from moray_surgeon.engine.rename import rename_symbol
from moray_surgeon.engine.replace_call import replace_call
from moray_surgeon.engine.split_class import split_class
from moray_surgeon.engine.type_hints import add_type_hints

if TYPE_CHECKING:
    from moray_surgeon.engine.base import EditResult
    from moray_surgeon.engine.indexer import Indexer


class PythonBackend(LanguageBackend):
    """Surgery atoms for Python sources.

    Rule R2 lives here: the AST handle is a libcst ``Module`` and every rewrite
    goes through libcst transforms, so comments, blank lines, quoting and
    indentation survive byte-for-byte on untouched nodes. Project-aware atoms
    use Jedi reference resolution narrowed by the code index.
    """

    language = "python"
    suffixes = (".py", ".pyi")

    def parse(self, source: str) -> object:
        """Parse ``source`` into a libcst ``Module`` (format-preserving)."""
        return cst.parse_module(source)

    def verify_syntax(self, source: str) -> bool:
        """Report whether libcst can parse ``source``; never raises."""
        try:
            cst.parse_module(source)
        except cst.ParserSyntaxError:
            return False
        return True

    def rename_symbol(
        self,
        repo_root: Path,
        target_file: str,
        old_name: str,
        new_name: str,
        indexer: Indexer | None = None,
    ) -> EditResult:
        """Delegate to the Jedi-backed cross-file rename (writes files itself)."""
        return rename_symbol(repo_root, target_file, old_name, new_name, indexer)

    def extract_function(
        self, source: str, function_name: str, start_line: int, end_line: int
    ) -> EditResult:
        """Delegate to the libcst extraction atom (top-level statements only)."""
        return extract_function(source, function_name, start_line, end_line)

    def extract_variable(
        self,
        source: str,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
        variable_name: str,
    ) -> EditResult:
        """Delegate to the libcst variable-extraction atom (absolute coordinates)."""
        return extract_variable(
            source,
            start_line,
            start_column,
            end_line,
            end_column,
            variable_name,
        )

    def inline_variable(
        self, source: str, variable_name: str, repo_root: Path, target_file: Path
    ) -> EditResult:
        """Delegate to the Jedi-narrowed inline atom (may rewrite importers)."""
        return inline_variable(source, variable_name, repo_root, target_file)

    def inline_function(
        self, source: str, function_name: str, repo_root: Path, target_path: Path
    ) -> EditResult:
        """Delegate to the single-return inline atom (may rewrite importers)."""
        return inline_function(source, function_name, repo_root, target_path)

    def remove_dead_code(
        self, source: str, symbol_name: str, repo_root: Path, target_path: Path
    ) -> EditResult:
        """Delegate to the reference-counted dead-code atom (may rewrite importers)."""
        return remove_dead_code(source, symbol_name, repo_root, target_path)

    def split_class(
        self,
        source: str,
        class_name: str,
        methods_to_move: Sequence[str],
        new_class_name: str,
    ) -> EditResult:
        """Delegate to the class-splitting atom (new sibling class)."""
        return split_class(source, class_name, methods_to_move, new_class_name)

    def replace_call(self, source: str, old_call: str, new_call: str) -> EditResult:
        """Delegate to the callee-substitution atom (arguments untouched)."""
        return replace_call(source, old_call, new_call)

    def add_type_hints(self, source: str, function_name: str | None = None) -> EditResult:
        """Delegate to the conservative return-annotation atom (``-> None`` only)."""
        return add_type_hints(source, function_name)

    def add_import(
        self, source: str, module: str, name: str | None = None, alias: str | None = None
    ) -> EditResult:
        """Delegate to the sorted-import-block insertion atom."""
        return add_import(source, module, name, alias)

    def remove_import(self, source: str, module: str, name: str | None = None) -> EditResult:
        """Delegate to the import-removal atom."""
        return remove_import(source, module, name)

    def convert_to_match_case(
        self, source: str, start_line: int, end_line: int
    ) -> EditResult:
        """Delegate to the provably-equivalent if/elif → match rewrite atom."""
        return convert_to_match_case(source, start_line, end_line)

    def move_symbol(
        self,
        repo_root: Path,
        source_path: Path,
        target_path: Path,
        symbol_name: str,
        indexer: Indexer | None = None,
    ) -> EditResult:
        """Delegate to the project-aware move atom (rewrites all affected files)."""
        return move_symbol(repo_root, source_path, target_path, symbol_name, indexer)
