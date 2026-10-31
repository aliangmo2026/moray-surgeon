"""Shared project-reference guards for the surgical atoms.

The delete / inline / move family all need the same three proofs before they
may rewrite a symbol, and they all fail closed the same way:

* no string literal anywhere in the repository equals the symbol name (a
  potential ``getattr`` / ``setattr`` / registry key Jedi cannot see);
* no module star-imports the symbol's module (re-exports are invisible);
* Jedi's project-wide reference search *and* an independent libcst candidate
  scan agree the symbol is not used outside the module the atom edits.

The wording of every refusal is parameterised by the operation verb so each
atom can say what it was about to do when it refused.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import jedi
import libcst as cst
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider

from moray_surgeon.engine.inline_variable import _python_files
from moray_surgeon.engine.rename import _jedi_reference_budget


def plain_string_value(node: cst.SimpleString) -> str | None:
    """The unquoted value of a simple string, or None for exotic spellings."""
    match = re.match(r"^[rbuRBU]{0,2}(['\"])(.*?)\1$", node.value, re.DOTALL)
    return match.group(2) if match else None


def assert_no_dynamic_string_refs(
    repo_root: Path,
    target_path: Path,
    symbol_name: str,
    span: CodeRange,
    operation: str,
) -> None:
    """Refuse when any string literal equals the symbol name outside its span.

    Strings inside the symbol's own span (its docstring) are unaffected by the
    rewrite and are skipped.
    """
    pattern = re.compile(rf"['\"]{re.escape(symbol_name)}['\"]")
    for path in _python_files(repo_root):
        text = path.read_text(encoding="utf-8")
        if not pattern.search(text):
            continue
        wrapper = MetadataWrapper(cst.parse_module(text))
        scanner = _StringScan(
            symbol_name=symbol_name,
            positions=wrapper.resolve(PositionProvider),
            path=path,
            in_target=path.resolve() == target_path.resolve(),
            span=span,
            operation=operation,
        )
        wrapper.visit(scanner)


class _StringScan(cst.CSTVisitor):
    """Collect exact-match string literals that could be dynamic references."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        symbol_name: str,
        positions: Mapping[cst.CSTNode, CodeRange],
        path: Path,
        in_target: bool,
        span: CodeRange,
        operation: str,
    ) -> None:
        self._symbol_name = symbol_name
        self._positions = positions
        self._path = path
        self._in_target = in_target
        self._span = span
        self._operation = operation

    def visit_SimpleString(self, node: cst.SimpleString) -> bool:  # noqa: N802
        if plain_string_value(node) != self._symbol_name:
            return True
        position = self._positions[node]
        if self._in_target and self._span.start.line <= position.start.line <= self._span.end.line:
            return True  # inside the symbol; rewritten along with it
        raise ValueError(
            f"a {self._symbol_name!r} string literal at {self._path.name}:"
            f"{position.start.line} could be a dynamic reference "
            f"(getattr/setattr/registry); refusing to {self._operation}"
        )


def assert_no_star_import(repo_root: Path, target_path: Path, operation: str) -> None:
    """``from <this module> import *`` may re-export the symbol invisibly."""
    for path in _python_files(repo_root):
        if path.resolve() == target_path.resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if "import *" not in text:
            continue
        wrapper = MetadataWrapper(cst.parse_module(text))
        wrapper.visit(
            _StarScan(
                path=path,
                repo_root=repo_root,
                target_path=target_path,
                positions=wrapper.resolve(PositionProvider),
                operation=operation,
            )
        )


class _StarScan(cst.CSTVisitor):
    """Veto star imports of the target module; their re-exports are invisible."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        path: Path,
        repo_root: Path,
        target_path: Path,
        positions: Mapping[cst.CSTNode, CodeRange],
        operation: str,
    ) -> None:
        self._path = path
        self._repo_root = repo_root
        self._target_path = target_path
        self._positions = positions
        self._operation = operation

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:  # noqa: N802
        if not isinstance(node.names, cst.ImportStar):
            return True
        if node.module is None:
            raise ValueError(
                f"a 'from . import *' in {self._path.name} cannot be traced; "
                f"refusing to {self._operation}"
            )
        start = self._positions[node.module].start
        definitions = jedi.Script(
            path=str(self._path), project=jedi.Project(path=str(self._repo_root))
        ).goto(start.line, start.column, follow_imports=True)
        for definition in definitions:
            module_path = definition.module_path
            if module_path is None:
                continue
            if Path(module_path).resolve() == self._target_path.resolve():
                raise ValueError(
                    f"{self._path.name} star-imports {self._target_path.name}; the "
                    "symbol may be re-exported dynamically"
                )
        if not definitions:
            raise ValueError(
                f"a star import in {self._path.name} cannot be resolved; "
                f"refusing to {self._operation}"
            )
        return True


def assert_not_referenced_from_other_modules(
    repo_root: Path,
    target_path: Path,
    symbol_name: str,
    name_start: tuple[int, int],
    span: CodeRange,
    operation: str,
    *,
    include_same_file: bool,
) -> None:
    """Jedi and an independent candidate scan must agree on the symbol's uses.

    With ``include_same_file`` a reference from the edited module itself also
    refuses (the dead-code atom: any surviving use means "not dead"); without
    it, same-module references are expected — the inline atoms rewrite them —
    and only other modules refuse.
    """
    try:
        with _jedi_reference_budget(repo_root):
            project = jedi.Project(path=str(repo_root))
            script = jedi.Script(path=str(target_path), project=project)
            references = script.get_references(name_start[0], name_start[1], scope="project")
    except Exception as exc:
        raise ValueError(
            f"could not verify references for {symbol_name!r} ({exc}); "
            f"refusing to {operation}"
        ) from None

    for reference in references:
        if reference.module_path is None:
            raise ValueError(
                f"a reference to {symbol_name!r} could not be attributed to a "
                f"module; refusing to {operation}"
            )
        reference_path = Path(reference.module_path).resolve()
        in_target = reference_path == target_path.resolve()
        if in_target and not include_same_file:
            continue
        inside_own_span = in_target and span.start.line <= reference.line <= span.end.line
        if not inside_own_span:
            raise ValueError(
                f"{symbol_name!r} is referenced at {reference_path.name}:"
                f"{reference.line}; refusing to {operation}"
            )

    for path in _python_files(repo_root):
        if not include_same_file and path.resolve() == target_path.resolve():
            continue
        _audit_reference_candidates(
            repo_root, path, target_path, symbol_name, name_start, span, operation
        )


def _audit_reference_candidates(
    repo_root: Path,
    path: Path,
    target_path: Path,
    symbol_name: str,
    name_start: tuple[int, int],
    span: CodeRange,
    operation: str,
) -> None:
    """Second opinion: every spelled-like-the-symbol site must resolve elsewhere.

    Independent of ``get_references`` on purpose: if Jedi ever returned an
    incomplete reference set, this scan still catches the use (or refuses as
    uncertain instead of rewriting live code).
    """
    wrapper = MetadataWrapper(cst.parse_module(path.read_text(encoding="utf-8")))
    scanner = _CandidateScan(
        symbol_name=symbol_name,
        positions=wrapper.resolve(PositionProvider),
        path=path,
        in_target=path.resolve() == target_path.resolve(),
        span=span,
    )
    wrapper.visit(scanner)

    script = jedi.Script(path=str(path), project=jedi.Project(path=str(repo_root)))
    for line, column in scanner.sites:
        try:
            definitions = script.goto(line, column, follow_imports=True)
        except Exception as exc:
            raise ValueError(
                f"could not verify a use of {symbol_name!r} in {path.name} ({exc}); "
                f"refusing to {operation}"
            ) from None
        for definition in definitions:
            module_path = definition.module_path
            if module_path is None:
                continue
            if Path(module_path).resolve() == target_path.resolve() and (
                definition.line,
                definition.column,
            ) == name_start:
                raise ValueError(
                    f"{symbol_name!r} is referenced at {path.name}:{line}; "
                    f"refusing to {operation}"
                )
        if not definitions:
            raise ValueError(
                f"a use of {symbol_name!r} in {path.name} (line {line}) cannot be "
                f"resolved; refusing to {operation} rather than guess"
            )


class _CandidateScan(cst.CSTVisitor):
    """Collect every site spelled like the symbol that must resolve elsewhere."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        symbol_name: str,
        positions: Mapping[cst.CSTNode, CodeRange],
        path: Path,
        in_target: bool,
        span: CodeRange,
    ) -> None:
        self._symbol_name = symbol_name
        self._positions = positions
        self._path = path
        self._in_target = in_target
        self._span = span
        self.sites: list[tuple[int, int]] = []

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:  # noqa: N802
        if isinstance(node.names, cst.ImportStar):
            return True
        for alias in node.names:
            if not (isinstance(alias.name, cst.Name) and alias.name.value == self._symbol_name):
                continue
            if node.module is None:
                self.sites.append((1, 0))
            else:
                module_start = self._positions[node.module].start
                self.sites.append((module_start.line, module_start.column))
        return True

    def visit_Name(self, node: cst.Name) -> bool:  # noqa: N802
        if node.value != self._symbol_name:
            return True
        start = self._positions[node].start
        if self._in_target and self._span.start.line <= start.line <= self._span.end.line:
            return True  # inside the symbol; rewritten along with it
        self.sites.append((start.line, start.column))
        return True
