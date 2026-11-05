"""Tests for the LanguageBackend seam: registry, ABC contract and dispatch routing.

The routing table is exercised end to end through ``execute_plan`` with a
recording backend, so a schema/engine drift (an instruction type with no atom
method, or unpacked arguments that stop matching an atom signature) fails here
instead of surfacing as a wrong edit in some future backend.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import libcst as cst
import pytest

from moray_surgeon.engine.backends import (
    LanguageBackend,
    PythonBackend,
    backend_for,
    register_backend,
)
from moray_surgeon.engine.backends import base as backends_base
from moray_surgeon.engine.base import EditResult
from moray_surgeon.engine.dispatcher import execute_plan
from moray_surgeon.exceptions import UnsupportedLanguageError
from moray_surgeon.schema import (
    AddImport,
    AddTypeHints,
    ConvertToMatchCase,
    ExtractFunction,
    ExtractVariable,
    InlineFunction,
    InlineVariable,
    MoveSymbol,
    RefactorPlan,
    RemoveDeadCode,
    RemoveImport,
    RenameSymbol,
    ReplaceCall,
    SplitClass,
)

_PY_SOURCE = 'def old_name():\n    return old_name()\n'


class RecordingBackend(LanguageBackend):
    """Implements every atom by recording the call and echoing the source."""

    language = "recording"
    suffixes = (".rec",)

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def _record(self, name: str, args: tuple[object, ...], source: str) -> EditResult:
        self.calls.append((name, args))
        return EditResult(new_source=source, changes=1)

    def parse(self, source: str) -> object:
        self.calls.append(("parse", (source,)))
        return object()

    def verify_syntax(self, source: str) -> bool:
        self.calls.append(("verify_syntax", (source,)))
        return True

    def rename_symbol(
        self,
        repo_root: Path,
        target_file: str,
        old_name: str,
        new_name: str,
        indexer: object | None = None,
    ) -> EditResult:
        return self._record(
            "rename_symbol", (repo_root, target_file, old_name, new_name, indexer), ""
        )

    def extract_function(
        self, source: str, function_name: str, start_line: int, end_line: int
    ) -> EditResult:
        return self._record(
            "extract_function", (source, function_name, start_line, end_line), source
        )

    def extract_variable(
        self,
        source: str,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
        variable_name: str,
    ) -> EditResult:
        return self._record(
            "extract_variable",
            (source, start_line, start_column, end_line, end_column, variable_name),
            source,
        )

    def inline_variable(
        self, source: str, variable_name: str, repo_root: Path, target_file: Path
    ) -> EditResult:
        return self._record(
            "inline_variable", (source, variable_name, repo_root, target_file), source
        )

    def inline_function(
        self, source: str, function_name: str, repo_root: Path, target_path: Path
    ) -> EditResult:
        return self._record(
            "inline_function", (source, function_name, repo_root, target_path), source
        )

    def remove_dead_code(
        self, source: str, symbol_name: str, repo_root: Path, target_path: Path
    ) -> EditResult:
        return self._record(
            "remove_dead_code", (source, symbol_name, repo_root, target_path), source
        )

    def split_class(
        self,
        source: str,
        class_name: str,
        methods_to_move: Sequence[str],
        new_class_name: str,
    ) -> EditResult:
        return self._record(
            "split_class", (source, class_name, methods_to_move, new_class_name), source
        )

    def replace_call(self, source: str, old_call: str, new_call: str) -> EditResult:
        return self._record("replace_call", (source, old_call, new_call), source)

    def add_type_hints(self, source: str, function_name: str | None = None) -> EditResult:
        return self._record("add_type_hints", (source, function_name), source)

    def add_import(
        self, source: str, module: str, name: str | None = None, alias: str | None = None
    ) -> EditResult:
        return self._record("add_import", (source, module, name, alias), source)

    def remove_import(self, source: str, module: str, name: str | None = None) -> EditResult:
        return self._record("remove_import", (source, module, name), source)

    def convert_to_match_case(
        self, source: str, start_line: int, end_line: int
    ) -> EditResult:
        return self._record("convert_to_match_case", (source, start_line, end_line), source)

    def move_symbol(
        self,
        repo_root: Path,
        source_path: Path,
        target_path: Path,
        symbol_name: str,
        indexer: object | None = None,
    ) -> EditResult:
        return self._record(
            "move_symbol", (repo_root, source_path, target_path, symbol_name, indexer), ""
        )


@pytest.fixture
def clean_registry(monkeypatch):
    """Swap in an empty registry so test registrations never leak."""
    monkeypatch.setattr(backends_base, "_BACKENDS", {})
    return backends_base._BACKENDS


def test_language_backend_is_abstract():
    with pytest.raises(TypeError):
        LanguageBackend()  # type: ignore[abstract]


def test_python_backend_implements_the_full_contract():
    backend = PythonBackend()

    assert isinstance(backend, LanguageBackend)
    assert backend.language == "python"
    assert backend.suffixes == (".py", ".pyi")


def test_parse_returns_libcst_module_preserving_comments():
    module = PythonBackend().parse("def f():\n    return 1  # keep me\n")

    assert isinstance(module, cst.Module)
    assert "# keep me" in module.code


def test_verify_syntax_accepts_valid_python():
    assert PythonBackend().verify_syntax(_PY_SOURCE) is True
    assert PythonBackend().verify_syntax("") is True  # an empty module parses


def test_verify_syntax_rejects_broken_python():
    assert PythonBackend().verify_syntax("def f(:\n    pass\n") is False


def test_backend_for_resolves_python_suffixes(tmp_path):
    assert isinstance(backend_for(tmp_path / "mod.py"), PythonBackend)
    # Suffix lookups are case-insensitive, and stub files are Python too.
    assert isinstance(backend_for(tmp_path / "mod.PYI"), PythonBackend)


def test_backend_for_unknown_suffix_raises_with_registered_list(tmp_path):
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        backend_for(tmp_path / "main.rs")

    assert "'.rs'" in str(excinfo.value)
    assert ".pyi" in str(excinfo.value)  # the registered suffix list is in the message
    assert ".ts" in str(excinfo.value)  # TypeScript is registered too


def test_backend_for_suffixless_file_raises(tmp_path):
    with pytest.raises(UnsupportedLanguageError):
        backend_for(tmp_path / "Makefile")


def test_register_backend_claims_its_suffixes(clean_registry):
    backend = RecordingBackend()
    register_backend(backend)

    assert backend_for(Path("x.rec")) is backend


def test_register_backend_last_claim_wins(clean_registry):
    first, second = PythonBackend(), PythonBackend()
    register_backend(first)
    register_backend(second)

    assert backend_for(Path("x.py")) is second


# (atom method, instruction factory) — one case per single-file instruction type.
_ROUTING_CASES: list[tuple[str, Callable[[str], object]]] = [
    (
        "rename_symbol",
        lambda tf: RenameSymbol(id="i1", target_file=tf, old_name="old", new_name="new"),
    ),
    (
        "extract_function",
        lambda tf: ExtractFunction(
            id="i1", target_file=tf, function_name="f", start_line=1, end_line=2
        ),
    ),
    (
        "convert_to_match_case",
        lambda tf: ConvertToMatchCase(id="i1", target_file=tf, start_line=1, end_line=3),
    ),
    (
        "extract_variable",
        lambda tf: ExtractVariable(
            id="i1",
            target_file=tf,
            start_line=1,
            start_column=0,
            end_line=1,
            end_column=5,
            variable_name="v",
        ),
    ),
    (
        "inline_variable",
        lambda tf: InlineVariable(id="i1", target_file=tf, variable_name="v"),
    ),
    (
        "inline_function",
        lambda tf: InlineFunction(id="i1", target_file=tf, function_name="f"),
    ),
    (
        "remove_dead_code",
        lambda tf: RemoveDeadCode(id="i1", target_file=tf, function_name="f"),
    ),
    (
        "split_class",
        lambda tf: SplitClass(
            id="i1",
            target_file=tf,
            class_name="C",
            methods_to_move=["m"],
            new_class_name="D",
        ),
    ),
    (
        "replace_call",
        lambda tf: ReplaceCall(id="i1", target_file=tf, old_call="f()", new_call="g()"),
    ),
    (
        "add_type_hints",
        lambda tf: AddTypeHints(id="i1", target_file=tf, function_name="f"),
    ),
    ("add_import", lambda tf: AddImport(id="i1", target_file=tf, module="os")),
    ("remove_import", lambda tf: RemoveImport(id="i1", target_file=tf, module="os")),
]


@pytest.mark.parametrize(
    ("method", "factory"),
    _ROUTING_CASES,
    ids=[method for method, _ in _ROUTING_CASES],
)
def test_execute_plan_routes_every_single_file_instruction(
    clean_registry, tmp_path, method, factory
):
    (tmp_path / "mod.rec").write_text("placeholder\n", encoding="utf-8")
    backend = RecordingBackend()
    register_backend(backend)
    plan = RefactorPlan(instructions=[factory("mod.rec")])  # type: ignore[arg-type]

    execute_plan(plan, tmp_path)

    assert [name for name, _ in backend.calls] == [method]


def test_execute_plan_hands_atoms_exact_unpacked_arguments(clean_registry, tmp_path):
    source = "x = 1\n"
    (tmp_path / "mod.rec").write_text(source, encoding="utf-8")
    backend = RecordingBackend()
    register_backend(backend)
    plan = RefactorPlan(
        instructions=[
            ExtractVariable(
                id="i1",
                target_file="mod.rec",
                start_line=1,
                start_column=0,
                end_line=1,
                end_column=5,
                variable_name="v",
            )
        ]
    )

    execute_plan(plan, tmp_path)

    assert backend.calls == [
        ("extract_variable", (source, 1, 0, 1, 5, "v")),
    ]


def test_execute_plan_gives_rename_repository_context(clean_registry, tmp_path):
    (tmp_path / "mod.rec").write_text("placeholder\n", encoding="utf-8")
    backend = RecordingBackend()
    register_backend(backend)
    plan = RefactorPlan(
        instructions=[
            RenameSymbol(id="i1", target_file="mod.rec", old_name="old", new_name="new")
        ]
    )

    execute_plan(plan, tmp_path)

    name, args = backend.calls[0]
    assert name == "rename_symbol"
    assert args[0] == tmp_path.resolve()  # repo_root, resolved
    assert args[1] == "mod.rec"  # the instruction's target_file string, as authored
    assert args[2] == "old"
    assert args[3] == "new"
    assert args[4] is not None  # the shared indexer handle


def test_execute_plan_routes_move_symbol(clean_registry, tmp_path):
    (tmp_path / "a.rec").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.rec").write_text("y = 2\n", encoding="utf-8")
    backend = RecordingBackend()
    register_backend(backend)
    plan = RefactorPlan(
        instructions=[
            MoveSymbol(id="i1", source_file="a.rec", target_file="b.rec", symbol_name="x")
        ]
    )

    execute_plan(plan, tmp_path)

    name, args = backend.calls[0]
    assert name == "move_symbol"
    assert args[0] == tmp_path.resolve()
    assert args[1] == (tmp_path / "a.rec").resolve()
    assert args[2] == (tmp_path / "b.rec").resolve()
    assert args[3] == "x"
    assert args[4] is not None


def test_execute_plan_rejects_unsupported_language_before_touching_the_file(tmp_path):
    """Schema accepts any suffix; the dispatcher is the language gatekeeper."""
    (tmp_path / "app.rs").write_text("fn main() {}\n", encoding="utf-8")
    plan = RefactorPlan(
        instructions=[
            ReplaceCall(id="i1", target_file="app.rs", old_call="a()", new_call="b()")
        ]
    )

    with pytest.raises(UnsupportedLanguageError) as excinfo:
        execute_plan(plan, tmp_path)

    assert "'.rs'" in str(excinfo.value)
    # Nothing was read, parsed or written: the file is byte-for-byte intact.
    assert (tmp_path / "app.rs").read_text(encoding="utf-8") == "fn main() {}\n"


def test_execute_plan_rejects_move_to_unregistered_language(tmp_path):
    (tmp_path / "a.py").write_text(_PY_SOURCE, encoding="utf-8")
    (tmp_path / "b.rs").write_text("// rust\n", encoding="utf-8")
    plan = RefactorPlan(
        instructions=[
            MoveSymbol(id="i1", source_file="a.py", target_file="b.rs", symbol_name="old_name")
        ]
    )

    with pytest.raises(UnsupportedLanguageError, match=r"'\.rs'"):
        execute_plan(plan, tmp_path)


def test_execute_plan_rejects_move_across_different_backends(clean_registry, tmp_path):
    """Both suffixes registered but owned by different languages: refuse the move."""
    (tmp_path / "a.py").write_text(_PY_SOURCE, encoding="utf-8")
    (tmp_path / "b.rec").write_text("placeholder\n", encoding="utf-8")
    register_backend(PythonBackend())
    register_backend(RecordingBackend())
    plan = RefactorPlan(
        instructions=[
            MoveSymbol(id="i1", source_file="a.py", target_file="b.rec", symbol_name="old_name")
        ]
    )

    with pytest.raises(UnsupportedLanguageError, match="differ"):
        execute_plan(plan, tmp_path)


def test_apply_rejects_instruction_types_outside_the_schema(tmp_path):
    """Schema/engine drift fails loudly instead of silently editing nothing."""
    with pytest.raises(TypeError, match="unsupported instruction type"):
        PythonBackend().apply(
            object(),  # type: ignore[arg-type]
            "source",
            tmp_path,
            tmp_path / "mod.py",
            indexer=None,
        )
