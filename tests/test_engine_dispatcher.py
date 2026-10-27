from __future__ import annotations

from pathlib import Path

import pytest

from moray_surgeon import exceptions as root_exceptions
from moray_surgeon.engine import execute_plan
from moray_surgeon.exceptions import (
    InstructionError,
    MorayError,
    NoOpEditError,
    PathTraversalError,
)
from moray_surgeon.schema import (
    AddTypeHints,
    ExtractFunction,
    Instruction,
    Noop,
    RefactorPlan,
    RenameSymbol,
    ReplaceCall,
)
from moray_surgeon.schema import exceptions as schema_exceptions


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _plan(*instructions: Instruction) -> RefactorPlan:
    return RefactorPlan(instructions=list(instructions))


def _unvalidated_plan(*instructions: Instruction) -> RefactorPlan:
    """Build a plan without schema validation.

    ``RenameSymbol``'s own validator rejects ``..`` and absolute paths, so the
    dispatcher's containment guard can only be exercised by a plan that skipped
    validation (``model_construct``), e.g. one deserialized by another layer.
    """
    return RefactorPlan.model_construct(instructions=list(instructions))


def _unsafe_rename(target_file: str, old_name: str, new_name: str) -> RenameSymbol:
    """A ``RenameSymbol`` whose ``target_file`` never went through validation."""
    return RenameSymbol.model_construct(
        target_file=target_file, old_name=old_name, new_name=new_name
    )


def test_rename_symbol_rewrites_file(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "def old_name():\n    return old_name()\n")

    log = execute_plan(
        _plan(RenameSymbol(target_file="app.py", old_name="old_name", new_name="new_name")),
        repo,
    )

    assert _read(target) == "def new_name():\n    return new_name()\n"
    assert log == ["[0] rename_symbol: 2 change(s) in app.py"]


def test_chained_instructions_apply_in_order(tmp_path):
    repo = tmp_path / "repo"
    target = _write(
        repo / "app.py",
        "import os\n\n\ndef fetch():\n    return os.getcwd()\n",
    )

    log = execute_plan(
        _plan(
            RenameSymbol(target_file="app.py", old_name="fetch", new_name="load"),
            ReplaceCall(target_file="app.py", old_call="os.getcwd", new_call="pathlib.Path.cwd"),
        ),
        repo,
    )

    assert _read(target) == "import os\n\n\ndef load():\n    return pathlib.Path.cwd()\n"
    assert log == [
        "[0] rename_symbol: 1 change(s) in app.py",
        "[1] replace_call: 1 change(s) in app.py",
    ]


def test_add_type_hints_dispatch(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "def helper():\n    pass\n")

    log = execute_plan(_plan(AddTypeHints(target_file="app.py")), repo)

    assert _read(target) == "def helper() -> None:\n    pass\n"
    assert log == ["[0] add_type_hints: 1 change(s) in app.py"]


def test_dotdot_target_raises_path_traversal(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    victim = _write(tmp_path / "evil.py", "VICTIM = 1\n")

    with pytest.raises(PathTraversalError):
        execute_plan(_unvalidated_plan(_unsafe_rename("../evil.py", "VICTIM", "PWNED")), repo)

    assert _read(victim) == "VICTIM = 1\n"
    assert not (repo / "evil.py").exists()


def test_nested_dotdot_target_raises_path_traversal(tmp_path):
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)

    # sub/../../../x.py escapes repo even though the escaped file does not exist;
    # the containment check must win over the existence check.
    with pytest.raises(PathTraversalError):
        execute_plan(
            _unvalidated_plan(_unsafe_rename("sub/../../../x.py", "VICTIM", "PWNED")), repo
        )

    assert not any(repo.rglob("*.py"))


def test_missing_target_raises_file_not_found(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(FileNotFoundError):
        execute_plan(
            _plan(RenameSymbol(target_file="missing.py", old_name="a", new_name="b")),
            repo,
        )


def test_directory_target_raises_file_not_found(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg.py").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        execute_plan(
            _plan(RenameSymbol(target_file="pkg.py", old_name="a", new_name="b")),
            repo,
        )


def test_replace_call_without_match_raises_noop_edit_error(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "print('hi')\n")

    with pytest.raises(NoOpEditError):
        execute_plan(
            _plan(
                ReplaceCall(
                    target_file="app.py",
                    old_call="json.dumps",
                    new_call="orjson.dumps",
                )
            ),
            repo,
        )

    assert _read(target) == "print('hi')\n"


def test_rename_symbol_without_match_raises_noop_edit_error(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "def present():\n    return 1\n")

    with pytest.raises(NoOpEditError):
        execute_plan(
            _plan(RenameSymbol(target_file="app.py", old_name="absent", new_name="other")),
            repo,
        )

    assert _read(target) == "def present():\n    return 1\n"


def test_invalid_extract_range_raises_instruction_error(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "x = 1\ny = 2\n")

    plan = _plan(
        ExtractFunction(target_file="app.py", function_name="helper", start_line=2, end_line=1)
    )

    with pytest.raises(InstructionError) as excinfo:
        execute_plan(plan, repo)

    assert excinfo.value.instruction_index == 0
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert isinstance(excinfo.value, MorayError)
    assert _read(target) == "x = 1\ny = 2\n"


def test_failure_stops_the_plan_and_keeps_earlier_edits(tmp_path):
    repo = tmp_path / "repo"
    first = _write(repo / "first.py", "def old():\n    return old()\n")
    _write(repo / "second.py", "VALUE = 1\n")
    third = _write(repo / "third.py", "NAME = 2\n")

    plan = _plan(
        RenameSymbol(target_file="first.py", old_name="old", new_name="new"),
        ReplaceCall(
            target_file="second.py",
            old_call="json.dumps",
            new_call="orjson.dumps",
        ),
        RenameSymbol(target_file="third.py", old_name="NAME", new_name="OTHER"),
    )

    with pytest.raises(NoOpEditError):
        execute_plan(plan, repo)

    assert _read(first) == "def new():\n    return new()\n"
    assert _read(third) == "NAME = 2\n"


def test_instruction_error_index_points_at_the_failing_instruction(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "ok.py", "def old():\n    return old()\n")
    _write(repo / "bad.py", "x = 1\n")

    plan = _plan(
        RenameSymbol(target_file="ok.py", old_name="old", new_name="new"),
        ExtractFunction(target_file="bad.py", function_name="helper", start_line=5, end_line=4),
    )

    with pytest.raises(InstructionError) as excinfo:
        execute_plan(plan, repo)

    assert excinfo.value.instruction_index == 1


def test_noop_is_skipped_and_does_not_shift_indexes(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "def old():\n    return old()\n")

    log = execute_plan(
        _plan(Noop(), RenameSymbol(target_file="app.py", old_name="old", new_name="new")),
        repo,
    )

    assert log == ["[0] noop: skipped", "[1] rename_symbol: 2 change(s) in app.py"]
    assert _read(target) == "def new():\n    return new()\n"


def test_noop_only_plan_writes_nothing_and_counts_no_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    log = execute_plan(_plan(Noop(message="nothing to do")), repo)

    assert log == ["[0] noop: skipped"]
    assert all("change(s)" not in entry for entry in log)
    assert not any(repo.iterdir())


def test_domain_errors_share_a_single_moray_error_base():
    assert root_exceptions.MorayError is schema_exceptions.MorayError
    assert issubclass(PathTraversalError, MorayError)
    assert issubclass(NoOpEditError, MorayError)
    assert issubclass(InstructionError, MorayError)
