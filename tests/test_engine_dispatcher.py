from __future__ import annotations

from pathlib import Path

import pytest

from moray_surgeon import exceptions as root_exceptions
from moray_surgeon.engine import execute_plan
from moray_surgeon.exceptions import (
    CircularDependencyError,
    InstructionError,
    MorayError,
    NoOpEditError,
    PathTraversalError,
    UnknownDependencyError,
)
from moray_surgeon.schema import (
    AddImport,
    AddTypeHints,
    ExtractFunction,
    ExtractVariable,
    InlineFunction,
    InlineVariable,
    Instruction,
    MoveSymbol,
    Noop,
    RefactorPlan,
    RemoveDeadCode,
    RemoveImport,
    RenameSymbol,
    ReplaceCall,
    SplitClass,
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
        id="unvalidated", target_file=target_file, old_name=old_name, new_name=new_name
    )


def test_rename_symbol_rewrites_file(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "def old_name():\n    return old_name()\n")

    log = execute_plan(
        _plan(
            RenameSymbol(id="s1", target_file="app.py", old_name="old_name", new_name="new_name")
        ),
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
            RenameSymbol(id="s2", target_file="app.py", old_name="fetch", new_name="load"),
            ReplaceCall(
                id="s3", target_file="app.py", old_call="os.getcwd", new_call="pathlib.Path.cwd"
            ),
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

    log = execute_plan(_plan(AddTypeHints(id="s4", target_file="app.py")), repo)

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
            _plan(RenameSymbol(id="s5", target_file="missing.py", old_name="a", new_name="b")),
            repo,
        )


def test_directory_target_raises_file_not_found(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg.py").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        execute_plan(
            _plan(RenameSymbol(id="s6", target_file="pkg.py", old_name="a", new_name="b")),
            repo,
        )


def test_replace_call_without_match_raises_noop_edit_error(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "print('hi')\n")

    with pytest.raises(NoOpEditError):
        execute_plan(
            _plan(
                ReplaceCall(
                    id="s7",
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
            _plan(RenameSymbol(id="s8", target_file="app.py", old_name="absent", new_name="other")),
            repo,
        )

    assert _read(target) == "def present():\n    return 1\n"


def test_invalid_extract_range_raises_instruction_error(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "x = 1\ny = 2\n")

    plan = _plan(
        ExtractFunction(
            id="s9", target_file="app.py", function_name="helper", start_line=2, end_line=1
        )
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
        RenameSymbol(id="s10", target_file="first.py", old_name="old", new_name="new"),
        ReplaceCall(
            id="s11",
            target_file="second.py",
            old_call="json.dumps",
            new_call="orjson.dumps",
        ),
        RenameSymbol(id="s12", target_file="third.py", old_name="NAME", new_name="OTHER"),
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
        RenameSymbol(id="s13", target_file="ok.py", old_name="old", new_name="new"),
        ExtractFunction(
            id="s14", target_file="bad.py", function_name="helper", start_line=5, end_line=4
        ),
    )

    with pytest.raises(InstructionError) as excinfo:
        execute_plan(plan, repo)

    assert excinfo.value.instruction_index == 1


def test_noop_is_skipped_and_does_not_shift_indexes(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "def old():\n    return old()\n")

    log = execute_plan(
        _plan(
            Noop(id="s15"),
            RenameSymbol(id="s16", target_file="app.py", old_name="old", new_name="new"),
        ),
        repo,
    )

    assert log == ["[0] noop: skipped", "[1] rename_symbol: 2 change(s) in app.py"]
    assert _read(target) == "def new():\n    return new()\n"


def test_noop_only_plan_writes_nothing_and_counts_no_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    log = execute_plan(_plan(Noop(id="s17", message="nothing to do")), repo)

    assert log == ["[0] noop: skipped"]
    assert all("change(s)" not in entry for entry in log)
    assert not any(repo.iterdir())


def test_domain_errors_share_a_single_moray_error_base():
    assert root_exceptions.MorayError is schema_exceptions.MorayError
    assert issubclass(PathTraversalError, MorayError)
    assert issubclass(NoOpEditError, MorayError)
    assert issubclass(InstructionError, MorayError)


# --------------------------------------------------------------------------- #
# Controlled imports
# --------------------------------------------------------------------------- #


def test_add_import_instruction_updates_the_file(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "import sys\n\nX = 1\n")

    log = execute_plan(_plan(AddImport(id="s18", target_file="app.py", module="os")), repo)

    assert log == ["[0] add_import: 1 change(s) in app.py"]
    assert _read(target) == "import os\nimport sys\n\nX = 1\n"


def test_add_import_merges_a_name_into_an_existing_from_statement(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "from collections import OrderedDict\n")

    execute_plan(
        _plan(AddImport(id="s19", target_file="app.py", module="collections", name="Counter")),
        repo,
    )

    assert _read(target) == "from collections import Counter, OrderedDict\n"


def test_remove_import_instruction_updates_the_file(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "import os\nimport sys\n")

    log = execute_plan(_plan(RemoveImport(id="s20", target_file="app.py", module="os")), repo)

    assert log == ["[0] remove_import: 1 change(s) in app.py"]
    assert _read(target) == "import sys\n"


def test_add_import_of_an_existing_import_is_a_noop_error(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "import os\n")

    with pytest.raises(NoOpEditError, match="add_import"):
        execute_plan(_plan(AddImport(id="s21", target_file="app.py", module="os")), repo)

    assert _read(target) == "import os\n"


def test_remove_import_of_a_missing_import_is_a_noop_error(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "import os\n")

    with pytest.raises(NoOpEditError, match="remove_import"):
        execute_plan(_plan(RemoveImport(id="s22", target_file="app.py", module="json")), repo)

    assert _read(target) == "import os\n"


def test_add_import_with_invalid_module_is_an_instruction_error(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "app.py", "import os\n")
    unsafe = AddImport.model_construct(id="unsafe", target_file="app.py", module="not a module")

    with pytest.raises(InstructionError) as excinfo:
        execute_plan(_plan(unsafe), repo)

    assert excinfo.value.instruction_index == 0


def test_import_instructions_run_in_plan_order(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "import sys\n")

    log = execute_plan(
        _plan(
            AddImport(id="s23", target_file="app.py", module="os"),
            RemoveImport(id="s24", target_file="app.py", module="sys"),
        ),
        repo,
    )

    assert log == [
        "[0] add_import: 1 change(s) in app.py",
        "[1] remove_import: 1 change(s) in app.py",
    ]
    assert _read(target) == "import os\n"


# --------------------------------------------------------------------------- #
# Dependency ordering (DAG)
# --------------------------------------------------------------------------- #


def test_dependencies_override_the_authored_order(tmp_path):
    repo = tmp_path / "repo"
    target = _write(
        repo / "app.py",
        "import os\n\n\ndef fetch():\n    return os.getcwd()\n",
    )

    log = execute_plan(
        _plan(
            AddImport(
                id="import-pathlib",
                target_file="app.py",
                module="pathlib",
                depends_on=["replace"],
            ),
            ReplaceCall(
                id="replace",
                target_file="app.py",
                old_call="os.getcwd",
                new_call="pathlib.Path.cwd",
                depends_on=["rename"],
            ),
            RenameSymbol(id="rename", target_file="app.py", old_name="fetch", new_name="load"),
        ),
        repo,
    )

    assert log == [
        "[0] rename_symbol: 1 change(s) in app.py",
        "[1] replace_call: 1 change(s) in app.py",
        "[2] add_import: 1 change(s) in app.py",
    ]
    assert _read(target) == (
        "import os\nimport pathlib\n\n\ndef load():\n    return pathlib.Path.cwd()\n"
    )


def test_independent_steps_keep_their_authored_order(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "import sys\n")

    log = execute_plan(
        _plan(
            AddImport(id="first", target_file="app.py", module="os"),
            RemoveImport(id="second", target_file="app.py", module="sys"),
        ),
        repo,
    )

    assert log == [
        "[0] add_import: 1 change(s) in app.py",
        "[1] remove_import: 1 change(s) in app.py",
    ]
    assert _read(target) == "import os\n"


def test_cyclic_plan_is_rejected_before_any_file_is_touched(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "def old():\n    return old()\n")

    plan = _plan(
        RenameSymbol(
            id="a", target_file="app.py", old_name="old", new_name="new", depends_on=["b"]
        ),
        RenameSymbol(
            id="b", target_file="app.py", old_name="new", new_name="newer", depends_on=["a"]
        ),
    )

    with pytest.raises(CircularDependencyError, match="a -> b -> a"):
        execute_plan(plan, repo)

    assert _read(target) == "def old():\n    return old()\n"


def test_unknown_dependency_is_rejected_before_any_file_is_touched(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "def old():\n    return old()\n")

    plan = _plan(
        RenameSymbol(
            id="a",
            target_file="app.py",
            old_name="old",
            new_name="new",
            depends_on=["missing"],
        )
    )

    with pytest.raises(UnknownDependencyError, match="missing"):
        execute_plan(plan, repo)

    assert _read(target) == "def old():\n    return old()\n"


def test_instruction_error_index_is_the_execution_position(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "app.py", "x = 1\ny = 2\n")

    plan = _plan(
        ExtractFunction(
            id="bad",
            target_file="app.py",
            function_name="helper",
            start_line=2,
            end_line=1,
            depends_on=["first"],
        ),
        RenameSymbol(id="first", target_file="app.py", old_name="x", new_name="z"),
    )

    with pytest.raises(InstructionError) as excinfo:
        execute_plan(plan, repo)

    # The rename is executed first, so the failing step is at execution index 1
    # even though it is authoring index 0.
    assert excinfo.value.instruction_index == 1


def test_extract_variable_dispatch(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "# price times count\nprint(price * count)\n")

    log = execute_plan(
        _plan(
            ExtractVariable(
                id="s1",
                target_file="app.py",
                start_line=2,
                start_column=6,
                end_line=2,
                end_column=19,
                variable_name="unit",
            )
        ),
        repo,
    )

    assert _read(target) == "# price times count\nunit = price * count\nprint(unit)\n"
    assert log == ["[0] extract_variable: 2 change(s) in app.py"]


def test_inline_variable_dispatch(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "total = price * count\nprint(total)\n")

    log = execute_plan(
        _plan(InlineVariable(id="s1", target_file="app.py", variable_name="total")),
        repo,
    )

    assert _read(target) == "print((price * count))\n"
    assert log == ["[0] inline_variable: 2 change(s) in app.py"]


def test_extract_then_inline_round_trip(tmp_path):
    """extract_variable hoists an expression; inlining it restores the reads."""
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "print(price * count)\n")
    original = _read(target)

    plan = _plan(
        ExtractVariable(
            id="extract",
            target_file="app.py",
            start_line=1,
            start_column=6,
            end_line=1,
            end_column=19,
            variable_name="total",
        ),
        InlineVariable(
            id="inline",
            target_file="app.py",
            variable_name="total",
            depends_on=["extract"],
        ),
    )
    execute_plan(plan, repo)

    # The inline reproduces the original expression, now with explicit grouping.
    assert _read(target) == "print((price * count))\n"
    assert original == "print(price * count)\n"


def test_refused_inline_raises_instruction_error(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "x = 1\nprint(x)\nx = 2\n")

    plan = _plan(InlineVariable(id="s1", target_file="app.py", variable_name="x"))

    with pytest.raises(InstructionError, match="inline_variable failed"):
        execute_plan(plan, repo)

    # A refusal must leave the file byte-for-byte untouched.
    assert _read(target) == "x = 1\nprint(x)\nx = 2\n"


def test_remove_dead_code_dispatch(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "def dead():\n    return 1\n\n\ndef live():\n    return 2\n")

    log = execute_plan(
        _plan(RemoveDeadCode(id="s1", target_file="app.py", function_name="dead")),
        repo,
    )

    assert _read(target) == "def live():\n    return 2\n"
    assert log == ["[0] remove_dead_code: 1 change(s) in app.py"]


def test_remove_dead_code_class_name_dispatch(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "class Legacy:\n    pass\n\n\ndef live():\n    return 2\n")

    log = execute_plan(
        _plan(RemoveDeadCode(id="s1", target_file="app.py", class_name="Legacy")),
        repo,
    )

    assert _read(target) == "def live():\n    return 2\n"
    assert log == ["[0] remove_dead_code: 1 change(s) in app.py"]


def test_move_symbol_dispatch(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "models.py", "class User:\n    pass\n")
    _write(repo / "entities.py", "")
    _write(repo / "service.py", "from models import User\nprint(User)\n")

    log = execute_plan(
        _plan(
            MoveSymbol(
                id="s1", source_file="models.py", target_file="entities.py", symbol_name="User"
            )
        ),
        repo,
    )

    assert _read(repo / "models.py") == "\n"
    assert _read(repo / "entities.py") == "class User:\n    pass"
    assert _read(repo / "service.py") == "from entities import User\nprint(User)\n"
    assert log == ["[0] move_symbol: 3 change(s) (models.py -> entities.py)"]


def test_refused_move_raises_instruction_error_and_touches_nothing(tmp_path):
    repo = tmp_path / "repo"
    models = _write(repo / "models.py", "class User:\n    pass\n")
    entities = _write(repo / "entities.py", "User = 1\n")

    plan = _plan(
        MoveSymbol(id="s1", source_file="models.py", target_file="entities.py", symbol_name="User")
    )

    with pytest.raises(InstructionError, match="move_symbol failed"):
        execute_plan(plan, repo)

    assert _read(models) == "class User:\n    pass\n"
    assert _read(entities) == "User = 1\n"


def test_inline_function_dispatch(tmp_path):
    repo = tmp_path / "repo"
    target = _write(
        repo / "app.py",
        "def add(a, b):\n    return a + b\n\n\nx = add(1, 2)\ny = add(x, 5)\n",
    )

    log = execute_plan(
        _plan(InlineFunction(id="s1", target_file="app.py", function_name="add")),
        repo,
    )

    assert _read(target) == "x = (1 + 2)\ny = (x + 5)\n"
    assert log == ["[0] inline_function: 3 change(s) in app.py"]


def test_split_class_dispatch(tmp_path):
    repo = tmp_path / "repo"
    math_src = (
        "class Math:\n"
        "    @staticmethod\n"
        "    def add(a, b):\n"
        "        return a + b\n"
        "\n"
        "    def use(self):\n"
        "        return 1\n"
    )
    target = _write(repo / "app.py", math_src)

    log = execute_plan(
        _plan(
            SplitClass(
                id="s1",
                target_file="app.py",
                class_name="Math",
                methods_to_move=["add"],
                new_class_name="MathOps",
            )
        ),
        repo,
    )

    text = _read(target)
    assert "def add" in text.split("class MathOps:")[1]
    assert log == ["[0] split_class: 2 change(s) in app.py"]


def test_refused_inline_function_raises_instruction_error(tmp_path):
    repo = tmp_path / "repo"
    target = _write(repo / "app.py", "def f(a):\n    x = a\n    return x\n\n\nv = f(1)\n")

    plan = _plan(InlineFunction(id="s1", target_file="app.py", function_name="f"))

    with pytest.raises(InstructionError, match="inline_function failed"):
        execute_plan(plan, repo)

    # A refusal must leave the file byte-for-byte untouched.
    assert _read(target) == "def f(a):\n    x = a\n    return x\n\n\nv = f(1)\n"


def test_refused_split_class_raises_instruction_error(tmp_path):
    repo = tmp_path / "repo"
    target = _write(
        repo / "app.py",
        "class Math:\n    def add(self, a):\n        return self.value + a\n",
    )
    before = _read(target)

    plan = _plan(
        SplitClass(
            id="s1",
            target_file="app.py",
            class_name="Math",
            methods_to_move=["add"],
            new_class_name="MathOps",
        )
    )

    with pytest.raises(InstructionError, match="split_class failed"):
        execute_plan(plan, repo)

    assert _read(target) == before


def test_move_symbol_missing_target_file_raises_file_not_found(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "models.py", "class User:\n    pass\n")

    plan = _plan(
        MoveSymbol(
            id="s1", source_file="models.py", target_file="missing.py", symbol_name="User"
        )
    )

    with pytest.raises(FileNotFoundError):
        execute_plan(plan, repo)
