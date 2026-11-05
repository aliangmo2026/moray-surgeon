"""Tests for multi-process instruction execution and dispatcher batching.

The interesting properties are not "it is faster" but "it is the same": a
parallel batch must produce byte-identical files and the same log order as a
sequential run, and a failing worker must leave every file of the batch
untouched. The batching policy (file-disjoint, dependency-free) is tested with a
recording double so the exact batch boundaries can be asserted.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from moray_surgeon.engine import concurrency as concurrency_module
from moray_surgeon.engine import dispatcher as dispatcher_module
from moray_surgeon.engine import execute_plan
from moray_surgeon.engine.base import EditResult
from moray_surgeon.engine.concurrency import (
    DEFAULT_WORKER_CAP,
    MAX_WORKERS_ENV,
    InstructionTask,
    apply_task,
    atomic_write_text,
    default_max_workers,
    execute_parallel,
    is_parallel_safe,
    resolve_max_workers,
)
from moray_surgeon.exceptions import (
    ConcurrencyError,
    NoOpEditError,
    UnsupportedLanguageError,
)
from moray_surgeon.schema import (
    AddImport,
    AddTypeHints,
    ConvertToMatchCase,
    ExtractFunction,
    ExtractVariable,
    InlineFunction,
    InlineVariable,
    MoveSymbol,
    Noop,
    RefactorPlan,
    RemoveDeadCode,
    RemoveImport,
    RenameSymbol,
    ReplaceCall,
    SplitClass,
)

_SOURCE = "def f(x):\n    return x\n\n\ndef g(y):\n    return y + 1\n"


def _write(path: Path, source: str = _SOURCE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _hints(file_name: str, step: str, **kwargs: object) -> AddTypeHints:
    return AddTypeHints(id=step, target_file=file_name, function_name=None, **kwargs)  # type: ignore[arg-type]


def _task(index: int, root: Path, file_name: str, instruction: object) -> InstructionTask:
    return InstructionTask(
        index=index,
        instruction=instruction,  # type: ignore[arg-type]
        repo_root=root,
        target=root / file_name,
    )


# --------------------------------------------------------------------------- #
# Worker-count resolution
# --------------------------------------------------------------------------- #


def test_default_workers_respect_a_small_machine(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 3)

    assert default_max_workers() == 3


def test_default_workers_are_capped(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 64)

    assert default_max_workers() == DEFAULT_WORKER_CAP


def test_default_workers_survive_an_unknown_cpu_count(monkeypatch):
    """``os.cpu_count()`` may return None; the pool still needs a size."""
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    assert default_max_workers() == 1


def test_resolve_max_workers_prefers_the_explicit_value(monkeypatch):
    monkeypatch.setenv(MAX_WORKERS_ENV, "2")

    assert resolve_max_workers(5) == 5


def test_resolve_max_workers_clamps_below_one(monkeypatch):
    assert resolve_max_workers(0) == 1
    assert resolve_max_workers(-4) == 1


def test_resolve_max_workers_reads_the_environment(monkeypatch):
    monkeypatch.setenv(MAX_WORKERS_ENV, "3")

    assert resolve_max_workers() == 3


def test_resolve_max_workers_ignores_garbage_with_a_warning(monkeypatch, caplog):
    monkeypatch.setenv(MAX_WORKERS_ENV, "many")
    monkeypatch.setattr(os, "cpu_count", lambda: 2)

    with caplog.at_level("WARNING"):
        assert resolve_max_workers() == 2

    assert MAX_WORKERS_ENV in caplog.text
    assert "not an integer" in caplog.text


def test_resolve_max_workers_ignores_non_positive_environment_values(monkeypatch, caplog):
    monkeypatch.setenv(MAX_WORKERS_ENV, "0")
    monkeypatch.setattr(os, "cpu_count", lambda: 2)

    with caplog.at_level("WARNING"):
        assert resolve_max_workers() == 2

    assert "must be positive" in caplog.text


# --------------------------------------------------------------------------- #
# Parallel-safety classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "instruction",
    [
        AddImport(id="s", target_file="a.py", module="os"),
        AddTypeHints(id="s", target_file="a.py", function_name=None),
        ConvertToMatchCase(id="s", target_file="a.py", start_line=1, end_line=2),
        ExtractFunction(id="s", target_file="a.py", function_name="h", start_line=1, end_line=1),
        ExtractVariable(
            id="s", target_file="a.py", start_line=1, start_column=0,
            end_line=1, end_column=1, variable_name="v",
        ),
        RemoveImport(id="s", target_file="a.py", module="os"),
        ReplaceCall(id="s", target_file="a.py", old_call="f()", new_call="g()"),
        SplitClass(
            id="s", target_file="a.py", class_name="A", methods_to_move=["m"],
            new_class_name="B",
        ),
    ],
)
def test_single_file_atoms_are_parallel_safe(instruction):
    assert is_parallel_safe(instruction)


@pytest.mark.parametrize(
    "instruction",
    [
        RenameSymbol(id="s", target_file="a.py", old_name="a", new_name="b"),
        MoveSymbol(id="s", source_file="a.py", target_file="b.py", symbol_name="f"),
        InlineVariable(id="s", target_file="a.py", variable_name="v"),
        InlineFunction(id="s", target_file="a.py", function_name="f"),
        RemoveDeadCode(id="s", target_file="a.py", function_name="f"),
    ],
)
def test_project_aware_atoms_stay_in_the_parent(instruction):
    """They read or write other files, so a worker could see a stale tree."""
    assert not is_parallel_safe(instruction)


# --------------------------------------------------------------------------- #
# Worker entry point and atomic writes
# --------------------------------------------------------------------------- #


def test_apply_task_rewrites_source_without_writeback(tmp_path):
    target = _write(tmp_path / "a.py")

    result = apply_task(
        _task(0, tmp_path, "a.py", AddTypeHints(id="s", target_file="a.py", function_name=None))
    )

    assert result.changes == 2
    assert "-> None" in result.new_source
    # The worker must not touch the disk: the parent owns every write.
    assert target.read_text(encoding="utf-8") == _SOURCE


def test_apply_task_rejects_an_unregistered_language(tmp_path):
    target = tmp_path / "a.rs"
    target.write_text("fn main() {}\n", encoding="utf-8")

    with pytest.raises(UnsupportedLanguageError, match=r"'\.rs'"):
        apply_task(
            _task(
                0,
                tmp_path,
                "a.rs",
                ReplaceCall(id="s", target_file="a.rs", old_call="a()", new_call="b()"),
            )
        )


def test_atomic_write_replaces_content_and_leaves_no_temp_file(tmp_path):
    target = _write(tmp_path / "a.py", "old\n")

    atomic_write_text(target, "new\n")

    assert target.read_text(encoding="utf-8") == "new\n"
    assert [entry.name for entry in tmp_path.iterdir()] == ["a.py"]


def test_atomic_write_keeps_the_original_when_the_rename_fails(tmp_path, monkeypatch):
    target = _write(tmp_path / "a.py", "old\n")

    def explode(source: object, destination: object) -> None:
        raise OSError("rename refused")

    monkeypatch.setattr(os, "replace", explode)

    with pytest.raises(OSError, match="rename refused"):
        atomic_write_text(target, "new\n")

    assert target.read_text(encoding="utf-8") == "old\n"
    assert [entry.name for entry in tmp_path.iterdir()] == ["a.py"]


# --------------------------------------------------------------------------- #
# execute_parallel
# --------------------------------------------------------------------------- #


def test_execute_parallel_without_tasks_is_a_no_op():
    assert execute_parallel([], max_workers=4) == {}


def test_execute_parallel_keys_results_by_task_index(tmp_path):
    for name in ("a.py", "b.py", "c.py"):
        _write(tmp_path / name)
    tasks = [
        _task(
            index,
            tmp_path,
            name,
            AddTypeHints(id=f"s{index}", target_file=name, function_name=None),
        )
        for index, name in enumerate(("a.py", "b.py", "c.py"))
    ]

    results = execute_parallel(tasks, max_workers=2)

    assert sorted(results) == [0, 1, 2]
    assert all(result.changes == 2 for result in results.values())
    # Nothing was written by the pool.
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == _SOURCE


def test_execute_parallel_single_task_never_starts_a_pool(tmp_path, monkeypatch):
    _write(tmp_path / "a.py")

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("a single task must not spawn a process pool")

    monkeypatch.setattr("moray_surgeon.engine.concurrency.ProcessPoolExecutor", explode)

    results = execute_parallel(
        [_task(0, tmp_path, "a.py", AddTypeHints(id="s", target_file="a.py", function_name=None))],
        max_workers=4,
    )

    assert results[0].changes == 2


def test_execute_parallel_falls_back_when_the_pool_cannot_start(tmp_path, monkeypatch):
    _write(tmp_path / "a.py")

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("spawn is not allowed here")

    monkeypatch.setattr("moray_surgeon.engine.concurrency.ProcessPoolExecutor", explode)
    tasks = [
        _task(0, tmp_path, "a.py", AddTypeHints(id="s", target_file="a.py", function_name=None))
    ]

    results = execute_parallel(tasks, max_workers=4)

    assert results[0].changes == 2


def test_execute_parallel_wraps_a_worker_failure(tmp_path):
    _write(tmp_path / "a.py")
    _write(tmp_path / "b.py")
    tasks = [
        _task(0, tmp_path, "a.py", AddTypeHints(id="s0", target_file="a.py", function_name=None)),
        _task(
            1,
            tmp_path,
            "b.py",
            ExtractFunction(
                id="s1", target_file="b.py", function_name="h", start_line=900, end_line=920
            ),
        ),
    ]

    with pytest.raises(ConcurrencyError) as excinfo:
        execute_parallel(tasks, max_workers=2)

    error = excinfo.value
    assert error.instruction_index == 1
    assert "b.py" in str(error)
    assert "cancelled" in str(error)
    assert isinstance(error.__cause__, ValueError)


class _RecordingFuture:
    """Stand-in for a concurrent.futures Future that records cancellation."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> bool:
        self.cancelled = True
        return True


def test_cancellation_reaches_every_pending_task():
    """The failure path must cancel queued work instead of letting it run."""
    futures = (_RecordingFuture(), _RecordingFuture())

    concurrency_module._cancel_pending({future: None for future in futures})  # type: ignore[dict-item]

    assert [future.cancelled for future in futures] == [True, True]


# --------------------------------------------------------------------------- #
# Dispatcher batching policy (recording double)
# --------------------------------------------------------------------------- #


class _RecordingParallel:
    """Record the batch shape and return a marker edit for every task."""

    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def __call__(self, tasks, max_workers) -> dict[int, EditResult]:
        self.batches.append([task.index for task in tasks])
        return {
            task.index: EditResult(
                new_source=task.target.read_text(encoding="utf-8") + "# parallel\n",
                changes=1,
            )
            for task in tasks
        }


def _install_recorder(monkeypatch) -> _RecordingParallel:
    recorder = _RecordingParallel()
    monkeypatch.setattr(dispatcher_module, "execute_parallel", recorder)
    return recorder


def test_disjoint_independent_instructions_share_one_batch(tmp_path, monkeypatch):
    for name in ("a.py", "b.py", "c.py"):
        _write(tmp_path / name)
    recorder = _install_recorder(monkeypatch)
    plan = RefactorPlan(
        instructions=[
            _hints("a.py", "s0"),
            _hints("b.py", "s1"),
            _hints("c.py", "s2"),
        ]
    )

    log = execute_plan(plan, tmp_path, max_workers=4)

    assert recorder.batches == [[0, 1, 2]]
    assert [line.split(":")[0] for line in log] == [
        "[0] add_type_hints",
        "[1] add_type_hints",
        "[2] add_type_hints",
    ]
    assert all(
        "# parallel" in (tmp_path / name).read_text(encoding="utf-8")
        for name in ("a.py", "b.py", "c.py")
    )


def test_two_instructions_on_one_file_never_share_a_batch(tmp_path, monkeypatch):
    _write(tmp_path / "a.py")
    _write(tmp_path / "b.py")
    recorder = _install_recorder(monkeypatch)
    plan = RefactorPlan(
        instructions=[_hints("a.py", "s0"), _hints("a.py", "s1"), _hints("b.py", "s2")]
    )

    execute_plan(plan, tmp_path, max_workers=4)

    assert recorder.batches == [[1, 2]]


def test_a_dependency_edge_splits_the_batch(tmp_path, monkeypatch):
    for name in ("a.py", "b.py", "c.py", "d.py"):
        _write(tmp_path / name)
    recorder = _install_recorder(monkeypatch)
    plan = RefactorPlan(
        instructions=[
            _hints("a.py", "s0"),
            _hints("b.py", "s1", depends_on=["s0"]),
            _hints("c.py", "s2"),
            _hints("d.py", "s3"),
        ]
    )

    execute_plan(plan, tmp_path, max_workers=4)

    # s1 cannot join s0; s2/s3 can join s1 once the batch restarted.
    assert recorder.batches == [[1, 2, 3]]


def test_project_aware_instruction_flushes_the_batch(tmp_path, monkeypatch):
    _write(tmp_path / "a.py")
    _write(tmp_path / "b.py", "def old_name():\n    return 1\n")
    _write(tmp_path / "c.py")
    recorder = _install_recorder(monkeypatch)
    plan = RefactorPlan(
        instructions=[
            _hints("a.py", "s0"),
            RenameSymbol(id="s1", target_file="b.py", old_name="old_name", new_name="new_name"),
            _hints("c.py", "s2"),
        ]
    )

    execute_plan(plan, tmp_path, max_workers=4)

    assert recorder.batches == []
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "def new_name():\n    return 1\n"


def test_noop_splits_the_batch_and_keeps_its_log_position(tmp_path, monkeypatch):
    _write(tmp_path / "a.py")
    _write(tmp_path / "b.py")
    recorder = _install_recorder(monkeypatch)
    plan = RefactorPlan(
        instructions=[_hints("a.py", "s0"), Noop(id="s1"), _hints("b.py", "s2")]
    )

    log = execute_plan(plan, tmp_path, max_workers=4)

    assert recorder.batches == []
    assert log[1] == "[1] noop: skipped"
    assert log[0].startswith("[0] add_type_hints")
    assert log[2].startswith("[2] add_type_hints")


# --------------------------------------------------------------------------- #
# End-to-end behaviour
# --------------------------------------------------------------------------- #


def test_parallel_run_matches_a_sequential_run_byte_for_byte(tmp_path):
    templates = {
        "a.py": _SOURCE,
        "b.py": "def alpha(x):\n    return x\n",
        "c.py": "def beta():\n    return old()\n",
        "d.py": "def gamma(v):\n    return v * 2\n",
        "e.py": "def delta():\n    return old()\n",
        "f.py": "def epsilon(z):\n    return z\n",
    }
    sequential_root = tmp_path / "sequential"
    parallel_root = tmp_path / "parallel"
    for name, source in templates.items():
        _write(sequential_root / name, source)
        _write(parallel_root / name, source)

    instructions = [
        AddTypeHints(id="s0", target_file="a.py", function_name=None),
        AddTypeHints(id="s1", target_file="b.py", function_name=None),
        ReplaceCall(id="s2", target_file="c.py", old_call="old", new_call="new"),
        AddTypeHints(id="s3", target_file="d.py", function_name=None),
        ReplaceCall(id="s4", target_file="e.py", old_call="old", new_call="new"),
        AddTypeHints(id="s5", target_file="f.py", function_name=None),
    ]

    sequential_log = execute_plan(
        RefactorPlan(instructions=instructions), sequential_root, max_workers=1
    )
    parallel_log = execute_plan(
        RefactorPlan(instructions=instructions), parallel_root, max_workers=4
    )

    assert parallel_log == sequential_log
    for name in templates:
        assert (parallel_root / name).read_bytes() == (sequential_root / name).read_bytes()


def test_parallel_failure_raises_concurrency_error_and_writes_nothing(tmp_path):
    for name in ("a.py", "b.py", "c.py"):
        _write(tmp_path / name)
    before = {name: (tmp_path / name).read_bytes() for name in ("a.py", "b.py", "c.py")}
    plan = RefactorPlan(
        instructions=[
            _hints("a.py", "s0"),
            ExtractFunction(
                id="s1", target_file="b.py", function_name="h", start_line=900, end_line=920
            ),
            _hints("c.py", "s2"),
        ]
    )

    with pytest.raises(ConcurrencyError) as excinfo:
        execute_plan(plan, tmp_path, max_workers=4)

    assert excinfo.value.instruction_index == 1
    assert {name: (tmp_path / name).read_bytes() for name in before} == before


def test_a_zero_change_batch_is_rejected_before_any_write(tmp_path):
    _write(tmp_path / "a.py", _SOURCE)
    _write(tmp_path / "b.py", "def f(x) -> None:\n    return x\n")
    plan = RefactorPlan(
        instructions=[
            _hints("a.py", "s0"),
            AddTypeHints(id="s1", target_file="b.py", function_name=None),
        ]
    )

    with pytest.raises(NoOpEditError, match=r"changed nothing in b\.py"):
        execute_plan(plan, tmp_path, max_workers=4)

    # The batch is all-or-nothing: the valid first edit was never written.
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == _SOURCE


def test_max_workers_one_disables_the_pool(tmp_path, monkeypatch):
    _write(tmp_path / "a.py")
    _write(tmp_path / "b.py")

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("max_workers=1 must stay in-process")

    monkeypatch.setattr(dispatcher_module, "execute_parallel", explode)
    plan = RefactorPlan(instructions=[_hints("a.py", "s0"), _hints("b.py", "s1")])

    log = execute_plan(plan, tmp_path, max_workers=1)

    assert len(log) == 2
    assert "-> None" in (tmp_path / "a.py").read_text(encoding="utf-8")


def test_environment_variable_disables_the_pool(tmp_path, monkeypatch):
    _write(tmp_path / "a.py")
    _write(tmp_path / "b.py")
    monkeypatch.setenv(MAX_WORKERS_ENV, "1")

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("MORAY_MAX_WORKERS=1 must stay in-process")

    monkeypatch.setattr(dispatcher_module, "execute_parallel", explode)
    plan = RefactorPlan(instructions=[_hints("a.py", "s0"), _hints("b.py", "s1")])

    assert len(execute_plan(plan, tmp_path)) == 2
