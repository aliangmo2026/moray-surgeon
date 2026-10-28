from __future__ import annotations

import pytest

from moray_surgeon.engine import topological_sort
from moray_surgeon.exceptions import (
    CircularDependencyError,
    DependencyError,
    DuplicateInstructionIdError,
    MorayError,
    UnknownDependencyError,
)
from moray_surgeon.schema import Noop


def _step(step_id: str, *depends_on: str) -> Noop:
    return Noop(id=step_id, depends_on=list(depends_on))


def _ids(instructions: list[Noop]) -> list[str]:
    return [instruction.id for instruction in instructions]


def test_dependencies_run_before_their_dependents():
    ordered = topological_sort([_step("c", "b"), _step("b", "a"), _step("a")])

    assert _ids(ordered) == ["a", "b", "c"]


def test_plan_without_dependencies_keeps_the_authored_order():
    ordered = topological_sort([_step("c"), _step("a"), _step("b")])

    assert _ids(ordered) == ["c", "a", "b"]


def test_ready_ties_follow_the_authored_position():
    # "a" is first in the JSON but cannot run until "c"; "b" is ready immediately.
    ordered = topological_sort([_step("a", "c"), _step("b"), _step("c")])

    assert _ids(ordered) == ["b", "c", "a"]


def test_diamond_dependency_orders_the_join_last():
    ordered = topological_sort(
        [_step("d", "b", "c"), _step("b", "a"), _step("c", "a"), _step("a")]
    )

    assert _ids(ordered)[0] == "a"
    assert _ids(ordered)[-1] == "d"
    assert set(_ids(ordered)[1:3]) == {"b", "c"}


def test_returns_the_same_instruction_objects_in_a_new_list():
    steps = [_step("b", "a"), _step("a")]

    ordered = topological_sort(steps)

    assert ordered is not steps
    assert all(any(item is original for original in steps) for item in ordered)


def test_sort_is_deterministic_across_runs():
    steps = [_step("d", "a", "c"), _step("c", "b"), _step("b"), _step("a"), _step("e")]

    assert _ids(topological_sort(steps)) == _ids(topological_sort(steps))


def test_duplicate_ids_are_rejected():
    with pytest.raises(DuplicateInstructionIdError, match="positions 0 and 1"):
        topological_sort([_step("a"), _step("a")])


def test_unknown_dependency_is_rejected():
    with pytest.raises(UnknownDependencyError, match="ghost"):
        topological_sort([_step("a", "ghost")])


def test_cycle_names_one_concrete_path():
    with pytest.raises(CircularDependencyError, match="a -> b -> a"):
        topological_sort([_step("a", "b"), _step("b", "a")])


def test_cycle_through_three_steps_names_the_full_loop():
    with pytest.raises(CircularDependencyError, match="a -> b -> c -> a"):
        topological_sort([_step("a", "b"), _step("b", "c"), _step("c", "a")])


def test_self_dependency_is_a_cycle():
    with pytest.raises(CircularDependencyError, match="a -> a"):
        topological_sort([_step("a", "a")])


def test_a_step_downstream_of_a_cycle_still_reports_the_cycle():
    with pytest.raises(CircularDependencyError, match="a -> b -> a"):
        topological_sort([_step("a", "b"), _step("b", "a"), _step("late", "a")])


def test_repeated_dependency_is_counted_once():
    # Schema validation rejects repeats; model_construct bypasses it, and the
    # sorter must not strand the step on a double-counted indegree.
    step = Noop.model_construct(id="a", depends_on=["b", "b"])

    ordered = topological_sort([step, _step("b")])

    assert _ids(ordered) == ["b", "a"]


def test_errors_share_the_moray_base_and_the_dependency_base():
    for error in (DuplicateInstructionIdError, UnknownDependencyError, CircularDependencyError):
        assert issubclass(error, DependencyError)
        assert issubclass(error, MorayError)
