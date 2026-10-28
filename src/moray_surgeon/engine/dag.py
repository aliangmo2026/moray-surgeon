"""Topologically order a plan whose steps declare dependencies.

A realistic multi-step refactor has an order the model knows better than the JSON
list: add the import the new call needs, rename the class before rewriting its
callers, run the rename that changes a signature before the call-site rewrite
that relies on it. Every instruction carries a unique ``id`` and may list the ids
it ``depends_on``; this module turns that graph into one deterministic execution
sequence.

The algorithm is Kahn's, with the ready queue implemented as a min-heap keyed on
each instruction's original list position. Two properties follow:

* A plan with no ``depends_on`` edges executes in exactly the order it was
  written — v0.2.0 plans keep their behaviour byte for byte.
* Ties between independent steps are broken by original position, so the same
  plan always produces the same order, on every run and every interpreter.

Validation happens before any file is touched: repeated ids, ids that name no
step, and cycles all raise subclasses of :class:`DependencyError`.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from typing import TYPE_CHECKING

from moray_surgeon.exceptions import (
    CircularDependencyError,
    DuplicateInstructionIdError,
    UnknownDependencyError,
)

if TYPE_CHECKING:
    from moray_surgeon.schema.instructions import Instruction


def topological_sort(instructions: Sequence[Instruction]) -> list[Instruction]:
    """Return ``instructions`` in dependency order.

    Args:
        instructions: The plan's instructions, in authoring order.

    Returns:
        A new list holding the same instruction objects. Steps with no unmet
        dependency come first; among the currently runnable steps the one that
        appears earliest in ``instructions`` is chosen, so the order is stable.

    Raises:
        DuplicateInstructionIdError: Two instructions share an ``id``.
        UnknownDependencyError: A ``depends_on`` entry names no instruction.
        CircularDependencyError: The dependency graph contains a cycle; the
            message names one concrete cycle (``step1 -> step2 -> step1``).
    """
    ids = [instruction.id for instruction in instructions]
    positions = _unique_positions(ids)

    indegree = [0] * len(instructions)
    dependents: dict[str, list[int]] = {step_id: [] for step_id in ids}
    for index, instruction in enumerate(instructions):
        # ``dict.fromkeys`` de-duplicates while preserving the authored order;
        # schema validation rejects repeats, but model_construct can bypass it.
        for dependency in dict.fromkeys(instruction.depends_on):
            if dependency not in positions:
                raise UnknownDependencyError(
                    f"instruction {instruction.id!r} depends on unknown id {dependency!r}"
                )
            indegree[index] += 1
            dependents[dependency].append(index)

    ready = [(index, ids[index]) for index in range(len(instructions)) if indegree[index] == 0]
    heapq.heapify(ready)
    ordered: list[Instruction] = []
    while ready:
        index, step_id = heapq.heappop(ready)
        ordered.append(instructions[index])
        for dependent in dependents[step_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, (dependent, ids[dependent]))

    if len(ordered) != len(instructions):
        blocked = [step_id for step_id, degree in zip(ids, indegree, strict=True) if degree > 0]
        dependencies = {
            instruction.id: list(dict.fromkeys(instruction.depends_on))
            for instruction in instructions
        }
        cycle = _find_cycle(blocked, dependencies, frozenset(blocked))
        raise CircularDependencyError(
            "circular dependency between instructions: " + " -> ".join(cycle)
        )
    return ordered


def _unique_positions(ids: Sequence[str]) -> dict[str, int]:
    """Map every instruction id to its first position, rejecting duplicates."""
    positions: dict[str, int] = {}
    for index, step_id in enumerate(ids):
        previous = positions.get(step_id)
        if previous is not None:
            raise DuplicateInstructionIdError(
                f"duplicate instruction id {step_id!r} at positions {previous} and {index}"
            )
        positions[step_id] = index
    return positions


def _find_cycle(
    blocked: Sequence[str],
    dependencies: dict[str, list[str]],
    blocked_set: frozenset[str],
) -> list[str]:
    """Return one concrete cycle among the nodes Kahn's algorithm could not emit.

    ``blocked`` contains every node still holding a non-zero indegree, which is
    the set of cycle members plus anything downstream of them. The DFS follows
    ``depends_on`` edges, so a back edge to a node on the current stack closes a
    cycle; the stack slice from that node is the cycle, closed by repeating it.
    """
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for dependency in dependencies[node]:
            if dependency not in blocked_set:
                continue
            if state.get(dependency) == 1:
                return [*stack[stack.index(dependency) :], dependency]
            if state.get(dependency, 0) == 0:
                found = visit(dependency)
                if found is not None:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for node in blocked:
        if state.get(node, 0) == 0:
            found = visit(node)
            if found is not None:
                return found
    # Unreachable for a well-formed graph: a non-empty residual always contains a
    # cycle. Returning the residual keeps the message useful if that ever changes.
    return list(blocked)
