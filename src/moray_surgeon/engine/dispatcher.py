"""Execute a validated refactor plan against a repository, instruction by instruction.

The dispatcher owns ordering, path containment, batching and error classification
— and, since the LanguageBackend split, *nothing else*: every per-language
operation lives behind :class:`~moray_surgeon.engine.backends.base.LanguageBackend`,
which is resolved from each target file's suffix. Adding a language means
registering a backend; this module never changes again.

Execution has two modes. A *batch* is a run of consecutive, file-disjoint,
DAG-independent instructions whose atoms depend on nothing but their own target
file; those are applied in worker processes and written back by this process
(see :mod:`moray_surgeon.engine.concurrency`). Everything else — project-aware
atoms (rename/move/inline/dead-code), instructions that share a file, steps
joined by a ``depends_on`` edge — runs here in plan order, and acts as a barrier
that flushes the pending batch first. That keeps the observable result identical
to a strictly sequential run while the CPU-bound parsing happens in parallel.

Invariants:

* The plan is topologically sorted on ``id``/``depends_on`` first, so a step
  never runs before the steps it declares. A plan with no dependencies keeps its
  authored order. The first failure aborts the run; there is no transactional
  rollback here, so earlier edits stay on disk.
* Every path is resolved with ``pathlib`` and must stay inside ``repo_root``.
* The backend is resolved *before* any byte is read, so a target whose language
  has no registered backend fails with ``UnsupportedLanguageError`` and leaves
  the file untouched.
* A parallel batch is validated and only then written, all-or-nothing; a worker
  failure raises ``ConcurrencyError`` and leaves every file of the batch intact.
* Every read and write is UTF-8, and every write is atomic (temp file + rename).
* Nothing is printed: the returned log list is the caller's to render.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from moray_surgeon.engine.backends import backend_for
from moray_surgeon.engine.concurrency import (
    InstructionTask,
    atomic_write_text,
    execute_parallel,
    is_parallel_safe,
    resolve_max_workers,
)
from moray_surgeon.engine.dag import topological_sort
from moray_surgeon.engine.indexer import Indexer, get_indexer
from moray_surgeon.exceptions import (
    InstructionError,
    NoOpEditError,
    PathTraversalError,
    UnsupportedLanguageError,
)
from moray_surgeon.schema.instructions import (
    Instruction,
    MoveSymbol,
    Noop,
    RefactorPlan,
)

if TYPE_CHECKING:
    from moray_surgeon.engine.backends.base import LanguageBackend


def execute_plan(
    plan: RefactorPlan,
    repo_root: Path,
    indexer: Indexer | None = None,
    max_workers: int | None = None,
) -> list[str]:
    """Execute every instruction in dependency order; raise on the first failure.

    Args:
        plan: A plan from the schema layer. Its ``instructions`` are first
            topologically sorted on ``depends_on``; steps without dependencies
            keep their authored order.
        repo_root: Directory every ``target_file`` is resolved against.
        indexer: Optional code-index handle shared by the project-aware atoms.
            Omitting it uses the process-wide handle for ``repo_root``, which the
            orchestrator has already refreshed for this attempt.
        max_workers: Parallel pool size. ``None`` reads ``MORAY_MAX_WORKERS`` and
            otherwise uses ``min(cpu_count, 8)``; ``1`` disables the pool. Batches
            smaller than two instructions always run in this process regardless.

    Returns:
        One human-readable log line per applied instruction, in execution order.
        ``Noop`` entries are reported as skipped and never count as changes. The
        bracketed index is the execution position, not the position in the JSON.

    Raises:
        DuplicateInstructionIdError: Two instructions share an id.
        UnknownDependencyError: ``depends_on`` names an id that is not in the plan.
        CircularDependencyError: The plan's dependency graph contains a cycle.
        PathTraversalError: ``target_file`` resolves outside ``repo_root``.
        FileNotFoundError: ``target_file`` is missing or not a regular file.
        UnsupportedLanguageError: No backend registered for a target's suffix.
        InstructionError: An atom rejected the instruction input (e.g. an invalid
            line range); carries the failing ``instruction_index``.
        NoOpEditError: An atom matched nothing (``changes == 0``).
        ConcurrencyError: A worker process failed while applying a parallel
            batch; the batch's files were left untouched.
        OSError: Reading or writing the target file failed (permissions, ...).
    """
    root = repo_root.resolve()
    handle = indexer or get_indexer(root)
    workers = resolve_max_workers(max_workers)
    ordered = topological_sort(plan.instructions)
    log: dict[int, str] = {}
    batch = _ParallelBatch(root, handle, workers)

    for index, instruction in enumerate(ordered):
        if isinstance(instruction, Noop):
            batch.flush(log)
            log[index] = f"[{index}] noop: skipped"
            continue

        if workers <= 1 or not is_parallel_safe(instruction):
            batch.flush(log)
            log[index] = _execute_instruction(instruction, index, root, handle)
            continue

        # Resolve before batching: a path/traversal/language failure must be
        # reported at this step's position, after the earlier batch has been
        # written — exactly where a sequential run would stop.
        try:
            target = _resolve_target(instruction.target_file, root)
            backend_for(target)
        except Exception:
            batch.flush(log)
            raise

        if not batch.can_join(instruction, target):
            batch.flush(log)
        batch.add(index, instruction, target, instruction.target_file)

    batch.flush(log)
    return [log[index] for index in range(len(ordered))]


@dataclass(frozen=True)
class _PendingEdit:
    """One resolved instruction waiting to be applied by a batch flush.

    ``target_file`` keeps the authored repository-relative path so error and log
    messages can quote it after the path has been resolved for the worker.
    """

    index: int
    instruction: Instruction
    target: Path
    target_file: str


class _ParallelBatch:
    """Consecutive instructions that may travel to worker processes together.

    Membership is decided while the dispatcher walks the topologically sorted
    plan: a single-file action, a target no current member touches, and no
    ``depends_on`` edge to any current member. Because the walk is consecutive
    and dependency-respecting, refusing an edge to a member is enough for
    transitive edges too: any intermediate step is either a member (the edge is
    refused) or sits between the two instructions, which ends the batch.

    A flush of a one-instruction batch falls back to the sequential path; running
    a worker for a single file would only add spawn latency.
    """

    def __init__(self, root: Path, handle: Indexer, workers: int) -> None:
        self._root = root
        self._handle = handle
        self._workers = workers
        self._edits: list[_PendingEdit] = []
        self._files: set[Path] = set()
        self._ids: set[str] = set()

    def can_join(self, instruction: Instruction, target: Path) -> bool:
        """Report whether ``instruction`` can join without breaking isolation."""
        if not self._edits:
            return True
        return target not in self._files and not self._ids.intersection(
            instruction.depends_on
        )

    def add(
        self, index: int, instruction: Instruction, target: Path, target_file: str
    ) -> None:
        """Append an instruction that :meth:`can_join` accepted."""
        self._edits.append(_PendingEdit(index, instruction, target, target_file))
        self._files.add(target)
        self._ids.add(instruction.id)

    def flush(self, log: dict[int, str]) -> None:
        """Apply and write the pending batch, then reset it.

        Two or more members run in a process pool; the results are validated
        (no zero-change instruction) before any write, so the batch is
        all-or-nothing. Writes happen in plan order, after every worker finished.
        """
        if not self._edits:
            return
        edits = self._edits
        self._edits = []
        self._files.clear()
        self._ids.clear()

        if self._workers <= 1 or len(edits) == 1:
            for edit in edits:
                log[edit.index] = _execute_instruction(
                    edit.instruction, edit.index, self._root, self._handle
                )
            return

        tasks = [
            InstructionTask(edit.index, edit.instruction, self._root, edit.target)
            for edit in edits
        ]
        results = execute_parallel(tasks, self._workers)
        for edit in edits:
            if results[edit.index].changes == 0:
                raise NoOpEditError(
                    f"instruction {edit.index} ({edit.instruction.action}) changed "
                    f"nothing in {edit.target_file}"
                )
        for edit in edits:
            atomic_write_text(edit.target, results[edit.index].new_source)
            log[edit.index] = _log_line(
                edit.index, edit.instruction, results[edit.index].changes, edit.target
            )


def _execute_instruction(
    instruction: Instruction, index: int, root: Path, indexer: Indexer
) -> str:
    """Run one instruction in this process and return its log line."""
    if isinstance(instruction, Noop):  # pragma: no cover - callers filter Noop out
        raise TypeError("Noop is reported by execute_plan and never applied")
    if isinstance(instruction, MoveSymbol):
        return _execute_move_symbol(instruction, index, root, indexer)

    target = _resolve_target(instruction.target_file, root)
    backend = backend_for(target)
    return _apply_single_file(instruction, index, target, backend, root, indexer)


def _apply_single_file(
    instruction: Instruction,
    index: int,
    target: Path,
    backend: LanguageBackend,
    root: Path,
    indexer: Indexer,
) -> str:
    """Apply a single-file atom to ``target``, write it atomically and log it."""
    if isinstance(instruction, Noop):  # pragma: no cover - callers filter Noop out
        raise TypeError("Noop is reported by execute_plan and never applied")
    source = target.read_text(encoding="utf-8")
    try:
        result = backend.apply(instruction, source, root, target, indexer)
    except ValueError as exc:
        raise InstructionError(
            f"{instruction.action} failed for {instruction.target_file}: {exc}",
            instruction_index=index,
        ) from exc

    if result.changes == 0:
        raise NoOpEditError(
            f"instruction {index} ({instruction.action}) changed nothing in "
            f"{instruction.target_file}"
        )

    atomic_write_text(target, result.new_source)
    return _log_line(index, instruction, result.changes, target)


def _log_line(index: int, instruction: Instruction, changes: int, target: Path) -> str:
    """Format the log line shared by the sequential and parallel paths."""
    return f"[{index}] {instruction.action}: {changes} change(s) in {target.name}"


def _execute_move_symbol(
    instruction: MoveSymbol, index: int, root: Path, indexer: Indexer
) -> str:
    """Run the project-aware move atom through its language backend.

    Unlike the single-file atoms, a move touches the source, the target and
    every importer file, so it manages its own writes inside the caller's
    transaction (the same contract ``rename_symbol`` has).
    """
    source_path = _resolve_target(instruction.source_file, root)
    target_path = _resolve_target(instruction.target_file, root)
    # A move is a same-language operation: both endpoints must resolve to the
    # same backend. The second lookup also rejects unsupported suffixes before
    # the backend touches any file.
    backend = backend_for(source_path)
    if backend_for(target_path) is not backend:
        raise UnsupportedLanguageError(
            f"move_symbol needs one language for both endpoints; "
            f"{instruction.source_file!r} and {instruction.target_file!r} differ"
        )
    try:
        result = backend.move_symbol(
            root, source_path, target_path, instruction.symbol_name, indexer
        )
    except ValueError as exc:
        raise InstructionError(
            f"{instruction.action} failed for {instruction.source_file}: {exc}",
            instruction_index=index,
        ) from exc
    if result.changes == 0:  # pragma: no cover - a move always changes both files
        raise NoOpEditError(
            f"instruction {index} ({instruction.action}) changed nothing in "
            f"{instruction.source_file}"
        )
    return (
        f"[{index}] {instruction.action}: {result.changes} change(s) "
        f"({source_path.name} -> {target_path.name})"
    )


def _resolve_target(target_file: str, repo_root: Path) -> Path:
    """Resolve ``target_file`` under ``repo_root`` and prove it cannot escape.

    ``resolve()`` is what makes the guard meaningful: it collapses ``..`` and
    follows symlinks *before* containment is checked, so both
    ``sub/../../x.py`` and an in-repo symlink pointing at an outside file are
    caught. Containment is checked first so an escape is always reported as
    ``PathTraversalError``, even when the escaped path does not exist.
    """
    target = (repo_root / target_file).resolve()
    if not target.is_relative_to(repo_root):
        raise PathTraversalError(f"target_file {target_file!r} resolves outside the repository")
    if not target.is_file():
        raise FileNotFoundError(
            f"target_file {target_file!r} does not exist or is not a regular file"
        )
    return target
