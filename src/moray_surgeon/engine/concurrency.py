"""Multi-process execution of independent, single-file refactor instructions.

libcst parsing and rewriting are CPU-bound pure functions of one file's source
text, which makes a batch of file-disjoint instructions embarrassingly parallel.
The dispatcher collects such a batch (see :func:`is_parallel_safe`), submits one
task per instruction to a :class:`~concurrent.futures.ProcessPoolExecutor`, and
writes the returned sources itself once every task has succeeded.

The split of duties is what keeps the engine's safety contract intact:

* Workers only *read* their target and return an :class:`EditResult`; the
  parent process performs every write. A batch therefore stays all-or-nothing:
  one failing worker cancels the rest and no byte of the batch reaches disk.
* Only atoms whose output depends on nothing but the target file may travel to
  a worker. The project-aware atoms (rename, move, inline, dead-code removal)
  read other files to prove safety, so they run in the parent as barriers — a
  child would either see a stale tree or race the parent's writes.
* Every write goes through :func:`atomic_write_text` (temp file + ``os.replace``),
  so an interrupted run can never leave a half-written module behind.

Worker processes are spawned per batch and shut down before returning; the pool
size comes from ``MORAY_MAX_WORKERS`` and defaults to ``min(cpu_count, 8)`` so a
refactor cannot pin every core on a laptop. ``MORAY_MAX_WORKERS=1`` disables the
pool entirely, which is also the escape hatch for hosts where spawning is not
allowed (restricted sandboxes, frozen executables).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from moray_surgeon.engine.backends import backend_for
from moray_surgeon.exceptions import ConcurrencyError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from concurrent.futures import Future

    from moray_surgeon.engine.base import EditResult
    from moray_surgeon.schema.instructions import Instruction

logger = logging.getLogger(__name__)

MAX_WORKERS_ENV = "MORAY_MAX_WORKERS"
# A hard ceiling on the *default*: refactors are usually short bursts, and
# oversubscribing a 64-core build machine buys nothing but context switches.
DEFAULT_WORKER_CAP = 8

# Atoms whose result is a pure function of the target file's source. The
# project-aware atoms are deliberately absent: they scan or rewrite other files,
# so the dispatcher keeps them in the parent process where sequential visibility
# is guaranteed. Unknown/new actions are absent too, which makes "not reviewed
# for parallel safety" the safe default.
PARALLEL_SAFE_ACTIONS: frozenset[str] = frozenset(
    {
        "add_import",
        "add_type_hints",
        "convert_to_match_case",
        "extract_function",
        "extract_variable",
        "remove_import",
        "replace_call",
        "split_class",
    }
)


def default_max_workers() -> int:
    """Return the conservative default pool size for this machine.

    Returns:
        ``min(os.cpu_count() or 1, 8)``: never more than eight processes even on
        a many-core host, and at least one on platforms that cannot report a
        CPU count.
    """
    return min(os.cpu_count() or 1, DEFAULT_WORKER_CAP)


def resolve_max_workers(explicit: int | None = None) -> int:
    """Resolve the pool size from an explicit value or ``MORAY_MAX_WORKERS``.

    ``explicit`` wins over the environment so embedders (tests, the benchmark)
    can force a mode. Unset, empty, non-numeric and non-positive environment
    values fall back to :func:`default_max_workers` with a warning instead of
    failing the refactor: a typo in an environment variable must not stop work.

    Args:
        explicit: Caller-supplied pool size. Values below 1 are clamped to 1
            (which means "run everything in this process").

    Returns:
        A pool size of at least 1.
    """
    if explicit is not None:
        return max(1, explicit)
    raw = os.environ.get(MAX_WORKERS_ENV, "").strip()
    if not raw:
        return default_max_workers()
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d workers", MAX_WORKERS_ENV, raw,
                       default_max_workers())
        return default_max_workers()
    if value < 1:
        logger.warning("%s=%r must be positive; using %d workers", MAX_WORKERS_ENV, raw,
                       default_max_workers())
        return default_max_workers()
    return value


def is_parallel_safe(instruction: Instruction) -> bool:
    """Report whether ``instruction`` may run in a worker process.

    Args:
        instruction: Any schema instruction; ``Noop`` is handled by the
            dispatcher before this predicate is consulted.

    Returns:
        True only for actions listed in :data:`PARALLEL_SAFE_ACTIONS`, i.e. atoms
        that neither read nor write any file other than their target.
    """
    return instruction.action in PARALLEL_SAFE_ACTIONS


@dataclass(frozen=True)
class InstructionTask:
    """One unit of parallel work: a resolved target plus its instruction.

    Everything here must be picklable, because the task crosses a process
    boundary. Paths are already resolved and containment-checked by the
    dispatcher, so a worker never re-derives (and never re-trusts) a repo path.

    Attributes:
        index: Execution position in the plan's topologically sorted order.
        instruction: The schema instruction to apply.
        repo_root: Resolved repository root, passed through to the atom.
        target: Resolved, containment-checked path of ``instruction.target_file``.
    """

    index: int
    instruction: Instruction
    repo_root: Path
    target: Path


def apply_task(task: InstructionTask) -> EditResult:
    """Apply one task in the current process and return its un-written result.

    This is the worker entry point; it is also used as the in-process fallback,
    so both paths run byte-identical code. The target is read with UTF-8 and
    *not* modified — the parent owns every write.

    Args:
        task: The resolved task to apply.

    Returns:
        The atom's result; ``new_source`` still needs to be written.

    Raises:
        UnsupportedLanguageError: The current process has no backend for the
            target's suffix (e.g. a backend registered only in the parent).
        ValueError: The atom rejected the instruction input.
        OSError: The target could not be read.
    """
    backend = backend_for(task.target)
    source = task.target.read_text(encoding="utf-8")
    # ``apply`` accepts the index handle for project-aware atoms; every action
    # allowed in a worker ignores it, and an Indexer must not cross processes.
    return backend.apply(task.instruction, source, task.repo_root, task.target, None)


def execute_parallel(
    tasks: Sequence[InstructionTask], max_workers: int
) -> dict[int, EditResult]:
    """Run ``tasks`` across a process pool and return their results.

    Nothing is written and nothing is ordered: results are keyed by task index so
    the caller can apply them in plan order. The pool is always shut down before
    returning, cancelling every task that had not started yet.

    Args:
        tasks: Independent single-file tasks; they must not target the same file
            (the dispatcher guarantees this when it forms a batch).
        max_workers: Pool size. ``1`` (or a single task) runs everything in this
            process, which is also what happens when the pool cannot be created.

    Returns:
        ``{task.index: EditResult}`` with one entry per task.

    Raises:
        ConcurrencyError: A worker raised. Every pending task is cancelled before
            this is raised, and ``instruction_index``/``__cause__`` identify the
            failing step. No file was written.
    """
    if not tasks:
        return {}
    if max_workers <= 1 or len(tasks) == 1:
        return {task.index: apply_task(task) for task in tasks}

    try:
        executor = ProcessPoolExecutor(max_workers=min(max_workers, len(tasks)))
    except OSError as exc:  # pragma: no cover - depends on the host's spawn support
        logger.warning("cannot start a process pool (%s); running in-process", exc)
        return {task.index: apply_task(task) for task in tasks}

    futures: dict[Future[EditResult], InstructionTask] = {}
    try:
        futures = {executor.submit(apply_task, task): task for task in tasks}
        results: dict[int, EditResult] = {}
        for future in as_completed(futures):
            task = futures[future]
            try:
                results[task.index] = future.result()
            except Exception as exc:
                # Cancel everything still queued before propagating: the
                # remaining work cannot change the outcome of this batch.
                _cancel_pending(futures)
                raise ConcurrencyError(
                    f"instruction {task.index} ({task.instruction.action}) failed in "
                    f"worker process {task.target.name}: {exc}; "
                    f"the remaining tasks were cancelled",
                    instruction_index=task.index,
                ) from exc
        return results
    finally:
        _cancel_pending(futures)
        # ``wait`` reaps the workers that were already running; queued futures
        # were cancelled above and will never start.
        executor.shutdown(wait=True, cancel_futures=True)


def _cancel_pending(futures: dict[Future[EditResult], InstructionTask]) -> None:
    """Best-effort cancellation of every task that has not started running."""
    for future in futures:
        future.cancel()


def atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path``'s contents with ``text`` without a partial-write window.

    The text goes to a temporary file in the *same directory* (so ``os.replace``
    is a same-filesystem rename, which is atomic on POSIX and Windows), the
    original mode bits are copied over, and only then is the file swapped in.

    Newline handling deliberately matches ``Path.write_text``: both callers (the
    sequential and the parallel path) must produce byte-identical files.

    Args:
        path: Existing file to overwrite.
        text: New contents, written as UTF-8.

    Raises:
        OSError: The temp file could not be created/written or the rename failed.
            The original file is never left half-written and the temp file is
            removed before the error propagates.
    """
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copymode(path, temp_path)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
