"""Execute a validated refactor plan against a repository, instruction by instruction.

The dispatcher owns ordering, path containment and error classification. Most
atoms are source-in / source-out; ``rename_symbol`` is project-aware (Jedi
resolves references across files) and rewrites its affected files in place,
inside the caller's transaction.

Invariants:

* Instructions run strictly in order and the first failure aborts the run.
  There is no transactional rollback, so earlier edits stay on disk.
* Every path is resolved with ``pathlib`` and must stay inside ``repo_root``.
* Every read and write is UTF-8.
* Nothing is printed: the returned log list is the caller's to render.
"""

from __future__ import annotations

from pathlib import Path

from moray_surgeon.engine.base import EditResult
from moray_surgeon.engine.extract import extract_function
from moray_surgeon.engine.rename import rename_symbol
from moray_surgeon.engine.replace_call import replace_call
from moray_surgeon.engine.type_hints import add_type_hints
from moray_surgeon.exceptions import (
    InstructionError,
    NoOpEditError,
    PathTraversalError,
)
from moray_surgeon.schema.instructions import (
    AddTypeHints,
    ExtractFunction,
    Instruction,
    Noop,
    RefactorPlan,
    RenameSymbol,
    ReplaceCall,
)


def execute_plan(plan: RefactorPlan, repo_root: Path) -> list[str]:
    """Execute every instruction in order; raise on the first failure.

    Args:
        plan: A plan from the schema layer. Its ``instructions`` list is applied
            verbatim; no reordering, deduplication or pre-validation happens here.
        repo_root: Directory every ``target_file`` is resolved against.

    Returns:
        One human-readable log line per applied instruction. ``Noop`` entries are
        reported as skipped and never count as changes.

    Raises:
        PathTraversalError: ``target_file`` resolves outside ``repo_root``.
        FileNotFoundError: ``target_file`` is missing or not a regular file.
        InstructionError: An atom rejected the instruction input (e.g. an invalid
            line range); carries the failing ``instruction_index``.
        NoOpEditError: An atom matched nothing (``changes == 0``).
        OSError: Reading or writing the target file failed (permissions, ...).
    """
    root = repo_root.resolve()
    log: list[str] = []

    for index, instruction in enumerate(plan.instructions):
        if isinstance(instruction, Noop):
            log.append(f"[{index}] noop: skipped")
            continue

        target = _resolve_target(instruction.target_file, root)
        source = target.read_text(encoding="utf-8")
        try:
            result = _apply(instruction, source, root)
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

        target.write_text(result.new_source, encoding="utf-8")
        log.append(f"[{index}] {instruction.action}: {result.changes} change(s) in {target.name}")

    return log


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


def _apply(instruction: Instruction, source: str, repo_root: Path) -> EditResult:
    """Dispatch one non-noop instruction to its atomic engine operation.

    ``rename_symbol`` needs the repository root for Jedi; the remaining atoms
    consume the source the dispatcher already read.
    """
    if isinstance(instruction, RenameSymbol):
        return rename_symbol(
            repo_root,
            instruction.target_file,
            instruction.old_name,
            instruction.new_name,
        )
    if isinstance(instruction, ExtractFunction):
        return extract_function(
            source,
            instruction.function_name,
            instruction.start_line,
            instruction.end_line,
        )
    if isinstance(instruction, ReplaceCall):
        return replace_call(source, instruction.old_call, instruction.new_call)
    if isinstance(instruction, AddTypeHints):
        return add_type_hints(source, instruction.function_name)
    raise TypeError(f"unsupported instruction type: {type(instruction).__name__}")
