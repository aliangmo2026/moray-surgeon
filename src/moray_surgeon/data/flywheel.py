"""Dataset flywheel: mine real Git history into fine-tuning samples.

The engine's contract is *intent in, ``RefactorPlan`` JSON out*. This module
turns real commits into exactly that pair: it compares the parent and the commit
with libcst, recognises the AST changes the engine can express (pure symbol
renames, callee replacements, import additions/removals and ``-> None`` return
annotations), and emits the canonical plan that would have produced them. The
user message carries the repository's code map at the parent commit and the full
pre-change file, so the sample mirrors what the runtime model actually sees.

Two output layouts are supported:

* ``openai`` (default) — ``{"messages": [system, user, assistant]}``, the strict
  OpenAI/TRL chat format. The system message is the exact prompt
  :class:`~moray_surgeon.intent.parser.IntentParser` sends at inference time.
* ``alpaca`` — ``{"instruction", "input", "output"}`` for classic instruction
  tuning.

Only high-precision patterns are emitted. A commit that renames a function *and*
rewrites its body produces no rename sample; a call whose arguments also changed
produces no replacement sample. Silence is better than wrong supervision.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import libcst as cst

from moray_surgeon.engine.repo_map import generate_repo_map
from moray_surgeon.exceptions import FlywheelError, MorayError
from moray_surgeon.intent import IntentParser
from moray_surgeon.schema import (
    AddImport,
    AddTypeHints,
    RefactorPlan,
    RemoveImport,
    RenameSymbol,
    ReplaceCall,
)

if TYPE_CHECKING:
    from moray_surgeon.schema import Instruction

logger = logging.getLogger(__name__)

_FORMATS: tuple[str, ...] = ("openai", "alpaca")
_GIT_TIMEOUT_SECONDS = 120
_MAX_SOURCE_CHARS = 200_000
_MAX_SNIPPET_CHARS = 8_000
_REPO_MAP_FILES = 50
_TRUNCATION_MARKER = "\n# ... (truncated by moray-surgeon export-sft)"


@dataclass(frozen=True)
class SftSample:
    """One supervised fine-tuning example.

    Attributes:
        commit: Full SHA of the mined commit.
        subject: Commit subject line.
        path: Repository-relative file the plan targets.
        repo_root: Repository root shown in the user message (runtime parity).
        instruction: Natural-language request derived from the detected changes.
        context: Commit metadata, parent-commit repo map and pre-change source.
        plan: The validated plan that expresses the detected changes.
    """

    commit: str
    subject: str
    path: str
    repo_root: str
    instruction: str
    context: str
    plan: RefactorPlan

    @property
    def plan_json(self) -> str:
        """Canonical assistant answer: the complete plan JSON.

        Defaults are kept: ``action`` is the discriminator and must be present
        for ``RefactorPlan.model_validate_json`` to rebuild the plan.
        """
        document = self.plan.model_dump(mode="json")
        return json.dumps(document, ensure_ascii=False, separators=(",", ":"))

    def as_openai(self, system_prompt: str) -> dict[str, object]:
        """Render an OpenAI/TRL chat record (only the ``messages`` key)."""
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._user_content()},
                {"role": "assistant", "content": self.plan_json},
            ]
        }

    def as_alpaca(self) -> dict[str, object]:
        """Render an Alpaca-style instruction/input/output record."""
        return {
            "instruction": self.instruction,
            "input": f"Repository root: {self.repo_root}\n\n{self.context}",
            "output": self.plan_json,
        }

    def _user_content(self) -> str:
        """Mirror ``IntentParser._user_prompt`` so training matches inference."""
        return "\n\n".join(
            [
                f"Repository root: {self.repo_root}",
                f"Refactoring intent:\n{self.instruction}",
                "Additional context (may include the previous attempt's diff and "
                f"verification errors):\n{self.context}",
                "Return the RefactorPlan JSON object now.",
            ]
        )


@dataclass(frozen=True)
class ExportReport:
    """Counters returned by :func:`export_sft`.

    Attributes:
        commits_scanned: Recent commits examined (including skipped merges/roots).
        files_analyzed: Modified Python files whose before/after pair was parsed.
        samples: JSONL records written.
        actions: ``(action, count)`` pairs across every emitted instruction.
        output: Path of the written JSONL file.
        fmt: Output layout (``openai`` or ``alpaca``).
    """

    commits_scanned: int
    files_analyzed: int
    samples: int
    actions: tuple[tuple[str, int], ...]
    output: Path
    fmt: str


def collect_samples(
    repo_root: Path,
    *,
    branch: str = "HEAD",
    steps: int = 50,
    include_repo_map: bool = True,
    repo_map_limit: int = _REPO_MAP_FILES,
) -> list[SftSample]:
    """Mine the most recent ``steps`` commits into samples (no file written).

    Args:
        repo_root: Any path inside the target repository.
        branch: Revision to walk (default ``HEAD``; a branch name or SHA).
        steps: Number of most recent commits to inspect.
        include_repo_map: Build a parent-commit code map for each sample.
        repo_map_limit: Maximum files summarised into that map.

    Returns:
        Samples in commit order (oldest first), deterministic for a given repo.

    Raises:
        FlywheelError: ``steps`` is invalid, the revision does not resolve, or the
            repository cannot be read.
    """
    samples, _, _ = _collect(
        repo_root,
        branch=branch,
        steps=steps,
        include_repo_map=include_repo_map,
        repo_map_limit=repo_map_limit,
    )
    return samples


def export_sft(
    repo_root: Path,
    *,
    branch: str = "HEAD",
    steps: int = 50,
    output: Path = Path("moray-sft.jsonl"),
    fmt: str = "openai",
    include_repo_map: bool = True,
    repo_map_limit: int = _REPO_MAP_FILES,
) -> ExportReport:
    """Write the mined samples as JSONL and return export counters.

    Args:
        repo_root: Any path inside the target repository.
        branch: Revision to walk.
        steps: Number of most recent commits to inspect.
        output: Destination ``.jsonl`` path (parents are created).
        fmt: ``openai`` (messages) or ``alpaca`` (instruction/input/output).
        include_repo_map: Include the parent-commit code map in each sample.
        repo_map_limit: Maximum files in that map.

    Returns:
        Counters describing the export.

    Raises:
        FlywheelError: The format is unsupported, the revision is invalid, or the
            repository/archive cannot be read.
        OSError: The output file cannot be written.
    """
    if fmt not in _FORMATS:
        raise FlywheelError(
            f"unsupported format {fmt!r}; choose one of {', '.join(_FORMATS)}"
        )
    samples, commits_scanned, files_analyzed = _collect(
        repo_root,
        branch=branch,
        steps=steps,
        include_repo_map=include_repo_map,
        repo_map_limit=repo_map_limit,
    )
    system_prompt = IntentParser().system_prompt
    destination = output.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            record = sample.as_openai(system_prompt) if fmt == "openai" else sample.as_alpaca()
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    actions = Counter(
        instruction.action
        for sample in samples
        for instruction in sample.plan.instructions
    )
    return ExportReport(
        commits_scanned=commits_scanned,
        files_analyzed=files_analyzed,
        samples=len(samples),
        actions=tuple(sorted(actions.items())),
        output=destination,
        fmt=fmt,
    )


def detect_instructions(
    before_source: str, after_source: str, path: str
) -> list[Instruction]:
    """Detect the engine instructions that explain an AST-level diff.

    Supported, deliberately high-precision patterns:

    * a top-level function/class/module variable whose name changed while its
      signature and body stayed identical → ``rename_symbol``;
    * calls whose callee changed while the argument list stayed identical →
      ``replace_call``;
    * a function that gained exactly a ``-> None`` return annotation while its
      parameters/body stayed identical → ``add_type_hints``;
    * module-level imports added or removed → ``add_import`` / ``remove_import``.

    Args:
        before_source: The file at the parent commit.
        after_source: The file at the commit.
        path: Repository-relative path used as every instruction's target.

    Returns:
        Validated instructions in a deterministic order. Empty when either side
        does not parse or no supported pattern is present.
    """
    before_module = _parse(before_source)
    after_module = _parse(after_source)
    if before_module is None or after_module is None:
        return []
    renderer = cst.Module(body=[])
    renames = _rename_pairs(before_module, after_module, renderer)
    replace_pairs = [
        pair
        for pair in _replacement_pairs(before_module, after_module, renderer)
        if pair not in set(renames)
    ]
    hints = _added_none_hints(before_module, after_module, renderer)
    removed_imports, added_imports = _import_changes(before_module, after_module)

    instructions: list[Instruction] = []
    instruction: Instruction

    def add(instruction: Instruction) -> None:
        """Append with the next id (models are built with a placeholder id)."""
        instructions.append(
            instruction.model_copy(update={"id": f"step{len(instructions) + 1}"})
        )

    for old, new in renames:
        try:
            instruction = RenameSymbol(
                id="pending",
                target_file=path,
                old_name=old,
                new_name=new,
                reasoning=f"The commit renames {old!r} to {new!r} in {path}.",
            )
        except ValueError:
            logger.debug("skipping an invalid rename", exc_info=True)
            continue
        add(instruction)
    for old, new in replace_pairs:
        try:
            instruction = ReplaceCall(
                id="pending",
                target_file=path,
                old_call=old,
                new_call=new,
                reasoning=f"The commit replaces calls to {old!r} with {new!r}.",
            )
        except ValueError:
            logger.debug("skipping an invalid call replacement", exc_info=True)
            continue
        add(instruction)
    for function_name in hints:
        try:
            instruction = AddTypeHints(
                id="pending",
                target_file=path,
                function_name=function_name,
                reasoning=(
                    f"The commit adds a '-> None' return annotation to "
                    f"{function_name!r}."
                ),
            )
        except ValueError:
            logger.debug("skipping an invalid type-hint instruction", exc_info=True)
            continue
        add(instruction)
    for spec in removed_imports:
        try:
            instruction = RemoveImport(
                id="pending",
                target_file=path,
                module=spec.module,
                name=spec.name,
                reasoning=f"The commit removes the import {_import_text(spec)!r}.",
            )
        except ValueError:
            logger.debug("skipping an invalid removed import", exc_info=True)
            continue
        add(instruction)
    for spec in added_imports:
        try:
            instruction = AddImport(
                id="pending",
                target_file=path,
                module=spec.module,
                name=spec.name,
                alias=spec.alias,
                reasoning=f"The commit adds the import {_import_text(spec)!r}.",
            )
        except ValueError:
            logger.debug("skipping an invalid added import", exc_info=True)
            continue
        add(instruction)
    return instructions


# --------------------------------------------------------------------------- #
# Git plumbing
# --------------------------------------------------------------------------- #


def _collect(
    repo_root: Path,
    *,
    branch: str,
    steps: int,
    include_repo_map: bool,
    repo_map_limit: int,
) -> tuple[list[SftSample], int, int]:
    """Shared implementation for ``collect_samples`` and ``export_sft``."""
    if steps < 1:
        raise FlywheelError(f"steps must be at least 1, got {steps}")
    root = _repo_root(repo_root)
    shas = _recent_commits(root, branch, steps)
    samples: list[SftSample] = []
    files_analyzed = 0
    for sha in shas:
        parents = _parents(root, sha)
        if len(parents) != 1:
            # Root commits have no "before"; merge diffs are not single-parent.
            continue
        subject = _subject(root, sha)
        changed = _changed_python_files(root, sha)
        if not changed:
            continue
        repo_map = (
            _parent_repo_map(root, parents[0], changed, repo_map_limit)
            if include_repo_map
            else ""
        )
        for path in changed:
            files_analyzed += 1
            before = _show(root, f"{sha}^:{path}")
            after = _show(root, f"{sha}:{path}")
            if before is None or after is None:
                continue
            if len(before) > _MAX_SOURCE_CHARS or len(after) > _MAX_SOURCE_CHARS:
                logger.debug("skipping %s in %s: source too large", path, sha)
                continue
            instructions = detect_instructions(before, after, path)
            if not instructions:
                continue
            plan = RefactorPlan(
                instructions=instructions,
                summary=f"Detected {len(instructions)} change(s) in {path}.",
            )
            samples.append(
                SftSample(
                    commit=sha,
                    subject=subject,
                    path=path,
                    repo_root=str(root),
                    instruction=_instruction_text(instructions, path),
                    context=_context(sha, subject, path, before, repo_map),
                    plan=plan,
                )
            )
    return samples, len(shas), files_analyzed


def _repo_root(path: Path) -> Path:
    """Resolve the repository root from any path inside it."""
    candidate = path.expanduser().resolve()
    if not candidate.exists():
        raise FlywheelError(f"{path} does not exist")
    result = _git(candidate, "rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        raise FlywheelError(f"{path} is not inside a git working tree")
    return Path(result.stdout.strip()).resolve()


def _recent_commits(root: Path, branch: str, steps: int) -> list[str]:
    """Return up to ``steps`` recent commit SHAs, oldest first."""
    if not branch or branch.startswith("-"):
        raise FlywheelError(f"invalid revision {branch!r}")
    result = _git(root, "rev-list", f"--max-count={steps}", branch, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise FlywheelError(f"cannot read history for {branch!r}: {detail}")
    shas = result.stdout.split()
    shas.reverse()
    return shas


def _parents(root: Path, sha: str) -> list[str]:
    """Return the parent SHAs of ``sha``."""
    output = _git(root, "rev-list", "--parents", "-n", "1", sha).stdout.strip()
    return output.split()[1:]


def _subject(root: Path, sha: str) -> str:
    """Return the commit subject line."""
    return _git(root, "show", "-s", "--format=%s", sha).stdout.strip()


def _changed_python_files(root: Path, sha: str) -> list[str]:
    """Return modified ``.py`` paths in ``sha`` (added/deleted/renamed skipped)."""
    result = _git(
        root,
        "-c",
        "core.quotepath=false",
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        sha,
    )
    paths: list[str] = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status = fields[0][:1]
        if status != "M":
            continue
        path = fields[-1]
        if path.endswith(".py"):
            paths.append(path)
    return paths


def _show(root: Path, revision_path: str) -> str | None:
    """Read one blob from a revision, or None when it does not exist."""
    result = _git(root, "show", revision_path, check=False)
    if result.returncode != 0:
        return None
    return result.stdout


def _parent_repo_map(
    root: Path, parent: str, changed: Sequence[str], limit: int
) -> str:
    """Render the code map of the parent tree (best effort, never fatal)."""
    workspace = Path(tempfile.mkdtemp(prefix="moray-flywheel-"))
    try:
        try:
            archive = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "archive",
                    "--format=tar",
                    parent,
                    "--",
                    "*.py",
                ],
                capture_output=True,
                timeout=_GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("repo map for %s timed out", parent)
            return ""
        if archive.returncode != 0 or not archive.stdout:
            logger.warning("repo map for %s unavailable", parent)
            return ""
        with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
            members: list[tarfile.TarInfo] = []
            for member in tar.getmembers():
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise FlywheelError(
                        f"unsafe path in git archive for {parent}: {member.name!r}"
                    )
                if member.isfile() or member.isdir():
                    members.append(member)
            try:
                tar.extractall(workspace, members=members, filter="data")
            except TypeError:  # pragma: no cover - Python < 3.11.4
                tar.extractall(workspace, members=members)
        available = sorted(
            path.relative_to(workspace).as_posix()
            for path in workspace.rglob("*.py")
            if path.is_file()
        )
        ordered = [path for path in changed if path in available]
        ordered.extend(path for path in available if path not in ordered)
        if not ordered:
            return ""
        return generate_repo_map(workspace, ordered[:limit])
    except (OSError, tarfile.TarError, MorayError) as exc:
        logger.warning("repo map for %s unavailable: %s", parent, exc)
        return ""
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _git(
    cwd: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run one git command with a timeout; never use a shell."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FlywheelError(
            f"git {' '.join(args)} timed out after {_GIT_TIMEOUT_SECONDS}s"
        ) from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
        raise FlywheelError(f"git {' '.join(args)} failed: {detail}")
    return result


# --------------------------------------------------------------------------- #
# AST analysis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Symbol:
    """A top-level definition paired by name across two revisions."""

    name: str
    kind: str
    signature: str
    body: str


@dataclass(frozen=True)
class _ImportSpec:
    """One module-level import binding."""

    module: str
    name: str | None
    alias: str | None


def _parse(source: str) -> cst.Module | None:
    """Parse Python source, returning None on a syntax error."""
    try:
        return cst.parse_module(source)
    except cst.ParserSyntaxError:
        return None


def _symbols(module: cst.Module, renderer: cst.Module) -> list[_Symbol]:
    """Collect top-level functions, classes and simple module assignments."""
    symbols: list[_Symbol] = []
    for statement in module.body:
        if isinstance(statement, cst.FunctionDef):
            symbols.append(
                _Symbol(
                    name=statement.name.value,
                    kind="function",
                    signature=_function_signature(statement, renderer),
                    body=renderer.code_for_node(statement.body),
                )
            )
        elif isinstance(statement, cst.ClassDef):
            signature = "|".join(
                [
                    "class",
                    "".join(
                        renderer.code_for_node(decorator)
                        for decorator in statement.decorators
                    ),
                    ",".join(
                        renderer.code_for_node(base) for base in statement.bases
                    ),
                    ",".join(
                        renderer.code_for_node(keyword)
                        for keyword in statement.keywords
                    ),
                ]
            )
            symbols.append(
                _Symbol(
                    name=statement.name.value,
                    kind="class",
                    signature=signature,
                    body=renderer.code_for_node(statement.body),
                )
            )
        elif isinstance(statement, cst.SimpleStatementLine):
            for small in statement.body:
                if not isinstance(small, cst.Assign) or len(small.targets) != 1:
                    continue
                target = small.targets[0].target
                if not isinstance(target, cst.Name):
                    continue
                symbols.append(
                    _Symbol(
                        name=target.value,
                        kind="variable",
                        signature="",
                        body=renderer.code_for_node(small.value),
                    )
                )
    return symbols


def _function_signature(node: cst.FunctionDef, renderer: cst.Module) -> str:
    """Render everything about a function except its name and body."""
    parts = [
        "async" if node.asynchronous is not None else "def",
        renderer.code_for_node(node.params),
        "".join(renderer.code_for_node(item) for item in node.decorators),
    ]
    if node.returns is not None:
        parts.append("-> " + renderer.code_for_node(node.returns.annotation))
    return "|".join(parts)


def _rename_pairs(
    before: cst.Module, after: cst.Module, renderer: cst.Module
) -> list[tuple[str, str]]:
    """Pair a disappeared symbol with an identical-looking new one."""
    before_symbols = _symbols(before, renderer)
    after_symbols = _symbols(after, renderer)
    before_counts = Counter(symbol.name for symbol in before_symbols)
    after_counts = Counter(symbol.name for symbol in after_symbols)
    before_unique = {
        symbol.name: symbol
        for symbol in before_symbols
        if before_counts[symbol.name] == 1
    }
    after_unique = {
        symbol.name: symbol
        for symbol in after_symbols
        if after_counts[symbol.name] == 1
    }
    removed = sorted(set(before_unique) - set(after_unique))
    added = [symbol for name, symbol in after_unique.items() if name not in before_unique]
    used: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for old_name in removed:
        old = before_unique[old_name]
        candidates = [
            new
            for new in added
            if new.name not in used
            and new.kind == old.kind
            and new.signature == old.signature
            and new.body == old.body
        ]
        if len(candidates) != 1:
            continue
        new = candidates[0]
        used.add(new.name)
        pairs.append((old.name, new.name))
    return pairs


class _CallCollector(cst.CSTVisitor):
    """Collect ``(callee_text, rendered_arguments)`` for every call in a module."""

    def __init__(self, renderer: cst.Module) -> None:
        self._renderer = renderer
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def visit_Call(self, node: cst.Call) -> None:  # noqa: N802 - libcst hook name
        """Record one call expression; nested calls are visited too."""
        callee = self._renderer.code_for_node(node.func).strip()
        arguments = tuple(self._renderer.code_for_node(arg) for arg in node.args)
        self.calls.append((callee, arguments))


def _call_counts(
    module: cst.Module, renderer: cst.Module
) -> Counter[tuple[str, tuple[str, ...]]]:
    """Count ``(callee_text, rendered_arguments)`` pairs in a module."""
    collector = _CallCollector(renderer)
    module.visit(collector)
    return Counter(collector.calls)


def _replacement_pairs(
    before: cst.Module, after: cst.Module, renderer: cst.Module
) -> list[tuple[str, str]]:
    """Detect callees that were fully replaced with identical argument lists."""
    before_counts = _call_counts(before, renderer)
    after_counts = _call_counts(after, renderer)
    before_callees = {callee for callee, _ in before_counts}
    after_callees = {callee for callee, _ in after_counts}
    pairs: list[tuple[str, str]] = []
    for old in sorted(before_callees - after_callees):
        removed = {
            arguments: count
            for (callee, arguments), count in before_counts.items()
            if callee == old
        }
        replacements: set[str] = set()
        consistent = True
        for arguments, removed_count in removed.items():
            deltas = {
                callee: after_counts.get((callee, arguments), 0)
                - before_counts.get((callee, arguments), 0)
                for callee in after_callees
            }
            candidates = [
                callee
                for callee, delta in deltas.items()
                if delta == removed_count and callee != old
            ]
            if len(candidates) != 1:
                consistent = False
                break
            replacements.add(candidates[0])
        if consistent and len(replacements) == 1:
            pairs.append((old, replacements.pop()))
    return pairs


def _function_shape(
    node: cst.FunctionDef, renderer: cst.Module
) -> tuple[str, str, str, str]:
    """Render a function ignoring its return annotation."""
    return (
        "async" if node.asynchronous is not None else "def",
        renderer.code_for_node(node.params),
        "".join(renderer.code_for_node(item) for item in node.decorators),
        renderer.code_for_node(node.body),
    )


def _functions_by_name(module: cst.Module) -> dict[str, cst.FunctionDef]:
    """Index unique top-level functions by name."""
    functions: dict[str, cst.FunctionDef] = {}
    counts: Counter[str] = Counter()
    for statement in module.body:
        if isinstance(statement, cst.FunctionDef):
            counts[statement.name.value] += 1
            functions[statement.name.value] = statement
    return {
        name: node for name, node in functions.items() if counts[name] == 1
    }


def _added_none_hints(
    before: cst.Module, after: cst.Module, renderer: cst.Module
) -> list[str]:
    """Return functions that gained exactly a ``-> None`` annotation."""
    before_functions = _functions_by_name(before)
    after_functions = _functions_by_name(after)
    names: list[str] = []
    for name in sorted(set(before_functions) & set(after_functions)):
        old = before_functions[name]
        new = after_functions[name]
        if old.returns is not None or new.returns is None:
            continue
        annotation = new.returns.annotation
        if not isinstance(annotation, cst.Name) or annotation.value != "None":
            continue
        if _function_shape(old, renderer) != _function_shape(new, renderer):
            continue
        names.append(name)
    return names


def _dotted(node: cst.BaseExpression | None) -> str | None:
    """Render a dotted import target, or None for an unsupported form."""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr.value}" if base is not None else None
    return None


def _import_alias_name(alias: cst.ImportAlias) -> str | None:
    """Return the local alias of an import when it is a simple name."""
    if alias.asname is None or not isinstance(alias.asname.name, cst.Name):
        return None
    return alias.asname.name.value


def _import_specs(module: cst.Module) -> Iterator[_ImportSpec]:
    """Yield module-level import bindings; relative and star imports are skipped."""
    for statement in module.body:
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for small in statement.body:
            if isinstance(small, cst.Import):
                for alias in small.names:
                    module_name = _dotted(alias.name)
                    if module_name is None:
                        continue
                    yield _ImportSpec(
                        module=module_name,
                        name=None,
                        alias=_import_alias_name(alias),
                    )
            elif isinstance(small, cst.ImportFrom):
                if small.relative or small.module is None:
                    continue
                if isinstance(small.names, cst.ImportStar):
                    continue
                module_name = _dotted(small.module)
                if module_name is None:
                    continue
                for alias in small.names:
                    if not isinstance(alias.name, cst.Name):
                        continue  # from x import * has no concrete binding
                    yield _ImportSpec(
                        module=module_name,
                        name=alias.name.value,
                        alias=_import_alias_name(alias),
                    )


def _import_changes(
    before: cst.Module, after: cst.Module
) -> tuple[list[_ImportSpec], list[_ImportSpec]]:
    """Return removed and added imports, deterministic and duplicate-safe."""
    before_counts = Counter(_import_specs(before))
    after_counts = Counter(_import_specs(after))

    def key(spec: _ImportSpec) -> tuple[str, str, str]:
        return (spec.module, spec.name or "", spec.alias or "")

    removed = sorted(
        (spec for spec, count in (before_counts - after_counts).items() if count == 1),
        key=key,
    )
    added = sorted(
        (spec for spec, count in (after_counts - before_counts).items() if count == 1),
        key=key,
    )
    return removed, added


def _import_text(spec: _ImportSpec) -> str:
    """Render an import for the natural-language instruction."""
    if spec.name is None:
        return f"import {spec.module}" + (f" as {spec.alias}" if spec.alias else "")
    text = f"from {spec.module} import {spec.name}"
    return text + (f" as {spec.alias}" if spec.alias else "")


def _instruction_text(instructions: Sequence[Instruction], path: str) -> str:
    """Summarise the detected instructions as one natural-language request."""
    parts: list[str] = []
    for instruction in instructions:
        if isinstance(instruction, RenameSymbol):
            parts.append(
                f"Rename `{instruction.old_name}` to `{instruction.new_name}` "
                f"in `{path}`."
            )
        elif isinstance(instruction, ReplaceCall):
            parts.append(
                f"Replace calls to `{instruction.old_call}` with "
                f"`{instruction.new_call}` in `{path}`."
            )
        elif isinstance(instruction, AddTypeHints):
            target = instruction.function_name or "every function"
            parts.append(
                f"Add a `-> None` return annotation to `{target}` in `{path}`."
            )
        elif isinstance(instruction, RemoveImport):
            spec = _ImportSpec(instruction.module, instruction.name, None)
            parts.append(f"Remove `{_import_text(spec)}` from `{path}`.")
        elif isinstance(instruction, AddImport):
            spec = _ImportSpec(instruction.module, instruction.name, instruction.alias)
            parts.append(f"Add `{_import_text(spec)}` to `{path}`.")
    return " ".join(parts) if parts else f"Apply the detected refactor to {path}."


def _context(
    sha: str, subject: str, path: str, before_source: str, repo_map: str
) -> str:
    """Build the user-message context: commit, repo map and pre-change source."""
    sections = [f"【Commit】 {sha[:12]} {subject}".strip()]
    if repo_map.strip():
        sections.append(f"【Project Code Map】\n{repo_map}")
    sections.append(
        f"【Before File: {path}】\n````python\n{_truncate(before_source)}\n````"
    )
    return "\n\n".join(sections)


def _truncate(source: str) -> str:
    """Bound the before-file snippet included in the user message."""
    if len(source) <= _MAX_SNIPPET_CHARS:
        return source.rstrip("\n")
    return source[:_MAX_SNIPPET_CHARS].rstrip("\n") + _TRUNCATION_MARKER
