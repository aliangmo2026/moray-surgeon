"""Run the two stress bombings and write docs/STRESS_TEST_REPORT.md.

Bombing A sends a 50-instruction ``RefactorPlan`` (two project-wide renames,
twenty add-import/replace-call pairs, four rename-preserving call rewrites and
four re-export removals) through the real orchestrator, with the real ruff,
mypy and pytest gates and the AST scope guard enabled.

Bombing B mixes exactly one poisoned instruction into a small otherwise-valid
plan and proves the failure path: the attempt is rolled back, the tree returns
to HEAD, and the feedback handed to the next prompt contains the captured diff
and the gate diagnostics. B-1 poisons a type-correct call into a mypy error;
B-2 poisons it into a runtime (pytest) failure, so both gates are exercised.

Usage::

    python tests/stress_test/run_bombing.py                # full run + report
    python tests/stress_test/run_bombing.py --keep         # keep the work dirs
    python tests/stress_test/run_bombing.py --only-a       # skip disaster runs

The report is written to ``docs/STRESS_TEST_REPORT.md`` at the repository root
by default; the generated project lives in a temporary directory and is not
committed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # executed as a script from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import moray_surgeon
from moray_surgeon import orchestrator as orchestrator_module
from moray_surgeon.orchestrator import Orchestrator
from moray_surgeon.schema import (
    AddImport,
    RefactorPlan,
    RemoveImport,
    RenameSymbol,
    ReplaceCall,
)
from moray_surgeon.transaction import TransactionManager
from moray_surgeon.verify import Verifier, VerifyResult

if __package__ in (None, ""):
    from generate_project import (  # type: ignore[no-redef]
        ProjectStats,
        generate_project,
        init_git_repository,
    )
else:
    from tests.stress_test.generate_project import (
        ProjectStats,
        generate_project,
        init_git_repository,
    )

_GATES_TIMEOUT = 600
_BASELINE_TARGET = "stresspkg"
_REPORT_TAIL_CHARS = 2500
_DIFF_LIMIT_CHARS = 12000


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #


def _module(index: int, package: str) -> str:
    return f"stresspkg/{package}/m{index:02d}.py"


def build_bombing_a() -> RefactorPlan:
    """Build the 50-instruction extreme refactor plan.

    The two renames keep their imports sorted (``AbstractEntity`` sorts before
    ``BaseRepository``; ``verify_record`` after ``reject_record``) because the
    rename atom is deliberately not an import sorter. The project enables ruff's
    isort rules, so an order-breaking name would be caught by the gate rather
    than silently accepted.
    """
    instructions = [
        RenameSymbol(
            id="rename-base",
            target_file="stresspkg/core/base.py",
            old_name="BaseEntity",
            new_name="AbstractEntity",
            reasoning="Make the root entity abstract by name.",
        ),
        RenameSymbol(
            id="rename-validator",
            target_file="stresspkg/core/validation.py",
            old_name="validate_record",
            new_name="verify_record",
            reasoning="Align the validator with the verify_* naming convention.",
        ),
    ]

    for index in range(20):
        instructions.append(
            AddImport(
                id=f"import-{index:02d}",
                target_file=_module(index, "domain"),
                module="stresspkg.core.validation",
                name="encode_payload",
                reasoning="Bring the shared encoder into scope.",
            )
        )
    for index in range(20):
        instructions.append(
            ReplaceCall(
                id=f"encode-{index:02d}",
                target_file=_module(index, "domain"),
                old_call="json.dumps",
                new_call="encode_payload",
                depends_on=[f"import-{index:02d}"],
                reasoning="Route JSON encoding through the shared helper.",
            )
        )
    for index in range(20, 24):
        instructions.append(
            ReplaceCall(
                id=f"format-{index:02d}",
                target_file=_module(index, "domain"),
                old_call="str",
                new_call="format_key",
                reasoning="Format keys canonically instead of relying on str().",
            )
        )

    for name in ("DeprecatedPolicy", "LegacyRecord", "ObsoleteService", "RetiredHandler"):
        instructions.append(
            RemoveImport(
                id=f"drop-{name}",
                target_file="stresspkg/__init__.py",
                module="stresspkg.core.legacy",
                name=name,
                reasoning="Drop a deprecated re-export from the package root.",
            )
        )

    return RefactorPlan(
        instructions=instructions,
        summary="50-instruction extreme refactor across the 50-module ring",
    )


def build_bombing_b(poison: str) -> RefactorPlan:
    """Build a small valid plan with one poisoned ``validate_record`` call.

    Args:
        poison: Replacement callee for the last instruction. ``"Record"`` makes
            mypy reject the argument type; ``"reject_record"`` keeps the types
            valid but raises at runtime, so pytest is the gate that catches it.
    """
    return RefactorPlan(
        instructions=[
            AddImport(id="import-00", target_file=_module(0, "domain"),
                      module="stresspkg.core.validation", name="encode_payload"),
            ReplaceCall(id="encode-00", target_file=_module(0, "domain"), old_call="json.dumps",
                        new_call="encode_payload", depends_on=["import-00"]),
            AddImport(id="import-20", target_file=_module(20, "domain"),
                      module="stresspkg.core.validation", name="encode_payload"),
            ReplaceCall(id="encode-20", target_file=_module(20, "domain"), old_call="json.dumps",
                        new_call="encode_payload", depends_on=["import-20"]),
            ReplaceCall(id="format-21", target_file=_module(21, "domain"), old_call="str",
                        new_call="format_key"),
            ReplaceCall(
                id="poison",
                target_file=_module(4, "domain"),
                old_call="validate_record",
                new_call=poison,
                reasoning="POISONED: one bad instruction among good ones.",
            ),
        ],
        summary=f"small valid plan plus one poisoned instruction ({poison})",
    )


# --------------------------------------------------------------------------- #
# Harness plumbing
# --------------------------------------------------------------------------- #


class ReplayParser:
    """Return a scripted plan, recording the prompt context each round.

    On a round past the supplied plans it raises :class:`StopAfterFailure`; by
    then the failed attempt has already been rolled back, so the recorded
    context is exactly the evidence the next model call would have received.
    """

    def __init__(self, plans: list[RefactorPlan]) -> None:
        self._plans = plans
        self.contexts: list[str] = []
        self.rounds = 0

    def parse(self, intent: str, repo_root: Path, extra_context: str = "") -> RefactorPlan:
        self.contexts.append(extra_context)
        self.rounds += 1
        if self.rounds > len(self._plans):
            raise StopAfterFailureError(self.contexts[-1])
        return self._plans[self.rounds - 1]


class StopAfterFailureError(Exception):
    """Raised by :class:`ReplayParser` after a failed round was captured."""

    def __init__(self, context: str) -> None:
        super().__init__("failure evidence captured")
        self.context = context


class TimedVerifier(Verifier):
    """Real verifier that records how long the three gates took."""

    def __init__(
        self,
        repo_root: Path,
        python_executable: str = sys.executable,
        mypy_target: str = "src",
        pytest_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(repo_root, python_executable, mypy_target, pytest_args, env)
        self.seconds = 0.0
        self.result: VerifyResult | None = None

    def verify(self) -> VerifyResult:
        start = time.perf_counter()
        try:
            self.result = super().verify()
            return self.result
        finally:
            self.seconds = time.perf_counter() - start


@dataclass
class Timings:
    """Wall-clock measurements for one bombing, in seconds."""

    total: float = 0.0
    apply: float = 0.0
    scope_snapshot: float = 0.0
    scope_verify: float = 0.0
    verification: float = 0.0
    rollback: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "total": round(self.total, 3),
            "apply": round(self.apply, 3),
            "scope_snapshot": round(self.scope_snapshot, 3),
            "scope_verify": round(self.scope_verify, 3),
            "verification": round(self.verification, 3),
            "rollback": round(self.rollback, 3),
        }


@contextmanager
def _timed_global(module: object, name: str, timings: Timings, field_name: str) -> Iterator[None]:
    """Patch a module-level callable and record its cumulative wall time."""
    original = getattr(module, name)

    def wrapper(*args: object, **kwargs: object) -> object:
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            setattr(
                timings,
                field_name,
                getattr(timings, field_name) + time.perf_counter() - start,
            )

    setattr(module, name, wrapper)
    try:
        yield
    finally:
        setattr(module, name, original)


@contextmanager
def _timed_rollback(timings: Timings) -> Iterator[None]:
    """Patch ``TransactionManager.rollback`` to time the safety net itself."""
    original = TransactionManager.rollback

    def wrapper(self: TransactionManager) -> None:
        start = time.perf_counter()
        try:
            original(self)
        finally:
            timings.rollback += time.perf_counter() - start

    TransactionManager.rollback = wrapper  # type: ignore[method-assign]
    try:
        yield
    finally:
        TransactionManager.rollback = original  # type: ignore[method-assign]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=True,
    )
    return result.stdout


def _configure_git_environment() -> None:
    """Isolate git from the host configuration for the whole harness."""
    os.environ.update(
        {
            "GIT_AUTHOR_NAME": "moray stress",
            "GIT_AUTHOR_EMAIL": "stress@example.com",
            "GIT_COMMITTER_NAME": "moray stress",
            "GIT_COMMITTER_EMAIL": "stress@example.com",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "commit.gpgsign",
            "GIT_CONFIG_VALUE_0": "false",
            "GIT_CONFIG_KEY_1": "core.autocrlf",
            "GIT_CONFIG_VALUE_1": "false",
        }
    )


def _verifier(root: Path) -> TimedVerifier:
    return TimedVerifier(
        root,
        python_executable=sys.executable,
        mypy_target=_BASELINE_TARGET,
        pytest_args=["-q"],
    )


def _run_baseline_gates(root: Path) -> dict[str, Any]:
    """Run ruff/mypy/pytest once before any bombing to prove a green baseline."""
    verifier = Verifier(
        root,
        python_executable=sys.executable,
        mypy_target=_BASELINE_TARGET,
        pytest_args=["-q"],
    )
    start = time.perf_counter()
    result = verifier.verify()
    return {
        "ok": result.ok,
        "stage": result.stage,
        "seconds": round(time.perf_counter() - start, 3),
        "stdout": result.stdout[-_REPORT_TAIL_CHARS:],
        "stderr": result.stderr[-_REPORT_TAIL_CHARS:],
    }


def _plan_composition(plan: RefactorPlan) -> dict[str, int]:
    return dict(sorted(Counter(item.action for item in plan.instructions).items()))


def _remove_tree(path: Path) -> None:
    """Remove a tree even when git's read-only object files block unlink on Windows."""
    if not path.exists():
        return

    def on_error(function: Callable[[str], object], target: str, _error: BaseException) -> None:
        try:
            os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
            function(target)
        except OSError:
            pass

    shutil.rmtree(path, onexc=on_error)


def _new_project(workdir: Path, name: str) -> tuple[Path, ProjectStats]:
    """Create a fresh project directory, replacing any previous run of it."""
    root = workdir / name
    # The directory is inside the harness workdir and named by us; replacing it
    # makes repeated runs deterministic instead of failing on "nothing to
    # commit" against the previous tree.
    _remove_tree(root)
    stats = generate_project(root)
    init_git_repository(root)
    return root, stats


def _baseline_head(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD").strip()


# --------------------------------------------------------------------------- #
# Bombing A
# --------------------------------------------------------------------------- #


def run_bombing_a(workdir: Path) -> dict[str, Any]:
    """Execute the 50-instruction plan against a fresh project."""
    root, stats = _new_project(workdir, "bombing-a")
    plan = build_bombing_a()
    baseline_head = _baseline_head(root)
    baseline = _run_baseline_gates(root)

    parser = ReplayParser([plan])
    verifier = _verifier(root)
    timings = Timings()
    started = time.perf_counter()
    with _timed_global(orchestrator_module, "execute_plan", timings, "apply"), _timed_global(
        orchestrator_module, "snapshot_repository", timings, "scope_snapshot"
    ), _timed_global(
        orchestrator_module, "verify_repository_scope", timings, "scope_verify"
    ), _timed_rollback(timings):
        report = Orchestrator(parser, verifier, max_attempts=1).run(
            "extreme refactor: 50 instructions across the ring", root
        )
    timings.total = time.perf_counter() - started
    timings.verification = verifier.seconds

    commit_stats = _git(root, "show", "--stat", "--oneline", "--no-color", "HEAD")
    diff_numstat = _git(root, "diff", "--numstat", baseline_head, "HEAD")
    numstat_rows = [line.split("\t") for line in diff_numstat.splitlines() if line.strip()]
    files_changed = len(numstat_rows)
    insertions = sum(int(row[0]) for row in numstat_rows if row[0].isdigit())
    deletions = sum(int(row[1]) for row in numstat_rows if row[1].isdigit())
    status = _git(root, "status", "--porcelain")

    return {
        "ok": bool(report.success),
        "attempts": report.attempts,
        "plan": _plan_composition(plan),
        "instructions": len(plan.instructions),
        "stats": stats.as_dict(),
        "baseline": baseline,
        "timings": timings.as_dict(),
        "log": report.log,
        "stage": report.last_verify.stage if report.last_verify else None,
        "verify_stdout": (report.last_verify.stdout if report.last_verify else "")[
            -_REPORT_TAIL_CHARS:
        ],
        "commit": commit_stats,
        "files_changed": files_changed,
        "insertions": insertions,
        "deletions": deletions,
        "status_clean": status.strip() == "",
    }


# --------------------------------------------------------------------------- #
# Bombing B
# --------------------------------------------------------------------------- #


_STAGE_PATTERN = re.compile(r"\[(mypy|pytest|ruff|scope) failed\]")


def _extract_diff(context: str) -> str:
    """Pull the captured git diff out of the retry context."""
    marker = "Code changes from that attempt (git diff):\n"
    if marker not in context:
        return context[:_DIFF_LIMIT_CHARS]
    tail = context.split(marker, 1)[1]
    for end_marker in ("\n\nVerification error:", "\n\nPrevious attempt failed"):
        if end_marker in tail:
            tail = tail.split(end_marker, 1)[0]
    return tail.strip()


def _extract_feedback(context: str) -> str:
    marker = "Verification error:\n"
    if marker not in context:
        return ""
    return context.split(marker, 1)[1].strip()


def run_bombing_b(workdir: Path, name: str, poison: str) -> dict[str, Any]:
    """Run a plan with one poisoned instruction and capture the recovery path."""
    root, stats = _new_project(workdir, name)
    plan = build_bombing_b(poison)
    baseline_head = _baseline_head(root)
    parser = ReplayParser([plan])
    verifier = _verifier(root)
    timings = Timings()
    started = time.perf_counter()
    context = ""
    failure_stage = "unknown"
    with _timed_global(orchestrator_module, "execute_plan", timings, "apply"), _timed_global(
        orchestrator_module, "snapshot_repository", timings, "scope_snapshot"
    ), _timed_global(
        orchestrator_module, "verify_repository_scope", timings, "scope_verify"
    ), _timed_rollback(timings):
        try:
            Orchestrator(parser, verifier, max_attempts=2).run(
                "recover from one poisoned instruction", root
            )
        except StopAfterFailureError as exc:
            context = exc.context
    timings.total = time.perf_counter() - started
    timings.verification = verifier.seconds

    match = _STAGE_PATTERN.search(context)
    if match:
        failure_stage = match.group(1)
    status = _git(root, "status", "--porcelain")
    head_after = _baseline_head(root)
    log_count = int(_git(root, "rev-list", "--count", "HEAD").strip())
    return {
        "ok": timings.rollback < 5.0 and status.strip() == "" and head_after == baseline_head,
        "stage": failure_stage,
        "stats": stats.as_dict(),
        "timings": timings.as_dict(),
        "diff": _extract_diff(context),
        "feedback": _extract_feedback(context),
        "status_clean": status.strip() == "",
        "head_restored": head_after == baseline_head,
        "commit_count": log_count,
        "plan": _plan_composition(plan),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _package_version(name: str) -> str:
    """Installed distribution version, or ``unknown`` for a source checkout."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _environment() -> dict[str, str]:
    """Identify the harness, interpreter and dependency versions."""
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "moray_surgeon": moray_surgeon.__version__,
        "libcst": _package_version("libcst"),
        "pydantic": _package_version("pydantic"),
        "jedi": _package_version("jedi"),
    }


def _fence(text: str, language: str = "") -> str:
    """Wrap text in a fenced block, choosing a fence the content cannot close."""
    body = text.rstrip()
    fence = "```"
    while fence in body:
        fence += "`"
    return f"{fence}{language}\n{body}\n{fence}"


def _table(rows: list[tuple[str, object]]) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    return "\n".join(lines)


def _timing_table(timings: dict[str, float]) -> str:
    return _table(
        [
            ("Total round", f"{timings['total']:.3f} s"),
            ("AST apply (all instructions)", f"{timings['apply']:.3f} s"),
            ("Scope snapshot (before)", f"{timings['scope_snapshot']:.3f} s"),
            ("Scope verification (after)", f"{timings['scope_verify']:.3f} s"),
            ("Verification gates", f"{timings['verification']:.3f} s"),
            ("Git rollback", f"{timings['rollback']:.3f} s"),
        ]
    )


def write_report(results: dict[str, Any], path: Path) -> None:
    """Render the stress report from the collected evidence."""
    environment = results["environment"]
    a = results["bombing_a"]
    b1 = results["bombing_b_mypy"]
    b2 = results["bombing_b_pytest"]
    stats = a["stats"]

    lines: list[str] = []
    lines.append("# STRESS TEST REPORT")
    lines.append("")
    lines.append(
        "Generated by `tests/stress_test/run_bombing.py`. Every number below is a real "
        f"measurement against a synthetic project of {stats['lines']:,} lines and "
        f"{stats['classes']} classes. Bombing A executes one 50-instruction plan behind the "
        "real ruff/mypy/pytest gates plus the AST scope guard. Bombing B injects exactly one "
        "poisoned instruction into five valid ones and proves the transaction rolls back to "
        "HEAD and that the feedback handed to the next model call contains the captured diff "
        "and the failing gate's diagnostics."
    )
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(_table(list(environment.items())))
    lines.append("")
    lines.append("## Test bed")
    lines.append("")
    lines.append(
        "Generated by [`tests/stress_test/generate_project.py`]"
        "(../tests/stress_test/generate_project.py): "
        "50 modules in `stresspkg/domain` + `stresspkg/services`, 300 classes across six "
        "inheritance roles (entity/service/repository/handler/policy/snapshot), a runtime "
        "dependency ring closed by lazy imports, and typing-only back-edges under "
        "`if TYPE_CHECKING:`."
    )
    lines.append("")
    lines.append(
        _table(
            [
                ("Modules", stats["modules"]),
                ("Classes", stats["classes"]),
                ("Files", stats["files"]),
                ("Lines", f"{stats['lines']:,}"),
                ("Baseline gates", "PASS" if a["baseline"]["ok"] else "FAIL"),
                ("Baseline gate time", f"{a['baseline']['seconds']:.3f} s"),
            ]
        )
    )
    lines.append("")
    lines.append("Baseline verification output (tail):")
    lines.append("")
    lines.append(_fence(a["baseline"]["stdout"] or a["baseline"]["stderr"] or "(no output)"))
    lines.append("")
    lines.append("## Bombing A — 50-instruction extreme refactor")
    lines.append("")
    lines.append(
        "One `RefactorPlan` with 50 instructions: two project-wide `rename_symbol` "
        "operations, 20 `add_import` + 20 `replace_call` pairs (the import is required for "
        "the call to type-check), four `replace_call` rewrites in files that already import "
        "the helper, and four `remove_import` operations dropping deprecated re-exports."
    )
    lines.append("")
    lines.append("Plan composition: " + ", ".join(
        f"`{action}` x {count}" for action, count in a["plan"].items()
    ))
    lines.append("")
    lines.append(f"**Result: {'PASS' if a['ok'] else 'FAIL'}** — attempts: {a['attempts']}, "
                 f"verification stage: `{a['stage']}`.")
    lines.append("")
    lines.append(_timing_table(a["timings"]))
    lines.append("")
    lines.append(
        _table(
            [
                ("Changed files (commit)", a["files_changed"]),
                ("Insertions", f"+{a['insertions']}"),
                ("Deletions", f"-{a['deletions']}"),
                ("Working tree clean after commit", "yes" if a["status_clean"] else "no"),
            ]
        )
    )
    lines.append("")
    lines.append("Commit stat:")
    lines.append("")
    lines.append(_fence(a["commit"]))
    lines.append("")
    lines.append("Dispatcher log (one line per instruction, in execution order):")
    lines.append("")
    lines.append(_fence("\n".join(a["log"])))
    lines.append("")
    lines.append("Verification output (tail):")
    lines.append("")
    lines.append(_fence(a["verify_stdout"] or "(no output)"))
    lines.append("")
    lines.append("## Bombing B — disaster recovery")
    lines.append("")
    lines.append(
        "Both scenarios run a six-instruction plan in which five instructions are valid and "
        "exactly one is poisoned. The engine must roll back the entire attempt and hand the "
        "model a prompt containing the captured diff plus the failing gate's diagnostics."
    )
    lines.append("")
    for title, result, poison in (
        ("B-1 — mypy gate (type-correct syntax, wrong type)", b1, "Record"),
        ("B-2 — pytest gate (types valid, runtime failure)", b2, "reject_record"),
    ):
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"Poisoned instruction: `replace_call(validate_record -> {poison})` "
                     "on `stresspkg/domain/m04.py`.")
        lines.append("")
        lines.append(f"**Result: {'PASS' if result['ok'] else 'FAIL'}** — the attempt failed at "
                     f"`{result['stage']}` and the repository was restored.")
        lines.append("")
        lines.append(_timing_table(result["timings"]))
        lines.append("")
        lines.append(
            _table(
                [
                    ("Rollback under 5 s", "yes" if result["timings"]["rollback"] < 5 else "no"),
                    ("Working tree clean", "yes" if result["status_clean"] else "no"),
                    ("HEAD restored", "yes" if result["head_restored"] else "no"),
                    ("Commits after run (baseline is 1)", result["commit_count"]),
                ]
            )
        )
        lines.append("")
        lines.append("Feedback handed to the next prompt (gate diagnostics):")
        lines.append("")
        lines.append(_fence(result["feedback"] or "(no feedback captured)"))
        lines.append("")
        lines.append("Captured worktree diff (before rollback):")
        lines.append("")
        lines.append(_fence(result["diff"] or "(no diff captured)", "diff"))
        lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append(_fence(
        "# from a checkout of moray-surgeon:\n"
        "python -m pip install -e \".[dev]\"\n"
        "python tests/stress_test/run_bombing.py --keep --workdir /tmp/moray-stress\n"
        "\n"
        "# or generate and gate the project yourself:\n"
        "python tests/stress_test/generate_project.py --target /tmp/moray-stress\n"
        "cd /tmp/moray-stress && python -m ruff check . && python -m mypy stresspkg"
        " && python -m pytest -q\n"
        "\n"
        "# moray-surgeon's own suite and lint, from the repository root:\n"
        "python -m pytest -q\n"
        "python -m ruff check .",
        "bash",
    ))
    lines.append("")
    lines.append("## Notes and known limits")
    lines.append("")
    lines.append(
        "- Instructions execute **sequentially, in dependency order** (ties keep the authored "
        "order), inside one git transaction. The 50-instruction run is a throughput and safety "
        "test, not a parallel one; deterministic ordering is a deliberate engine invariant."
    )
    lines.append(
        "- `rename_symbol` raises Jedi's built-in project-search budget for the duration of "
        "the search. Jedi otherwise stops after 30 parsed files and silently leaves call "
        "sites unrenamed; the fix is covered by a regression test with 32 modules."
    )
    lines.append(
        "- The rename atom does not reorder imports by design. Bombing A uses names that "
        "preserve isort order (`AbstractEntity`, `verify_record`), so the project's ruff "
        "isort gate remains a real gate rather than a silenced one."
    )
    lines.append(
        "- The AST scope guard fingerprints every function, class and module-level "
        "statement before the plan and re-verifies every changed file afterwards; imports "
        "are excluded because the import atoms are allowed to rewrite them."
    )
    lines.append(
        "- `docs/STRESS_TEST_REPORT.md` is generated evidence and is intentionally kept in "
        "the repository. Re-run the harness to refresh every number."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="directory for the generated projects (default: a temp directory)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "STRESS_TEST_REPORT.md",
        help="path of the markdown report",
    )
    parser.add_argument("--only-a", action="store_true", help="run only bombing A")
    parser.add_argument("--only-b", action="store_true", help="run only bombing B")
    parser.add_argument("--keep", action="store_true", help="keep the generated work dirs")
    parser.add_argument("--json", type=Path, default=None, help="also write raw evidence JSON")
    args = parser.parse_args(argv)

    _configure_git_environment()
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="moray-stress-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"work dir: {workdir}")

    results: dict[str, Any] = {"environment": _environment()}
    if not args.only_b:
        print("bombing A: 50 instructions ...")
        results["bombing_a"] = run_bombing_a(workdir)
        print(f"  -> {'PASS' if results['bombing_a']['ok'] else 'FAIL'}")
    if not args.only_a:
        print("bombing B-1: mypy red ...")
        results["bombing_b_mypy"] = run_bombing_b(workdir, "bombing-b1", "Record")
        print(f"  -> {'PASS' if results['bombing_b_mypy']['ok'] else 'FAIL'}")
        print("bombing B-2: pytest red ...")
        results["bombing_b_pytest"] = run_bombing_b(workdir, "bombing-b2", "reject_record")
        print(f"  -> {'PASS' if results['bombing_b_pytest']['ok'] else 'FAIL'}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    if {"bombing_a", "bombing_b_mypy", "bombing_b_pytest"} <= set(results):
        write_report(results, args.report)
        print(f"report written: {args.report}")

    all_ok = all(
        results[key]["ok"]
        for key in ("bombing_a", "bombing_b_mypy", "bombing_b_pytest")
        if key in results
    )
    if not args.keep:
        for child in workdir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
