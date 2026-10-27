"""Close the loop: intent → plan → transactional apply → verify, with auto-fix.

Each attempt runs inside its own :class:`TransactionManager`. Verification
failure raises an internal signal that makes the context manager roll the
working tree back to HEAD, so the next attempt always re-plans from a clean
baseline instead of patching up a broken tree.

When an attempt fails, the worktree diff is captured *before* the rollback and
handed to the next parse together with the verification feedback. The model then
sees both what it changed and why it broke, while the repository it plans against
is back at HEAD.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from moray_surgeon.engine.dispatcher import execute_plan
from moray_surgeon.engine.repo_map import generate_repo_map
from moray_surgeon.intent import IntentParser
from moray_surgeon.schema import Noop, RefactorPlan
from moray_surgeon.transaction import TransactionManager
from moray_surgeon.verify import Verifier, VerifyResult

if TYPE_CHECKING:
    from moray_surgeon.ui import LiveTrace

_DIFF_TIMEOUT_SECONDS = 30
_MAX_DIFF_CHARS = 12000
# Card-mandated feedback for an empty plan; the full-width comma is intentional.
_NOOP_PLAN_FEEDBACK = "你的计划为空，请重新根据指令生成具体的修改动作"  # noqa: RUF001
_NOOP_DIFF = "(no diff: the previous plan made no changes)"


@dataclass(frozen=True)
class RefactorReport:
    """Outcome of one orchestrated refactor run.

    Attributes:
        success: True when a round applied its plan and verification passed, or
            when a dry run captured the diff and rolled back cleanly.
        attempts: Number of plan/apply/verify rounds consumed.
        plan: The last plan parsed, or None if parsing never succeeded.
        log: Dispatcher log lines of the successful round; empty on failure.
        last_verify: Verification result of the last round, success or failure.
        diff: Unified diff captured by a dry run; empty for normal runs.
    """

    success: bool
    attempts: int
    plan: RefactorPlan | None
    log: list[str]
    last_verify: VerifyResult | None
    diff: str = ""


class _VerificationFailed(Exception):  # noqa: N818 - internal control-flow signal, not an API error
    """Internal signal: roll this attempt back and ask the model again."""

    def __init__(self, verify: VerifyResult, diff: str) -> None:
        super().__init__("verification failed")
        self.verify = verify
        self.diff = diff


class _DryRunComplete(Exception):  # noqa: N818 - internal control-flow signal, not an API error
    """Dry run finished: the transaction must roll back instead of committing."""

    def __init__(self, diff: str) -> None:
        super().__init__("dry run complete")
        self.diff = diff


class _NullTrace:
    """No-op trace for tests, embeddings and the MCP server."""

    def for_attempt(self, attempt: int, max_attempts: int, intent: str) -> _NullTrace:
        return self

    @contextmanager
    def step(self, label: str) -> Iterator[_NullTrace]:
        yield self

    def success(self, message: str) -> None:
        """Do nothing."""

    def error(self, message: str) -> None:
        """Do nothing."""

    def info(self, message: str) -> None:
        """Do nothing."""


_NULL_TRACE = _NullTrace()


class Orchestrator:
    """Run the plan/apply/verify loop, one fresh transaction per attempt."""

    def __init__(
        self,
        parser: IntentParser,
        verifier: Verifier,
        max_attempts: int = 3,
        trace: LiveTrace | None = None,
    ) -> None:
        """Store collaborators; the orchestrator owns no I/O of its own.

        Args:
            parser: Turns intent text into a validated plan.
            verifier: Runs the project gates and never raises on a red gate.
            max_attempts: Maximum plan/apply/verify rounds.
            trace: Optional LiveTrace template. The orchestrator clones it per
                attempt via ``for_attempt``; None disables all narration.
        """
        self._parser = parser
        self._verifier = verifier
        self._max_attempts = max_attempts
        self._trace = trace

    def run(
        self,
        intent: str,
        repo_root: Path,
        context: str = "",
        target_files: list[str] | None = None,
        dry_run: bool = False,
    ) -> RefactorReport:
        """Apply and verify until success or ``max_attempts`` is exhausted.

        A plan made only of ``noop`` instructions consumes an attempt and is
        retried with explicit feedback instead of aborting the whole run.

        Args:
            intent: Original natural-language request.
            repo_root: Repository to refactor.
            context: Extra caller-supplied context for the model. On retries the
                previous attempt's diff and verification feedback are appended.
            target_files: Files summarised into a 【Project Code Map】 block that
                is prepended to the context. None/empty disables the map.
            dry_run: Apply the plan, capture the diff, then roll back without
                running verification or committing.

        Returns:
            A report; ``success=False`` only after exhausting the attempts. A dry
            run returns ``success=True`` with the captured diff and a
            rolled-back log.

        Raises:
            IntentParserError: The model was unreachable or produced no plan.
            FileNotFoundError: A plan targets a missing file.
            NoOpEditError, InstructionError, PathTraversalError: The engine
                rejected the plan; another model round cannot repair that.
            GitError: Transaction setup, commit or rollback failed.
        """
        base_context = context
        if target_files:
            repo_map = generate_repo_map(repo_root, target_files)
            base_context = f"【Project Code Map】\n{repo_map}\n\n{context}".strip()

        last_failure: _VerificationFailed | None = None
        last_verify: VerifyResult | None = None
        last_plan: RefactorPlan | None = None
        log: list[str] = []

        for attempt in range(1, self._max_attempts + 1):
            trace = self._trace_for(attempt, intent)
            attempt_context = base_context
            if last_failure is not None:
                trace.info("Retrying with error context + code diff...")
                attempt_context = _compose_context(base_context, last_failure)

            with trace.step("Requesting plan from LLM..."):
                plan = self._parser.parse(intent, repo_root, attempt_context)
            trace.success("Plan received.")
            last_plan = plan
            if not plan.instructions or all(
                isinstance(item, Noop) for item in plan.instructions
            ):
                # A noop-only plan is a failed attempt, not a hard error: the
                # model gets explicit feedback and consumes one auto-fix round.
                failure = _VerificationFailed(
                    VerifyResult(ok=False, stage="noop", stdout=_NOOP_PLAN_FEEDBACK),
                    _NOOP_DIFF,
                )
                trace.error("Plan contained only noop instructions; retrying.")
                last_failure = failure
                last_verify = failure.verify
                continue
            modified = _modified_file_count(plan)
            verify: VerifyResult | None = None

            try:
                # A fresh transaction per round is what makes auto-fix safe: a
                # failed verify rolls back to HEAD before the next plan arrives.
                with TransactionManager(
                    repo_root, commit_message=f"moray-surgeon: {intent[:60]}"
                ):
                    with trace.step("Applying AST instructions..."):
                        log = execute_plan(plan, repo_root)
                    trace.success(f"Plan applied. Modified {modified} file(s).")
                    if dry_run:
                        diff = _capture_diff(repo_root)
                        trace.info("Dry run: collecting the diff and rolling back...")
                        raise _DryRunComplete(diff)
                    with trace.step("Running verification (ruff, mypy, pytest)..."):
                        verify = self._verifier.verify()
                    if not verify.ok:
                        # Capture before raising: __exit__ is about to reset the
                        # worktree, and this diff is the next prompt's evidence.
                        failure = _VerificationFailed(verify, _capture_diff(repo_root))
                        trace.error("Test failed! Rollback initiated.")
                        raise failure
            except _DryRunComplete as exc:
                trace.success("Dry run complete. Rolled back.")
                return RefactorReport(
                    success=True,
                    attempts=attempt,
                    plan=plan,
                    log=["Dry run complete. Rolled back."],
                    last_verify=None,
                    diff=exc.diff,
                )
            except _VerificationFailed as exc:
                last_failure = exc
                last_verify = exc.verify
                continue

            trace.success("Verification passed! Commit created.")
            return RefactorReport(
                success=True,
                attempts=attempt,
                plan=plan,
                log=log,
                last_verify=verify,
            )

        return RefactorReport(
            success=False,
            attempts=self._max_attempts,
            plan=last_plan,
            log=[],
            last_verify=last_verify,
        )

    def _trace_for(self, attempt: int, intent: str) -> LiveTrace | _NullTrace:
        """Return this attempt's trace, cloning the injected template if possible."""
        if self._trace is None:
            return _NULL_TRACE
        factory = getattr(self._trace, "for_attempt", None)
        if callable(factory):
            return factory(attempt, self._max_attempts, intent)
        return self._trace


def _modified_file_count(plan: RefactorPlan) -> int:
    """Count distinct files touched by non-noop instructions."""
    return len(
        {item.target_file for item in plan.instructions if not isinstance(item, Noop)}
    )


def _capture_diff(repo_root: Path) -> str:
    """Read the uncommitted worktree diff without touching git state."""
    try:
        completed = subprocess.run(
            ["git", "diff", "--no-color"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_DIFF_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(git diff unavailable: {exc})"
    if completed.returncode != 0:
        return f"(git diff failed: {completed.stderr.strip()})"
    diff = completed.stdout.strip()
    if not diff:
        return "(no text diff captured)"
    if len(diff) > _MAX_DIFF_CHARS:
        return diff[:_MAX_DIFF_CHARS] + "\n... (diff truncated)"
    return diff


def _compose_context(context: str, failure: _VerificationFailed) -> str:
    """Combine caller context with the failed attempt's diff and feedback."""
    sections: list[str] = []
    if context.strip():
        sections.append(context.strip())
    sections.append(
        "Previous attempt failed verification and was rolled back.\n"
        f"Code changes from that attempt (git diff):\n{failure.diff}\n\n"
        f"Verification error:\n{failure.verify.feedback()}\n\n"
        "Generate a new RefactorPlan that fixes this failure. Do not repeat the "
        "same changes."
    )
    return "\n\n".join(sections)
