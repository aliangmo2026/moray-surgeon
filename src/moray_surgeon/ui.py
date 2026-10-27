"""Rich-based presentation for plans, reports and the live orchestration trace.

Dynamic content (names, messages, verification feedback) is rendered with
``markup=False`` or as ``Text`` so a symbol like ``old[0]`` or a bracketed
failure header is never mistaken for Rich markup.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from moray_surgeon.schema import (
    AddTypeHints,
    ExtractFunction,
    Instruction,
    Noop,
    RefactorPlan,
    RenameSymbol,
    ReplaceCall,
)

if TYPE_CHECKING:
    from moray_surgeon.orchestrator import RefactorReport

console = Console()

_BANNER = "moray-surgeon — AST refactoring, zero hallucination"


def banner() -> None:
    """Show the product banner."""
    console.print(Panel(_BANNER, border_style="cyan"))


def show_plan(plan: RefactorPlan) -> None:
    """Render every instruction of ``plan`` as a table row."""
    table = Table(title="Refactor plan")
    table.add_column("#", justify="right", style="dim")
    table.add_column("action")
    table.add_column("target")
    table.add_column("detail")
    for index, instruction in enumerate(plan.instructions):
        table.add_row(
            str(index),
            Text(instruction.action),
            Text(_target(instruction)),
            Text(_detail(instruction)),
        )
    if plan.summary:
        table.caption = plan.summary
    console.print(table)


def show_diff(diff_text: str) -> None:
    """Render a unified diff with syntax highlighting and line numbers."""
    console.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=True))


def show_report(report: RefactorReport) -> None:
    """Render success logs or the rollback/failure feedback."""
    if report.success:
        console.print(f"[green]✔[/green] refactor succeeded in {report.attempts} attempt(s)")
        for line in report.log:
            console.print(line, markup=False)
        return

    console.print("[red]✘[/red] refactor failed; workspace rolled back")
    if report.last_verify is not None:
        console.print(report.last_verify.feedback(), markup=False)


class LiveTrace:
    """Attempt-scoped narration for the orchestrator loop.

    Pass an instance to ``Orchestrator(..., trace=...)``. The orchestrator clones
    it per attempt through :meth:`for_attempt`, so every line carries the right
    ``[Attempt n/m]`` prefix. Without an injected trace nothing is printed, which
    keeps tests and the MCP server silent.
    """

    def __init__(
        self,
        attempt: int,
        max_attempts: int,
        intent: str,
        console: Console | None = None,
    ) -> None:
        """Bind the trace to one attempt and (optionally) a console.

        Args:
            attempt: 1-based attempt number.
            max_attempts: Number of attempts the orchestrator may run.
            intent: The original request, kept for context/debugging.
            console: Console to write to; defaults to a fresh one on stdout.
        """
        self._attempt = attempt
        self._max_attempts = max_attempts
        self._intent = intent
        self._console = console if console is not None else Console()

    def for_attempt(self, attempt: int, max_attempts: int, intent: str) -> LiveTrace:
        """Clone this trace for another attempt, sharing the same console."""
        return LiveTrace(attempt, max_attempts, intent, console=self._console)

    @contextmanager
    def step(self, label: str) -> Iterator[LiveTrace]:
        """Announce a long step and keep a spinner alive while it runs."""
        line = self._prefix()
        line.append(label, style="dim")
        if not self._console.is_terminal:
            # Piped output: Rich's Live view is a no-op, so print the trace line
            # explicitly to keep the narration complete in logs and tests.
            self._console.print(line)
        with self._console.status(line, spinner="dots"):
            yield self

    def success(self, message: str) -> None:
        """Print a green success line."""
        line = self._prefix()
        line.append(f"✅ {message}", style="green")
        self._console.print(line)

    def error(self, message: str) -> None:
        """Print a red failure line."""
        line = self._prefix()
        line.append(f"❌ {message}", style="red")
        self._console.print(line)

    def info(self, message: str) -> None:
        """Print a dim informational line."""
        line = self._prefix()
        line.append(message, style="dim")
        self._console.print(line)

    def _prefix(self) -> Text:
        """Literal ``[Attempt n/m]`` prefix; Text bypasses Rich markup parsing."""
        return Text(f"[Attempt {self._attempt}/{self._max_attempts}] ", style="bold cyan")


def _target(instruction: Instruction) -> str:
    if isinstance(instruction, Noop):
        return "-"
    return instruction.target_file


def _detail(instruction: Instruction) -> str:
    if isinstance(instruction, RenameSymbol):
        return f"{instruction.old_name} -> {instruction.new_name}"
    if isinstance(instruction, ExtractFunction):
        return (
            f"{instruction.function_name} "
            f"(lines {instruction.start_line}-{instruction.end_line})"
        )
    if isinstance(instruction, AddTypeHints):
        return instruction.function_name or "whole file"
    if isinstance(instruction, ReplaceCall):
        return f"{instruction.old_call} -> {instruction.new_call}"
    if isinstance(instruction, Noop):
        return instruction.message
    return ""
