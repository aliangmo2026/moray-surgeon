"""Typer command-line interface for moray-surgeon."""

from __future__ import annotations

import shlex
from pathlib import Path

import typer

from moray_surgeon import __version__
from moray_surgeon.exceptions import MorayError
from moray_surgeon.intent import IntentParser
from moray_surgeon.orchestrator import Orchestrator
from moray_surgeon.ui import LiveTrace, banner, show_diff, show_plan, show_report
from moray_surgeon.verify import Verifier

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def version() -> None:
    """Print the installed moray-surgeon version."""
    typer.echo(__version__)


@app.command()
def refactor(
    intent: str = typer.Argument(..., help="Natural-language refactoring request."),
    path: Path = typer.Option(Path("."), "--path", help="Repository root."),
    model: str = typer.Option("qwen2.5-coder:7b", "--model", help="Ollama model tag."),
    ollama: str = typer.Option("http://localhost:11434", "--ollama", help="Ollama base URL."),
    max_attempts: int = typer.Option(3, "--max-attempts", help="Plan/verify rounds."),
    context: str = typer.Option("", "--context", help="Extra context for the model."),
    mypy_target: str = typer.Option("src", "--mypy-target", help="Path passed to mypy."),
    pytest_args: str = typer.Option(
        "-q", "--pytest-args", help="Space-separated pytest arguments."
    ),
    files: str = typer.Option(
        "", "--files", help="Comma-separated files for the code map."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the diff and roll back without committing."
    ),
) -> None:
    """Plan, apply, verify and auto-fix one refactoring intent."""
    if not (path / ".git").exists():
        typer.echo(f"error: {path} is not a git repository", err=True)
        raise typer.Exit(code=2)

    banner()
    try:
        target_files = _target_files(files, path)
        orchestrator = Orchestrator(
            IntentParser(model=model, base_url=ollama),
            Verifier(
                path,
                mypy_target=mypy_target,
                pytest_args=shlex.split(pytest_args),
            ),
            max_attempts=max_attempts,
            trace=LiveTrace(attempt=1, max_attempts=max_attempts, intent=intent),
        )
        report = orchestrator.run(intent, path, context, target_files, dry_run)
    except (MorayError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if report.plan is not None:
        show_plan(report.plan)
    if dry_run:
        show_diff(report.diff)
    show_report(report)
    raise typer.Exit(code=0 if report.success else 1)


@app.command()
def mcp() -> None:
    """Start the moray-surgeon MCP server over stdio."""
    # Imported lazily so ordinary CLI commands do not load the MCP SDK.
    from moray_surgeon.mcp_server import main as run_mcp_server

    run_mcp_server()


_SKIP_DIRS = {
    ".git",
    ".venv",
    ".tox",
    "venv",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
}


def _target_files(files: str, repo: Path) -> list[str]:
    """Parse --files, or scan the repository for up to 50 Python files."""
    explicit = [item.strip() for item in files.split(",") if item.strip()]
    if explicit:
        return explicit
    root = repo.resolve()
    relative = [
        path.relative_to(root) for path in sorted(root.rglob("*.py")) if path.is_file()
    ]
    kept = [path for path in relative if not _SKIP_DIRS & set(path.parts)]
    return [path.as_posix() for path in kept[:50]]
