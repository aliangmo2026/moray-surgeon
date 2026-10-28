"""Typer command-line interface for moray-surgeon."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import typer

from moray_surgeon import __version__
from moray_surgeon.audit import run_audit
from moray_surgeon.exceptions import MorayError
from moray_surgeon.intent import IntentParser
from moray_surgeon.orchestrator import Orchestrator
from moray_surgeon.schema import RefactorPlan
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
    model: str = typer.Option(
        "qwen2.5-coder:7b",
        "--model",
        help="Ollama model tag, or model name for a cloud API.",
    ),
    base_url: str = typer.Option(
        "http://localhost:11434",
        "--base-url",
        "--ollama",
        help="Ollama base URL, or cloud API root (OpenAI-compatible .../v1 or Anthropic Messages).",
    ),
    api_type: str = typer.Option(
        "ollama",
        "--api-type",
        help=(
            "Transport dialect: 'ollama' (local), 'openai' (/chat/completions), "
            "or 'anthropic' (Messages API)."
        ),
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help=(
            "API key for --api-type openai (Bearer) or anthropic (x-api-key). "
            "Prefer the MORAY_API_KEY environment variable: a flag is visible "
            "in shell history."
        ),
    ),
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

    # The environment is the preferred credential channel: a flag is visible in
    # shell history and process listings. An empty flag falls back to the env.
    resolved_api_key = api_key or os.environ.get("MORAY_API_KEY") or None

    banner()
    try:
        target_files = _target_files(files, path)
        orchestrator = Orchestrator(
            IntentParser(
                model=model,
                base_url=base_url,
                api_key=resolved_api_key,
                api_type=api_type,
            ),
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
def audit(
    path: Path = typer.Option(Path("."), "--path", help="Repository root."),
    base_ref: str = typer.Option(
        "HEAD", "--base-ref", help="Git ref the working tree is compared against."
    ),
    plan: Path | None = typer.Option(
        None, "--plan", help="RefactorPlan JSON authorising the diff (enables the scope gate)."
    ),
    json_output: Path | None = typer.Option(
        None, "--json-output", help="Write the JSON audit report to this file."
    ),
    mypy_target: str = typer.Option("src", "--mypy-target", help="Path passed to mypy."),
    pytest_args: str = typer.Option(
        "-q", "--pytest-args", help="Space-separated pytest arguments."
    ),
) -> None:
    """Audit the working tree against a base ref without modifying any file.

    Exits 0 with "Audit passed" when the scope gate and every verification gate
    are green, and 1 with a JSON report on stdout otherwise. Use it as a pull
    request gate with --base-ref origin/main (or the PR base SHA).
    """
    if not (path / ".git").exists():
        typer.echo(f"error: {path} is not a git repository", err=True)
        raise typer.Exit(code=2)

    try:
        report = run_audit(
            path,
            base_ref=base_ref,
            plan=_load_plan(plan),
            plan_path=str(plan) if plan is not None else None,
            verifier=Verifier(
                path, mypy_target=mypy_target, pytest_args=shlex.split(pytest_args)
            ),
        )
    except (MorayError, OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(report.to_json() + "\n", encoding="utf-8")

    if report.ok:
        typer.echo("Audit passed")
        if json_output is not None:
            typer.echo(f"report: {json_output}")
        raise typer.Exit(code=0)

    typer.echo(report.to_json())
    typer.echo(
        f"error: audit failed ({len(report.scope_violations)} scope violation(s), "
        f"verification stage: {report.verify.stage})",
        err=True,
    )
    raise typer.Exit(code=1)


@app.command()
def mcp() -> None:
    """Start the moray-surgeon MCP server over stdio."""
    # Imported lazily so ordinary CLI commands do not load the MCP SDK.
    from moray_surgeon.mcp_server import main as run_mcp_server

    run_mcp_server()


def _load_plan(plan_path: Path | None) -> RefactorPlan | None:
    """Load and schema-validate the optional authorising plan.

    ``run_audit`` validates the dependency graph as well, so a cyclic plan is a
    setup error (exit 2) rather than a confusing audit result.
    """
    if plan_path is None:
        return None
    return RefactorPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))


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
