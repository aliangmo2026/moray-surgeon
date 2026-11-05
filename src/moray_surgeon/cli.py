"""Typer command-line interface for moray-surgeon."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import typer

from moray_surgeon import __version__
from moray_surgeon.audit import run_audit
from moray_surgeon.engine.indexer import Indexer
from moray_surgeon.engine.indexer import clear_cache as clear_index_cache
from moray_surgeon.exceptions import CacheError, MorayError
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
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Delete the local code index before running (it is rebuilt on demand).",
    ),
    no_differential: bool = typer.Option(
        False,
        "--no-differential",
        help=(
            "Skip the runtime-equivalence gate: do not execute changed functions "
            "before and after the refactor in the differential sandbox."
        ),
    ),
) -> None:
    """Plan, apply, verify and auto-fix one refactoring intent."""
    if not (path / ".git").exists():
        typer.echo(f"error: {path} is not a git repository", err=True)
        raise typer.Exit(code=2)

    if clear_cache:
        try:
            removed = clear_index_cache(path)
        except CacheError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo("Code index cleared." if removed else "No code index to clear.")

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
            differential=not no_differential,
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


@app.command("time-travel")
def time_travel(
    intent: str = typer.Option(
        ..., "--intent", help="Refactoring request applied to the base commit."
    ),
    path: Path = typer.Option(Path("."), "--path", help="Repository root."),
    branch: str | None = typer.Option(
        None, "--branch", help="Branch whose last N commits are rewritten (default: current)."
    ),
    steps: int = typer.Option(
        1, "--steps", min=1, help="Number of most recent commits to rewrite."
    ),
    model: str = typer.Option("qwen2.5-coder:7b", "--model", help="Model name."),
    base_url: str = typer.Option(
        "http://localhost:11434",
        "--base-url",
        "--ollama",
        help="Ollama base URL or cloud API root.",
    ),
    api_type: str = typer.Option(
        "ollama", "--api-type", help="'ollama', 'openai' or 'anthropic'."
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="Model credential; prefer MORAY_API_KEY."
    ),
    max_attempts: int = typer.Option(3, "--max-attempts", help="Plan/verify rounds."),
    mypy_target: str = typer.Option(
        "src", "--mypy-target", help="Path passed to mypy inside the temporary worktree."
    ),
    pytest_args: str = typer.Option(
        "-q", "--pytest-args", help="Space-separated pytest arguments."
    ),
    no_differential: bool = typer.Option(
        False, "--no-differential", help="Skip the runtime-equivalence gate."
    ),
    skip_verify: bool = typer.Option(
        False,
        "--skip-verify",
        help=(
            "DANGEROUS: accept the refactor without ruff/mypy/pytest. The AST "
            "scope gate still runs. Prefer fixing the historical checkout."
        ),
    ),
    push: bool = typer.Option(False, "--push", help="Push the rewritten branch."),
    pr: bool = typer.Option(
        False, "--pr", help="Push and open a review pull request (needs GITHUB_TOKEN)."
    ),
    remote: str = typer.Option("origin", "--remote", help="Remote used by --push/--pr."),
    force: bool = typer.Option(
        False, "--force", help="Replace an existing moray-time-travel/<branch> branch."
    ),
) -> None:
    """Apply one refactor across the last N commits on a new branch.

    The original branch and working tree are never modified. The rewrite happens
    in a temporary worktree; a conflict aborts and removes it. A rewritten
    history cannot be merged normally, so --pr opens a review PR whose body
    carries the force-push command a maintainer runs after approval.
    """
    from moray_surgeon.time_travel import (
        SkipVerifier,
        open_time_travel_pull_request,
        push_rewritten_branch,
        run_time_travel,
    )

    resolved_api_key = api_key or os.environ.get("MORAY_API_KEY") or None
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if pr and not token:
        typer.echo("error: GITHUB_TOKEN is required for --pr", err=True)
        raise typer.Exit(code=1)
    if skip_verify:
        typer.echo(
            "warning: --skip-verify disables ruff/mypy/pytest for the rewritten "
            "history; the AST scope gate still applies.",
            err=True,
        )

    def build(worktree: Path) -> Orchestrator:
        parser = IntentParser(
            model=model,
            base_url=base_url,
            api_key=resolved_api_key,
            api_type=api_type,
        )
        verifier: Verifier = (
            SkipVerifier()
            if skip_verify
            else Verifier(
                worktree,
                mypy_target=mypy_target,
                pytest_args=shlex.split(pytest_args),
            )
        )
        return Orchestrator(
            parser,
            verifier,
            max_attempts=max_attempts,
            differential=not no_differential,
        )

    try:
        result = run_time_travel(
            path, branch, steps, intent, orchestrator_factory=build, force=force
        )
    except (MorayError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"new branch: {result.new_branch}")
    typer.echo(
        f"replayed {result.replayed_commits} of {result.requested_steps} commit(s); "
        f"{result.base_commit[:12]} -> {result.rewritten_tip[:12]}"
    )
    typer.echo(f"inspect: git log --oneline {result.branch}..{result.new_branch}")

    if push or pr:
        try:
            push_rewritten_branch(path, remote, result.new_branch)
        except (MorayError, OSError) as exc:
            typer.echo(
                f"error: {exc}\nthe rewritten branch is still local: {result.new_branch}",
                err=True,
            )
            raise typer.Exit(code=1) from exc
        typer.echo(f"pushed {result.new_branch} to {remote}")

    if pr:
        try:
            url = open_time_travel_pull_request(
                path,
                remote=remote,
                base_branch=result.branch,
                head_branch=result.new_branch,
                intent=intent,
                result=result,
                token=token,
            )
        except (MorayError, OSError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"pull request: {url}")
    elif not push:
        typer.echo(f"push with: git push {remote} {result.new_branch}")
    raise typer.Exit(code=0)


@app.command("install-hook")
def install_hook(
    path: Path = typer.Option(Path("."), "--path", help="Repository root."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace a pre-commit hook that was not installed by moray-surgeon.",
    ),
) -> None:
    """Install a pre-commit hook that blocks commits failing `moray-surgeon audit`."""
    from moray_surgeon.hooks import install_pre_commit_hook

    try:
        hook = install_pre_commit_hook(path, force=force)
    except (MorayError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"installed {hook}")
    typer.echo(
        "The hook runs `moray-surgeon audit` before every commit; "
        "bypass one commit with `git commit --no-verify`."
    )
    raise typer.Exit(code=0)


@app.command("export-sft")
def export_sft(
    path: Path = typer.Option(Path("."), "--path", help="Repository root."),
    branch: str = typer.Option("HEAD", "--branch", help="Revision to walk."),
    steps: int = typer.Option(
        50, "--steps", min=1, help="Number of most recent commits to mine."
    ),
    output: Path = typer.Option(
        Path("moray-sft.jsonl"), "--output", help="Destination JSONL file."
    ),
    fmt: str = typer.Option(
        "openai",
        "--format",
        help="'openai' (messages) or 'alpaca' (instruction/input/output).",
    ),
    no_repo_map: bool = typer.Option(
        False, "--no-repo-map", help="Omit the parent-commit code map (faster)."
    ),
) -> None:
    """Export real commit refactors as a fine-tuning JSONL dataset."""
    from moray_surgeon.data.flywheel import export_sft as run_export

    try:
        report = run_export(
            path,
            branch=branch,
            steps=steps,
            output=output,
            fmt=fmt,
            include_repo_map=not no_repo_map,
        )
    except (MorayError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"commits scanned: {report.commits_scanned}")
    typer.echo(f"files analyzed: {report.files_analyzed}")
    typer.echo(f"samples: {report.samples}")
    if report.actions:
        typer.echo(
            "actions: "
            + ", ".join(f"{name}={count}" for name, count in report.actions)
        )
    else:
        typer.echo("actions: none")
    typer.echo(f"wrote: {report.output} ({report.fmt})")
    if report.samples == 0:
        typer.echo(
            "warning: no trainable AST changes were found in the selected range",
            err=True,
        )
    raise typer.Exit(code=0)


@app.command()
def bot(
    issue: int = typer.Option(..., "--issue", help="GitHub issue number to resolve."),
    repo: str = typer.Option(..., "--repo", help="Target repository as owner/name."),
    work_dir: Path | None = typer.Option(
        None, "--work-dir", help="Keep the clone under this directory (default: temp dir)."
    ),
    close_on_failure: bool = typer.Option(
        False,
        "--close-on-failure",
        help=(
            "Also close the issue when the refactor fails "
            "(the needs-human label is always added)."
        ),
    ),
) -> None:
    """Refactor a GitHub issue and open a verified pull request.

    Credentials and endpoints come from the environment: GITHUB_TOKEN is
    required; GITHUB_API_URL/GITHUB_SERVER_URL cover GitHub Enterprise; the
    MORAY_* variables configure the model. See docs/github_action_example.yml.
    """
    # Imported lazily so the other CLI commands do not load PyGithub.
    from moray_surgeon.bot import BotConfig, run_github_bot

    try:
        config = BotConfig.from_env(close_on_failure=close_on_failure or None)
        result = run_github_bot(issue, repo, config=config, work_dir=work_dir)
    except (MorayError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if result.success:
        typer.echo(f"pull request: {result.pull_request_url}")
        raise typer.Exit(code=0)
    typer.echo(f"bot stopped: {result.message}", err=True)
    raise typer.Exit(code=1)


@app.command()
def mcp() -> None:
    """Start the moray-surgeon MCP server over stdio."""
    # Imported lazily so ordinary CLI commands do not load the MCP SDK.
    from moray_surgeon.mcp_server import main as run_mcp_server

    run_mcp_server()


@app.command()
def lsp(
    model: str = typer.Option(
        "qwen2.5-coder:7b",
        "--model",
        help="Ollama model tag, or model name for a cloud API (AI code action).",
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
            "Transport dialect for the AI code action: 'ollama' (local), "
            "'openai' (/chat/completions), or 'anthropic' (Messages API)."
        ),
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help=(
            "API key for --api-type openai or anthropic. Prefer the "
            "MORAY_API_KEY environment variable."
        ),
    ),
) -> None:
    """Start the moray-surgeon language server over stdio (editors launch this)."""
    # Imported lazily so ordinary CLI commands do not load pygls.
    from moray_surgeon.lsp_server import LspServerConfig
    from moray_surgeon.lsp_server import main as run_lsp_server

    # The environment is the preferred credential channel: a flag is visible in
    # shell history and process listings.
    resolved_api_key = api_key or os.environ.get("MORAY_API_KEY") or None
    run_lsp_server(
        LspServerConfig(
            model=model, base_url=base_url, api_type=api_type, api_key=resolved_api_key
        )
    )


@app.command()
def cache(
    path: Path = typer.Option(Path("."), "--path", help="Repository root."),
    clear: bool = typer.Option(
        False, "--clear-cache", "--clear", help="Delete the local code index."
    ),
    show_stats: bool = typer.Option(
        False, "--stats", help="Print the size and contents of the index."
    ),
) -> None:
    """Inspect or clear the local code index used by rename/move.

    The index lives in ``<repo>/.moray-cache/index.sqlite3`` and is rebuilt on
    demand, so clearing it never loses information: the next refactor recreates
    it from the sources.
    """
    try:
        if clear:
            removed = clear_index_cache(path)
            typer.echo("Code index cleared." if removed else "No code index to clear.")
            if not show_stats:
                return
        summary = Indexer(path).summary()
    except CacheError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"index: {summary.path}")
    typer.echo(f"files: {summary.files}")
    typer.echo(f"symbols: {summary.symbols}")
    typer.echo(f"size: {summary.size_bytes} bytes")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address of the dashboard."),
    port: int = typer.Option(8080, "--port", help="Port of the dashboard."),
    no_open: bool = typer.Option(
        False, "--no-open", help="Do not open the system browser automatically."
    ),
) -> None:
    """Start the web dashboard (http://127.0.0.1:8080) and open it in a browser."""
    # Imported lazily so ordinary CLI commands do not load FastAPI/uvicorn.
    from moray_surgeon.web_server import run_server

    run_server(host=host, port=port, open_browser=not no_open)


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
