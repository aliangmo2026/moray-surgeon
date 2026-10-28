"""MCP server exposing moray-surgeon as three tools over stdio.

stdio transport means stdout belongs to JSON-RPC: this module must never print.
Diagnostics go through ``logging`` to a file (``MORAY_MCP_LOG``, default
``~/.moray-surgeon/mcp.log``), and every tool returns a string instead of letting
an exception escape into the protocol loop.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from moray_surgeon.engine.dispatcher import execute_plan as apply_plan
from moray_surgeon.intent import IntentParser, IntentParserError
from moray_surgeon.orchestrator import Orchestrator
from moray_surgeon.schema import Noop, RefactorPlan
from moray_surgeon.transaction import TransactionManager
from moray_surgeon.verify import Verifier, VerifyResult

# The SDK renamed its high-level server in 2.x (FastMCP -> MCPServer). Declaring
# "mcp>=1.0.0" means both ranges are installable, so import the name that exists
# while keeping one `mcp = FastMCP("moray-surgeon")` construction below.
try:  # mcp 1.x
    # Under mcp 2.x the module exists without this attribute, which is exactly
    # the case the except branch below handles; mypy only sees the 2.x install.
    from mcp.server.fastmcp import FastMCP  # type: ignore[attr-defined]
except ModuleNotFoundError as exc:  # mcp 2.x
    if exc.name != "mcp.server.fastmcp":
        raise
    from mcp.server.mcpserver import MCPServer as FastMCP

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.propagate = False

_LOG_HANDLER_NAME = "moray-surgeon-mcp-file"

mcp = FastMCP("moray-surgeon")

_ACTIONS_MARKDOWN = """# moray-surgeon actions

A plan is one JSON object:
{"instructions": [...], "summary": "short description"}

Steps execute in dependency order. A step with no `depends_on` keeps the order it
appears in; a step whose dependency has not run yet waits for it.

Common fields:
- `id`: unique step id inside the plan, e.g. "step1" (letters, digits, `._:-`).
- `depends_on`: optional list of step ids that must execute before this one.
- `action`: one of the actions below.
- `target_file`: repository-relative POSIX path ending in `.py`; no absolute
  paths and no `..`.
- `reasoning`: optional short explanation.

## rename_symbol
Rename every identifier spelled `old_name` in `target_file`.
Coarse-grained: attribute names (`obj.old_name`) and keyword argument names
(`call(old_name=1)`) are renamed too. Fields: `old_name`, `new_name`
(valid Python identifiers, different from each other).

## extract_function
Move module-level statements fully inside `[start_line, end_line]` into a new
parameterless function and insert a call where they were.
Fields: `function_name`, `start_line`, `end_line` (1-based, start <= end).
Only top-level statements can move.

## add_type_hints
Add `-> None` to functions that have no return annotation.
Fields: `function_name` (string or null; null means the whole file).

## replace_call
Replace the callee of every call whose callee text equals `old_call`.
Arguments and formatting are preserved.
Fields: `old_call`, `new_call` (a valid Python expression).

## add_import
Add one module-level import to `target_file`, keeping the import block sorted.
Fields: `module` (dotted path), `name` (string or null), `alias` (string or null).
`name` null writes `import module`; otherwise `from module import name`.
Matching imports that already exist are reported as a no-op.

## remove_import
Remove a module-level import from `target_file`.
Fields: `module` (dotted path), `name` (string or null).
`name` null removes every import of `module`; otherwise only that name is
dropped from `from module import ...` statements.

## noop
Explicitly do nothing. Use it when the intent requires no edit.
Fields: `message` (optional).

Example:
{"instructions": [{"id": "step1", "action": "add_import", "target_file": "src/app.py",
"module": "logging"}, {"id": "step2", "action": "replace_call", "target_file":
"src/app.py", "old_call": "print", "new_call": "logging.info", "depends_on":
["step1"]}], "summary": "route print through logging"}
"""

_REPO_HINT = (
    "Next step: pass the absolute path to a git repository with at least one HEAD "
    "commit, and make sure every target_file exists."
)


class _ToolError(Exception):
    """Bad tool input that should become a message for the calling model."""


class _RollbackRequiredError(Exception):
    """Verification failed: the enclosing transaction must roll back."""

    def __init__(self, verify: VerifyResult) -> None:
        super().__init__("verification failed")
        self.verify = verify


@mcp.tool()
def list_available_actions() -> str:
    """List every supported RefactorPlan action and its fields.

    Call this before building a plan so action names and required fields match
    the schema exactly.
    """
    return _ACTIONS_MARKDOWN


@mcp.tool()
def execute_plan(plan_json: str, repo_path: str) -> str:
    """Apply a RefactorPlan JSON to a git repository and commit only if verification passes.

    Args:
        plan_json: JSON text matching the RefactorPlan schema.
        repo_path: Absolute path to a git repository with at least one commit.

    Returns:
        A success message with the dispatcher log, or an actionable error string.
    """
    try:
        return _execute_plan_impl(plan_json, repo_path)
    except Exception as exc:  # the tool boundary must never crash the JSON-RPC loop
        logger.exception("execute_plan tool failed unexpectedly")
        return (
            f"Unexpected error while applying the plan: {exc}\n\n"
            f"{_REPO_HINT} Check `git status` in the repository before retrying."
        )


@mcp.tool()
def refactor_smart(intent: str, repo_path: str) -> str:
    """Turn a natural-language refactoring request into an applied, verified change.

    Uses the local Ollama model through IntentParser and the auto-fix
    Orchestrator (up to 3 rounds, each in its own transaction).

    Args:
        intent: The user's request, e.g. "rename parse_args to parse_cli_options".
        repo_path: Absolute path to a git repository with at least one commit.

    Returns:
        A success message with the attempt count and log, or an actionable error
        string describing the rollback and what to try next.
    """
    try:
        return _refactor_smart_impl(intent, repo_path)
    except Exception as exc:  # the tool boundary must never crash the JSON-RPC loop
        logger.exception("refactor_smart tool failed unexpectedly")
        return (
            f"Unexpected error while refactoring: {exc}\n\n"
            f"{_REPO_HINT} Then retry with a smaller, more specific intent."
        )


def _execute_plan_impl(plan_json: str, repo_path: str) -> str:
    """Validate, apply, verify and report one plan; all failures become strings."""
    try:
        plan = RefactorPlan.model_validate_json(plan_json)
    except ValueError as exc:
        logger.info("execute_plan rejected invalid plan JSON: %s", exc)
        return (
            "Invalid plan format: the JSON did not validate as a RefactorPlan.\n"
            f"{exc}\n\n"
            "Next step: call list_available_actions() to see the exact action names "
            "and fields, fix the JSON (raw JSON only, no markdown fences), then call "
            "execute_plan again."
        )

    if not plan.instructions or all(isinstance(item, Noop) for item in plan.instructions):
        return (
            "Nothing to do: the plan contains no edit instructions. Ask the user for a "
            "concrete refactoring, then send at least one rename_symbol, "
            "extract_function, add_type_hints or replace_call instruction."
        )

    try:
        root = _resolve_repo(repo_path)
    except _ToolError as exc:
        return f"Error: {exc}\n\n{_REPO_HINT}"
    except Exception as exc:
        logger.exception("execute_plan could not resolve repo_path")
        return f"Error: cannot use repo_path {repo_path!r}: {exc}\n\n{_REPO_HINT}"

    try:
        with TransactionManager(
            root,
            commit_message=f"moray-surgeon(mcp): {plan.summary or 'execute plan'}",
        ):
            log = apply_plan(plan, root)
            verify = Verifier(root).verify()
            if not verify.ok:
                raise _RollbackRequiredError(verify)
    except _RollbackRequiredError as exc:
        logger.info("execute_plan verification failed; transaction rolled back")
        return (
            "Failed. Rolled back. Error:\n"
            f"{exc.verify.feedback()}\n\n"
            "Next step: fix the reported problems or adjust the plan, then call "
            "execute_plan again. The repository is back at HEAD."
        )
    except Exception as exc:
        logger.exception("execute_plan tool failed")
        return (
            f"Error while applying the plan: {exc}\n\n"
            f"{_REPO_HINT} No commit was created; partial edits were rolled back."
        )

    modified = "\n".join(log) if log else "no files changed"
    return f"Success! Modified files:\n{modified}"


def _refactor_smart_impl(intent: str, repo_path: str) -> str:
    """Plan, apply, verify and auto-fix; all failures become strings."""
    try:
        root = _resolve_repo(repo_path)
    except _ToolError as exc:
        return f"Error: {exc}\n\n{_REPO_HINT}"
    except Exception as exc:
        logger.exception("refactor_smart could not resolve repo_path")
        return f"Error: cannot use repo_path {repo_path!r}: {exc}\n\n{_REPO_HINT}"

    try:
        report = Orchestrator(IntentParser(), Verifier(root)).run(intent, root)
    except IntentParserError as exc:
        logger.info("refactor_smart could not obtain a plan: %s", exc)
        return (
            f"Could not reach the Ollama model: {exc}\n\n"
            "Next step: start Ollama, run `ollama pull qwen2.5-coder:7b` if the model "
            "is missing, then call refactor_smart again."
        )
    except Exception as exc:
        logger.exception("refactor_smart tool failed")
        return (
            f"Error while refactoring: {exc}\n\n"
            f"{_REPO_HINT} Then retry with a smaller, more specific intent."
        )

    if report.success:
        lines = "\n".join(report.log) if report.log else "no files changed"
        return f"Success! Attempts: {report.attempts}. Modified files:\n{lines}"

    feedback = (
        report.last_verify.feedback()
        if report.last_verify is not None
        else "verification did not run"
    )
    return (
        "Failed. Rolled back. Error:\n"
        f"{feedback}\n\n"
        f"Tried {report.attempts} attempt(s). Next step: fix the code or make the "
        "intent more specific, then call refactor_smart again."
    )


def _resolve_repo(repo_path: str) -> Path:
    """Resolve and validate the repository root, raising a user-facing error."""
    path = Path(repo_path).expanduser().resolve()
    if not path.is_dir():
        raise _ToolError(f"repo_path is not a directory: {repo_path}")
    if not (path / ".git").exists():
        raise _ToolError(f"repo_path is not a git repository root (no .git): {path}")
    return path


def _configure_file_logging() -> None:
    """Attach one file handler; stdout/stderr stay clean for JSON-RPC.

    The handler is tagged by name because hosts (pytest, IDEs, MCP clients) may
    attach their own FileHandlers to this logger; only ours decides whether the
    file sink is already configured.
    """
    if any(handler.get_name() == _LOG_HANDLER_NAME for handler in logger.handlers):
        return
    log_path = Path(
        os.environ.get("MORAY_MCP_LOG") or Path.home() / ".moray-surgeon" / "mcp.log"
    )
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError:
        logger.warning("cannot open %s; MCP logging stays disabled", log_path)
        return
    handler.set_name(_LOG_HANDLER_NAME)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main() -> None:
    """Run the MCP server over stdio (Cursor/Claude Desktop launch this)."""
    _configure_file_logging()
    logger.info("moray-surgeon MCP server starting")
    try:
        mcp.run(transport="stdio")
    finally:
        logger.info("moray-surgeon MCP server stopped")
