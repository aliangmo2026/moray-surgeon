"""Serverless HTTP entry point for cloud platforms (e.g. WorkBuddy).

The platform invokes :func:`handler` with an event dict and expects an
API-Gateway-style response (``statusCode`` / ``headers`` / JSON ``body``).
The payload is JSON with:

    {"action": "refactor" | "audit", "repo_path": "...", "intent": "...", ...}

The model endpoint is *not* configurable from the payload: it comes only from
deployment environment variables (``MORAY_API_TYPE``, ``MORAY_MODEL``,
``MORAY_BASE_URL``, ``MORAY_API_KEY``, ``MORAY_TIMEOUT``), so an untrusted
request can never point the server at an arbitrary URL.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from moray_surgeon.audit import run_audit
from moray_surgeon.exceptions import GitError, MorayError
from moray_surgeon.intent import IntentParser, IntentParserError
from moray_surgeon.orchestrator import Orchestrator, RefactorReport
from moray_surgeon.verify import Verifier, VerifyResult

_ACTIONS = ("refactor", "audit")
_REPO_HINT = (
    "Hint: this handler runs in a cloud sandbox. `repo_path` must point to a "
    "Git repository that is actually reachable from the sandbox (for example a "
    "directory the platform mounted or cloned for this invocation); a local "
    "path on the caller's machine cannot be accessed here."
)


def _response(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a payload in an API-Gateway-style response."""
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(payload, ensure_ascii=False),
    }


def _error(status: int, message: str) -> dict[str, Any]:
    """Build an error response with a machine-readable code."""
    return _response(status, {"ok": False, "error": message})


def _extract_payload(event: object) -> dict[str, Any]:
    """Pull the JSON payload out of a raw or API-Gateway-style event.

    Supports a plain dict event, a JSON-string ``body`` (API Gateway), and a
    base64-encoded body (``isBase64Encoded``).

    Raises:
        ValueError: The event is not a JSON object or its body is not JSON.
    """
    if isinstance(event, (bytes, str)):
        raw = event.decode("utf-8") if isinstance(event, bytes) else event
        return json.loads(raw)
    if not isinstance(event, dict):
        raise ValueError("event must be a JSON object")
    body = event.get("body")
    if body is None:
        return event
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    if isinstance(body, str):
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise ValueError("event body must be a JSON object")
        return parsed
    if isinstance(body, dict):
        return body
    raise ValueError("event body must be a JSON object or string")


def _resolve_repo(repo_path: str) -> Path:
    """Validate ``repo_path`` against what the sandbox can actually see.

    Raises:
        ValueError: The path is missing, not a directory, or not a Git root —
            with a cloud-specific hint the caller can act on.
    """
    path = Path(repo_path).expanduser()
    if not path.is_dir():
        raise ValueError(
            f"repo_path is not reachable or not a directory: {repo_path!r}. {_REPO_HINT}"
        )
    if not (path / ".git").exists():
        raise ValueError(f"repo_path is not a Git repository root (no .git): {path}. {_REPO_HINT}")
    return path


def _build_parser() -> IntentParser:
    """Construct the intent parser from deployment environment variables."""
    return IntentParser(
        model=os.environ.get("MORAY_MODEL", "qwen2.5-coder:7b"),
        base_url=os.environ.get("MORAY_BASE_URL", "http://localhost:11434"),
        timeout=float(os.environ.get("MORAY_TIMEOUT", "120")),
        api_key=os.environ.get("MORAY_API_KEY"),
        api_type=os.environ.get("MORAY_API_TYPE", "ollama"),
    )


def _verify_to_dict(verify: VerifyResult | None) -> dict[str, Any] | None:
    """Serialise a VerifyResult for JSON output."""
    if verify is None:
        return None
    return {
        "ok": verify.ok,
        "stage": verify.stage,
        "stdout": verify.stdout,
        "stderr": verify.stderr,
    }


def _report_to_dict(report: RefactorReport) -> dict[str, Any]:
    """Serialise a RefactorReport for JSON output."""
    return {
        "ok": report.success,
        "attempts": report.attempts,
        "log": report.log,
        "diff": report.diff,
        "verify": _verify_to_dict(report.last_verify),
    }


def _handle_refactor(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the plan/apply/verify loop for one intent."""
    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        return _error(400, "refactor requires a non-empty `intent` string.")
    root = _resolve_repo(str(payload["repo_path"]))

    dry_run = bool(payload.get("dry_run", False))
    target_files = payload.get("target_files")
    context = str(payload.get("context", ""))
    max_attempts = int(payload.get("max_attempts", 3))

    try:
        orchestrator = Orchestrator(_build_parser(), Verifier(root), max_attempts=max_attempts)
        report = orchestrator.run(
            intent,
            root,
            context=context,
            target_files=target_files if isinstance(target_files, list) else None,
            dry_run=dry_run,
        )
    except IntentParserError as exc:
        return _error(
            502, f"Model unreachable: {exc}. Configure MORAY_BASE_URL / MORAY_API_KEY and retry."
        )
    except (GitError, FileNotFoundError, OSError, MorayError) as exc:
        return _error(500, f"Refactor failed: {exc}")

    body = _report_to_dict(report)
    body["action"] = "refactor"
    return _response(200, body)


def _handle_audit(payload: dict[str, Any]) -> dict[str, Any]:
    """Audit the working tree against a base ref without editing anything."""
    root = _resolve_repo(str(payload["repo_path"]))
    base_ref = str(payload.get("base_ref", "HEAD"))

    try:
        report = run_audit(root, base_ref=base_ref)
    except (GitError, MorayError) as exc:
        return _error(500, f"Audit failed: {exc}")

    body = report.as_dict()
    body["action"] = "audit"
    return _response(200, body)


def dispatch(payload: dict[str, Any]) -> dict[str, Any]:
    """Route one parsed payload to its action handler."""
    action = payload.get("action")
    if action not in _ACTIONS:
        return _error(400, f"Unsupported action {action!r}; expected one of {list(_ACTIONS)}.")
    if not isinstance(payload.get("repo_path"), str):
        return _error(400, "`repo_path` is required and must be a string.")

    try:
        if action == "refactor":
            return _handle_refactor(payload)
        return _handle_audit(payload)
    except ValueError as exc:
        return _error(400, str(exc))


def handler(event: object, context: object = None) -> dict[str, Any]:
    """Standard Serverless HTTP handler: event dict in, API-Gateway response out.

    Args:
        event: Raw platform event. May be the payload itself or an API-Gateway
            envelope with a JSON (optionally base64) ``body``.
        context: Platform-provided invocation context; unused.

    Returns:
        ``{"statusCode": int, "headers": {...}, "body": "<json string>"}``.
        Every failure — including an unreachable ``repo_path`` — becomes an
        error response rather than an exception, so the platform never sees a
        crash.
    """
    try:
        payload = _extract_payload(event)
    except (ValueError, json.JSONDecodeError) as exc:
        return _error(400, f"Invalid event payload: {exc}")
    return dispatch(payload)
