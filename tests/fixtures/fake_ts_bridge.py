"""Fake TypeScript bridge used to test the JSON-RPC protocol without Node.

It speaks exactly the protocol documented in
``moray_surgeon.engine.backends.typescript_backend``: one JSON-RPC 2.0 request
line on stdin, one response line on stdout. The optional first argument selects
a failure mode (``escape``, ``sleep``, ``missing-deps``); everything else behaves
like a tiny, deterministic language server.
"""

from __future__ import annotations

import json
import sys
import time

_INVALID_PARAMS = -32602
_APPLICATION_ERROR = -32000
_INTERNAL_ERROR = -32603
_METHOD_NOT_FOUND = -32601


def _respond(request_id: object, result: dict[str, object]) -> None:
    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)


def _error(request_id: object, code: int, message: str) -> None:
    payload = {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    print(json.dumps(payload), flush=True)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    raw = sys.stdin.read()
    line = next((part for part in raw.splitlines() if part.strip()), "")
    if not line:
        _error(None, _INVALID_PARAMS, "empty request")
        return 0
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        _error(None, _INVALID_PARAMS, "request is not valid JSON")
        return 0

    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "ping":
        _respond(request_id, {"pong": True, "node": "fake", "typeScript": "fake"})
    elif method == "parse":
        if mode == "parse-bad-statements":
            _respond(request_id, {"statements": "1", "diagnostics": []})
        elif mode == "parse-bad-diagnostics":
            _respond(request_id, {"statements": 1})
        elif mode == "parse-bad-diagnostic":
            _respond(request_id, {"statements": 1, "diagnostics": ["oops"]})
        elif mode == "parse-bad-message":
            diagnostic = {"code": 1, "message": 2, "line": 1, "column": 0}
            _respond(request_id, {"statements": 1, "diagnostics": [diagnostic]})
        else:
            _respond(request_id, {"statements": 1, "diagnostics": []})
    elif method == "verify":
        if mode == "verify-bad-ok":
            _respond(request_id, {"ok": "yes"})
        else:
            _respond(request_id, {"ok": params.get("source") != "BROKEN"})
    elif method in {
        "extract_function",
        "extract_variable",
        "replace_call",
        "add_import",
        "remove_import",
    }:
        _respond(
            request_id,
            {"source": f"{params.get('source', '')}/* fake */", "changes": 1},
        )
    elif method == "rename":
        if mode == "escape":
            _respond(
                request_id,
                {"changes": 1, "files": [{"path": "../evil.ts", "text": "pwned\n"}]},
            )
        elif mode == "rename-no-files":
            _respond(request_id, {"changes": 1})
        elif mode == "rename-bad-entry":
            _respond(request_id, {"changes": 1, "files": ["oops"]})
        elif mode == "rename-no-path":
            _respond(request_id, {"changes": 1, "files": [{"text": "x"}]})
        elif mode == "rename-no-text":
            _respond(request_id, {"changes": 1, "files": [{"path": "app.ts"}]})
        elif mode == "rename-empty":
            _respond(request_id, {"changes": 1, "files": []})
        elif mode == "rename-wrong-target":
            _respond(
                request_id,
                {"changes": 1, "files": [{"path": "other.ts", "text": "x"}]},
            )
        else:
            _respond(
                request_id,
                {
                    "changes": 1,
                    "files": [
                        {
                            "path": params.get("targetFile", "app.ts"),
                            "text": "// renamed\n",
                        }
                    ],
                },
            )
    elif method == "fail-application":
        _error(request_id, _APPLICATION_ERROR, "no definition of 'old' found")
    elif method == "fail-internal":
        _error(request_id, _INTERNAL_ERROR, "bridge exploded")
    elif method == "bad-json":
        print("this is not JSON", flush=True)
    elif method == "two-lines":
        print('{"jsonrpc": "2.0", "id": 1, "result": {}}', flush=True)
        print('{"jsonrpc": "2.0", "id": 1, "result": {}}', flush=True)
    elif method == "wrong-id":
        print('{"jsonrpc": "2.0", "id": 999, "result": {}}', flush=True)
    elif method == "not-object":
        print("[]", flush=True)
    elif method == "wrong-version":
        print(json.dumps({"jsonrpc": "1.0", "id": request_id, "result": {}}), flush=True)
    elif method == "error-not-object":
        print(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "error": "boom"}),
            flush=True,
        )
    elif method == "no-result":
        print(json.dumps({"jsonrpc": "2.0", "id": request_id}), flush=True)
    elif method == "sleep":
        time.sleep(10)
    elif method == "missing-deps":
        print(
            "Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'ts-morph'",
            file=sys.stderr,
            flush=True,
        )
        return 1
    elif method == "exit-error":
        print("fatal: bridge crashed", file=sys.stderr, flush=True)
        return 1
    else:
        _error(request_id, _METHOD_NOT_FOUND, f"unknown method {method!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
