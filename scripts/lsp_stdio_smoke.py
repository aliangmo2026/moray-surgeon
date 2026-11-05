"""End-to-end stdio smoke test: drive `moray-surgeon lsp` with real JSON-RPC frames.

Usage: python scripts/lsp_stdio_smoke.py

Spawns ``python -m moray_surgeon lsp`` as a binary, speaks framed JSON-RPC to it
(initialize -> initialized -> didOpen -> codeAction -> codeAction/resolve ->
shutdown -> exit) and prints the responses. A fast check that server, engine and
the preview pipeline cooperate without any editor.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SAMPLE = (
    "def grade(x):\n"
    "    if x == 1:\n"
    "        label = 'one'\n"
    "    elif x == 2:\n"
    "        label = 'two'\n"
    "    else:\n"
    "        label = 'many'\n"
    "    return label\n"
)


def frame(payload: dict[str, object]) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def main() -> None:
    project = Path(tempfile.mkdtemp(prefix="moray-lsp-smoke-"))
    (project / "app.py").write_bytes(SAMPLE.encode("utf-8"))  # LF on disk, like a real repo

    proc = subprocess.Popen(
        [sys.executable, "-m", "moray_surgeon", "lsp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(project),
    )

    responses: list[dict[str, object]] = []
    buffer = bytearray()

    def reader() -> None:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                break
            buffer.extend(chunk)
            header_end = bytes(buffer).find(b"\r\n\r\n")
            if header_end == -1:
                continue
            headers = bytes(buffer[:header_end]).decode("ascii")
            length = None
            for line in headers.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    length = int(line.split(":", 1)[1].strip())
            if length is None:
                continue
            body_start = header_end + 4
            if len(buffer) < body_start + length:
                continue
            body = bytes(buffer[body_start : body_start + length])
            del buffer[: body_start + length]
            responses.append(json.loads(body.decode("utf-8")))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    uri = (project / "app.py").as_uri()
    proc.stdin.write(frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"processId": None, "rootUri": project.as_uri(), "capabilities": {}}}))
    proc.stdin.write(frame({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
    proc.stdin.write(frame({
        "jsonrpc": "2.0", "method": "textDocument/didOpen",
        "params": {"textDocument": {"uri": uri, "languageId": "python", "version": 1, "text": SAMPLE}},
    }))
    proc.stdin.write(frame({
        "jsonrpc": "2.0", "id": 2, "method": "textDocument/codeAction",
        "params": {
            "textDocument": {"uri": uri},
            "range": {"start": {"line": 1, "character": 8}, "end": {"line": 1, "character": 8}},
            "context": {"diagnostics": []},
        },
    }))
    proc.stdin.flush()

    deadline = time.time() + 30
    init = code_action = None
    while time.time() < deadline:
        for msg in list(responses):
            if msg.get("id") == 1:
                init = msg
            if msg.get("id") == 2:
                code_action = msg
        if code_action is not None:
            break
        time.sleep(0.2)

    print("== initialize result capabilities (excerpt) ==")
    caps = (init or {}).get("result", {}).get("capabilities", {})
    print("codeActionProvider:", caps.get("codeActionProvider"))

    print("== codeAction response ==")
    actions = (code_action or {}).get("result", [])
    for title in (a["title"] for a in actions):
        print(" -", title)

    convert = next(a for a in actions if "match-case" in a["title"])
    print("convert data:", json.dumps(convert["data"])[:160], "...")

    proc.stdin.write(frame({
        "jsonrpc": "2.0", "id": 3, "method": "codeAction/resolve", "params": convert,
    }))
    proc.stdin.flush()

    resolved = None
    deadline = time.time() + 60
    while time.time() < deadline:
        for msg in list(responses):
            if msg.get("id") == 3:
                resolved = msg
        if resolved is not None:
            break
        time.sleep(0.2)

    result = (resolved or {}).get("result", {})
    edit = result.get("edit") or {}
    changes = edit.get("changes") or {}
    print("== resolve result ==")
    print("title:", result.get("title"))
    for edit_uri, edits in changes.items():
        print("uri:", edit_uri)
        print("edits:", json.dumps(edits, ensure_ascii=False))
    disk_ok = (project / "app.py").read_bytes() == SAMPLE.encode("utf-8")
    print("disk restored byte-for-byte:", disk_ok)

    proc.stdin.write(frame({"jsonrpc": "2.0", "id": 4, "method": "shutdown", "params": None}))
    proc.stdin.write(frame({"jsonrpc": "2.0", "method": "exit"}))
    proc.stdin.flush()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("exit code:", proc.returncode)
    err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    print("stderr (should be quiet):", repr(err[:200]))


if __name__ == "__main__":
    main()
