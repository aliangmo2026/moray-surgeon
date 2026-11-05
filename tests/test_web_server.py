"""Tests for the built-in web console: routes, SSE bus, engine wiring, CLI.

Everything runs offline: the endpoint's ``build_orchestrator`` is patched with
fakes, so no model, git command or verifier subprocess is ever touched here.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from moray_surgeon import cli as cli_module
from moray_surgeon import web_server
from moray_surgeon.intent import IntentParserError
from moray_surgeon.orchestrator import Orchestrator, RefactorReport
from moray_surgeon.schema import RefactorPlan, RenameSymbol
from moray_surgeon.verify import VerifyResult
from moray_surgeon.web_server import EventBus, RefactorRequest, create_app, run_server

runner = CliRunner()


class FakeOrchestrator:
    """Stands in for the real orchestrator; records calls, replays a report."""

    def __init__(self, report: RefactorReport, error: Exception | None = None) -> None:
        self._report = report
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        intent: str,
        repo_root: Path,
        context: str = "",
        target_files: list[str] | None = None,
        dry_run: bool = False,
    ) -> RefactorReport:
        self.calls.append(
            {
                "intent": intent,
                "repo_root": repo_root,
                "target_files": target_files,
                "dry_run": dry_run,
            }
        )
        if self._error is not None:
            raise self._error
        return self._report


def _report(success: bool = True) -> RefactorReport:
    plan = RefactorPlan(
        instructions=[
            RenameSymbol(
                id="s1", target_file="app.py", old_name="old_name", new_name="new_name"
            )
        ]
    )
    return RefactorReport(
        success=success,
        attempts=1,
        plan=plan,
        log=["app.py: replaced 2 name(s)."],
        last_verify=VerifyResult(ok=True, stage="all", stdout="all green"),
        diff="--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-def old_name():\n+def new_name():",
        architecture_diff="+ module app",
    )


def _fake_repo(tmp_path: Path) -> Path:
    """A directory that only pretends to be a git repo (no git subprocess)."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "app.py").write_text("def old_name():\n    return 1\n", encoding="utf-8")
    return repo


# --------------------------------------------------------------------------- /
# index page and health
# ---------------------------------------------------------------------------


def test_index_serves_console_html():
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    for marker in ('id="repo"', 'id="intent"', 'id="execute"', "moray-surgeon"):
        assert marker in response.text


def test_create_app_fails_loudly_without_static_asset(monkeypatch, tmp_path):
    monkeypatch.setattr(web_server, "_STATIC_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="web console assets"):
        create_app()


def test_health_reports_version():
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == web_server.__version__


# --------------------------------------------------------------------------- /
# POST /api/refactor
# ---------------------------------------------------------------------------


def test_refactor_returns_full_report(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)
    fake = FakeOrchestrator(_report())
    monkeypatch.setattr(web_server, "build_orchestrator", lambda *a, **k: fake)
    client = TestClient(create_app())

    response = client.post(
        "/api/refactor",
        json={"intent": "rename old_name", "repo_path": str(repo)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["attempts"] == 1
    assert "def new_name" in body["diff"]
    assert body["architecture_diff"] == "+ module app"
    assert body["verify_stage"] == "all"
    assert body["verify_ok"] is True
    assert body["log"] == ["app.py: replaced 2 name(s)."]
    assert body["plan"]["instructions"][0]["new_name"] == "new_name"
    # The engine ran against the resolved repo with a scanned code map.
    call = fake.calls[0]
    assert call["intent"] == "rename old_name"
    assert Path(call["repo_root"]) == repo.resolve()
    assert call["target_files"] == ["app.py"]
    assert call["dry_run"] is False


def test_refactor_failed_run_is_data_not_error(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)
    failed = _report(success=False)
    fake = FakeOrchestrator(failed)
    monkeypatch.setattr(web_server, "build_orchestrator", lambda *a, **k: fake)
    client = TestClient(create_app())

    response = client.post(
        "/api/refactor", json={"intent": "impossible wish", "repo_path": str(repo)}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False


def test_refactor_maps_engine_errors_to_502(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)
    fake = FakeOrchestrator(_report(), error=IntentParserError("connection refused"))
    monkeypatch.setattr(web_server, "build_orchestrator", lambda *a, **k: fake)
    client = TestClient(create_app())

    response = client.post(
        "/api/refactor", json={"intent": "rename", "repo_path": str(repo)}
    )

    assert response.status_code == 502
    assert "connection refused" in response.json()["detail"]


def test_refactor_rejects_missing_directory(tmp_path):
    client = TestClient(create_app())

    response = client.post(
        "/api/refactor",
        json={"intent": "x", "repo_path": str(tmp_path / "nope")},
    )

    assert response.status_code == 400
    assert "not a directory" in response.json()["detail"]


def test_refactor_rejects_non_git_directory(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    client = TestClient(create_app())

    response = client.post(
        "/api/refactor", json={"intent": "x", "repo_path": str(plain)}
    )

    assert response.status_code == 400
    assert "not a git repository" in response.json()["detail"]


def test_refactor_rejects_blank_intent(tmp_path):
    client = TestClient(create_app())

    response = client.post(
        "/api/refactor",
        json={"intent": "", "repo_path": str(_fake_repo(tmp_path))},
    )

    assert response.status_code == 422


def test_refactor_rejects_unknown_fields(tmp_path):
    """R4: extra='forbid' must survive the HTTP boundary."""
    client = TestClient(create_app())

    response = client.post(
        "/api/refactor",
        json={
            "intent": "x",
            "repo_path": str(_fake_repo(tmp_path)),
            "hacker_field": True,
        },
    )

    assert response.status_code == 422


def test_refactor_rejects_out_of_range_attempts(tmp_path):
    client = TestClient(create_app())

    response = client.post(
        "/api/refactor",
        json={"intent": "x", "repo_path": str(_fake_repo(tmp_path)), "max_attempts": 0},
    )

    assert response.status_code == 422


def test_refactor_serializes_concurrent_runs(tmp_path):
    app = create_app()
    client = TestClient(app)

    with app.state.run_lock:
        response = client.post(
            "/api/refactor", json={"intent": "x", "repo_path": str(_fake_repo(tmp_path))}
        )

    assert response.status_code == 409
    assert "already running" in response.json()["detail"]


# --------------------------------------------------------------------------- /
# SSE stream and EventBus
# ---------------------------------------------------------------------------


def test_stream_replays_history_as_sse_frames():
    """The backlog reaches a browser that connects after the events fired.

    This runs a real uvicorn on an ephemeral port instead of the starlette
    TestClient: the stream is infinite by design (one connection serves every
    run), and the TestClient deadlocks closing a stream it abandons early,
    because its ASGI receive channel never delivers ``http.disconnect``. A
    real socket does, which is also the deployment shape the endpoint targets.
    """
    import httpx
    import uvicorn

    app = create_app()
    bus: EventBus = app.state.bus
    bus.publish("info", "hello")
    bus.publish("success", "done")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10.0
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started, "uvicorn did not come up within 10s"

        collected: list[str] = []
        with (
            httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0) as client,
            client.stream("GET", "/api/stream") as response,
        ):
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            for chunk in response.iter_text():
                collected.append(chunk)
                if "done" in "".join(collected):
                    break

        text = "".join(collected)
        assert "event: info" in text
        assert "event: success" in text
        # Payloads are JSON-encoded single lines even with embedded text.
        assert f"data: {json.dumps('hello')}" in text
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_publish_without_bound_loop_only_records_history():
    bus = EventBus()

    bus.publish("info", "bootstrap")

    queue, backlog = bus.subscribe()
    assert backlog == ["event: info\ndata: " + json.dumps("bootstrap") + "\n\n"]
    assert queue.empty()


def test_publish_fans_out_to_subscribers_via_bound_loop():
    bus = EventBus()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        bus.bind_loop(loop)
        received: list[str] = []

        async def consume() -> None:
            queue, _backlog = bus.subscribe()
            received.append(await asyncio.wait_for(queue.get(), timeout=2.0))

        future = asyncio.run_coroutine_threadsafe(consume(), loop)
        for _ in range(500):
            if bus._queues:
                break
            time.sleep(0.01)
        bus.publish("success", "hello")
        future.result(timeout=2.0)

        assert "event: success" in received[0]
        assert json.dumps("hello") in received[0]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


# --------------------------------------------------------------------------- /
# engine wiring and server startup
# ---------------------------------------------------------------------------


def test_build_orchestrator_builds_real_engine(tmp_path):
    repo = _fake_repo(tmp_path)
    payload = RefactorRequest(intent="rename it", repo_path=str(repo))

    orchestrator = web_server.build_orchestrator(payload, repo, EventBus())

    assert isinstance(orchestrator, Orchestrator)


def test_build_orchestrator_rejects_unknown_api_type(tmp_path):
    repo = _fake_repo(tmp_path)
    payload = RefactorRequest(intent="x", repo_path=str(repo), api_type="carrier-pigeon")

    with pytest.raises(IntentParserError):
        web_server.build_orchestrator(payload, repo, EventBus())


def test_open_browser_waits_for_listener(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(web_server.webbrowser, "open", opened.append)
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        web_server._open_browser_when_ready(
            "127.0.0.1", port, "http://ready/", timeout_seconds=2.0
        )
    finally:
        server.close()

    assert opened == ["http://ready/"]


def test_open_browser_gives_up_without_listener(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(web_server.webbrowser, "open", opened.append)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    web_server._open_browser_when_ready(
        "127.0.0.1", port, "http://never/", timeout_seconds=0.3
    )

    assert opened == []


def test_run_server_starts_uvicorn_with_defaults(monkeypatch):
    recorded: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        recorded["app"] = app
        recorded.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.setattr(web_server, "_open_browser_when_ready", lambda *a, **k: None)

    run_server(host="127.0.0.1", port=8123, open_browser=False)

    assert recorded["host"] == "127.0.0.1"
    assert recorded["port"] == 8123
    assert isinstance(recorded["app"], type(create_app()))


# --------------------------------------------------------------------------- /
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_web_command_invokes_run_server(monkeypatch):
    recorded: dict[str, Any] = {}

    def fake_run_server(host: str, port: int, open_browser: bool) -> None:
        recorded.update(host=host, port=port, open_browser=open_browser)

    monkeypatch.setattr(web_server, "run_server", fake_run_server)

    result = runner.invoke(cli_module.app, ["web", "--no-open", "--port", "9001"])

    assert result.exit_code == 0
    assert recorded == {"host": "127.0.0.1", "port": 9001, "open_browser": False}


def test_cli_web_command_defaults(monkeypatch):
    recorded: dict[str, Any] = {}

    def fake_run_server(host: str, port: int, open_browser: bool) -> None:
        recorded.update(host=host, port=port, open_browser=open_browser)

    monkeypatch.setattr(web_server, "run_server", fake_run_server)

    result = runner.invoke(cli_module.app, ["web"])

    assert result.exit_code == 0
    assert recorded == {"host": "127.0.0.1", "port": 8080, "open_browser": True}
