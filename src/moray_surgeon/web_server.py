"""Built-in web console: a FastAPI dashboard for refactoring from the browser.

The console adds no engine logic of its own — it wraps the existing
:class:`~moray_surgeon.orchestrator.Orchestrator` exactly like the CLI does.
One refactor runs at a time on a worker thread (``asyncio.to_thread``), its
narration (plan request -> AST apply -> verification) is forwarded to browsers
as Server-Sent Events through a :class:`EventBus`, and the final report — plan,
git diff, architecture change and verification verdict — comes back as JSON
from ``POST /api/refactor``.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
import webbrowser
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from moray_surgeon import __version__
from moray_surgeon.exceptions import MorayError
from moray_surgeon.intent import IntentParser
from moray_surgeon.orchestrator import Orchestrator, RefactorReport
from moray_surgeon.verify import Verifier

if TYPE_CHECKING:
    from moray_surgeon.ui import LiveTrace

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DEFAULT_PORT = 8080
# Heartbeat interval for the SSE stream: comments keep intermediaries from
# closing an idle connection and give the client a liveness signal.
_SSE_KEEPALIVE_SECONDS = 15.0

# Mirrors cli._SKIP_DIRS (kept local to avoid importing the CLI module);
# .moray-cache is added because it is this tool's own artifact directory.
_SKIP_DIRS = {
    ".git",
    ".venv",
    ".tox",
    "venv",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
    ".moray-cache",
}


def _sse_frame(event: str, data: str) -> str:
    """Render one Server-Sent Events frame.

    The payload is JSON-encoded so newlines (a diff is full of them) can never
    break the one-line-per-``data:`` rule of the SSE grammar.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class EventBus:
    """Thread-safe, replay-capable broadcaster feeding the SSE endpoint.

    The orchestrator narrates from a worker thread while FastAPI consumes on
    the event loop, so :meth:`publish` marshals frames with
    ``call_soon_threadsafe``. Frames are also kept in a replay buffer: a
    browser that connects mid-run receives everything published so far, which
    is what keeps the log panel correct across page refreshes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: list[str] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: list[asyncio.Queue[str]] = []

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the event loop that owns the SSE consumers."""
        self._loop = loop

    def clear(self) -> None:
        """Drop the replay buffer; called when a new run starts."""
        with self._lock:
            self._history.clear()

    def publish(self, event: str, data: str) -> None:
        """Record one event and fan it out to live subscribers.

        Safe to call from any thread. Without a bound loop (no run started in
        this process yet) the frame is still recorded, so a subscriber that
        attaches later can replay it.
        """
        frame = _sse_frame(event, data)
        with self._lock:
            self._history.append(frame)
            loop = self._loop
        if loop is None:
            return
        # A closed loop is not an error here: the replay buffer still holds the
        # frame, which is what late subscribers receive.
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(self._fanout, frame)

    def subscribe(self) -> tuple[asyncio.Queue[str], list[str]]:
        """Register a subscriber and snapshot the replay buffer atomically.

        Returning the queue and the backlog from one critical section means a
        frame is delivered exactly once at the boundary: anything already in
        the backlog is not duplicated into the queue, and anything published
        after is not lost.
        """
        queue: asyncio.Queue[str] = asyncio.Queue()
        with self._lock:
            self._queues.append(queue)
            return queue, list(self._history)

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        """Remove a subscriber that disconnected."""
        with self._lock:
            if queue in self._queues:
                self._queues.remove(queue)

    def _fanout(self, frame: str) -> None:
        for queue in list(self._queues):
            queue.put_nowait(frame)


class _WebTrace:
    """Orchestrator narration adapter: forwards every beat to the event bus.

    Mirrors the ``LiveTrace`` interface the orchestrator expects
    (``for_attempt``/``step``/``success``/``error``/``info``) so the web console
    watches exactly the pipeline the CLI prints.
    """

    def __init__(self, bus: EventBus, attempt: int, max_attempts: int, intent: str) -> None:
        self._bus = bus
        self._attempt = attempt
        self._max_attempts = max_attempts
        self._intent = intent

    def for_attempt(self, attempt: int, max_attempts: int, intent: str) -> _WebTrace:
        """Clone this trace for another attempt (orchestrator contract)."""
        return _WebTrace(self._bus, attempt, max_attempts, intent)

    @contextmanager
    def step(self, label: str) -> Iterator[_WebTrace]:
        """Announce a pipeline stage as a progress event."""
        self._bus.publish("progress", f"[{self._attempt}/{self._max_attempts}] {label}")
        yield self

    def success(self, message: str) -> None:
        """Report a green beat (plan received, gates passed, ...)."""
        self._bus.publish("success", message)

    def error(self, message: str) -> None:
        """Report a red beat (gate failed, rollback, ...)."""
        self._bus.publish("error", message)

    def info(self, message: str) -> None:
        """Report an informational beat."""
        self._bus.publish("info", message)


class RefactorRequest(BaseModel):
    """Body of ``POST /api/refactor``.

    Frozen + forbid per project rule R4: the payload is an instruction record,
    not a mutable bag, and typos in field names must fail loudly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: str = Field(min_length=1)
    repo_path: str
    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434"
    api_type: str = "ollama"
    api_key: str | None = None
    max_attempts: int = Field(default=3, ge=1, le=10)
    dry_run: bool = False


class RefactorResponse(BaseModel):
    """JSON report returned by ``POST /api/refactor``.

    ``diff`` carries the git diff of a dry run (or of the failed last attempt
    reported by the engine), ``architecture_diff`` the before/after structure
    comparison when moray-scribe is available.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    attempts: int
    log: list[str]
    diff: str
    architecture_diff: str
    plan: dict[str, Any] | None
    verify_stage: str | None
    verify_ok: bool | None
    verify_feedback: str


def _load_index_html() -> str:
    """Read the single-page console shipped next to this module."""
    path = _STATIC_DIR / "index.html"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"web console assets missing: {path} ({exc})") from exc


def _resolve_repo(repo_path: str) -> Path:
    """Validate the request's repository root (400 when unusable).

    This mirrors the CLI's git check; the engine's own path-containment gates
    keep every target file inside this root.
    """
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        raise HTTPException(status_code=400, detail=f"{repo_path} is not a directory")
    if not (repo / ".git").exists():
        raise HTTPException(status_code=400, detail=f"{repo_path} is not a git repository")
    return repo


def _scan_target_files(repo: Path) -> list[str]:
    """Inventory up to 50 Python files for the code map (the CLI default)."""
    root = repo.resolve()
    relative = [p.relative_to(root) for p in sorted(root.rglob("*.py")) if p.is_file()]
    kept = [p for p in relative if not _SKIP_DIRS & set(p.parts)]
    return [p.as_posix() for p in kept[:50]]


def build_orchestrator(payload: RefactorRequest, repo: Path, trace_bus: EventBus) -> Orchestrator:
    """Wire the engine exactly like the CLI does (public so tests can patch it).

    Falls back to ``MORAY_API_KEY`` when the request carries no key, mirroring
    the CLI credential rules: the secret lives server-side and the browser
    never needs to send it. ``mypy`` targets ``src`` when the repository has
    one, else the repository root (flat demo repositories must verify too).
    """
    api_key = payload.api_key or os.environ.get("MORAY_API_KEY") or None
    mypy_target = "src" if (repo / "src").is_dir() else "."
    return Orchestrator(
        parser=IntentParser(
            model=payload.model,
            base_url=payload.base_url,
            api_key=api_key,
            api_type=payload.api_type,
        ),
        verifier=Verifier(repo, mypy_target=mypy_target),
        max_attempts=payload.max_attempts,
        trace=cast(
            "LiveTrace",
            # The orchestrator duck-types its trace (it probes for_attempt before
            # use), so a structurally compatible adapter is safe here even though
            # the declared parameter type is LiveTrace.
            _WebTrace(
                trace_bus,
                attempt=1,
                max_attempts=payload.max_attempts,
                intent=payload.intent,
            ),
        ),
    )


def _run_refactor(payload: RefactorRequest, repo: Path, trace_bus: EventBus) -> RefactorReport:
    """Worker-thread body: build the orchestrator and run the full pipeline."""
    orchestrator = build_orchestrator(payload, repo, trace_bus)
    return orchestrator.run(
        payload.intent,
        repo,
        target_files=_scan_target_files(repo),
        dry_run=payload.dry_run,
    )


def _report_to_response(report: RefactorReport) -> RefactorResponse:
    """Flatten an orchestrator report into the API response model."""
    verify = report.last_verify
    if verify is None:
        verify_stage: str | None = None
        verify_ok: bool | None = None
        verify_feedback = ""
    elif verify.ok:
        verify_stage, verify_ok, verify_feedback = verify.stage, True, verify.stdout
    else:
        verify_stage, verify_ok, verify_feedback = verify.stage, False, verify.feedback()
    return RefactorResponse(
        success=report.success,
        attempts=report.attempts,
        log=list(report.log),
        diff=report.diff,
        architecture_diff=report.architecture_diff,
        plan=report.plan.model_dump(mode="json") if report.plan is not None else None,
        verify_stage=verify_stage,
        verify_ok=verify_ok,
        verify_feedback=verify_feedback,
    )


def create_app(bus: EventBus | None = None) -> FastAPI:
    """Assemble the FastAPI application (one refactor at a time per app)."""
    bus = bus if bus is not None else EventBus()
    index_html = _load_index_html()
    app = FastAPI(title="moray-surgeon console", version=__version__)
    app.state.bus = bus
    app.state.run_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the single-page console."""
        return HTMLResponse(index_html)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        """Liveness probe; the page also reads the version from here."""
        return {"status": "ok", "version": __version__}

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        """Stream run narration as Server-Sent Events."""
        queue, backlog = bus.subscribe()

        async def event_source() -> AsyncIterator[str]:
            try:
                for frame in backlog:
                    yield frame
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        frame = await asyncio.wait_for(
                            queue.get(), timeout=_SSE_KEEPALIVE_SECONDS
                        )
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield frame
            finally:
                bus.unsubscribe(queue)

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/api/refactor", response_model=RefactorResponse)
    async def refactor(payload: RefactorRequest) -> RefactorResponse:
        """Run one orchestrated refactor and return the full report."""
        repo = _resolve_repo(payload.repo_path)
        if not app.state.run_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409, detail="A refactor is already running; wait for it to finish."
            )
        try:
            bus.clear()
            bus.bind_loop(asyncio.get_running_loop())
            bus.publish("info", f"Run started | repo: {repo}")
            try:
                # The engine blocks for minutes (LLM + ruff/mypy/pytest); the
                # event loop must keep serving SSE while it works.
                report = await asyncio.to_thread(_run_refactor, payload, repo, bus)
            except MorayError as exc:
                bus.publish("error", f"Engine error: {exc}")
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            except OSError as exc:
                bus.publish("error", f"Filesystem error: {exc}")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            response = _report_to_response(report)
            verdict = "success" if response.success else "failed"
            bus.publish(
                "success" if response.success else "error",
                f"Run finished: {verdict} after {response.attempts} attempt(s).",
            )
            bus.publish("result", response.model_dump_json())
            return response
        finally:
            app.state.run_lock.release()

    return app


def _open_browser_when_ready(host: str, port: int, url: str, timeout_seconds: float = 30.0) -> None:
    """Poll until the server accepts TCP connections, then open the browser."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.1)


def run_server(
    host: str = "127.0.0.1", port: int = _DEFAULT_PORT, open_browser: bool = True
) -> None:
    """Start the dashboard in the foreground; Ctrl+C stops it.

    Args:
        host: Bind address (the loopback default keeps the console local).
        port: TCP port; 8080 unless overridden.
        open_browser: Launch the system browser as soon as the port is live.
    """
    # Imported lazily so library imports and the CLI's other subcommands never
    # pay for the web stack.
    import uvicorn

    console = Console()
    url = f"http://{host}:{port}/"
    console.print(f"[bold]moray-surgeon[/] web console -> [link={url}]{url}[/link]")
    console.print("[dim]Press Ctrl+C to stop.[/]")
    if open_browser:
        threading.Thread(
            target=_open_browser_when_ready, args=(host, port, url), daemon=True
        ).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="warning", access_log=False)
