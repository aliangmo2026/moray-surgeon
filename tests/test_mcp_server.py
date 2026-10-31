from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from moray_surgeon import mcp_server
from moray_surgeon.intent import IntentParserError
from moray_surgeon.mcp_server import (
    execute_plan as mcp_execute_plan,
)
from moray_surgeon.mcp_server import (
    list_available_actions,
    refactor_smart,
)
from moray_surgeon.orchestrator import RefactorReport
from moray_surgeon.schema import Noop, RefactorPlan, RenameSymbol
from moray_surgeon.verify import VerifyResult

_APP_SOURCE = "def old_name() -> int:\n    return old_name()\n"
_PASSING_TEST = (
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def test_app_exists():\n"
    "    assert Path('src/app.py').is_file()\n"
)
_FAILING_TEST = (
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def test_keeps_old_name():\n"
    "    source = Path('src/app.py').read_text(encoding='utf-8')\n"
    "    assert 'old_name' in source\n"
)

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "Moray Test",
    "GIT_AUTHOR_EMAIL": "moray@example.com",
    "GIT_COMMITTER_NAME": "Moray Test",
    "GIT_COMMITTER_EMAIL": "moray@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_COUNT": "2",
    "GIT_CONFIG_KEY_0": "commit.gpgsign",
    "GIT_CONFIG_VALUE_0": "false",
    "GIT_CONFIG_KEY_1": "core.autocrlf",
    "GIT_CONFIG_VALUE_1": "false",
}


@pytest.fixture(autouse=True)
def _isolated_git_environment(monkeypatch):
    for name, value in _GIT_ENV.items():
        monkeypatch.setenv(name, value)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _init_repo(tmp_path: Path, test_source: str) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    _write(repo / "src" / "app.py", _APP_SOURCE)
    _write(repo / "tests" / "test_source.py", test_source)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _rename_plan_json() -> str:
    plan = RefactorPlan(
        instructions=[
            RenameSymbol(
                id="s1", target_file="src/app.py", old_name="old_name", new_name="new_name"
            )
        ],
        summary="rename helper",
    )
    return plan.model_dump_json()


def _commit_count(repo: Path) -> int:
    return int(_git(repo, "rev-list", "--count", "HEAD").stdout.strip())


def test_list_available_actions_documents_every_action():
    text = list_available_actions()

    for action in (
        "rename_symbol",
        "extract_function",
        "extract_variable",
        "inline_variable",
        "remove_dead_code",
        "move_symbol",
        "inline_function",
        "split_class",
        "add_type_hints",
        "replace_call",
        "noop",
    ):
        assert action in text
    assert "target_file" in text
    assert "JSON" in text


def test_fastmcp_exposes_three_tools():
    tools = asyncio.run(mcp_server.mcp.list_tools())

    assert {"list_available_actions", "execute_plan", "refactor_smart"} <= {
        tool.name for tool in tools
    }


def test_tools_never_write_to_stdout(capsys):
    list_available_actions()
    mcp_execute_plan("{not valid json", str(Path(".")))

    assert capsys.readouterr().out == ""


def test_execute_plan_rejects_invalid_json(tmp_path):
    result = mcp_execute_plan("{not valid json", str(tmp_path))

    assert "Invalid plan format" in result
    assert "list_available_actions" in result


def test_execute_plan_reports_non_git_repo(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    result = mcp_execute_plan(_rename_plan_json(), str(plain))

    assert result.startswith("Error:")
    assert "git repository" in result


def test_execute_plan_reports_noop_plan(tmp_path):
    plan = RefactorPlan(instructions=[Noop(id="s2", message="nothing to do")])

    result = mcp_execute_plan(plan.model_dump_json(), str(tmp_path))

    assert "Nothing to do" in result


def test_execute_plan_applies_and_commits(tmp_path):
    repo = _init_repo(tmp_path, _PASSING_TEST)

    result = mcp_execute_plan(_rename_plan_json(), str(repo))

    assert result.startswith("Success!")
    assert "rename_symbol" in result
    assert _read(repo / "src" / "app.py") == "def new_name() -> int:\n    return new_name()\n"
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert _commit_count(repo) == 2
    assert "moray-surgeon(mcp)" in _git(repo, "log", "-1", "--pretty=%s").stdout


def test_execute_plan_rolls_back_on_verification_failure(tmp_path):
    repo = _init_repo(tmp_path, _FAILING_TEST)

    result = mcp_execute_plan(_rename_plan_json(), str(repo))

    assert "Failed. Rolled back." in result
    assert "pytest failed" in result
    assert _read(repo / "src" / "app.py") == _APP_SOURCE
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert _commit_count(repo) == 1


class _FakeOrchestrator:
    def __init__(self, report: RefactorReport) -> None:
        self._report = report

    def run(self, intent, repo_root, context=""):
        return self._report


def _patch_orchestrator(monkeypatch, orchestrator_factory) -> None:
    monkeypatch.setattr(mcp_server, "IntentParser", lambda: object())
    monkeypatch.setattr(mcp_server, "Verifier", lambda path: object())
    monkeypatch.setattr(mcp_server, "Orchestrator", orchestrator_factory)


def test_refactor_smart_reports_success(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path, _PASSING_TEST)
    report = RefactorReport(
        success=True,
        attempts=1,
        plan=None,
        log=["[0] rename_symbol: 2 change(s) in app.py"],
        last_verify=VerifyResult(ok=True, stage="all"),
    )
    _patch_orchestrator(monkeypatch, lambda *args, **kwargs: _FakeOrchestrator(report))

    result = refactor_smart("rename old_name", str(repo))

    assert result.startswith("Success!")
    assert "Attempts: 1" in result
    assert "[0] rename_symbol" in result


def test_refactor_smart_reports_rollback(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path, _PASSING_TEST)
    report = RefactorReport(
        success=False,
        attempts=3,
        plan=None,
        log=[],
        last_verify=VerifyResult(ok=False, stage="pytest", stdout="assertion failed at line 5"),
    )
    _patch_orchestrator(monkeypatch, lambda *args, **kwargs: _FakeOrchestrator(report))

    result = refactor_smart("rename old_name", str(repo))

    assert "Failed. Rolled back." in result
    assert "assertion failed at line 5" in result
    assert "Tried 3 attempt(s)" in result


def test_refactor_smart_reports_ollama_failure(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path, _PASSING_TEST)

    class _RaisingOrchestrator:
        def run(self, intent, repo_root, context=""):
            raise IntentParserError("connection refused")

    _patch_orchestrator(monkeypatch, lambda *args, **kwargs: _RaisingOrchestrator())

    result = refactor_smart("rename old_name", str(repo))

    assert "Could not reach the Ollama model" in result
    assert "connection refused" in result
    assert "ollama pull" in result


def test_execute_plan_boundary_catches_unexpected_exception(monkeypatch):
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(mcp_server, "_execute_plan_impl", boom)

    result = mcp_execute_plan("{}", ".")

    assert "Unexpected error" in result
    assert "kaboom" in result


def test_refactor_smart_boundary_catches_unexpected_exception(monkeypatch):
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(mcp_server, "_refactor_smart_impl", boom)

    result = refactor_smart("rename", ".")

    assert "Unexpected error" in result
    assert "kaboom" in result


def test_main_runs_stdio_and_logs_to_a_file(tmp_path, monkeypatch, capsys):
    log_file = tmp_path / "mcp.log"
    monkeypatch.setenv("MORAY_MCP_LOG", str(log_file))
    transports: list[str] = []

    def fake_run(*, transport: str = "stdio") -> None:
        transports.append(transport)

    monkeypatch.setattr(mcp_server.mcp, "run", fake_run)

    try:
        mcp_server.main()
    finally:
        # Close only our handler; pytest attaches its own handlers to this logger.
        for handler in list(mcp_server.logger.handlers):
            if handler.get_name() == mcp_server._LOG_HANDLER_NAME:
                mcp_server.logger.removeHandler(handler)
                handler.close()

    assert transports == ["stdio"]
    log_text = log_file.read_text(encoding="utf-8")
    assert "starting" in log_text
    assert "stopped" in log_text
    assert capsys.readouterr().out == ""
