from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from moray_surgeon.verify import Verifier, VerifyResult


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _green_repo(root: Path) -> Path:
    """Build a minimal repository that all three gates accept."""
    repo = root / "repo"
    _write(repo / "src" / "good.py", "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write(repo / "tests" / "test_good.py", "def test_good():\n    assert 1 + 1 == 2\n")
    return repo


def test_all_stages_pass(tmp_path):
    repo = _green_repo(tmp_path)

    result = Verifier(repo).verify()

    assert result.ok is True
    assert result.stage == "all"
    assert "All checks passed" in result.stdout
    assert "no issues found" in result.stdout
    assert "passed" in result.stdout
    assert result.stderr == ""


def test_pytest_failure_returns_pytest_stage(tmp_path):
    repo = _green_repo(tmp_path)
    # A bare "assert 1 == 2" is flagged by ruff's PLR0133, which would turn this
    # fixture into a ruff failure; raising AssertionError is ruff-clean.
    _write(repo / "tests" / "test_fail.py", "def test_fail():\n    raise AssertionError('boom')\n")

    result = Verifier(repo).verify()

    assert result.ok is False
    assert result.stage == "pytest"
    # Real pytest writes its failure report to stdout, not stderr; feedback()
    # must therefore fall back to stdout.
    assert "AssertionError" in result.stdout
    feedback = result.feedback()
    assert feedback.startswith("[pytest failed]\n")
    assert "AssertionError" in feedback


def test_ruff_failure_returns_ruff_stage(tmp_path):
    repo = _green_repo(tmp_path)
    _write(repo / "src" / "bad.py", "import os\n\n\ndef unused():\n    return 1\n")

    result = Verifier(repo).verify()

    assert result.ok is False
    assert result.stage == "ruff"
    assert "F401" in result.stdout
    assert result.feedback().startswith("[ruff failed]\n")


def test_ruff_failure_prevents_later_stages(tmp_path):
    repo = _green_repo(tmp_path)
    _write(repo / "src" / "bad.py", "import os\n")
    _write(repo / "tests" / "test_fail.py", "def test_fail():\n    raise AssertionError('boom')\n")

    result = Verifier(repo).verify()

    assert result.ok is False
    assert result.stage == "ruff"
    # pytest never started, so it never created its cache directory.
    assert not (repo / ".pytest_cache").exists()


def test_mypy_failure_returns_mypy_stage(tmp_path):
    repo = _green_repo(tmp_path)
    _write(repo / "src" / "bad.py", "def f(a: int) -> str:\n    return a\n")

    result = Verifier(repo).verify()

    assert result.ok is False
    assert result.stage == "mypy"
    assert "Incompatible return value" in result.stdout


def test_timeout_returns_failed_result_without_raising(tmp_path, monkeypatch):
    repo = _green_repo(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs: object):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = Verifier(repo).verify()

    assert result.ok is False
    assert result.stage == "ruff"
    assert result.stderr == "timed out"
    command, kwargs = calls[0]
    # The stage runs the current interpreter with "-m", never a bare "python".
    assert Path(command[0]).resolve() == Path(sys.executable).resolve()
    assert command[1:] == ["-m", "ruff", "check", "."]
    assert kwargs["timeout"] == 300
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    assert kwargs.get("shell", False) is False
    assert Path(kwargs["cwd"]).resolve() == repo.resolve()


def test_timeout_reports_the_current_stage(tmp_path, monkeypatch):
    repo = _green_repo(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs: object):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 0, stdout="ruff ok\n", stderr="")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = Verifier(repo).verify()

    assert result.ok is False
    assert result.stage == "mypy"
    assert result.stderr == "timed out"
    assert len(calls) == 2  # ruff passed, mypy timed out, pytest never started


def test_missing_executable_returns_failed_result(tmp_path):
    repo = _green_repo(tmp_path)
    missing = str(tmp_path / "missing-python.exe")

    result = Verifier(repo, python_executable=missing).verify()

    assert result.ok is False
    assert result.stage == "ruff"
    assert "command not found" in result.stderr
    assert "missing-python.exe" in result.stderr


def test_explicit_python_executable_is_honoured(tmp_path):
    repo = _green_repo(tmp_path)

    result = Verifier(repo, python_executable=sys.executable).verify()

    assert result.ok is True
    assert result.stage == "all"


def test_run_command_executes_direct_executable_path(tmp_path):
    repo = _green_repo(tmp_path)

    result = Verifier(repo)._run_command(
        [sys.executable, "-c", "print('direct-exec')"], "pytest"
    )

    assert result.ok is True
    assert result.stage == "pytest"
    assert result.stdout.strip() == "direct-exec"


def test_missing_repo_root_is_reported_not_raised(tmp_path):
    result = Verifier(tmp_path / "does-not-exist").verify()

    assert result.ok is False
    assert result.stage == "ruff"
    assert "failed to execute" in result.stderr


def test_feedback_prefers_stderr_and_truncates():
    result = VerifyResult(ok=False, stage="mypy", stdout="stdout noise", stderr="e" * 5000)

    feedback = result.feedback()

    assert feedback.startswith("[mypy failed]\n")
    assert len(feedback) == len("[mypy failed]\n") + 4000
    assert feedback.endswith("e")
    assert "stdout noise" not in feedback


def test_feedback_falls_back_to_stdout():
    result = VerifyResult(ok=False, stage="pytest", stdout="E   assert 1 == 2", stderr="")

    assert "assert 1 == 2" in result.feedback()


def test_verify_result_is_frozen():
    result = VerifyResult(ok=True, stage="all")

    with pytest.raises(FrozenInstanceError):
        result.ok = False


def test_custom_mypy_target_pytest_args_and_env(tmp_path, monkeypatch):
    repo = _green_repo(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs: object):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = Verifier(
        repo,
        mypy_target="app",
        pytest_args=["-q", "-x"],
        env={"PYTHONPATH": "app"},
    ).verify()

    assert result.ok is True
    assert result.stage == "all"
    assert len(calls) == 3
    assert calls[1][0][3:] == ["app"]
    assert calls[2][0][3:] == ["-q", "-x"]
    assert calls[0][1]["env"] == {"PYTHONPATH": "app"}
