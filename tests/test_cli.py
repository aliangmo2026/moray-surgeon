from __future__ import annotations

import json
import os
import secrets
import subprocess
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from moray_surgeon import __version__, ui
from moray_surgeon import cli as cli_module
from moray_surgeon.cli import _target_files, app
from moray_surgeon.orchestrator import RefactorReport
from moray_surgeon.schema import Noop, RefactorPlan, RenameSymbol
from moray_surgeon.verify import VerifyResult

runner = CliRunner()


def _output(result) -> str:
    return result.output + (result.stderr or "")


def _recording_console(monkeypatch) -> Console:
    console = Console(record=True, width=120)
    monkeypatch.setattr(ui, "console", console)
    return console


def test_version_command():
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_refactor_rejects_non_git_directory(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    result = runner.invoke(app, ["refactor", "rename x", "--path", str(plain)])

    assert result.exit_code == 2
    assert "not a git repository" in _output(result)


def test_refactor_help_lists_all_options():
    result = runner.invoke(app, ["refactor", "--help"])

    assert result.exit_code == 0
    output = _output(result)
    for option in (
        "--path",
        "--model",
        "--ollama",
        "--max-attempts",
        "--context",
        "--mypy-target",
        "--pytest-args",
        "--files",
        "--dry-run",
    ):
        assert option in output


def test_banner_mentions_zero_hallucination(monkeypatch):
    console = _recording_console(monkeypatch)

    ui.banner()

    assert "zero hallucination" in console.export_text()


def test_show_plan_renders_every_instruction(monkeypatch):
    console = _recording_console(monkeypatch)
    plan = RefactorPlan(
        instructions=[
            RenameSymbol(id="s1", target_file="app.py", old_name="old", new_name="new"),
            Noop(id="s2", message="nothing to do"),
        ]
    )

    ui.show_plan(plan)

    text = console.export_text()
    assert "Refactor plan" in text
    assert "rename_symbol" in text
    assert "app.py" in text
    assert "old -> new" in text
    assert "noop" in text
    assert "nothing to do" in text


def test_show_report_success_prints_logs_literally(monkeypatch):
    console = _recording_console(monkeypatch)
    report = RefactorReport(
        success=True,
        attempts=2,
        plan=None,
        log=["[0] rename_symbol: 1 change(s) in app.py"],
        last_verify=VerifyResult(ok=True, stage="all"),
    )

    ui.show_report(report)

    text = console.export_text()
    assert "succeeded" in text
    assert "2 attempt(s)" in text
    assert "[0] rename_symbol: 1 change(s) in app.py" in text


def test_show_report_failure_prints_feedback_literally(monkeypatch):
    console = _recording_console(monkeypatch)
    report = RefactorReport(
        success=False,
        attempts=3,
        plan=None,
        log=[],
        last_verify=VerifyResult(ok=False, stage="pytest", stdout="assertion failed at line 5"),
    )

    ui.show_report(report)

    text = console.export_text()
    assert "failed" in text
    assert "workspace rolled back" in text
    # The bracketed header must survive Rich markup parsing untouched.
    assert "[pytest failed]" in text
    assert "assertion failed at line 5" in text


def test_refactor_without_git_exits_before_any_output(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    result = runner.invoke(app, ["refactor", "rename x", "--path", str(plain)])

    assert result.exit_code == 2
    assert "zero hallucination" not in _output(result)


def _install_fake_orchestrator(monkeypatch, captured, report):
    class FakeOrchestrator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["trace"] = kwargs.get("trace")

        def run(self, intent, repo_root, context="", target_files=None, dry_run=False):
            captured.update(
                intent=intent,
                context=context,
                target_files=target_files,
                dry_run=dry_run,
            )
            return report

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrchestrator)


def test_refactor_dry_run_passes_through_and_renders_diff(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    diff = (
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-def old_name():\n"
        "+def new_name():\n"
    )
    report = RefactorReport(
        success=True,
        attempts=1,
        plan=None,
        log=["Dry run complete. Rolled back."],
        last_verify=None,
        diff=diff,
    )
    captured: dict[str, object] = {}
    _install_fake_orchestrator(monkeypatch, captured, report)

    result = runner.invoke(
        app,
        [
            "refactor",
            "rename x",
            "--path",
            str(repo),
            "--dry-run",
            "--files",
            "app.py, other.py",
        ],
    )

    assert result.exit_code == 0
    assert captured["dry_run"] is True
    assert captured["target_files"] == ["app.py", "other.py"]
    assert "Dry run complete. Rolled back." in result.output
    assert "+def new_name():" in result.output


def test_refactor_without_files_scans_python_sources(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "models.py").write_text("class M:\n    pass\n", encoding="utf-8")
    report = RefactorReport(
        success=True, attempts=1, plan=None, log=[], last_verify=None
    )
    captured: dict[str, object] = {}
    _install_fake_orchestrator(monkeypatch, captured, report)

    result = runner.invoke(app, ["refactor", "rename x", "--path", str(repo)])

    assert result.exit_code == 0
    assert captured["target_files"] == ["app.py", "pkg/models.py"]
    assert captured["dry_run"] is False


def test_auto_scan_caps_at_fifty_files(tmp_path):
    (tmp_path / ".git").mkdir()
    for index in range(55):
        (tmp_path / f"m{index:02d}.py").write_text("x = 1\n", encoding="utf-8")

    files = _target_files("", tmp_path)

    assert len(files) == 50
    assert files[0] == "m00.py"
    assert files[-1] == "m49.py"


def test_show_diff_renders_added_and_removed_lines(monkeypatch):
    console = _recording_console(monkeypatch)
    diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-def old_name():\n+def new_name():\n"

    ui.show_diff(diff)

    text = console.export_text()
    assert "-def old_name():" in text
    assert "+def new_name():" in text


def test_refactor_passes_verifier_options(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    captured: dict[str, object] = {}

    class FakeVerifier:
        def __init__(self, path, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(cli_module, "Verifier", FakeVerifier)
    report = RefactorReport(
        success=True, attempts=1, plan=None, log=[], last_verify=None
    )
    _install_fake_orchestrator(monkeypatch, {}, report)

    result = runner.invoke(
        app,
        [
            "refactor",
            "rename x",
            "--path",
            str(repo),
            "--mypy-target",
            "app",
            "--pytest-args",
            "-q -x",
        ],
    )

    assert result.exit_code == 0
    assert captured["mypy_target"] == "app"
    assert captured["pytest_args"] == ["-q", "-x"]


# --------------------------------------------------------------------------- #
# Model provider selection
# --------------------------------------------------------------------------- #


def _install_fake_parser(monkeypatch, captured):
    class FakeParser:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(cli_module, "IntentParser", FakeParser)


def _success_report() -> RefactorReport:
    return RefactorReport(success=True, attempts=1, plan=None, log=[], last_verify=None)


def test_refactor_help_lists_provider_options():
    result = runner.invoke(app, ["refactor", "--help"])

    assert result.exit_code == 0
    output = _output(result)
    for option in ("--base-url", "--ollama", "--api-type", "--api-key"):
        assert option in output
    assert "anthropic" in output


def test_refactor_passes_provider_options_to_the_parser(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    token = secrets.token_urlsafe(24)
    captured: dict[str, object] = {}
    _install_fake_parser(monkeypatch, captured)
    _install_fake_orchestrator(monkeypatch, {}, _success_report())
    monkeypatch.delenv("MORAY_API_KEY", raising=False)

    result = runner.invoke(
        app,
        [
            "refactor",
            "rename x",
            "--path",
            str(repo),
            "--api-type",
            "openai",
            "--base-url",
            "https://api.example.test/v1",
            "--model",
            "cloud-model",
            "--api-key",
            token,
        ],
    )

    assert result.exit_code == 0
    assert captured["api_type"] == "openai"
    assert captured["base_url"] == "https://api.example.test/v1"
    assert captured["model"] == "cloud-model"
    assert captured["api_key"] == token


def test_refactor_accepts_the_anthropic_api_type(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    captured: dict[str, object] = {}
    _install_fake_parser(monkeypatch, captured)
    _install_fake_orchestrator(monkeypatch, {}, _success_report())

    result = runner.invoke(
        app,
        [
            "refactor",
            "rename x",
            "--path",
            str(repo),
            "--api-type",
            "anthropic",
            "--base-url",
            "https://api.example.test/v1",
            "--model",
            "token-plan-model",
        ],
    )

    assert result.exit_code == 0
    assert captured["api_type"] == "anthropic"
    assert captured["base_url"] == "https://api.example.test/v1"
    assert captured["model"] == "token-plan-model"


def test_refactor_reads_the_api_key_from_the_environment(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    token = secrets.token_urlsafe(24)
    captured: dict[str, object] = {}
    _install_fake_parser(monkeypatch, captured)
    _install_fake_orchestrator(monkeypatch, {}, _success_report())
    monkeypatch.setenv("MORAY_API_KEY", token)

    result = runner.invoke(app, ["refactor", "rename x", "--path", str(repo)])

    assert result.exit_code == 0
    assert captured["api_key"] == token
    # The credential is consumed, never echoed into CLI output.
    assert token not in _output(result)


def test_refactor_flag_key_overrides_the_environment(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    flag_token = secrets.token_urlsafe(24)
    captured: dict[str, object] = {}
    _install_fake_parser(monkeypatch, captured)
    _install_fake_orchestrator(monkeypatch, {}, _success_report())
    monkeypatch.setenv("MORAY_API_KEY", secrets.token_urlsafe(24))

    result = runner.invoke(
        app, ["refactor", "rename x", "--path", str(repo), "--api-key", flag_token]
    )

    assert result.exit_code == 0
    assert captured["api_key"] == flag_token


def test_refactor_without_any_api_key_passes_none(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    captured: dict[str, object] = {}
    _install_fake_parser(monkeypatch, captured)
    _install_fake_orchestrator(monkeypatch, {}, _success_report())
    monkeypatch.delenv("MORAY_API_KEY", raising=False)

    result = runner.invoke(app, ["refactor", "rename x", "--path", str(repo)])

    assert result.exit_code == 0
    assert captured["api_key"] is None


def test_refactor_still_accepts_the_ollama_flag_alias(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    captured: dict[str, object] = {}
    _install_fake_parser(monkeypatch, captured)
    _install_fake_orchestrator(monkeypatch, {}, _success_report())

    result = runner.invoke(
        app, ["refactor", "rename x", "--path", str(repo), "--ollama", "http://localhost:9999"]
    )

    assert result.exit_code == 0
    assert captured["base_url"] == "http://localhost:9999"
    assert captured["api_type"] == "ollama"


# --------------------------------------------------------------------------- #
# audit command
# --------------------------------------------------------------------------- #

_APP_SOURCE = (
    "def old_name():\n"
    "    return old_name()\n"
    "\n"
    "\n"
    "def neighbour():\n"
    "    return 1\n"
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


def _init_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "audit-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "app.py").write_text(_APP_SOURCE, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _install_verifier(monkeypatch, result, captured=None):
    class FakeVerifier:
        def __init__(self, path, **kwargs: object) -> None:
            if captured is not None:
                captured["path"] = path
                captured.update(kwargs)

        def verify(self):
            return result

    monkeypatch.setattr(cli_module, "Verifier", FakeVerifier)


def _json_payload(text: str) -> dict:
    start = text.index("{")
    end = text.rindex("}")
    return json.loads(text[start : end + 1])


def _plan_file(path: Path) -> Path:
    plan = RefactorPlan(
        instructions=[
            RenameSymbol(
                id="step1",
                target_file="app.py",
                old_name="old_name",
                new_name="new_name",
            )
        ]
    )
    path.write_text(plan.model_dump_json(), encoding="utf-8")
    return path


def test_audit_help_lists_all_options():
    result = runner.invoke(app, ["audit", "--help"])

    assert result.exit_code == 0
    output = _output(result)
    for option in (
        "--path",
        "--base-ref",
        "--plan",
        "--json-output",
        "--mypy-target",
        "--pytest-args",
    ):
        assert option in output


def test_audit_rejects_non_git_directory(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    result = runner.invoke(app, ["audit", "--path", str(plain)])

    assert result.exit_code == 2
    assert "not a git repository" in _output(result)


def test_audit_passes_on_a_clean_tree(tmp_path, monkeypatch):
    repo = _init_git_repo(tmp_path)
    _install_verifier(monkeypatch, VerifyResult(ok=True, stage="all"))

    result = runner.invoke(app, ["audit", "--path", str(repo)])

    assert result.exit_code == 0
    assert "Audit passed" in result.output


def test_audit_reports_scope_violations_as_json_and_exits_one(tmp_path, monkeypatch):
    repo = _init_git_repo(tmp_path)
    tampered = _APP_SOURCE.replace("return 1", "return 999")
    (repo / "app.py").write_text(tampered, encoding="utf-8")
    plan = _plan_file(tmp_path / "plan.json")
    _install_verifier(monkeypatch, VerifyResult(ok=True, stage="all"))

    result = runner.invoke(
        app, ["audit", "--path", str(repo), "--plan", str(plan)]
    )

    assert result.exit_code == 1
    payload = _json_payload(_output(result))
    assert payload["ok"] is False
    assert payload["scope_violations"] == ["app.py: neighbour"]
    assert "audit failed" in _output(result)
    # The audit is read-only: the offending edit is still exactly as it was.
    assert (repo / "app.py").read_text(encoding="utf-8") == tampered


def test_audit_writes_the_report_file_on_success(tmp_path, monkeypatch):
    repo = _init_git_repo(tmp_path)
    report_path = tmp_path / "out" / "audit.json"
    _install_verifier(monkeypatch, VerifyResult(ok=True, stage="all"))

    result = runner.invoke(
        app, ["audit", "--path", str(repo), "--json-output", str(report_path)]
    )

    assert result.exit_code == 0
    assert "Audit passed" in result.output
    assert "report:" in result.output
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["files"] == []


def test_audit_fails_the_run_when_verification_is_red(tmp_path, monkeypatch):
    repo = _init_git_repo(tmp_path)
    _install_verifier(
        monkeypatch, VerifyResult(ok=False, stage="pytest", stdout="assertion failed")
    )

    result = runner.invoke(app, ["audit", "--path", str(repo)])

    assert result.exit_code == 1
    payload = _json_payload(_output(result))
    assert payload["verification"]["stage"] == "pytest"
    assert "assertion failed" in payload["verification"]["stdout"]


def test_audit_rejects_invalid_plan_json(tmp_path, monkeypatch):
    repo = _init_git_repo(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text("{not valid json", encoding="utf-8")
    _install_verifier(monkeypatch, VerifyResult(ok=True, stage="all"))

    result = runner.invoke(app, ["audit", "--path", str(repo), "--plan", str(plan)])

    assert result.exit_code == 2
    assert "error:" in _output(result)


def test_audit_rejects_a_cyclic_plan(tmp_path, monkeypatch):
    repo = _init_git_repo(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "instructions": [
                    {"id": "a", "action": "noop", "depends_on": ["b"]},
                    {"id": "b", "action": "noop", "depends_on": ["a"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    _install_verifier(monkeypatch, VerifyResult(ok=True, stage="all"))

    result = runner.invoke(app, ["audit", "--path", str(repo), "--plan", str(plan)])

    assert result.exit_code == 2
    assert "circular dependency" in _output(result)


def test_audit_passes_verifier_options(tmp_path, monkeypatch):
    repo = _init_git_repo(tmp_path)
    captured: dict[str, object] = {}
    _install_verifier(monkeypatch, VerifyResult(ok=True, stage="all"), captured)

    result = runner.invoke(
        app,
        [
            "audit",
            "--path",
            str(repo),
            "--mypy-target",
            "app",
            "--pytest-args",
            "-q -x",
        ],
    )

    assert result.exit_code == 0
    assert captured["mypy_target"] == "app"
    assert captured["pytest_args"] == ["-q", "-x"]
