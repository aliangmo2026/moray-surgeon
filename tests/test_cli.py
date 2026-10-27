from __future__ import annotations

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
            RenameSymbol(target_file="app.py", old_name="old", new_name="new"),
            Noop(message="nothing to do"),
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
