from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
from rich.console import Console

from moray_surgeon.intent import IntentParserError
from moray_surgeon.orchestrator import Orchestrator
from moray_surgeon.schema import Noop, RefactorPlan, RenameSymbol
from moray_surgeon.ui import LiveTrace
from moray_surgeon.verify import VerifyResult

_ORIGINAL = "def old_name():\n    return old_name()\n"

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


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
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
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo / "app.py", _ORIGINAL)
    _write(repo / "tests" / "test_ok.py", "def test_ok():\n    assert True\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _rename_plan() -> RefactorPlan:
    return RefactorPlan(
        instructions=[
            RenameSymbol(target_file="app.py", old_name="old_name", new_name="new_name")
        ]
    )


def _noop_plan() -> RefactorPlan:
    return RefactorPlan(instructions=[Noop(message="nothing to do")])


def _ok() -> VerifyResult:
    return VerifyResult(ok=True, stage="all", stdout="ok")


def _failed(feedback: str) -> VerifyResult:
    return VerifyResult(ok=False, stage="pytest", stdout=feedback)


class FakeParser:
    """Returns canned plans, recording intent, context and the file state seen."""

    def __init__(self, plans: list[RefactorPlan]) -> None:
        self._plans = list(plans)
        self.intents: list[str] = []
        self.contexts: list[str] = []
        self.snapshots: list[str] = []

    def parse(self, intent, repo_root, extra_context=""):
        self.intents.append(intent)
        self.contexts.append(extra_context)
        self.snapshots.append(_read(repo_root / "app.py"))
        index = min(len(self.intents) - 1, len(self._plans) - 1)
        return self._plans[index]


class FakeVerifier:
    """Replays results; the last one repeats once the list is exhausted."""

    def __init__(self, results: list[VerifyResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def verify(self):
        self.calls += 1
        return self._results[min(self.calls - 1, len(self._results) - 1)]


class FailingParser:
    def parse(self, intent, repo_root, extra_context=""):
        raise IntentParserError("ollama unreachable")


class FakeTrace:
    """Records every LiveTrace call so tests can assert ordering."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.attempts: list[tuple[int, int, str]] = []

    def for_attempt(self, attempt, max_attempts, intent):
        self.attempts.append((attempt, max_attempts, intent))
        return self

    @contextmanager
    def step(self, label):
        self.events.append(("step", label))
        yield self

    def success(self, message):
        self.events.append(("success", message))

    def error(self, message):
        self.events.append(("error", message))

    def info(self, message):
        self.events.append(("info", message))


def _commit_count(repo: Path) -> int:
    return int(_git(repo, "rev-list", "--count", "HEAD").stdout.strip())


def test_success_on_first_attempt(tmp_path):
    repo = _init_repo(tmp_path)
    parser = FakeParser([_rename_plan()])
    verifier = FakeVerifier([_ok()])

    report = Orchestrator(parser, verifier).run("rename old_name", repo)

    assert report.success is True
    assert report.attempts == 1
    assert report.plan is not None
    assert report.log == ["[0] rename_symbol: 2 change(s) in app.py"]
    assert report.last_verify is not None and report.last_verify.ok is True
    assert verifier.calls == 1
    assert parser.intents == ["rename old_name"]
    assert parser.contexts == [""]
    assert _read(repo / "app.py") == "def new_name():\n    return new_name()\n"
    assert _git(repo, "log", "-1", "--pretty=%s").stdout.strip() == (
        "moray-surgeon: rename old_name"
    )
    assert _git(repo, "status", "--porcelain").stdout == ""


def test_auto_fix_replans_from_clean_head_with_diff_context(tmp_path):
    repo = _init_repo(tmp_path)
    parser = FakeParser([_rename_plan()])
    verifier = FakeVerifier([_failed("assertion failed at line 5"), _ok()])

    report = Orchestrator(parser, verifier).run(
        "rename old_name", repo, context="house style"
    )

    assert report.success is True
    assert report.attempts == 2
    assert verifier.calls == 2
    assert parser.intents == ["rename old_name", "rename old_name"]
    assert parser.contexts[0] == "house style"

    repair = parser.contexts[1]
    assert "house style" in repair
    assert "Previous attempt failed verification" in repair
    assert "diff --git" in repair
    assert "-def old_name():" in repair
    assert "+def new_name():" in repair
    assert "assertion failed at line 5" in repair

    # The second plan was requested only after the first round was rolled back.
    assert parser.snapshots == [_ORIGINAL, _ORIGINAL]
    assert _read(repo / "app.py") == "def new_name():\n    return new_name()\n"
    # base + the successful round only: the failed round never committed.
    assert _commit_count(repo) == 2
    assert _git(repo, "status", "--porcelain").stdout == ""


def test_permanent_failure_exhausts_attempts_and_rolls_back(tmp_path):
    repo = _init_repo(tmp_path)
    parser = FakeParser([_rename_plan()])
    verifier = FakeVerifier([_failed("still broken")])

    report = Orchestrator(parser, verifier, max_attempts=3).run("rename old_name", repo)

    assert report.success is False
    assert report.attempts == 3
    assert report.log == []
    assert report.plan is not None
    assert report.last_verify is not None and report.last_verify.ok is False
    assert verifier.calls == 3
    assert parser.contexts[0] == ""
    assert "still broken" in parser.contexts[1]
    assert "still broken" in parser.contexts[2]
    assert "diff --git" in parser.contexts[2]
    assert _read(repo / "app.py") == _ORIGINAL
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert _commit_count(repo) == 1


def test_engine_error_propagates_without_retry(tmp_path):
    repo = _init_repo(tmp_path)
    plan = RefactorPlan(
        instructions=[
            RenameSymbol(target_file="missing.py", old_name="a", new_name="b")
        ]
    )
    parser = FakeParser([plan])
    verifier = FakeVerifier([_ok()])

    with pytest.raises(FileNotFoundError):
        Orchestrator(parser, verifier).run("rename it", repo)

    assert len(parser.intents) == 1
    assert verifier.calls == 0
    assert _read(repo / "app.py") == _ORIGINAL
    assert _commit_count(repo) == 1


def test_intent_parser_error_propagates_without_retry(tmp_path):
    repo = _init_repo(tmp_path)
    verifier = FakeVerifier([_ok()])

    with pytest.raises(IntentParserError, match="ollama unreachable"):
        Orchestrator(FailingParser(), verifier).run("rename it", repo)

    assert verifier.calls == 0
    assert _commit_count(repo) == 1


def test_trace_reports_success_sequence(tmp_path):
    repo = _init_repo(tmp_path)
    trace = FakeTrace()

    report = Orchestrator(
        FakeParser([_rename_plan()]), FakeVerifier([_ok()]), trace=trace
    ).run("rename old_name", repo)

    assert report.success is True
    assert trace.attempts == [(1, 3, "rename old_name")]
    assert trace.events == [
        ("step", "Requesting plan from LLM..."),
        ("success", "Plan received."),
        ("step", "Applying AST instructions..."),
        ("success", "Plan applied. Modified 1 file(s)."),
        ("step", "Running verification (ruff, mypy, pytest)..."),
        ("success", "Verification passed! Commit created."),
    ]


def test_trace_reports_rollback_then_retry(tmp_path):
    repo = _init_repo(tmp_path)
    trace = FakeTrace()

    report = Orchestrator(
        FakeParser([_rename_plan()]),
        FakeVerifier([_failed("boom"), _ok()]),
        trace=trace,
    ).run("rename old_name", repo)

    assert report.success is True
    assert trace.attempts == [(1, 3, "rename old_name"), (2, 3, "rename old_name")]
    assert trace.events == [
        ("step", "Requesting plan from LLM..."),
        ("success", "Plan received."),
        ("step", "Applying AST instructions..."),
        ("success", "Plan applied. Modified 1 file(s)."),
        ("step", "Running verification (ruff, mypy, pytest)..."),
        ("error", "Test failed! Rollback initiated."),
        ("info", "Retrying with error context + code diff..."),
        ("step", "Requesting plan from LLM..."),
        ("success", "Plan received."),
        ("step", "Applying AST instructions..."),
        ("success", "Plan applied. Modified 1 file(s)."),
        ("step", "Running verification (ruff, mypy, pytest)..."),
        ("success", "Verification passed! Commit created."),
    ]


def test_live_trace_renders_attempt_prefixed_lines():
    console = Console(record=True, width=120)
    trace = LiveTrace(2, 3, "rename old_name", console=console)

    trace.success("Verification passed! Commit created.")
    trace.error("Test failed! Rollback initiated.")
    trace.info("Retrying with error context + code diff...")
    with trace.step("Analyzing intent...") as step:
        assert step is trace

    text = console.export_text()
    assert text.count("[Attempt 2/3]") >= 3
    assert "Verification passed! Commit created." in text
    assert "Test failed! Rollback initiated." in text
    assert "Retrying with error context + code diff..." in text


def test_live_trace_for_attempt_switches_prefix():
    console = Console(record=True, width=120)
    template = LiveTrace(1, 3, "rename old_name", console=console)

    clone = template.for_attempt(2, 3, "rename old_name")
    clone.success("Verification passed! Commit created.")

    text = console.export_text()
    assert "[Attempt 2/3]" in text
    assert "[Attempt 1/3]" not in text


def test_repo_map_is_injected_and_survives_retry(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "models.py", "class Widget:\n    pass\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add models")
    parser = FakeParser([_rename_plan()])
    verifier = FakeVerifier([_failed("boom"), _ok()])

    report = Orchestrator(parser, verifier).run(
        "rename old_name", repo, context="house style", target_files=["models.py"]
    )

    assert report.success is True
    assert report.attempts == 2
    assert len(parser.contexts) == 2
    assert parser.contexts[0].startswith("【Project Code Map】")
    assert "class Widget:" in parser.contexts[0]
    assert "models.py" in parser.contexts[0]
    assert "house style" in parser.contexts[0]
    assert "【Project Code Map】" in parser.contexts[1]
    assert "boom" in parser.contexts[1]


def test_dry_run_rolls_back_without_verification_or_commit(tmp_path):
    repo = _init_repo(tmp_path)
    parser = FakeParser([_rename_plan()])
    verifier = FakeVerifier([_ok()])

    report = Orchestrator(parser, verifier).run("rename old_name", repo, dry_run=True)

    assert report.success is True
    assert report.attempts == 1
    assert report.log == ["Dry run complete. Rolled back."]
    assert report.last_verify is None
    assert "diff --git" in report.diff
    assert "-def old_name():" in report.diff
    assert "+def new_name():" in report.diff
    assert verifier.calls == 0
    assert _read(repo / "app.py") == _ORIGINAL
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert _commit_count(repo) == 1


def test_pure_noop_plan_is_retried_with_feedback(tmp_path):
    repo = _init_repo(tmp_path)
    parser = FakeParser([_noop_plan(), _rename_plan()])
    verifier = FakeVerifier([_ok()])

    report = Orchestrator(parser, verifier).run("rename old_name", repo)

    assert report.success is True
    assert report.attempts == 2
    assert verifier.calls == 1
    assert len(parser.contexts) == 2
    assert "你的计划为空" in parser.contexts[1]
    assert "no diff" in parser.contexts[1]
    assert _read(repo / "app.py") == "def new_name():\n    return new_name()\n"
    assert _commit_count(repo) == 2
    assert _git(repo, "status", "--porcelain").stdout == ""


def test_all_noop_plans_fail_without_touching_git(tmp_path):
    repo = _init_repo(tmp_path)
    parser = FakeParser([_noop_plan()])
    verifier = FakeVerifier([_ok()])

    report = Orchestrator(parser, verifier, max_attempts=2).run("rename old_name", repo)

    assert report.success is False
    assert report.attempts == 2
    assert verifier.calls == 0
    assert report.last_verify is not None and report.last_verify.stage == "noop"
    assert _read(repo / "app.py") == _ORIGINAL
    assert _commit_count(repo) == 1
