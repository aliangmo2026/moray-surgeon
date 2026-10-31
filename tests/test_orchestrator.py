from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
from rich.console import Console

from moray_surgeon.intent import IntentParserError
from moray_surgeon.orchestrator import Orchestrator, _failing_test_files
from moray_surgeon.schema import Noop, RefactorPlan, RenameSymbol, ReplaceCall, SplitClass
from moray_surgeon.ui import LiveTrace
from moray_surgeon.verify import Verifier, VerifyResult

_ORIGINAL = "def old_name():\n    return old_name()\n"
# The hint sentence the orchestrator must inject (full-width colon and all);
# built with chr() so the tests never spell the ambiguous punctuation themselves.
_TEST_REPAIR_HINT_SENTENCE = "注意" + chr(0xFF1A) + "测试代码也需要被重构"

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
            RenameSymbol(id="s1", target_file="app.py", old_name="old_name", new_name="new_name")
        ]
    )


def _noop_plan() -> RefactorPlan:
    return RefactorPlan(instructions=[Noop(id="s2", message="nothing to do")])


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
            RenameSymbol(id="s3", target_file="missing.py", old_name="a", new_name="b")
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
        ("step", "Recording AST scope fingerprints..."),
        ("step", "Recording architecture snapshot (moray-scribe)..."),
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
        ("step", "Recording AST scope fingerprints..."),
        ("step", "Recording architecture snapshot (moray-scribe)..."),
        ("step", "Applying AST instructions..."),
        ("success", "Plan applied. Modified 1 file(s)."),
        ("step", "Running verification (ruff, mypy, pytest)..."),
        ("error", "Test failed! Rollback initiated."),
        ("info", "Retrying with error context + code diff..."),
        ("step", "Requesting plan from LLM..."),
        ("success", "Plan received."),
        ("step", "Recording AST scope fingerprints..."),
        ("step", "Recording architecture snapshot (moray-scribe)..."),
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


def test_show_report_prints_architecture_change(monkeypatch):
    """The 架构变更 summary and its detail render after a successful run."""
    from moray_surgeon import ui
    from moray_surgeon.orchestrator import RefactorReport
    from moray_surgeon.schema import RefactorPlan

    # Full-width punctuation is built with chr() so the tests never spell the
    # ambiguous characters RUF001 would flag.
    summary = (
        "架构变更" + chr(0xFF1A) + "新增 1 个类" + chr(0xFF0C) + "移动了 1 个函数"
    )
    added_line = "  + 类 MathOps" + chr(0xFF08) + "app" + chr(0xFF09)
    moved_line = "  " + chr(0x2192) + " scale: app -> app.MathOps"
    console = Console(record=True, width=120)
    monkeypatch.setattr(ui, "console", console)
    report = RefactorReport(
        success=True,
        attempts=1,
        plan=RefactorPlan(),
        log=["[0] split_class: 2 change(s) in app.py"],
        last_verify=None,
        architecture_diff=(
            summary + "\n\n新增符号:\n" + added_line + "\n移动符号:\n" + moved_line + "\n"
        ),
    )

    ui.show_report(report)

    text = console.export_text()
    assert summary in text
    assert added_line in text
    assert moved_line in text


def test_show_report_silent_without_architecture_change(monkeypatch):
    """No architecture section renders when the diff is empty."""
    from moray_surgeon import ui
    from moray_surgeon.orchestrator import RefactorReport
    from moray_surgeon.schema import RefactorPlan

    console = Console(record=True, width=120)
    monkeypatch.setattr(ui, "console", console)
    report = RefactorReport(
        success=True,
        attempts=1,
        plan=RefactorPlan(),
        log=["[0] rename_symbol: 1 change(s) in app.py"],
        last_verify=None,
    )

    ui.show_report(report)

    assert "架构变更" not in console.export_text()


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


# --------------------------------------------------------------------------- #
# AST scope gate
# --------------------------------------------------------------------------- #

_APP_WITH_NEIGHBOUR = (
    "def old_name():\n"
    "    return old_name()\n"
    "\n"
    "\n"
    "def neighbour():\n"
    "    return 1\n"
)


def _repo_with_neighbour(tmp_path: Path) -> Path:
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", _APP_WITH_NEIGHBOUR)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add neighbour")
    return repo


def _collateral_edit_dispatcher(tamper_every_time: bool = False):
    """Fake dispatcher: renames the target and edits an unrelated function."""
    calls = 0

    def dispatch(plan, repo_root):
        nonlocal calls
        calls += 1
        original = _read(repo_root / "app.py")
        rewritten = original.replace("old_name", "new_name")
        if tamper_every_time or calls == 1:
            rewritten = rewritten.replace("return 1", "return 999")
        _write(repo_root / "app.py", rewritten)
        return ["[0] rename_symbol: 2 change(s) in app.py"]

    return dispatch


def test_scope_violation_rolls_back_and_retries_with_node_feedback(tmp_path, monkeypatch):
    repo = _repo_with_neighbour(tmp_path)
    parser = FakeParser([_rename_plan()])
    verifier = FakeVerifier([_ok()])
    monkeypatch.setattr(
        "moray_surgeon.orchestrator.execute_plan", _collateral_edit_dispatcher()
    )

    report = Orchestrator(parser, verifier).run("rename old_name", repo)

    assert report.success is True
    assert report.attempts == 2
    assert verifier.calls == 1
    assert "AST scope guard" in parser.contexts[1]
    assert "neighbour" in parser.contexts[1]
    assert _read(repo / "app.py") == _APP_WITH_NEIGHBOUR.replace("old_name", "new_name")
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert _commit_count(repo) == 3


def test_scope_violations_exhaust_the_attempts_and_restore_head(tmp_path, monkeypatch):
    repo = _repo_with_neighbour(tmp_path)
    parser = FakeParser([_rename_plan()])
    verifier = FakeVerifier([_ok()])
    monkeypatch.setattr(
        "moray_surgeon.orchestrator.execute_plan",
        _collateral_edit_dispatcher(tamper_every_time=True),
    )

    report = Orchestrator(parser, verifier, max_attempts=2).run("rename old_name", repo)

    assert report.success is False
    assert report.attempts == 2
    assert verifier.calls == 0
    assert report.last_verify is not None
    assert report.last_verify.stage == "scope"
    assert _read(repo / "app.py") == _APP_WITH_NEIGHBOUR
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert _commit_count(repo) == 2


def test_scope_guard_can_be_disabled_explicitly(tmp_path, monkeypatch):
    repo = _repo_with_neighbour(tmp_path)
    parser = FakeParser([_rename_plan()])
    verifier = FakeVerifier([_ok()])
    monkeypatch.setattr(
        "moray_surgeon.orchestrator.execute_plan", _collateral_edit_dispatcher()
    )

    report = Orchestrator(parser, verifier, scope_guard=False).run("rename old_name", repo)

    assert report.success is True
    assert report.attempts == 1
    assert "return 999" in _read(repo / "app.py")


# --------------------------------------------------------------------------
# Test-repair hint: a pytest failure naming files under tests/ must tell the
# model, explicitly, to send AST instructions for the test code.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        pytest.param(
            VerifyResult(
                ok=False,
                stage="pytest",
                stdout=(
                    "FAILED tests/test_app.py::test_fetch_data - NameError: name "
                    "'load_data' is not defined\n1 failed in 0.02s\n"
                ),
            ),
            ["tests/test_app.py"],
            id="summary-line",
        ),
        pytest.param(
            VerifyResult(
                ok=False, stage="pytest", stdout="tests/test_app.py:4: NameError\n1 failed"
            ),
            ["tests/test_app.py"],
            id="short-traceback",
        ),
        pytest.param(
            VerifyResult(
                ok=False,
                stage="pytest",
                stdout="ERROR collecting tests/test_app.py\nImportError: cannot import name",
                stderr="ImportError while importing test module 'tests/test_app.py'.",
            ),
            ["tests/test_app.py"],
            id="collection-error-deduplicated",
        ),
        pytest.param(
            VerifyResult(
                ok=False,
                stage="pytest",
                stdout="C:" + chr(92) + "repo" + chr(92) + "tests"
                + chr(92) + "test_old.py:5: AssertionError",
            ),
            ["tests/test_old.py"],
            id="windows-absolute-traceback",
        ),
        pytest.param(
            VerifyResult(
                ok=False,
                stage="pytest",
                stdout="FAILED tests/test_a.py::x\nFAILED tests/test_b.py::y\n",
            ),
            ["tests/test_a.py", "tests/test_b.py"],
            id="several-files-sorted",
        ),
        pytest.param(
            VerifyResult(ok=False, stage="pytest", stdout="FAILED subtests/test_x.py::x\n"),
            [],
            id="lookalike-directory-not-matched",
        ),
        pytest.param(
            VerifyResult(
                ok=False,
                stage="pytest",
                stdout="rootdir: C:" + chr(92) + "repo" + chr(92) + "tests"
                "\nconfigfile: pytest.ini\ncollected 0 items\n",
            ),
            [],
            id="rootdir-noise-not-matched",
        ),
        pytest.param(
            VerifyResult(ok=False, stage="mypy", stdout="tests/test_app.py:4: error: Name"),
            [],
            id="mypy-stage-never-hints",
        ),
        pytest.param(
            VerifyResult(ok=False, stage="scope", stdout="tests/test_app.py exploded"),
            [],
            id="scope-stage-never-hints",
        ),
        pytest.param(
            VerifyResult(ok=True, stage="all", stdout="tests/test_app.py passed"),
            [],
            id="green-run-never-hints",
        ),
    ],
)
def test_failing_test_files_detection(result, expected):
    """Only a red pytest gate that names tests/ files produces the repair list."""
    assert _failing_test_files(result) == expected


def test_pytest_failure_in_tests_dir_adds_the_repair_hint(tmp_path):
    repo = _init_repo(tmp_path)
    parser = FakeParser([_rename_plan()])
    verifier = FakeVerifier(
        [
            VerifyResult(
                ok=False,
                stage="pytest",
                stdout=(
                    "FAILED tests/test_ok.py::test_old_name - NameError: name "
                    "'new_name' is not defined\n1 failed in 0.01s\n"
                ),
            ),
            _ok(),
        ]
    )

    Orchestrator(parser, verifier).run("rename old_name", repo, context="house style")

    repair = parser.contexts[1]
    assert "house style" in repair
    assert "Previous attempt failed verification" in repair
    assert _TEST_REPAIR_HINT_SENTENCE in repair
    assert "- tests/test_ok.py" in repair
    assert "rename_symbol or replace_call" in repair
    # The hint is an addition: the plain failure feedback is still complete.
    assert "Verification error:" in repair
    assert "diff --git" in repair


def test_pytest_failure_outside_tests_gets_no_hint(tmp_path):
    repo = _init_repo(tmp_path)
    parser = FakeParser([_rename_plan()])
    verifier = FakeVerifier(
        [
            VerifyResult(
                ok=False,
                stage="pytest",
                stdout="app.py:1: error in production code\nFAILED app-only check",
            ),
            _ok(),
        ]
    )

    Orchestrator(parser, verifier).run("rename old_name", repo)

    assert "测试代码也需要被重构" not in parser.contexts[1]
    assert "Previous attempt failed verification" in parser.contexts[1]


# --------------------------------------------------------------------------
# End-to-end: source and tests are strongly bound, the source edit breaks the
# tests, and the second round repairs the test module with a real AST edit and
# a real pytest gate.
# --------------------------------------------------------------------------


_BOUND_SOURCE = "def fetch_data():\n    return 1\n"
# The star import is the classic tight coupling: pytest resolves the bare name
# at runtime, while ruff's default rule set stays silent about it — so a
# mismatch between the test and the source surfaces at the pytest gate, exactly
# where the test-repair hint lives.
_BOUND_TEST = "from app import *\n\n\ndef test_fetch_data():\n    assert fetch_data() == 1\n"


def _test_callee_plan() -> RefactorPlan:
    """A partial plan: the test's callee is renamed, the source is not."""
    return RefactorPlan(
        instructions=[
            ReplaceCall(
                id="s1",
                target_file="tests/test_app.py",
                old_call="fetch_data",
                new_call="load_data",
            )
        ]
    )


def _source_rename_plan() -> RefactorPlan:
    """Rename the source symbol; Jedi propagates it into the star-importing test."""
    return RefactorPlan(
        instructions=[
            RenameSymbol(
                id="s1", target_file="app.py", old_name="fetch_data", new_name="load_data"
            )
        ]
    )


def _bound_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo / "app.py", _BOUND_SOURCE)
    _write(repo / "conftest.py", "")
    _write(repo / "tests" / "test_app.py", _BOUND_TEST)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def test_broken_test_is_repaired_in_the_next_round(tmp_path):
    """A test-only rename breaks the suite; the hinted second round repairs it.

    Round one renames the callee inside the test module but not in the source,
    so pytest fails with a NameError inside tests/test_app.py. The repair
    feedback names the test file and demands test-code edits. The second plan —
    replanned from a clean HEAD — renames the source symbol; Jedi's project-wide
    rename propagates through the star import into the test module, and the
    suite turns green.
    """
    repo = _bound_repo(tmp_path)
    parser = FakeParser([_test_callee_plan(), _source_rename_plan()])
    verifier = Verifier(repo, mypy_target="app.py", pytest_args=["-q", "tests"])

    report = Orchestrator(parser, verifier).run(
        "rename fetch_data to load_data everywhere, including the tests", repo
    )

    assert report.success is True
    assert report.attempts == 2
    assert _read(repo / "app.py") == "def load_data():\n    return 1\n"
    assert _read(repo / "tests" / "test_app.py") == (
        "from app import *\n\n\ndef test_fetch_data():\n    assert load_data() == 1\n"
    )
    # The feedback named the broken test file and demanded test-code edits.
    assert _TEST_REPAIR_HINT_SENTENCE in parser.contexts[1]
    assert "tests/test_app.py" in parser.contexts[1]
    # The failed round rolled back; only the base and the fixed round committed.
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert _commit_count(repo) == 2


def test_unrepairable_test_failure_exhausts_attempts(tmp_path):
    repo = _bound_repo(tmp_path)
    # Every round repeats the test-only rename: the source never catches up,
    # the suite stays red, and every retry context carries the hint.
    parser = FakeParser([_test_callee_plan()])
    verifier = Verifier(repo, mypy_target="app.py", pytest_args=["-q", "tests"])

    report = Orchestrator(parser, verifier, max_attempts=2).run(
        "rename fetch_data to load_data", repo
    )

    assert report.success is False
    assert report.attempts == 2
    assert report.last_verify is not None and report.last_verify.stage == "pytest"
    assert _TEST_REPAIR_HINT_SENTENCE in parser.contexts[1]
    # Everything rolled back to the committed baseline.
    assert _read(repo / "app.py") == _BOUND_SOURCE
    assert _read(repo / "tests" / "test_app.py") == _BOUND_TEST
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert _commit_count(repo) == 1


# --------------------------------------------------------------------------
# Architecture before/after snapshots (moray-scribe).
# --------------------------------------------------------------------------


def _snapshot(nodes: list[tuple[str, str, str, str]]) -> str:
    """A canned moray-scribe snapshot in the architecture module's shape."""
    import json

    return json.dumps(
        {
            "nodes": [
                {"id": node_id, "kind": kind, "name": name, "module": module}
                for node_id, kind, name, module in nodes
            ],
            "edges": [],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def test_architecture_diff_captured_on_success(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    before = _snapshot([("app.build", "function", "build", "app")])
    after = _snapshot(
        [
            ("app.build", "function", "build", "app"),
            ("app.Widget", "class", "Widget", "app"),
        ]
    )
    captures = iter([before, after])
    monkeypatch.setattr(
        "moray_surgeon.orchestrator.capture_architecture", lambda root: next(captures)
    )
    parser = FakeParser([_rename_plan()])
    verifier = FakeVerifier([_ok()])

    report = Orchestrator(parser, verifier).run("rename old_name", repo)

    assert report.success is True
    assert "架构变更" in report.architecture_diff
    assert "Widget" in report.architecture_diff


def test_architecture_diff_empty_when_structure_unchanged(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    same = _snapshot([("app.build", "function", "build", "app")])
    monkeypatch.setattr("moray_surgeon.orchestrator.capture_architecture", lambda root: same)

    report = Orchestrator(FakeParser([_rename_plan()]), FakeVerifier([_ok()])).run(
        "rename old_name", repo
    )

    assert report.success is True
    assert report.architecture_diff == ""


def test_architecture_capture_failure_degrades_gracefully(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)

    def explode(root):
        raise RuntimeError("scribe exploded")

    monkeypatch.setattr("moray_surgeon.orchestrator.capture_architecture", explode)

    report = Orchestrator(FakeParser([_rename_plan()]), FakeVerifier([_ok()])).run(
        "rename old_name", repo
    )

    # The lens must never break the refactor.
    assert report.success is True
    assert report.architecture_diff == ""


def test_architecture_diff_empty_on_failure(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    before = _snapshot([("app.build", "function", "build", "app")])
    after = _snapshot(
        [
            ("app.build", "function", "build", "app"),
            ("app.Widget", "class", "Widget", "app"),
        ]
    )
    captures = iter([before, after])
    monkeypatch.setattr(
        "moray_surgeon.orchestrator.capture_architecture",
        lambda root: next(captures, before),
    )

    report = Orchestrator(FakeParser([_rename_plan()]), FakeVerifier([_failed("boom")])).run(
        "rename old_name", repo
    )

    # Rolled back: no architecture change is reported.
    assert report.success is False
    assert report.architecture_diff == ""


def test_architecture_diff_from_real_scribe(tmp_path):
    """With scribe installed, a class split produces a real architecture diff."""
    pytest.importorskip("moray_scribe")
    repo = _init_repo(tmp_path)
    _write(
        repo / "app.py",
        "class Math:\n"
        "    def scale(self, x):\n"
        "        return x * 2\n"
        "\n"
        "\n"
        "def helper():\n"
        "    return 1\n",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "math")
    parser = FakeParser(
        [
            RefactorPlan(
                instructions=[
                    SplitClass(
                        id="s1",
                        target_file="app.py",
                        class_name="Math",
                        methods_to_move=["scale"],
                        new_class_name="MathOps",
                    )
                ]
            )
        ]
    )

    report = Orchestrator(parser, FakeVerifier([_ok()]), scope_guard=False).run(
        "split Math", repo
    )

    assert report.success is True
    assert "架构变更" in report.architecture_diff
    assert "MathOps" in report.architecture_diff
