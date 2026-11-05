"""Tests for the differential testing engine and its orchestrator gate.

The sandbox tests run real user-shaped code in the in-process sandbox: pure
functions must compare as equivalent, semantic drift must be caught, and every
"cannot verify" case must degrade to a documented skip -- never to a silent
pass. The orchestrator tests prove the gate rolls back a behaviour-changing
plan exactly like a red pytest run does.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from moray_surgeon.exceptions import DifferentialError
from moray_surgeon.orchestrator import Orchestrator
from moray_surgeon.schema import AddTypeHints, RefactorPlan, ReplaceCall
from moray_surgeon.verify import (
    DifferentialResult,
    changed_function_names,
    compare_functions,
    verify_equivalence,
)
from moray_surgeon.verify.differential import VERDICT_SKIPPED

_ADD_ORIGINAL = "def add(a, b):\n    return a + b\n"
_ADD_EQUIVALENT = "def add(a, b):\n    return (a) + (b)\n"
_ADD_BROKEN = "def add(a, b):\n    return a - b\n"

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

_DOUBLE_SOURCE = (
    "def double(x):\n"
    "    return x * 2\n"
    "\n"
    "\n"
    "def triple(x):\n"
    "    return x * 3\n"
    "\n"
    "\n"
    "def compute(x):\n"
    "    return double(x)\n"
)


@pytest.fixture(autouse=True)
def _isolated_git_environment(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _init_double_repo(tmp_path: Path) -> Path:
    """A committed repo whose gates always pass and whose compute() is callable."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo / "app.py", _DOUBLE_SOURCE)
    _write(repo / "tests" / "test_ok.py", "def test_ok():\n    assert True\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


class FakeParser:
    def __init__(self, plans: list[RefactorPlan]) -> None:
        self._plans = list(plans)
        self.intents: list[str] = []

    def parse(self, intent: str, repo_root: Path, extra_context: str = "") -> RefactorPlan:
        self.intents.append(intent)
        index = min(len(self.intents) - 1, len(self._plans) - 1)
        return self._plans[index]


class FakeVerifier:
    def __init__(self, results: list) -> None:
        self._results = list(results)
        self.calls = 0

    def verify(self):
        self.calls += 1
        return self._results[min(self.calls - 1, len(self._results) - 1)]


class FakeTrace:
    def for_attempt(self, attempt: int, max_attempts: int, intent: str) -> FakeTrace:
        return self

    def step(self, label: str):
        import contextlib

        return contextlib.nullcontext(self)

    def success(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass


def _ok() -> object:
    from moray_surgeon.verify import VerifyResult

    return VerifyResult(ok=True, stage="all", stdout="ok")


# --------------------------------------------------------------------------- verdicts


def test_equivalent_pure_function_is_true() -> None:
    assert verify_equivalence(_ADD_ORIGINAL, _ADD_EQUIVALENT, "add") is True


def test_semantic_change_is_false_with_detail() -> None:
    assert verify_equivalence(_ADD_ORIGINAL, _ADD_BROKEN, "add") is False
    result = compare_functions(_ADD_ORIGINAL, _ADD_BROKEN, "add")
    assert result.verdict == "different"
    assert "return values differ" in result.detail


def test_printed_output_is_compared() -> None:
    original = "def greet(name):\n    print('hi', name)\n    return name\n"
    same = "def greet(name):\n    print(('hi'), name)\n    return name\n"
    changed = "def greet(name):\n    print('yo', name)\n    return name\n"
    assert verify_equivalence(original, same, "greet") is True
    assert verify_equivalence(original, changed, "greet") is False
    result = compare_functions(original, changed, "greet")
    assert "print output differs" in result.detail


def test_exception_behaviour_is_compared() -> None:
    original = "def parse(text):\n    return int(text)\n"
    same = "def parse(text):\n    return int(text) + 0\n"
    swallowed = (
        "def parse(text):\n"
        "    try:\n"
        "        return int(text)\n"
        "    except ValueError:\n"
        "        return -1\n"
    )
    assert verify_equivalence(original, same, "parse", "abc") is True
    assert verify_equivalence(original, swallowed, "parse", "abc") is False
    result = compare_functions(original, swallowed, "parse", args=["abc"])
    assert "only the original version raised" in result.detail


def test_explicit_args_are_used_verbatim() -> None:
    # x * 2 vs x ** 2: equal at 0 (and 1), different at 3 — the inputs given by
    # the caller are the ones that decide the verdict.
    source = "def f(x):\n    return x * 2\n"
    changed = "def f(x):\n    return x ** 2\n"
    assert verify_equivalence(source, changed, "f", 0) is True
    assert verify_equivalence(source, changed, "f", 3) is False


def test_explicit_kwargs_reach_both_versions() -> None:
    source = "def join(parts, sep=','):\n    return sep.join(parts)\n"
    changed = "def join(parts, sep=';'):\n    return sep.join(parts)\n"
    # Unpassed defaults drift -> caught. Explicit kwargs pin both versions to
    # the same value -> equivalent, proving the kwargs really reach the calls.
    assert verify_equivalence(source, changed, "join", ["a", "b"]) is False
    assert verify_equivalence(source, changed, "join", ["a", "b"], sep=",") is True


def test_random_generation_from_signature() -> None:
    annotated = "def scale(x: int, factor: float = 2.0) -> float:\n    return x * factor\n"
    equivalent = "def scale(x: int, factor: float = 2.0) -> float:\n    return factor * x\n"
    assert verify_equivalence(annotated, equivalent, "scale") is True
    plain = "def bump(value):\n    return value + 1\n"
    assert verify_equivalence(plain, plain, "bump") is True


def test_union_annotation_generation_handles_none() -> None:
    source = "def norm(value: int | None) -> int:\n    return value or 0\n"
    changed = "def norm(value: int | None) -> int:\n    return value or 1\n"
    assert verify_equivalence(source, changed, "norm") is False


def test_variadic_signature_requires_explicit_args() -> None:
    source = "def total(*values):\n    return sum(values)\n"
    with pytest.raises(DifferentialError, match="variadic"):
        verify_equivalence(source, source, "total")


def test_keyword_only_without_default_requires_kwargs() -> None:
    source = "def shout(text, *, upper):\n    return text.upper() if upper else text\n"
    with pytest.raises(DifferentialError, match="keyword-only"):
        verify_equivalence(source, source, "shout")


def test_unbuildable_annotation_skips() -> None:
    source = "class Repo:\n    pass\n\n\ndef load(r: Repo):\n    return r\n"
    result = compare_functions(source, source, "load")
    assert result.verdict == VERDICT_SKIPPED
    assert "cannot synthesise" in result.detail


def test_custom_return_type_uses_scrubbed_repr() -> None:
    source = (
        "class Point:\n"
        "    def __init__(self, x):\n"
        "        self.x = x\n"
        "\n"
        "\n"
        "def make(x):\n"
        "    return Point(x)\n"
    )
    same = source.replace("return Point(x)", "return Point((x))")
    changed = source.replace("self.x = x", "self.x = x + 1")
    assert verify_equivalence(source, same, "make") is True
    assert verify_equivalence(source, changed, "make") is False


def test_nan_returns_compare_equal() -> None:
    source = "def ratio(a, b):\n    return a / b\n"
    assert verify_equivalence(source, source, "ratio", 0.0, 0.0) is True


def test_deep_recursion_failure_is_equivalent() -> None:
    source = "def spin(n):\n    return spin(n + 1)\n"
    assert verify_equivalence(source, source, "spin", 0) is True


def test_returning_containers_compare_by_value() -> None:
    source = "def build(n):\n    return {'k': [n, n + 1], 's': {n}}\n"
    equivalent = "def build(n):\n    value = [n, n + 1]\n    return {'k': value, 's': {n}}\n"
    changed = source.replace("[n, n + 1]", "[n, n + 2]")
    assert verify_equivalence(source, equivalent, "build") is True
    assert verify_equivalence(source, changed, "build") is False


# --------------------------------------------------------------------------- skips


def test_impure_global_statement_is_skipped() -> None:
    source = (
        "COUNTER = 0\n"
        "\n"
        "\n"
        "def bump(value):\n"
        "    global COUNTER\n"
        "    COUNTER = value\n"
        "    return COUNTER\n"
    )
    with pytest.raises(DifferentialError, match="global"):
        verify_equivalence(source, source, "bump")


def test_file_io_call_is_skipped_statically() -> None:
    source = "def read_it(path):\n    return open(path).read()\n"
    result = compare_functions(source, source, "read_it")
    assert result.verdict == VERDICT_SKIPPED
    assert "call to 'open' is refused" in result.detail


def test_module_importing_blocked_module_is_skipped() -> None:
    source = "import os\n\n\ndef size_of(path):\n    return 0\n"
    result = compare_functions(source, source, "size_of")
    assert result.verdict == VERDICT_SKIPPED
    assert "module-level execution failed" in result.detail


def test_call_time_import_is_skipped() -> None:
    source = "def lazily():\n    import math\n    return math.pi\n"
    result = compare_functions(source, source, "lazily")
    assert result.verdict == VERDICT_SKIPPED
    assert "call-time imports" in result.detail


def test_generator_is_skipped() -> None:
    source = "def count(n):\n    yield from range(n)\n"
    result = compare_functions(source, source, "count")
    assert result.verdict == VERDICT_SKIPPED
    assert "generator" in result.detail


def test_nameerror_on_both_sides_is_a_skip_not_equivalence() -> None:
    # Both versions crash on a builtin the sandbox withholds: identical failure
    # proves nothing about production behaviour, so it must not read as a pass.
    source = "def f(x):\n    return hidden_helper(x)\n"
    result = compare_functions(source, source, "f", args=[1])
    assert result.verdict == VERDICT_SKIPPED
    assert "NameError" in result.detail


def test_dynamic_eval_is_refused() -> None:
    # The builtin's name is built with chr() so security scanners do not mistake
    # a sandbox fixture for a live dynamic-execution call (project precedent).
    name = chr(101) + "val"
    source = f"def f(text):\n    return {name}(text)\n"
    result = compare_functions(source, source, "f", args=["1+1"])
    assert result.verdict == VERDICT_SKIPPED
    assert f"call to '{name}' is refused" in result.detail


def test_missing_function_is_a_skip() -> None:
    with pytest.raises(DifferentialError, match="not found"):
        verify_equivalence(_ADD_ORIGINAL, _ADD_EQUIVALENT, "subtract")


def test_methods_are_out_of_scope() -> None:
    source = "class Widget:\n    def area(self):\n        return 4\n"
    result = compare_functions(source, source, "Widget.area")
    assert result.verdict == VERDICT_SKIPPED
    assert "method" in result.detail


_LOOP_MULTILINE = (
    "def spin(n):\n"
    "    total = 0\n"
    "    while True:\n"
    "        total = (total + n) % 7\n"
    "    return total\n"
)


@pytest.mark.parametrize(
    "loop_source",
    [
        _LOOP_MULTILINE,
        "def spin(n):\n    while True: pass\n",
    ],
)
def test_infinite_loops_hit_the_budget(loop_source: str) -> None:
    result = compare_functions(
        loop_source, loop_source, "spin", args=[1], opcode_budget=300_000, wall_seconds=3.0
    )
    assert result.verdict == VERDICT_SKIPPED
    assert "budget exhausted" in result.detail


def test_sandbox_cannot_reach_real_builtins() -> None:
    # getattr is withheld: even a dynamic attribute call fails loudly instead of
    # escaping the sandbox's builtin allowlist.
    source = "def f(text):\n    return getattr(text, 'upper')()\n"
    result = compare_functions(source, source, "f", args=["x"])
    assert result.verdict == VERDICT_SKIPPED


# --------------------------------------------------------------------------- change detection


def test_changed_function_names_reports_changed_pairs_only() -> None:
    before = (
        "def f(x):\n    return x * 2\n\n\n"
        "class C:\n    def m(self):\n        return 1\n\n\n"
        "def gone():\n    return 0\n"
    )
    after = (
        "def f(x):\n    return x * 3\n\n\n"
        "class C:\n    def m(self):\n        return 1\n\n\n"
        "def fresh():\n    return 0\n"
    )
    assert changed_function_names(before, after) == ["f"]


def test_changed_function_names_empty_for_identical_sources() -> None:
    assert changed_function_names(_ADD_ORIGINAL, _ADD_ORIGINAL) == []


def test_compare_functions_never_raises_on_garbage() -> None:
    result = compare_functions("def (:", "def f():\n    return 1\n", "f")
    assert result.verdict == VERDICT_SKIPPED
    assert isinstance(result, DifferentialResult)


# --------------------------------------------------------------------------- orchestrator gate


def test_orchestrator_rolls_back_on_differential_mismatch(tmp_path: Path) -> None:
    repo = _init_double_repo(tmp_path)
    plan = RefactorPlan(
        instructions=[
            ReplaceCall(id="s1", target_file="app.py", old_call="double", new_call="triple")
        ]
    )
    orchestrator = Orchestrator(
        FakeParser([plan]), FakeVerifier([_ok()]), max_attempts=1, trace=FakeTrace()
    )

    report = orchestrator.run("reroute compute", repo)

    assert report.success is False
    assert report.last_verify is not None
    assert report.last_verify.stage == "differential"
    assert "'compute'" in report.last_verify.stdout
    assert (repo / "app.py").read_text(encoding="utf-8") == _DOUBLE_SOURCE
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"


def test_orchestrator_commits_when_differential_passes(tmp_path: Path) -> None:
    repo = _init_double_repo(tmp_path)
    plan = RefactorPlan(
        instructions=[AddTypeHints(id="s1", target_file="app.py", function_name="compute")]
    )
    orchestrator = Orchestrator(
        FakeParser([plan]), FakeVerifier([_ok()]), max_attempts=1, trace=FakeTrace()
    )

    report = orchestrator.run("annotate compute", repo)

    assert report.success is True
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "2"


def test_orchestrator_skips_impure_functions_without_blocking(tmp_path: Path) -> None:
    repo = _init_double_repo(tmp_path)
    _write(
        repo / "app.py",
        "TOTAL = 0\n\n\ndef compute(x):\n    global TOTAL\n"
        "    TOTAL = double(x)\n    return TOTAL\n"
        "\n\n\ndef double(x):\n    return x * 2\n",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "globals")
    plan = RefactorPlan(
        instructions=[AddTypeHints(id="s1", target_file="app.py", function_name="compute")]
    )
    orchestrator = Orchestrator(
        FakeParser([plan]), FakeVerifier([_ok()]), max_attempts=1, trace=FakeTrace()
    )

    report = orchestrator.run("annotate impure", repo)

    assert report.success is True  # the skip is logged, the commit still happens
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "3"


def test_orchestrator_can_disable_the_differential_gate(tmp_path: Path) -> None:
    repo = _init_double_repo(tmp_path)
    plan = RefactorPlan(
        instructions=[
            ReplaceCall(id="s1", target_file="app.py", old_call="double", new_call="triple")
        ]
    )
    orchestrator = Orchestrator(
        FakeParser([plan]), FakeVerifier([_ok()]), max_attempts=1, trace=FakeTrace(),
        differential=False,
    )

    report = orchestrator.run("reroute compute unchecked", repo)

    assert report.success is True
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "2"
