"""Tests for the fine-tuning data flywheel: AST diff → plan → JSONL.

Every test builds a real temporary git repository. The mining path reads commits
with git plumbing, parses both sides with libcst and validates the emitted plan
against the schema; no model is involved.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from moray_surgeon.data.flywheel import (
    collect_samples,
    detect_instructions,
    export_sft,
)
from moray_surgeon.exceptions import FlywheelError
from moray_surgeon.schema import (
    AddImport,
    AddTypeHints,
    RefactorPlan,
    RemoveImport,
    RenameSymbol,
    ReplaceCall,
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

_BASE_APP = """\
from pkg import unused
from os import getcwd


def old_name(value):
    return value + 1


def caller(value):
    return old_name(value) + old_call(value)


def old_call(value):
    return value


def new_call(value):
    return value + 1


def typed():
    print("x")
"""


@pytest.fixture(autouse=True)
def _isolated_git_environment(monkeypatch):
    for name, value in _GIT_ENV.items():
        monkeypatch.setenv(name, value)


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result


def _rev(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", ref).stdout.strip()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _rev(repo, "HEAD")


def _history_repo(tmp_path: Path) -> Path:
    """A real repository whose commits exercise every supported pattern."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "--initial-branch=main")
    _write(repo / "app.py", _BASE_APP)
    _write(repo / "utils.py", "def helper():\n    return 1\n")
    _commit(repo, "base")

    renamed = _BASE_APP.replace("old_name", "new_name")
    _write(repo / "app.py", renamed)
    _commit(repo, "rename old_name to new_name")

    replaced = renamed.replace("old_call(value)", "new_call(value)")
    _write(repo / "app.py", replaced)
    _commit(repo, "replace old_call with new_call")

    imports = replaced.replace(
        "from pkg import unused", "from collections import deque"
    )
    _write(repo / "app.py", imports)
    _commit(repo, "update imports")

    hints = imports.replace("def typed():", "def typed() -> None:")
    _write(repo / "app.py", hints)
    _commit(repo, "annotate typed")

    _write(repo / "app.py", "# historical note\n" + hints)
    _commit(repo, "comment only")

    _write(repo / "app.py", "from .local import thing\n" + hints)
    _commit(repo, "relative import")

    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo / "feature.py", "FEATURE = 1\n")
    _commit(repo, "feature file")
    _git(repo, "checkout", "-q", "main")
    _write(repo / "main_extra.py", "MAIN = 2\n")
    _commit(repo, "main file")
    _git(repo, "merge", "-q", "--no-ff", "feature", "-m", "merge feature")
    return repo


def _samples_by_subject(repo: Path) -> dict[str, object]:
    return {sample.subject: sample for sample in collect_samples(repo, steps=50)}


# --------------------------------------------------------------------------- #
# Detection precision (source-in/source-out, no git)
# --------------------------------------------------------------------------- #


def test_detects_a_pure_function_rename():
    before = "def old(value):\n    return value + 1\n"
    after = "def new(value):\n    return value + 1\n"

    instructions = detect_instructions(before, after, "app.py")

    assert len(instructions) == 1
    instruction = instructions[0]
    assert isinstance(instruction, RenameSymbol)
    assert (instruction.old_name, instruction.new_name, instruction.target_file) == (
        "old",
        "new",
        "app.py",
    )


def test_a_rename_that_changes_the_body_is_not_mined():
    before = "def old(value):\n    return value + 1\n"
    after = "def new(value):\n    return value + 2\n"

    assert detect_instructions(before, after, "app.py") == []


def test_detects_a_pure_callee_replacement():
    before = "def caller(value):\n    return old_call(value)\n"
    after = "def caller(value):\n    return new_call(value)\n"

    instructions = detect_instructions(before, after, "app.py")

    assert len(instructions) == 1
    instruction = instructions[0]
    assert isinstance(instruction, ReplaceCall)
    assert (instruction.old_call, instruction.new_call) == ("old_call", "new_call")


def test_a_call_with_changed_arguments_is_not_mined():
    before = "def caller(value):\n    return old_call(value)\n"
    after = "def caller(value):\n    return new_call(value, 2)\n"

    assert detect_instructions(before, after, "app.py") == []


def test_detects_import_additions_and_removals():
    before = "from pkg import unused\nfrom os import getcwd\n"
    after = "from collections import deque\nfrom os import getcwd\n"

    instructions = detect_instructions(before, after, "app.py")

    assert [type(item) for item in instructions] == [RemoveImport, AddImport]
    removal, addition = instructions
    assert (removal.module, removal.name) == ("pkg", "unused")
    assert (addition.module, addition.name) == ("collections", "deque")


def test_relative_imports_are_not_mined():
    before = "import os\n"
    after = "from .local import thing\nimport os\n"

    assert detect_instructions(before, after, "app.py") == []


def test_detects_a_none_return_annotation():
    before = "def typed():\n    print('x')\n"
    after = "def typed() -> None:\n    print('x')\n"

    instructions = detect_instructions(before, after, "app.py")

    assert len(instructions) == 1
    instruction = instructions[0]
    assert isinstance(instruction, AddTypeHints)
    assert instruction.function_name == "typed"


def test_a_richer_return_annotation_is_not_mined():
    before = "def typed():\n    return 1\n"
    after = "def typed() -> int:\n    return 1\n"

    assert detect_instructions(before, after, "app.py") == []


def test_unparseable_sources_produce_nothing():
    assert detect_instructions("def broken(:\n", "def broken():\n", "app.py") == []


# --------------------------------------------------------------------------- #
# Git mining
# --------------------------------------------------------------------------- #


def test_mines_the_supported_commits_and_skips_the_rest(tmp_path):
    repo = _history_repo(tmp_path)

    samples = collect_samples(repo, steps=50)
    by_subject = {sample.subject: sample for sample in samples}

    assert set(by_subject) == {
        "rename old_name to new_name",
        "replace old_call with new_call",
        "update imports",
        "annotate typed",
    }
    assert "base" not in by_subject  # root commit has no parent
    assert "comment only" not in by_subject  # no AST change
    assert "relative import" not in by_subject  # schema cannot express it
    assert "merge feature" not in by_subject  # merge commits are skipped
    assert [sample.subject for sample in samples] == [
        "rename old_name to new_name",
        "replace old_call with new_call",
        "update imports",
        "annotate typed",
    ]

    rename = by_subject["rename old_name to new_name"]
    assert isinstance(rename.plan.instructions[0], RenameSymbol)
    assert rename.plan_json
    assert rename.instruction == "Rename `old_name` to `new_name` in `app.py`."

    replace = by_subject["replace old_call with new_call"]
    assert isinstance(replace.plan.instructions[0], ReplaceCall)
    imports = by_subject["update imports"]
    assert {type(item) for item in imports.plan.instructions} == {
        RemoveImport,
        AddImport,
    }
    hints = by_subject["annotate typed"]
    assert isinstance(hints.plan.instructions[0], AddTypeHints)


def test_context_contains_the_repo_map_and_the_before_file(tmp_path):
    repo = _history_repo(tmp_path)
    rename = _samples_by_subject(repo)["rename old_name to new_name"]

    context = rename.context
    assert "【Project Code Map】" in context
    assert "# File: utils.py" in context
    assert "helper" in context
    assert "【Before File: app.py】" in context
    assert "def old_name" in context
    # The answer must not leak into the input.
    assert "def new_name" not in context


def test_steps_limits_the_lookback_window(tmp_path):
    repo = _history_repo(tmp_path)

    assert collect_samples(repo, steps=4) == []
    assert [sample.subject for sample in collect_samples(repo, steps=6)] == [
        "annotate typed"
    ]


def test_mining_is_deterministic(tmp_path):
    repo = _history_repo(tmp_path)

    first = [sample.plan_json for sample in collect_samples(repo, steps=50)]
    second = [sample.plan_json for sample in collect_samples(repo, steps=50)]

    assert first == second


def test_repo_map_can_be_disabled(tmp_path):
    repo = _history_repo(tmp_path)

    samples = collect_samples(repo, steps=50, include_repo_map=False)

    assert samples
    assert all("【Project Code Map】" not in sample.context for sample in samples)


def test_plain_directory_is_rejected(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(FlywheelError, match="git working tree"):
        collect_samples(plain)


def test_unknown_revision_is_rejected(tmp_path):
    repo = _history_repo(tmp_path)

    with pytest.raises(FlywheelError, match="cannot read history"):
        collect_samples(repo, branch="does-not-exist")


# --------------------------------------------------------------------------- #
# JSONL output
# --------------------------------------------------------------------------- #


def test_openai_jsonl_is_strict_and_validates_as_refactor_plans(tmp_path):
    repo = _history_repo(tmp_path)
    output = tmp_path / "sft.jsonl"

    report = export_sft(repo, steps=50, output=output, fmt="openai")

    assert report.samples == 4
    assert report.fmt == "openai"
    assert report.output == output
    assert dict(report.actions)["rename_symbol"] == 1
    assert dict(report.actions)["replace_call"] == 1
    assert dict(report.actions)["add_type_hints"] == 1
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    for line in lines:
        record = json.loads(line)
        assert set(record) == {"messages"}
        messages = record["messages"]
        assert [message["role"] for message in messages] == [
            "system",
            "user",
            "assistant",
        ]
        system, user, assistant = (message["content"] for message in messages)
        assert "RefactorPlan" in system
        assert "JSON Schema" in system
        assert user.startswith("Repository root:")
        assert "Refactoring intent:" in user
        assert "【Project Code Map】" in user
        assert "【Before File: app.py】" in user
        assert "Return the RefactorPlan JSON object now." in user
        plan = RefactorPlan.model_validate_json(assistant)
        assert plan.instructions


def test_alpaca_jsonl_matches_the_compact_instruction_shape(tmp_path):
    repo = _history_repo(tmp_path)
    output = tmp_path / "sft-alpaca.jsonl"

    report = export_sft(
        repo, steps=50, output=output, fmt="alpaca", include_repo_map=False
    )

    assert report.samples == 4
    for line in output.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert set(record) == {"instruction", "input", "output"}
        assert record["instruction"]
        assert "【Before File: app.py】" in record["input"]
        assert "【Project Code Map】" not in record["input"]
        assert RefactorPlan.model_validate_json(record["output"]).instructions


def test_unknown_format_is_rejected_before_touching_git(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(FlywheelError, match="unsupported format"):
        export_sft(plain, output=tmp_path / "out.jsonl", fmt="csv")
