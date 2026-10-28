from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from moray_surgeon.audit import run_audit
from moray_surgeon.exceptions import CircularDependencyError, GitError
from moray_surgeon.schema import RefactorPlan, RenameSymbol
from moray_surgeon.verify import VerifyResult

_APP = (
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


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo / "app.py", _APP)
    _write(repo / "README.md", "# repo\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _rename_plan() -> RefactorPlan:
    return RefactorPlan(
        instructions=[
            RenameSymbol(
                id="step1",
                target_file="app.py",
                old_name="old_name",
                new_name="new_name",
            )
        ]
    )


class _FakeVerifier:
    def __init__(self, result: VerifyResult) -> None:
        self._result = result
        self.calls = 0

    def verify(self) -> VerifyResult:
        self.calls += 1
        return self._result


def _ok() -> VerifyResult:
    return VerifyResult(ok=True, stage="all", stdout="ok")


def _failed(stage: str = "pytest", stdout: str = "assertion failed") -> VerifyResult:
    return VerifyResult(ok=False, stage=stage, stdout=stdout)


def test_clean_tree_passes(tmp_path):
    repo = _init_repo(tmp_path)
    verifier = _FakeVerifier(_ok())

    report = run_audit(repo, verifier=verifier)

    assert report.ok is True
    assert report.changes == ()
    assert report.scope_violations == ()
    assert report.base_ref == "HEAD"
    assert report.base_commit == _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert report.plan_path is None
    assert verifier.calls == 1


def test_modified_file_reports_the_changed_nodes(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", _APP.replace("old_name", "new_name"))

    report = run_audit(repo, verifier=_FakeVerifier(_ok()))

    assert len(report.changes) == 1
    change = report.changes[0]
    assert change.path == "app.py"
    assert change.status == "modified"
    assert change.origin is None
    assert set(change.changed_nodes) == {"old_name", "new_name"}
    assert change.parse_error is False


def test_planned_change_passes_the_scope_gate(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", _APP.replace("old_name", "new_name"))

    report = run_audit(
        repo, plan=_rename_plan(), plan_path="plan.json", verifier=_FakeVerifier(_ok())
    )

    assert report.ok is True
    assert report.scope_violations == ()
    assert report.plan_path == "plan.json"


def test_unplanned_collateral_change_is_a_violation(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", _APP.replace("return 1", "return 999"))

    report = run_audit(repo, plan=_rename_plan(), verifier=_FakeVerifier(_ok()))

    assert report.ok is False
    assert report.scope_violations == ("app.py: neighbour",)


def test_created_file_is_a_violation_when_a_plan_is_supplied(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "extra.py", "def helper():\n    return 1\n")

    report = run_audit(repo, plan=_rename_plan(), verifier=_FakeVerifier(_ok()))

    assert report.ok is False
    assert report.changes[0].status == "added"
    assert any("extra.py: created outside the plan" in item for item in report.scope_violations)


def test_created_file_is_informational_without_a_plan(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "extra.py", "def helper():\n    return 1\n")

    report = run_audit(repo, verifier=_FakeVerifier(_ok()))

    assert report.ok is True
    assert report.changes[0].status == "added"
    assert "helper" in report.changes[0].changed_nodes


def test_deleted_file_is_a_violation_when_a_plan_is_supplied(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "app.py").unlink()

    report = run_audit(repo, plan=_rename_plan(), verifier=_FakeVerifier(_ok()))

    assert report.ok is False
    assert report.changes[0].status == "deleted"
    assert any("deleted outside the plan" in item for item in report.scope_violations)


def test_renamed_file_is_a_violation_when_a_plan_is_supplied(tmp_path):
    repo = _init_repo(tmp_path)
    _git(repo, "mv", "app.py", "moved.py")

    report = run_audit(repo, plan=_rename_plan(), verifier=_FakeVerifier(_ok()))

    assert report.ok is False
    assert report.changes[0].status == "renamed"
    assert report.changes[0].origin == "app.py"
    assert any("renamed from app.py" in item for item in report.scope_violations)


def test_non_python_change_is_a_violation_when_a_plan_is_supplied(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "README.md", "# changed\n")

    report = run_audit(repo, plan=_rename_plan(), verifier=_FakeVerifier(_ok()))

    assert report.ok is False
    assert any("non-Python file changed" in item for item in report.scope_violations)


def test_unparseable_python_is_a_violation(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "def broken(:\n")

    report = run_audit(repo, verifier=_FakeVerifier(_ok()))

    assert report.ok is False
    assert report.changes[0].parse_error is True
    assert report.scope_violations == ("app.py: does not parse as Python",)


def test_red_verification_gate_fails_the_audit(tmp_path):
    repo = _init_repo(tmp_path)

    report = run_audit(repo, verifier=_FakeVerifier(_failed()))

    assert report.ok is False
    assert report.scope_violations == ()
    assert report.verify.stage == "pytest"


def test_base_ref_audits_committed_changes(tmp_path):
    repo = _init_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _write(repo / "app.py", _APP.replace("old_name", "new_name"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "rename")

    report = run_audit(repo, base_ref=base, verifier=_FakeVerifier(_ok()))

    assert [change.path for change in report.changes] == ["app.py"]
    assert set(report.changes[0].changed_nodes) == {"old_name", "new_name"}
    assert report.base_commit == base


def test_audit_does_not_modify_the_worktree(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", _APP.replace("return 1", "return 999"))
    before = (repo / "app.py").read_text(encoding="utf-8")
    status_before = _git(repo, "status", "--porcelain").stdout

    run_audit(repo, plan=_rename_plan(), verifier=_FakeVerifier(_ok()))

    assert (repo / "app.py").read_text(encoding="utf-8") == before
    assert _git(repo, "status", "--porcelain").stdout == status_before


def test_non_git_directory_raises(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(GitError, match="not a git repository"):
        run_audit(plain, verifier=_FakeVerifier(_ok()))


def test_unknown_base_ref_raises(tmp_path):
    repo = _init_repo(tmp_path)

    with pytest.raises(GitError, match="cannot resolve base ref"):
        run_audit(repo, base_ref="no-such-ref", verifier=_FakeVerifier(_ok()))


def test_cyclic_plan_is_rejected(tmp_path):
    repo = _init_repo(tmp_path)
    plan = RefactorPlan.model_validate(
        {
            "instructions": [
                {"id": "a", "action": "noop", "depends_on": ["b"]},
                {"id": "b", "action": "noop", "depends_on": ["a"]},
            ]
        }
    )

    with pytest.raises(CircularDependencyError):
        run_audit(repo, plan=plan, verifier=_FakeVerifier(_ok()))


def test_json_report_is_serializable_and_bounded(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", _APP.replace("old_name", "new_name"))
    report = run_audit(
        repo,
        plan=_rename_plan(),
        verifier=_FakeVerifier(_failed(stdout="x" * 5000)),
    )

    data = json.loads(report.to_json())

    assert data["ok"] is False
    assert data["plan"] is None
    assert [item["path"] for item in data["files"]] == ["app.py"]
    assert data["files"][0]["status"] == "modified"
    assert data["scope_violations"] == []
    assert data["verification"]["stage"] == "pytest"
    assert len(data["verification"]["stdout"]) == 4000
