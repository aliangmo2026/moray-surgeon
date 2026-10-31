from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from moray_surgeon.audit import AuditReport
from moray_surgeon.cloud_entry import handler
from moray_surgeon.exceptions import GitError
from moray_surgeon.intent import IntentParserError
from moray_surgeon.orchestrator import RefactorReport
from moray_surgeon.verify import VerifyResult

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


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A minimal git repository with one committed file."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=root, check=True)
    return root


def _body(response: dict) -> dict:
    return json.loads(response["body"])


def test_refactor_success(repo: Path, monkeypatch):
    def fake_run(self, intent, repo_root, context="", target_files=None, dry_run=False):
        return RefactorReport(
            success=True,
            attempts=1,
            plan=None,
            log=["app.py"],
            last_verify=VerifyResult(ok=True, stage="all"),
        )

    monkeypatch.setattr("moray_surgeon.orchestrator.Orchestrator.run", fake_run)
    response = handler({"action": "refactor", "repo_path": str(repo), "intent": "rename x"})
    assert response["statusCode"] == 200
    payload = _body(response)
    assert payload["ok"] is True
    assert payload["log"] == ["app.py"]
    assert payload["verify"]["ok"] is True


def test_refactor_requires_intent(repo: Path):
    response = handler({"action": "refactor", "repo_path": str(repo)})
    assert response["statusCode"] == 400
    assert "intent" in _body(response)["error"]


def test_unreachable_repo_path_gets_friendly_error(tmp_path: Path):
    missing = tmp_path / "not-there"
    response = handler({"action": "audit", "repo_path": str(missing)})
    assert response["statusCode"] == 400
    error = _body(response)["error"]
    assert "not reachable" in error
    assert "cloud sandbox" in error


def test_non_git_directory_rejected(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    response = handler({"action": "audit", "repo_path": str(plain)})
    assert response["statusCode"] == 400
    assert "Git repository" in _body(response)["error"]


def test_audit_success(repo: Path, monkeypatch):
    fake = AuditReport(
        ok=True,
        repo_root=str(repo),
        base_ref="HEAD",
        base_commit="abc123",
        plan_path=None,
        changes=(),
        scope_violations=(),
        verify=VerifyResult(ok=True, stage="all"),
    )
    monkeypatch.setattr("moray_surgeon.cloud_entry.run_audit", lambda root, base_ref="HEAD": fake)
    response = handler({"action": "audit", "repo_path": str(repo), "base_ref": "HEAD"})
    assert response["statusCode"] == 200
    payload = _body(response)
    assert payload["ok"] is True
    assert payload["action"] == "audit"
    assert payload["base_commit"] == "abc123"


def test_audit_git_error_becomes_500(repo: Path, monkeypatch):
    monkeypatch.setattr(
        "moray_surgeon.cloud_entry.run_audit",
        lambda root, base_ref="HEAD": (_ for _ in ()).throw(GitError("bad ref")),
    )
    response = handler({"action": "audit", "repo_path": str(repo)})
    assert response["statusCode"] == 500
    assert "bad ref" in _body(response)["error"]


def test_unsupported_action_rejected():
    response = handler({"action": "deploy", "repo_path": "/tmp"})
    assert response["statusCode"] == 400
    assert "deploy" in _body(response)["error"]


def test_missing_repo_path_rejected():
    response = handler({"action": "audit"})
    assert response["statusCode"] == 400
    assert "repo_path" in _body(response)["error"]


def test_api_gateway_envelope_with_json_body(repo: Path):
    event = {"body": json.dumps({"action": "audit", "repo_path": str(repo / "nope")})}
    assert handler(event)["statusCode"] == 400


def test_api_gateway_envelope_with_base64_body(repo: Path):
    raw = json.dumps({"action": "audit", "repo_path": str(repo / "nope")}).encode()
    event = {"body": base64.b64encode(raw).decode(), "isBase64Encoded": True}
    assert handler(event)["statusCode"] == 400


def test_invalid_event_is_400_not_crash():
    assert handler({"body": "not json"})["statusCode"] == 400
    assert handler("just a string")["statusCode"] == 400


def test_model_unreachable_becomes_502(repo: Path, monkeypatch):
    def fake_run(self, intent, repo_root, context="", target_files=None, dry_run=False):
        raise IntentParserError("connection refused")

    monkeypatch.setattr("moray_surgeon.orchestrator.Orchestrator.run", fake_run)
    response = handler({"action": "refactor", "repo_path": str(repo), "intent": "rename x"})
    assert response["statusCode"] == 502
    assert "connection refused" in _body(response)["error"]


def test_build_skill_produces_small_zip(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_skill.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    zip_path = script.parents[1] / "moray-skill.zip"
    try:
        assert zip_path.stat().st_size < 3 * 1024 * 1024
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            assert "handler.py" in names
            assert "SKILL.md" in names
            assert "requirements.txt" in names
            assert "moray_surgeon/cloud_entry.py" in names
            manifest = archive.read("SKILL.md").decode("utf-8")
            assert "moray-surgeon" in manifest
            assert "refactor" in manifest
            assert "audit" in manifest
            # The platform parses a YAML frontmatter block at the very top.
            lines = manifest.splitlines()
            assert lines[0] == "---"
            assert "name: moray-surgeon" in lines
            required = (
                "version",
                "display_name",
                "display_name_en",
                "description_zh",
                "description_en",
            )
            for field in required:
                assert any(line.startswith(f"{field}: ") for line in lines), field
            assert "---" in lines[1:9]
            assert not any("__pycache__" in name for name in names)
            assert not any(name.startswith("moray_surgeon/tests") for name in names)
    finally:
        zip_path.unlink(missing_ok=True)
