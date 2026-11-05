"""Tests for the pre-commit guard: installation and a real blocking commit.

The last tests execute git for real: the installed hook runs
``python -m moray_surgeon audit`` inside the temporary repository, so a syntax
error is blocked by ruff before the commit object is created.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from moray_surgeon.exceptions import HookError
from moray_surgeon.hooks import install_pre_commit_hook

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


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "--initial-branch=main")
    _write(repo / "app.py", 'def greet(name: str) -> str:\n    return f"hello {name}"\n')
    _write(
        repo / "test_app.py",
        "from app import greet\n\n\ndef test_greet():\n    assert greet('x') == 'hello x'\n",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _commit_count(repo: Path) -> int:
    return int(_git(repo, "rev-list", "--count", "HEAD").stdout.strip())


def test_install_writes_an_lf_hook_that_runs_audit(tmp_path):
    repo = _init_repo(tmp_path)

    hook = install_pre_commit_hook(repo, python_executable=sys.executable)

    assert hook == (repo / ".git" / "hooks" / "pre-commit").resolve()
    data = hook.read_bytes()
    assert data.startswith(b"#!/bin/sh\n")
    assert b"\r\n" not in data  # a CRLF shebang breaks Git Bash
    assert b"# moray-surgeon pre-commit hook (managed)" in data
    assert b"moray_surgeon audit" in data
    assert b"--base-ref" in data
    assert b"--mypy-target" in data
    assert b"4b825dc642cb6eb9a060e54bf8d69288fbee4904" in data  # first-commit empty tree
    assert Path(sys.executable).as_posix().encode() in data


def test_reinstalling_is_idempotent(tmp_path):
    repo = _init_repo(tmp_path)

    first = install_pre_commit_hook(repo)
    second = install_pre_commit_hook(repo)

    assert first == second
    text = first.read_text(encoding="utf-8")
    assert text.count("# moray-surgeon pre-commit hook (managed)") == 1


def test_a_foreign_hook_requires_force(tmp_path):
    repo = _init_repo(tmp_path)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\necho foreign\n", encoding="utf-8")

    with pytest.raises(HookError, match="not installed by moray-surgeon"):
        install_pre_commit_hook(repo)

    assert hook.read_text(encoding="utf-8") == "#!/bin/sh\necho foreign\n"

    install_pre_commit_hook(repo, force=True)
    assert "moray-surgeon pre-commit hook" in hook.read_text(encoding="utf-8")


def test_core_hooks_path_is_honoured(tmp_path):
    repo = _init_repo(tmp_path)
    _git(repo, "config", "core.hooksPath", "custom-hooks")

    hook = install_pre_commit_hook(repo)

    assert hook == (repo / "custom-hooks" / "pre-commit").resolve()
    assert not (repo / ".git" / "hooks" / "pre-commit").exists()


def test_an_interpreter_with_spaces_is_shell_quoted(tmp_path):
    repo = _init_repo(tmp_path)
    interpreter = (tmp_path / "my python").as_posix()

    hook = install_pre_commit_hook(repo, python_executable=interpreter)

    assert f"'{interpreter}'" in hook.read_text(encoding="utf-8")


def test_a_plain_directory_is_not_a_repository(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(HookError, match="git working tree"):
        install_pre_commit_hook(plain)


def test_the_hook_blocks_a_syntax_error_commit(tmp_path):
    repo = _init_repo(tmp_path)
    install_pre_commit_hook(repo, python_executable=sys.executable)
    before = _commit_count(repo)
    _write(repo / "bad.py", "def broken(:\n    pass\n")
    _git(repo, "add", "bad.py")

    result = _git(repo, "commit", "-m", "add broken module", check=False)

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert _commit_count(repo) == before
    assert "audit failed" in output
    assert "bad.py" in output


def test_the_hook_allows_a_clean_commit(tmp_path):
    repo = _init_repo(tmp_path)
    install_pre_commit_hook(repo, python_executable=sys.executable)
    before = _commit_count(repo)
    _write(repo / "extra.py", "VALUE = 3\n")
    _git(repo, "add", "extra.py")

    result = _git(repo, "commit", "-m", "add extra", check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _commit_count(repo) == before + 1
