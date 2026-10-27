from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from moray_surgeon.exceptions import GitError
from moray_surgeon.transaction import TransactionManager

# Pinned per test so commits never depend on the machine's git identity, global
# hooks/signing, or line-ending conversion.
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
    """Give every git subprocess (library and helper) a hermetic environment."""
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


def _init_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    return repo


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def _head_subject(repo: Path) -> str:
    return _git(repo, "log", "-1", "--pretty=%s").stdout.strip()


def _status_lines(repo: Path) -> set[str]:
    return set(_git(repo, "status", "--porcelain").stdout.splitlines())


def _stash_lines(repo: Path) -> set[str]:
    return set(_git(repo, "stash", "list").stdout.splitlines())


def test_clean_repo_commits_transaction_changes(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "VALUE = 1\n")
    _commit_all(repo, "base")

    with TransactionManager(repo, "refactor: bump value") as transaction:
        assert isinstance(transaction, TransactionManager)
        _write(repo / "app.py", "VALUE = 2\n")
        _write(repo / "new_module.py", "NEW = True\n")

    assert _read(repo / "app.py") == "VALUE = 2\n"
    assert _git(repo, "show", "HEAD:app.py").stdout == "VALUE = 2\n"
    assert _git(repo, "show", "HEAD:new_module.py").stdout == "NEW = True\n"
    assert _head_subject(repo) == "refactor: bump value"
    assert _status_lines(repo) == set()
    assert _stash_lines(repo) == set()


def test_clean_repo_exception_restores_head(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "VALUE = 1\n")
    _commit_all(repo, "base")

    with pytest.raises(RuntimeError, match="boom"), TransactionManager(
        repo, "must not be committed"
    ):
        _write(repo / "app.py", "VALUE = 2\n")
        _write(repo / "broken.py", "BROKEN = True\n")
        raise RuntimeError("boom")

    assert _read(repo / "app.py") == "VALUE = 1\n"
    assert not (repo / "broken.py").exists()
    assert _head_subject(repo) == "base"
    assert _status_lines(repo) == set()


def test_dirty_repo_exception_restores_user_work(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "VALUE = 1\n")
    _commit_all(repo, "base")
    _write(repo / "app.py", "VALUE = 1\nUSER = 2\n")
    _write(repo / "scratch.py", "SC = 1\n")

    with pytest.raises(RuntimeError, match="boom"), TransactionManager(
        repo, "must not be committed"
    ):
        _write(repo / "app.py", "VALUE = 99\n")
        _write(repo / "generated.py", "G = 1\n")
        raise RuntimeError("boom")

    # Both the tracked edit and the untracked file come back byte-for-byte, and
    # nothing the transaction created survives.
    assert _read(repo / "app.py") == "VALUE = 1\nUSER = 2\n"
    assert _read(repo / "scratch.py") == "SC = 1\n"
    assert not (repo / "generated.py").exists()
    assert _status_lines(repo) == {" M app.py", "?? scratch.py"}
    assert _stash_lines(repo) == set()


def test_dirty_repo_success_commits_user_and_transaction_changes(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "VALUE = 1\n")
    _write(repo / "README.md", "base readme\n")
    _commit_all(repo, "base")
    _write(repo / "README.md", "base readme\nuser note\n")
    _write(repo / "scratch.py", "SC = 1\n")

    with TransactionManager(repo, "refactor: merge both"):
        _write(repo / "app.py", "VALUE = 2\n")

    assert _git(repo, "show", "HEAD:app.py").stdout == "VALUE = 2\n"
    assert _git(repo, "show", "HEAD:README.md").stdout == "base readme\nuser note\n"
    assert _git(repo, "show", "HEAD:scratch.py").stdout == "SC = 1\n"
    assert _head_subject(repo) == "refactor: merge both"
    assert _status_lines(repo) == set()
    assert _stash_lines(repo) == set()


def test_empty_repository_raises_git_error(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")

    with pytest.raises(GitError, match="空仓库无法使用事务机制"), TransactionManager(
        repo, "nothing to baseline"
    ):
        pass


def test_non_git_directory_raises_git_error(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(GitError) as excinfo, TransactionManager(plain, "no repo"):
        pass

    assert excinfo.value.stderr
    assert "rev-parse" in str(excinfo.value)


def test_chinese_commit_message_is_preserved(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "VALUE = 1\n")
    _commit_all(repo, "base")
    message = "重构: 重命名符号并提取函数"

    with TransactionManager(repo, message):
        _write(repo / "app.py", "VALUE = 2\n")

    assert _head_subject(repo) == message


def test_conflicting_snapshot_raises_and_keeps_user_work(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "app.py", "VALUE = 1\n")
    _commit_all(repo, "base")
    # The user's hunk and the transaction's rewrite overlap, so git cannot merge
    # them; the transaction must fail loudly instead of choosing a side.
    _write(repo / "app.py", "VALUE = 1\nUSER = 2\n")

    with pytest.raises(GitError) as excinfo, TransactionManager(repo, "must conflict"):
        _write(repo / "app.py", "VALUE = 999\n")

    assert excinfo.value.stderr
    assert _stash_lines(repo), "the snapshot must survive a failed commit"
    assert "USER = 2" in _git(repo, "show", "stash@{0}:app.py").stdout
    assert _read(repo / "app.py") == "VALUE = 999\n"
