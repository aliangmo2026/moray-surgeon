from __future__ import annotations

import json
from pathlib import Path

import pytest

from moray_surgeon.architecture import capture_architecture, diff_architecture


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "math.py").write_text(
        "class Math:\n"
        "    def scale(self, x):\n"
        "        return x * 2\n"
        "\n"
        "\n"
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (repo / "app.py").write_text("from math import helper\n\n\ny = helper()\n", encoding="utf-8")
    return repo


def test_capture_architecture_serializes_deterministically(tmp_path) -> None:
    """Two captures of the same tree are byte-identical JSON."""
    repo = _repo(tmp_path)

    first = capture_architecture(repo)
    second = capture_architecture(repo)

    assert first is not None and second is not None
    assert first == second
    snapshot = json.loads(first)
    ids = [entry["id"] for entry in snapshot["nodes"]]
    assert "math.Math" in ids
    assert "math.helper" in ids


def test_capture_architecture_includes_test_files(tmp_path) -> None:
    """Test modules are part of the architecture, not excluded like scribe's
    documentation defaults."""
    repo = _repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_math.py").write_text(
        "def test_scale():\n    assert True\n", encoding="utf-8"
    )

    snapshot = capture_architecture(repo)

    assert snapshot is not None
    assert "tests.test_math.test_scale" in snapshot


def test_diff_architecture_reports_moved_function(tmp_path) -> None:
    """A symbol leaving one module for another is reported as a move."""
    repo = _repo(tmp_path)
    before = capture_architecture(repo)
    (repo / "helpers.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (repo / "math.py").write_text(
        "class Math:\n    def scale(self, x):\n        return x * 2\n", encoding="utf-8"
    )
    (repo / "app.py").write_text(
        "from helpers import helper\n\n\ny = helper()\n", encoding="utf-8"
    )
    after = capture_architecture(repo)

    diff = diff_architecture(before, after)

    assert "架构变更" in diff.summary
    assert "移动了 1 个函数" in diff.summary
    assert "helpers" in diff.text
    assert "math" in diff.text


def test_diff_architecture_reports_new_class(tmp_path) -> None:
    """A newly created class shows up in the summary and the detail."""
    repo = _repo(tmp_path)
    before = capture_architecture(repo)
    (repo / "math.py").write_text(
        "class Math:\n"
        "    def scale(self, x):\n"
        "        return x * 2\n"
        "\n"
        "\n"
        "class MathOps:\n"
        "    def combine(a, b):\n"
        "        return a + b\n",
        encoding="utf-8",
    )
    after = capture_architecture(repo)

    diff = diff_architecture(before, after)

    assert "提取了 1 个新类" in diff.summary
    assert "MathOps" in diff.text


def test_diff_architecture_unchanged_is_empty(tmp_path) -> None:
    """Identical structures produce no report."""
    repo = _repo(tmp_path)
    snapshot = capture_architecture(repo)

    diff = diff_architecture(snapshot, snapshot)

    assert diff.summary == ""
    assert diff.text == ""


def test_diff_architecture_missing_side_is_empty(tmp_path) -> None:
    """A missing capture (scribe unavailable) disables the comparison."""
    assert diff_architecture(None, "{}").summary == ""
    assert diff_architecture("{}", None).text == ""


def test_capture_architecture_degrades_on_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file scribe cannot parse must not fail the refactor."""
    repo = _repo(tmp_path)

    def explode(path: Path, *, root: Path | None = None):
        raise RuntimeError("scanner exploded")

    monkeypatch.setattr("moray_scribe.scanner.extract_file", explode)

    assert capture_architecture(repo) is None


def test_capture_architecture_degrades_without_scribe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without moray-scribe installed the capture returns None."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name.startswith("moray_scribe"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert capture_architecture(_repo(tmp_path)) is None
