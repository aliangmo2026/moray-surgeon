from __future__ import annotations

from pathlib import Path

import pytest

from moray_surgeon.engine.repo_map import generate_repo_map
from moray_surgeon.exceptions import PathTraversalError

_SOURCE = (
    '"""Module doc."""\n'
    "\n"
    "\n"
    "def top_level(x: int) -> str:\n"
    '    """Convert x.\n'
    "\n"
    "    More detail.\n"
    '    """\n'
    "    def nested():\n"
    "        pass\n"
    "    return str(x)\n"
    "\n"
    "\n"
    "class User:\n"
    '    """Represents a user."""\n'
    "\n"
    "    def __init__(self, name: str):\n"
    "        self.name = name\n"
    "\n"
    "    async def load(self) -> None:\n"
    "        return None\n"
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_repo_map_extracts_classes_functions_methods_and_lines(tmp_path):
    _write(tmp_path / "app.py", _SOURCE)

    text = generate_repo_map(tmp_path, ["app.py"])

    assert text == (
        "# File: app.py\n"
        "def top_level(x: int) -> str: (line 4) # Convert x.\n"
        "class User: (line 14) # Represents a user.\n"
        "def __init__(self, name: str): (line 17)\n"
        "async def load(self) -> None: (line 20)"
    )
    assert "nested" not in text
    assert "More detail" not in text


def test_repo_map_only_reads_requested_files(tmp_path):
    _write(tmp_path / "a.py", "class A:\n    pass\n")
    _write(tmp_path / "b.py", "def build():\n    pass\n")
    _write(tmp_path / "c.py", "class C:\n    pass\n")

    text = generate_repo_map(tmp_path, ["a.py", "b.py"])

    assert "# File: a.py" in text
    assert "class A:" in text
    assert "# File: b.py" in text
    assert "def build():" in text
    assert "c.py" not in text
    assert "class C" not in text


def test_repo_map_renders_class_bases_and_async_methods(tmp_path):
    _write(
        tmp_path / "models.py",
        "class Repo(Base, metaclass=Meta):\n"
        "    async def load(self) -> None:\n"
        "        return None\n",
    )

    text = generate_repo_map(tmp_path, ["models.py"])

    assert "class Repo(Base, metaclass=Meta): (line 1)" in text
    assert "async def load(self) -> None: (line 2)" in text


def test_repo_map_rejects_path_traversal(tmp_path):
    with pytest.raises(PathTraversalError):
        generate_repo_map(tmp_path, ["../outside.py"])


def test_repo_map_returns_empty_string_for_no_targets(tmp_path):
    assert generate_repo_map(tmp_path, []) == ""


def test_repo_map_reports_unparseable_files(tmp_path):
    _write(tmp_path / "broken.py", "def broken(:\n    pass\n")

    text = generate_repo_map(tmp_path, ["broken.py"])

    assert "# File: broken.py" in text
    assert "unable to parse" in text


def test_repo_map_collapses_multiline_signatures_and_decorators(tmp_path):
    source = (
        "@first\n"
        "@second(arg=1)\n"
        "class Widget(Base, metaclass=Meta):\n"
        '    """A widget."""\n'
        "\n"
        "    def render(\n"
        "        self,\n"
        "        width: int,\n"
        '        label: str = "x",\n'
        "    ) -> list[str]:\n"
        "        return []\n"
    )
    _write(tmp_path / "widgets.py", source)

    text = generate_repo_map(tmp_path, ["widgets.py"])

    assert text.splitlines() == [
        "# File: widgets.py",
        "class Widget(Base, metaclass=Meta): (line 3) # A widget.",
        'def render(self, width: int, label: str = "x") -> list[str]: (line 6)',
    ]
    assert len(text) <= 2000


def test_repo_map_truncates_large_files(tmp_path):
    body = "".join(
        f"def handler_{index}(value: dict[str, int]) -> None:\n    pass\n"
        for index in range(80)
    )
    _write(tmp_path / "big.py", body)

    text = generate_repo_map(tmp_path, ["big.py"])

    assert len(text) <= 2000
    assert text.startswith("# File: big.py")
    assert text.endswith("... (truncated)")
