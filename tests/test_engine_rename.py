from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError
from pathlib import Path

import jedi
import pytest

from moray_surgeon import engine as engine_package
from moray_surgeon.engine.base import EditResult
from moray_surgeon.engine.rename import rename_symbol
from moray_surgeon.exceptions import PathTraversalError

_CROSS_A = (
    "class User:\n"
    '    """Represents a user."""\n'
    "\n"
    "    def __init__(self, name: str):\n"
    "        self.name = name\n"
    "\n"
    "    def get_name(self) -> str:\n"
    "        return self.name\n"
    "\n"
    "\n"
    "def make() -> User:\n"
    '    return User("x")\n'
)
_CROSS_B = (
    "from a import User\n"
    "\n"
    'user = User("bob")\n'
    "label = user.get_name()\n"
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_rename_class_across_files(tmp_path):
    a = _write(tmp_path / "a.py", _CROSS_A)
    b = _write(tmp_path / "b.py", _CROSS_B)

    result = rename_symbol(tmp_path, "a.py", "User", "Account")

    a_text = _read(a)
    b_text = _read(b)
    assert "class Account:" in a_text
    assert "def make() -> Account:" in a_text
    assert 'return Account("x")' in a_text
    assert "from a import Account" in b_text
    assert 'user = Account("bob")' in b_text
    assert "label = user.get_name()" in b_text
    assert "User" not in a_text
    assert "User" not in b_text
    assert result.new_source == a_text
    assert result.changes == 5


def test_local_variable_rename_does_not_touch_other_files(tmp_path):
    a = _write(tmp_path / "a.py", "def compute():\n    value = 1\n    return value\n")
    b = _write(tmp_path / "b.py", "value = 99\n")

    result = rename_symbol(tmp_path, "a.py", "value", "renamed")

    assert _read(a) == "def compute():\n    renamed = 1\n    return renamed\n"
    assert _read(b) == "value = 99\n"
    assert result.changes == 2


def test_rename_covers_more_files_than_jedis_default_parse_limit(tmp_path):
    """Jedi stops project reference search after 30 parsed files.

    The engine must raise that budget for the duration of the search; otherwise
    a project with more than 30 modules silently keeps half of its references.
    """
    _write(tmp_path / "shared.py", "def target():\n    return 1\n")
    modules = []
    for index in range(32):
        path = _write(
            tmp_path / f"m{index:02d}.py",
            "from shared import target\n"
            "\n"
            "\n"
            f"def use_{index}():\n"
            "    return target()\n",
        )
        modules.append(path)

    result = rename_symbol(tmp_path, "shared.py", "target", "renamed")

    assert result.changes == 65
    for path in modules:
        text = _read(path)
        assert "from shared import renamed" in text
        assert "return renamed()" in text
        assert "target" not in text.replace("target_file", "")


def test_rename_class_leaves_attributes_and_kwargs_alone(tmp_path):
    source = (
        "class User:\n"
        "    pass\n"
        "\n"
        "\n"
        "def touch(obj):\n"
        "    return obj.User\n"
        "\n"
        "\n"
        "def call(**kwargs):\n"
        "    return kwargs\n"
        "\n"
        "\n"
        "call(User=1)\n"
        "user = User()\n"
    )
    target = _write(tmp_path / "a.py", source)

    result = rename_symbol(tmp_path, "a.py", "User", "Account")

    text = _read(target)
    assert "class Account:" in text
    assert "user = Account()" in text
    assert "return obj.User" in text
    assert "call(User=1)" in text
    assert result.changes == 2


def test_rename_preserves_comments_docstrings_and_strings(tmp_path):
    source = (
        "# keep me\n"
        "class A:\n"
        '    """A is great."""\n'
        "    pass\n"
        "\n"
        "\n"
        "value = A()  # trailing\n"
        'text = "A"\n'
    )
    target = _write(tmp_path / "a.py", source)

    result = rename_symbol(tmp_path, "a.py", "A", "B")

    text = _read(target)
    assert "# keep me" in text
    assert '"""A is great."""' in text
    assert 'text = "A"' in text
    assert "class B:" in text
    assert "value = B()  # trailing" in text
    assert result.changes == 2


def test_missing_symbol_is_noop(tmp_path):
    target = _write(tmp_path / "a.py", "value = 1\n")

    result = rename_symbol(tmp_path, "a.py", "missing", "other")

    assert result.changes == 0
    assert result.new_source == "value = 1\n"
    assert _read(target) == "value = 1\n"


def test_falls_back_to_textual_rename_when_jedi_fails(tmp_path, monkeypatch, caplog):
    target = _write(
        tmp_path / "a.py",
        "def old_name():\n    return old_name()\n\n\nprint(old_name())\n",
    )

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("jedi exploded")

    monkeypatch.setattr(jedi, "Script", boom)

    with caplog.at_level(logging.WARNING):
        result = rename_symbol(tmp_path, "a.py", "old_name", "new_name")

    assert result.changes == 3
    assert "new_name" in _read(target)
    assert "falling back" in caplog.text


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    [
        ("A", "A"),
        ("", "B"),
        ("A", ""),
        ("not valid", "B"),
        ("A", "class"),
    ],
)
def test_rename_symbol_rejects_invalid_names(tmp_path, old_name, new_name):
    _write(tmp_path / "a.py", "A = 1\n")

    with pytest.raises(ValueError):
        rename_symbol(tmp_path, "a.py", old_name, new_name)


def test_rejects_target_outside_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(PathTraversalError):
        rename_symbol(repo, "../outside.py", "A", "B")


def test_missing_target_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        rename_symbol(tmp_path, "missing.py", "A", "B")


def test_edit_result_rejects_negative_changes() -> None:
    """EditResult guards the audit counter at construction time."""
    with pytest.raises(ValueError):
        EditResult(new_source="", changes=-1)


def test_edit_result_is_frozen() -> None:
    """Results are immutable so callers cannot rewrite an audit record."""
    result = EditResult(new_source="x = 1\n", changes=0)

    with pytest.raises(FrozenInstanceError):
        result.changes = 1


def test_engine_package_exposes_the_atoms_and_dispatcher() -> None:
    """The package re-exports every atom, the repo map and the executor."""
    assert set(engine_package.__all__) == {
        "add_import",
        "add_type_hints",
        "execute_plan",
        "extract_function",
        "extract_variable",
        "generate_repo_map",
        "get_ast_fingerprint",
        "inline_function",
        "inline_variable",
        "move_symbol",
        "split_class",
        "remove_dead_code",
        "remove_import",
        "rename_symbol",
        "replace_call",
        "snapshot_repository",
        "topological_sort",
        "verify_repository_scope",
        "verify_scope",
    }
    for name in engine_package.__all__:
        assert callable(getattr(engine_package, name))
