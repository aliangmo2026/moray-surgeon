from __future__ import annotations

import textwrap

import pytest

from moray_surgeon.engine.remove_dead_code import remove_dead_code


def _src(text: str) -> str:
    """Dedent a triple-quoted fixture that starts on its own line."""
    return textwrap.dedent(text).lstrip("\n")


def _run(tmp_path, source: str, symbol_name: str = "dead_thing", extra_files=()):
    (tmp_path / "app.py").write_text(source, encoding="utf-8")
    for name, content in extra_files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return remove_dead_code(source, symbol_name, tmp_path, tmp_path / "app.py")


def test_remove_dead_code_golden(tmp_path) -> None:
    """An unreferenced top-level function is deleted with its comment."""
    source = _src(
        """
        # compute tax
        def compute_tax(value):
            return value * 2


        def live():
            return 1
        """
    )

    result = _run(tmp_path, source, "compute_tax")

    # The comment above the first statement lives in the module header and is
    # handed to the next surviving statement instead of being destroyed.
    assert result.new_source == "# compute tax\ndef live():\n    return 1\n"
    assert result.changes == 1


def test_remove_dead_code_deletes_class_with_docstring(tmp_path) -> None:
    """A dead class goes away, docstring included."""
    source = _src(
        """
        class Widget:
            \"\"\"Does nothing useful.\"\"\"

            size = 1


        def live():
            return Widget.size
        """
    )
    # Widget.size is a live reference: keep it alive by reading it differently.
    source = source.replace("return Widget.size", "return 1")

    result = _run(tmp_path, source, "Widget")

    assert result.new_source == "def live():\n    return 1\n"


def test_remove_dead_code_allows_self_recursion(tmp_path) -> None:
    """A reference from inside the symbol dies with it."""
    source = _src(
        """
        def fact(value):
            return fact(value - 1)


        def live():
            return 1
        """
    )

    result = _run(tmp_path, source, "fact")

    assert result.new_source == "def live():\n    return 1\n"


def test_remove_dead_code_allows_prose_in_other_docstrings(tmp_path) -> None:
    """Docstring prose mentioning the name is not a reference."""
    source = _src(
        """
        def live():
            \"\"\"The dead helper used to live here.\"\"\"

            return 1


        def helper():
            return 2
        """
    )

    result = _run(tmp_path, source, "helper")

    assert "def helper():" not in result.new_source
    assert '"""The dead helper used to live here."""' in result.new_source


def test_remove_dead_code_transfers_header_comments(tmp_path) -> None:
    """Comments above the first statement live in the module header, not the node."""
    source = (
        "#!/usr/bin/env python\n# license: MIT\nimport os\n\n\ndef dead():\n    return 1\n"
        "\n\ndef live():\n    return os.sep\n"
    )

    result = _run(tmp_path, source, "dead")

    expected = (
        "#!/usr/bin/env python\n# license: MIT\nimport os\n\n\ndef live():\n    return os.sep\n"
    )
    assert result.new_source == expected


def test_remove_dead_code_empty_module(tmp_path) -> None:
    """Deleting the only statement leaves nothing behind."""
    result = _run(tmp_path, "def gone():\n    return 1\n", "gone")

    assert result.new_source.strip() == ""
    assert result.changes == 1


@pytest.mark.parametrize(
    ("source", "extra_files", "match"),
    [
        pytest.param(
            _src(
                """
                def dead_thing():
                    return 1


                def live():
                    return dead_thing()
                """
            ),
            (),
            "referenced at app.py",
            id="same-file-caller",
        ),
        pytest.param(
            "def dead_thing():\n    return 1\n",
            (("other.py", "from app import dead_thing\nprint(dead_thing)\n"),),
            "referenced at other.py",
            id="cross-file-import",
        ),
        pytest.param(
            "def dead_thing():\n    return 1\n",
            (("other.py", "import app\nprint(app.dead_thing)\n"),),
            "referenced at other.py",
            id="cross-file-attribute",
        ),
        pytest.param(
            '__all__ = ["dead_thing"]\n\n\ndef dead_thing():\n    return 1\n',
            (),
            "exported by __all__",
            id="all-export",
        ),
        pytest.param(
            '__all__ += ["dead_thing"]\n\n\ndef dead_thing():\n    return 1\n',
            (),
            "exported by __all__",
            id="all-augmented-export",
        ),
        pytest.param(
            "def dead_thing():\n    return 1\n\n\n__all__.append('other')\n",
            (),
            "manipulated dynamically",
            id="all-dynamic-manipulation",
        ),
        pytest.param(
            'def dead_thing():\n    return 1\n\n\nkey = "dead_thing"\n',
            (),
            "dynamic reference",
            id="getattr-string",
        ),
        pytest.param(
            "def dead_thing():\n    return 1\n",
            (("other.py", "name = 'dead_thing'\n"),),
            "dynamic reference",
            id="getattr-string-other-file",
        ),
        pytest.param(
            "import functools\n\n\n@functools.cache\ndef dead_thing():\n    return 1\n",
            (),
            "has decorators",
            id="decorated",
        ),
        pytest.param(
            "def dead_thing():\n    return 1\n",
            (("other.py", "from app import *\n"),),
            "star-imports",
            id="star-import",
        ),
        pytest.param(
            "def dead_thing():\n    return 1\n",
            (("other.py", "from . import *\n"),),
            "cannot be traced",
            id="relative-star-import",
        ),
        pytest.param(
            "def live():\n    return 1\n",
            (),
            "no top-level function or class",
            id="symbol-missing",
        ),
        pytest.param(
            "def dead_thing():\n    pass\n\n\ndef dead_thing():\n    pass\n",
            (),
            "defined 2 times",
            id="duplicate-definition",
        ),
        pytest.param(
            "def dead_thing():\n    return 1\n",
            (("other.py", "value = dead_thing\n"),),
            "cannot be resolved",
            id="unresolved-name-in-other-file",
        ),
    ],
)
def test_remove_dead_code_refuses_unsafe_deletions(tmp_path, source, extra_files, match) -> None:
    """Every shadow of doubt refuses the deletion; the file stays untouched."""
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, source, extra_files=extra_files)


def test_remove_dead_code_invalid_name(tmp_path) -> None:
    """The lookup target must be a real identifier."""
    with pytest.raises(ValueError, match="symbol_name"):
        _run(tmp_path, "x = 1\n", "not a name")
