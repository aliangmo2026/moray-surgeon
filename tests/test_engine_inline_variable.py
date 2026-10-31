from __future__ import annotations

import textwrap

import pytest

from moray_surgeon.engine.inline_variable import inline_variable


def _src(text: str) -> str:
    """Dedent a triple-quoted fixture that starts on its own line."""
    return textwrap.dedent(text).lstrip("\n")


def _run(tmp_path, source: str, variable_name: str = "x"):
    target = tmp_path / "app.py"
    target.write_text(source, encoding="utf-8")
    return inline_variable(source, variable_name, tmp_path, target)


_INLINE_GOLDEN_CASES = [
    pytest.param(
        _src(
            """
            # the invoice total
            total = price * count
            print(total)
            ship(total)
            """
        ),
        "total",
        _src(
            """
            # the invoice total
            print((price * count))
            ship((price * count))
            """
        ),
        3,
        id="reads-replaced-and-assignment-deleted-with-comment",
    ),
    pytest.param(
        _src(
            """
            tax = rate * amount
            final = tax + fee
            label = str(tax)
            """
        ),
        "tax",
        _src(
            """
            final = (rate * amount) + fee
            label = str((rate * amount))
            """
        ),
        3,
        id="non-atom-value-is-parenthesised-at-every-read",
    ),
    pytest.param(
        "x = get_value()\nuse(x)\n",
        "x",
        "use(get_value())\n",
        2,
        id="single-read-side-effecting-value-moves",
    ),
    pytest.param(
        "x = 1\ny = 2\nprint(y)\n",
        "x",
        "y = 2\nprint(y)\n",
        1,
        id="dead-store-is-just-deleted",
    ),
    pytest.param(
        _src(
            """
            def compute(price):
                markup = price + 1
                if markup > 10:
                    return markup
                return price
            """
        ),
        "markup",
        _src(
            """
            def compute(price):
                if (price + 1) > 10:
                    return (price + 1)
                return price
            """
        ),
        3,
        id="function-scope-inline",
    ),
    pytest.param(
        "x = 1; y = 2\nprint(x + y)\n",
        "x",
        "y = 2\nprint(1 + y)\n",
        2,
        id="semicolon-line-keeps-its-other-statement",
    ),
]


@pytest.mark.parametrize(
    ("source", "variable_name", "expected", "changes"),
    _INLINE_GOLDEN_CASES,
)
def test_inline_variable_golden(tmp_path, source, variable_name, expected, changes) -> None:
    """Every read becomes the value and the assignment line disappears."""
    result = _run(tmp_path, source, variable_name)

    assert result.new_source == expected
    assert result.changes == changes


def test_inline_variable_unrelated_same_name_in_other_file_is_untouched(tmp_path) -> None:
    """A different module binding the same name is not a reference to ours."""
    source = "x = 1\nprint(x)\n"
    (tmp_path / "other.py").write_text("x = 5\nprint(x)\n", encoding="utf-8")

    result = _run(tmp_path, source)

    assert result.new_source == "print(1)\n"


@pytest.mark.parametrize(
    ("source", "match"),
    [
        pytest.param(
            "x = 1\nprint(x)\nx = 2\n",
            "exactly one",
            id="reassigned",
        ),
        pytest.param(
            "x = 1\ndef f():\n    return x\n",
            "nested function or class",
            id="closure-capture",
        ),
        pytest.param(
            "print(x)\nx = 1\n",
            "cannot resolve to any binding",
            id="read-before-assignment",
        ),
        pytest.param(
            "x = x + 1\nprint(x)\n",
            "cannot resolve to any binding",
            id="self-reference",
        ),
        pytest.param(
            "x = get()\nprint(x)\nprint(x)\n",
            "side effect per read",
            id="impure-value-read-twice",
        ),
        pytest.param(
            "a = 1\nx = a + 1\nprint(x)\na = 2\nprint(x)\n",
            "is rebound between",
            id="free-name-rebound-before-a-read",
        ),
        pytest.param(
            "x = 1\nprint(x)\ndel x\n",
            "is deleted",
            id="deleted",
        ),
        pytest.param(
            "x: int = 5\nprint(x)\n",
            "no simple assignment",
            id="annotated-assignment-only",
        ),
        pytest.param(
            "y = 1\nprint(y)\n",
            "no simple assignment",
            id="name-never-assigned",
        ),
        pytest.param(
            "if cond: x = 1\nprint(x)\n",
            "one-line suite",
            id="assignment-inside-one-line-suite",
        ),
        pytest.param(
            "x = 1\nprint(x)\n",
            "variable_name",
            id="invalid-name",
        ),
    ],
)
def test_inline_variable_refuses_unsafe_shapes(tmp_path, source, match) -> None:
    """Anything ambiguous raises ValueError and leaves the source alone."""
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, source, "x" if match != "variable_name" else "not a name")


def test_inline_variable_refuses_cross_file_import(tmp_path) -> None:
    """``from app import x`` in another module blocks the inline."""
    source = "x = 1\nprint(x)\n"
    (tmp_path / "other.py").write_text("from app import x\nprint(x)\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"referenced from other\.py"):
        _run(tmp_path, source)


def test_inline_variable_refuses_cross_file_attribute(tmp_path) -> None:
    """``import app`` + ``app.x`` in another module blocks the inline."""
    source = "x = 1\nprint(x)\n"
    (tmp_path / "other.py").write_text("import app\nprint(app.x)\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"referenced from other\.py"):
        _run(tmp_path, source)


def test_inline_variable_refuses_relative_import_binding(tmp_path) -> None:
    """``from . import x`` may re-export the package namespace: refuse."""
    source = "x = 1\nprint(x)\n"
    (tmp_path / "other.py").write_text("from . import x\nprint(x)\n", encoding="utf-8")

    with pytest.raises(ValueError, match="relative import"):
        _run(tmp_path, source)


def test_inline_variable_refuses_unresolvable_read_in_other_module(tmp_path) -> None:
    """A same-name read Jedi cannot resolve is treated as a use: fail closed."""
    source = "x = 1\nprint(x)\n"
    (tmp_path / "other.py").write_text("from helpers import *\nprint(x)\n", encoding="utf-8")
    (tmp_path / "helpers.py").write_text("unrelated = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be resolved"):
        _run(tmp_path, source)


def test_inline_variable_refuses_lambda_param_shadowing(tmp_path) -> None:
    """A read inside a lambda is refused outright, shadowing or not.

    libcst scopes lambda bodies separately, so the closure guard fires first;
    either refusal keeps the inline from evaluating the value's names against
    the lambda's parameters.
    """
    source = "factor = base + 1\ng = lambda base: base * factor\n"

    with pytest.raises(ValueError, match="nested function or class"):
        _run(tmp_path, source, "factor")


def test_inline_variable_refuses_comprehension_target_shadowing(tmp_path) -> None:
    """A comprehension target that shadows a free name of the value blocks it."""
    source = "rate = base + 1\nout = [base * rate for base in items]\n"

    with pytest.raises(ValueError, match="binds names also used"):
        _run(tmp_path, source, "rate")


def test_inline_variable_allows_nested_comprehension_reads(tmp_path) -> None:
    """Reads through nested comprehensions are safe when nothing shadows."""
    source = "rate = 2\nout = [[v * rate for v in row] for row in rows]\n"

    result = _run(tmp_path, source, "rate")

    assert result.new_source == "out = [[v * 2 for v in row] for row in rows]\n"


def test_inline_variable_allows_multi_for_comprehension_reads(tmp_path) -> None:
    """``for`` clauses chained inside one comprehension are walked too."""
    source = "rate = 2\nout = [v * rate for row in rows for v in row]\n"

    result = _run(tmp_path, source, "rate")

    assert result.new_source == "out = [v * 2 for row in rows for v in row]\n"


def test_inline_variable_drops_comments_when_last_statement(tmp_path) -> None:
    """Deleting the block's final assignment leaves no comment dangling."""
    source = "def f():\n    y = 2\n    x = y + 1\n"

    result = _run(tmp_path, source, "x")

    assert result.new_source == "def f():\n    y = 2\n"
