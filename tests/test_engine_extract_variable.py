from __future__ import annotations

import textwrap

import pytest

from moray_surgeon.engine.extract_variable import extract_variable


def _src(text: str) -> str:
    """Dedent a triple-quoted fixture that starts on its own line."""
    return textwrap.dedent(text).lstrip("\n")


_EXTRACT_GOLDEN_CASES = [
    pytest.param(
        _src(
            """
            import os

            # Compute the price.
            print(price * count)
            """
        ),
        (4, 6, 4, 19),
        "unit",
        _src(
            """
            import os

            # Compute the price.
            unit = price * count
            print(unit)
            """
        ),
        2,
        id="module-level-comment-stays-put",
    ),
    pytest.param(
        _src(
            """
            def render():
                total = 0
                # add the tax
                print(total + tax)
                return total
            """
        ),
        (4, 10, 4, 21),
        "subtotal",
        _src(
            """
            def render():
                total = 0
                # add the tax
                subtotal = total + tax
                print(subtotal)
                return total
            """
        ),
        2,
        id="function-body-keeps-indentation-and-comment",
    ),
    pytest.param(
        _src(
            """
            report = build(
                "data.csv",
                style="wide",
            )
            print(report)
            """
        ),
        (2, 4, 2, 14),
        "source",
        _src(
            """
            source = "data.csv"
            report = build(
                source,
                style="wide",
            )
            print(report)
            """
        ),
        2,
        id="multi-line-statement-inserts-above-it",
    ),
    pytest.param(
        _src(
            """
            def ship(weight, *, insured=False):
                return weight

            ship(12, insured=price > 100)
            """
        ),
        (4, 17, 4, 28),
        "insured_now",
        _src(
            """
            def ship(weight, *, insured=False):
                return weight

            insured_now = price > 100
            ship(12, insured=insured_now)
            """
        ),
        2,
        id="keyword-argument-value",
    ),
    pytest.param(
        _src(
            """
            class Widget:
                parts = load()

                def size(self):
                    return self.length * self.width
            """
        ),
        (5, 15, 5, 39),
        "area",
        _src(
            """
            class Widget:
                parts = load()

                def size(self):
                    area = self.length * self.width
                    return area
            """
        ),
        2,
        id="class-method-body-two-statements-indented",
    ),
]


@pytest.mark.parametrize(
    ("source", "span", "variable_name", "expected", "changes"),
    _EXTRACT_GOLDEN_CASES,
)
def test_extract_variable_golden(source, span, variable_name, expected, changes) -> None:
    """The expression is hoisted above its statement; every other byte survives."""
    start_line, start_column, end_line, end_column = span
    result = extract_variable(
        source, start_line, start_column, end_line, end_column, variable_name
    )

    assert result.new_source == expected
    assert result.changes == changes


def test_extract_variable_result_round_trips_through_parser() -> None:
    """The rewritten source must still be a valid Python module."""
    import libcst

    result = extract_variable("y = a + b\nprint(y)\n", 1, 4, 1, 9, "value")

    assert libcst.parse_module(result.new_source) is not None


@pytest.mark.parametrize(
    ("source", "span"),
    [
        # end column one short of the expression's exclusive end
        pytest.param("print(price * count)\n", (1, 6, 1, 18), id="end-column-too-early"),
        # start inside the expression
        pytest.param("print(price * count)\n", (1, 7, 1, 19), id="start-column-too-late"),
        # beyond end of file
        pytest.param("y = a\n", (9, 0, 9, 1), id="past-end-of-file"),
        # a statement span, not an expression span
        pytest.param("y = a + b\n", (1, 0, 1, 9), id="whole-statement-is-not-an-expression"),
    ],
)
def test_extract_variable_rejects_range_that_matches_nothing(source, span) -> None:
    """Only an exact whole-expression span is accepted."""
    start_line, start_column, end_line, end_column = span

    with pytest.raises(ValueError, match="no expression occupies exactly"):
        extract_variable(source, start_line, start_column, end_line, end_column, "value")


def test_extract_variable_rejects_unordered_coordinates() -> None:
    """start must not come after end."""
    with pytest.raises(ValueError, match="must not come after"):
        extract_variable("y = a + b\n", 1, 9, 1, 4, "value")


@pytest.mark.parametrize("variable_name", ["", "not valid", "class", "3rd"])
def test_extract_variable_rejects_invalid_name(variable_name) -> None:
    """The generated assignment must be syntactically valid."""
    with pytest.raises(ValueError, match="variable_name"):
        extract_variable("y = a + b\n", 1, 4, 1, 9, variable_name)


def test_extract_variable_refuses_assignment_target() -> None:
    """Extracting an assignment target would rewrite a binding, not a value."""
    source = "existing = a + b\nprint(existing)\n"

    with pytest.raises(ValueError, match="assignment target"):
        extract_variable(source, 1, 0, 1, 8, "value")


def test_extract_variable_refuses_attribute_name() -> None:
    """``self.length``'s attribute name is not a value."""
    source = "def size(self):\n    return self.length\n"

    with pytest.raises(ValueError, match="attribute name"):
        extract_variable(source, 2, 16, 2, 22, "value")


def test_extract_variable_refuses_walrus() -> None:
    """A named expression binds a name; hoisting it moves the binding."""
    source = "y = (flag := True)\nprint(y, flag)\n"

    with pytest.raises(ValueError, match="NamedExpr"):
        extract_variable(source, 1, 5, 1, 17, "value")


def test_extract_variable_refuses_lambda_body() -> None:
    """A piece inside a lambda resolves its names in the lambda's scope."""
    source = "f = lambda v: v * factor\n"

    with pytest.raises(ValueError, match="lambda or comprehension"):
        extract_variable(source, 1, 14, 1, 24, "value")


def test_extract_variable_refuses_while_test() -> None:
    """A while test re-evaluates; hoisting it out would freeze the condition."""
    source = "while value > 0:\n    value -= 1\n"

    with pytest.raises(ValueError, match="while test"):
        extract_variable(source, 1, 6, 1, 15, "value")


def test_extract_variable_refuses_except_type() -> None:
    """An except type is evaluated lazily, only when an exception arrives."""
    source = "try:\n    pass\nexcept (ValueError, KeyError) as exc:\n    pass\n"

    with pytest.raises(ValueError, match="except type"):
        extract_variable(source, 3, 8, 3, 28, "value")


def test_extract_variable_refuses_match_pattern() -> None:
    """Match patterns evaluate while the subject is being matched."""
    source = "match subject:\n    case 0:\n        pass\n"

    with pytest.raises(ValueError, match="match pattern"):
        extract_variable(source, 2, 9, 2, 10, "value")


@pytest.mark.parametrize(
    ("source", "span", "reason"),
    [
        pytest.param(
            "value += 1\n",
            (1, 0, 1, 5),
            "augmented-assignment target",
            id="augassign-target",
        ),
        pytest.param(
            "value: int = load()\n",
            (1, 0, 1, 5),
            "annotated-assignment target",
            id="annassign-target",
        ),
        pytest.param(
            "for item in items:\n    pass\n",
            (1, 4, 1, 8),
            "for-loop target",
            id="for-target",
        ),
        pytest.param(
            "result = (flag := True)\nprint(flag)\n",
            (1, 10, 1, 14),
            "named-expression target",
            id="walrus-target",
        ),
        pytest.param(
            "ship(insured=True)\n",
            (1, 5, 1, 12),
            "keyword-argument name",
            id="keyword-name",
        ),
        pytest.param(
            "def load(value):\n    return value\n",
            (1, 9, 1, 14),
            "parameter name",
            id="param-name",
        ),
        pytest.param(
            "def handler():\n    pass\n",
            (1, 4, 1, 11),
            "definition name",
            id="function-name",
        ),
        pytest.param(
            "import os\n",
            (1, 7, 1, 9),
            "import name",
            id="import-name",
        ),
        pytest.param(
            "try:\n    pass\nexcept ValueError as exc:\n    pass\n",
            (3, 21, 3, 24),
            "as-binding name",
            id="except-name",
        ),
        pytest.param(
            "match subject:\n    case found:\n        pass\n",
            (2, 9, 2, 14),
            "match pattern",
            id="match-as-capture",
        ),
        pytest.param(
            "match items:\n    case [*rest]:\n        pass\n",
            (2, 11, 2, 15),
            "match pattern",
            id="match-star-capture",
        ),
        pytest.param(
            "match point:\n    case {**rest}:\n        pass\n",
            (2, 12, 2, 16),
            "match pattern",
            id="match-mapping-rest",
        ),
        pytest.param(
            "global counter\ncounter = 1\n",
            (1, 7, 1, 14),
            "global/nonlocal declaration",
            id="global-name",
        ),
        pytest.param(
            "def outer():\n    count = 0\n    def inner():\n        nonlocal count\n",
            (4, 17, 4, 22),
            "global/nonlocal declaration",
            id="nonlocal-name",
        ),
        pytest.param(
            "cache = {}\ndel cache[key]\n",
            (2, 4, 2, 9),
            "del operand",
            id="del-operand",
        ),
    ],
)
def test_extract_variable_refuses_binding_positions(source, span, reason) -> None:
    """Every binding position is named in the refusal, never rewritten."""
    start_line, start_column, end_line, end_column = span

    with pytest.raises(ValueError, match=reason):
        extract_variable(source, start_line, start_column, end_line, end_column, "value")


def test_extract_variable_refuses_one_line_suite() -> None:
    """A one-line while body re-evaluates too; hoisting would run it once."""
    source = "while waiting: print(value)\n"

    with pytest.raises(ValueError, match="one-line suite"):
        extract_variable(source, 1, 21, 1, 26, "value")
