from __future__ import annotations

import textwrap

import pytest

from moray_surgeon.engine.extract import extract_function


def _src(text: str) -> str:
    """Dedent a triple-quoted fixture that starts on its own line."""
    return textwrap.dedent(text).lstrip("\n")


_EXTRACT_GOLDEN_CASES = [
    pytest.param(
        _src(
            """
            import os

            # Compute totals.
            x = 1
            y = 2

            print(x + y)
            """
        ),
        "compute",
        4,
        5,
        _src(
            """
            import os

            # Compute totals.
            def compute():
                x = 1
                y = 2

            compute()

            print(x + y)
            """
        ),
        2,
        id="leading-comment-and-blank-lines",
    ),
    pytest.param(
        _src(
            """
            class Widget:
                # class comment
                name = "widget"

                def render(self):
                    return self.name


            print(Widget)
            """
        ),
        "build_widget",
        1,
        6,
        _src(
            """
            def build_widget():
                class Widget:
                    # class comment
                    name = "widget"

                    def render(self):
                        return self.name

            build_widget()


            print(Widget)
            """
        ),
        1,
        id="class-body-is-reindented",
    ),
    pytest.param(
        _src(
            """
            config = {
                "a": 1,  # keep
                "b": 2,
            }
            print(config)
            """
        ),
        "load_config",
        1,
        4,
        _src(
            """
            def load_config():
                config = {
                    "a": 1,  # keep
                    "b": 2,
                }

            load_config()
            print(config)
            """
        ),
        1,
        id="multiline-statement-and-inline-comment",
    ),
    pytest.param(
        _src(
            """
            class C:
              x = 1

            y = 2
            """
        ),
        "g",
        4,
        4,
        _src(
            """
            class C:
              x = 1

            def g():
              y = 2

            g()
            """
        ),
        1,
        id="two-space-module-indent-is-kept",
    ),
    pytest.param(
        _src(
            """
            # helper docs
            def helper(value):
                return value + 1


            result = helper(1)
            """
        ),
        "wrapped",
        2,
        3,
        _src(
            """
            # helper docs
            def wrapped():
                def helper(value):
                    return value + 1

            wrapped()


            result = helper(1)
            """
        ),
        1,
        id="nested-function-def-with-comment",
    ),
]


@pytest.mark.parametrize(
    ("source", "function_name", "start_line", "end_line", "expected", "changes"),
    _EXTRACT_GOLDEN_CASES,
)
def test_extract_function_golden(
    source,
    function_name,
    start_line,
    end_line,
    expected,
    changes,
) -> None:
    """The block is replaced in place by a def plus a call, formatting intact."""
    result = extract_function(source, function_name, start_line, end_line)

    assert result.new_source == expected
    assert result.changes == changes


@pytest.mark.parametrize(("start_line", "end_line"), [(3, 2), (0, 1), (1, 0)])
def test_extract_function_rejects_invalid_range(start_line, end_line) -> None:
    """Ranges must be positive and ordered."""
    with pytest.raises(ValueError):
        extract_function("x = 1\n", "helper", start_line, end_line)


def test_extract_function_requires_full_statement_containment() -> None:
    """A range covering only part of a multi-line statement matches nothing."""
    source = _src(
        """
        payload = build(
            1,
            2,
        )
        """
    )

    with pytest.raises(ValueError, match="fully inside"):
        extract_function(source, "helper", 2, 2)


def test_extract_function_rejects_range_beyond_end_of_file() -> None:
    """Line numbers past EOF cannot select any statement."""
    with pytest.raises(ValueError, match="fully inside"):
        extract_function("x = 1\n", "helper", 10, 20)


def test_extract_function_never_extracts_nested_scope() -> None:
    """Nested statements are invisible: only Module.body is inspected."""
    source = _src(
        """
        def outer():
            x = 1
            y = 2
        """
    )

    with pytest.raises(ValueError, match="fully inside"):
        extract_function(source, "helper", 2, 3)


def test_extract_function_refuses_hole_between_selected_statements() -> None:
    """An if/for/try between two selected statements would be reordered: refuse."""
    source = _src(
        """
        before = 1
        if flag:
            middle = 2
        after = 3
        """
    )

    with pytest.raises(ValueError, match="non-extractable"):
        extract_function(source, "helper", 1, 4)


def test_extract_function_rejects_range_with_only_non_extractable_stmt() -> None:
    """A module-level if is not one of the three supported statement kinds."""
    source = _src(
        """
        if flag:
            value = 1
        """
    )

    with pytest.raises(ValueError, match="fully inside"):
        extract_function(source, "helper", 1, 2)


@pytest.mark.parametrize("function_name", ["", "   ", "class", "not valid"])
def test_extract_function_rejects_invalid_function_name(function_name) -> None:
    """The generated def must be syntactically valid, so names are validated."""
    with pytest.raises(ValueError):
        extract_function("x = 1\n", function_name, 1, 1)
