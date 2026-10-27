from __future__ import annotations

import textwrap

import pytest

from moray_surgeon.engine.type_hints import add_type_hints


def _src(text: str) -> str:
    """Dedent a triple-quoted fixture that starts on its own line."""
    return textwrap.dedent(text).lstrip("\n")


_TYPE_HINT_GOLDEN_CASES = [
    pytest.param(
        _src(
            """
            def compute(value):
                # keep this
                return value * 2
            """
        ),
        None,
        _src(
            """
            def compute(value) -> None:
                # keep this
                return value * 2
            """
        ),
        1,
        id="single-function-keeps-inline-comment",
    ),
    pytest.param(
        _src(
            """
            def first():
                pass


            def second():
                pass
            """
        ),
        "second",
        _src(
            """
            def first():
                pass


            def second() -> None:
                pass
            """
        ),
        1,
        id="only-the-named-function",
    ),
    pytest.param(
        _src(
            """
            async def fetch():
                pass


            def outer():
                def inner():
                    pass
                return inner
            """
        ),
        None,
        _src(
            """
            async def fetch() -> None:
                pass


            def outer() -> None:
                def inner() -> None:
                    pass
                return inner
            """
        ),
        3,
        id="async-and-nested-functions",
    ),
    pytest.param(
        _src(
            """
            def known() -> int:
                return 1


            def unknown():
                pass
            """
        ),
        None,
        _src(
            """
            def known() -> int:
                return 1


            def unknown() -> None:
                pass
            """
        ),
        1,
        id="existing-return-annotation-is-not-counted",
    ),
    pytest.param(
        _src(
            """
            class Service:
                def start(self):
                    self.started = True


            @cache
            def handler():
                pass
            """
        ),
        None,
        _src(
            """
            class Service:
                def start(self) -> None:
                    self.started = True


            @cache
            def handler() -> None:
                pass
            """
        ),
        2,
        id="method-and-decorated-function",
    ),
]


@pytest.mark.parametrize(
    ("source", "function_name", "expected", "changes"),
    _TYPE_HINT_GOLDEN_CASES,
)
def test_add_type_hints_golden(source, function_name, expected, changes) -> None:
    """Only missing return hints change; everything else round-trips verbatim."""
    result = add_type_hints(source, function_name)

    assert result.new_source == expected
    assert result.changes == changes


def test_add_type_hints_all_annotated_is_noop() -> None:
    """A fully annotated module is untouched and reports zero changes."""
    source = _src(
        """
        def known() -> int:
            return 1


        def explicit() -> None:
            pass
        """
    )

    result = add_type_hints(source)

    assert result.changes == 0
    assert result.new_source == source


def test_add_type_hints_missing_named_function_is_noop() -> None:
    """An unknown target name silently matches nothing, like already-annotated code."""
    source = "def present():\n    pass\n"

    result = add_type_hints(source, "absent")

    assert result.changes == 0
    assert result.new_source == source


@pytest.mark.parametrize("function_name", ["", "   "])
def test_add_type_hints_rejects_empty_function_name(function_name) -> None:
    """An empty name is a malformed instruction, not a wildcard."""
    with pytest.raises(ValueError):
        add_type_hints("def f():\n    pass\n", function_name)
