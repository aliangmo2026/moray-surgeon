from __future__ import annotations

import textwrap

import pytest

from moray_surgeon.engine.split_class import split_class


def _src(text: str) -> str:
    """Dedent a triple-quoted fixture that starts on its own line."""
    return textwrap.dedent(text).lstrip("\n")


_MATH = _src(
    """
    class Math:
        \"\"\"Math helpers.\"\"\"

        version = 1

        # pure helper
        @staticmethod
        def add(a, b):
            \"\"\"Add two numbers.\"\"\"
            return a + b

        def scale(self, factor):
            return self.value * factor

        def describe(self):
            return f"Math v{self.version}"
    """
)


def test_split_class_moves_static_method(tmp_path) -> None:
    """The moved method keeps decorator, docstring and comments; a sibling
    class appears right after the original."""
    result = split_class(_MATH, "Math", ["add"], "MathOps")

    assert result.changes == 2
    source = result.new_source
    assert "class Math:" in source
    assert "class MathOps:" in source
    assert source.index("class Math:") < source.index("class MathOps:")
    # The method moved out of Math, verbatim.
    math_body = source[source.index("class Math:") : source.index("class MathOps:")]
    assert "def add" not in math_body
    ops_body = source[source.index("class MathOps:") :]
    assert "# pure helper" in ops_body
    assert "@staticmethod" in ops_body
    assert '"""Add two numbers."""' in ops_body
    assert "return a + b" in ops_body


def test_split_class_moves_several_methods_in_class_order(tmp_path) -> None:
    """Multiple methods move together, keeping their order in the class."""
    source = _src(
        """
        class Thing:
            def first(a):
                return a

            def second(b):
                return b

            def keep(self):
                return 1
        """
    )

    result = split_class(source, "Thing", ["second", "first"], "ThingPart")

    part = result.new_source[result.new_source.index("class ThingPart:") :]
    assert part.index("def first") < part.index("def second")
    assert "def keep" not in part


def test_split_class_inserts_pass_when_class_empties(tmp_path) -> None:
    """Moving every method leaves the original class with a ``pass`` body."""
    source = _src(
        """
        class Only:
            def alone(a):
                return a
        """
    )

    result = split_class(source, "Only", ["alone"], "OnlyOps")

    original = result.new_source[: result.new_source.index("class OnlyOps:")]
    assert "pass" in original
    assert "def alone" not in original


def test_split_class_refuses_receiver_using_method(tmp_path) -> None:
    """A method that touches ``self`` depends on instance state: refuse."""
    with pytest.raises(ValueError, match="uses its receiver 'self'"):
        split_class(_MATH, "Math", ["scale"], "MathOps")


def test_split_class_refuses_classmethod_using_cls(tmp_path) -> None:
    """A classmethod that reads ``cls`` is receiver-bound too."""
    source = _src(
        """
        class Config:
            @classmethod
            def create(cls, data):
                return cls(data)
        """
    )

    with pytest.raises(ValueError, match="uses its receiver 'cls'"):
        split_class(source, "Config", ["create"], "ConfigOps")


def test_split_class_refuses_super_call(tmp_path) -> None:
    """``super()`` would resolve a different MRO in the new class."""
    source = _src(
        """
        class Child(Parent):
            def greet(self):
                return super().greet()
        """
    )

    with pytest.raises(ValueError, match="super\\(\\)"):
        split_class(source, "Child", ["greet"], "ChildGreeter")


def test_split_class_allows_receiver_free_plain_method(tmp_path) -> None:
    """A plain method that only takes arguments may move."""
    source = _src(
        """
        class Widget:
            def combine(self, a, b):
                return a + b
        """
    )

    result = split_class(source, "Widget", ["combine"], "WidgetOps")

    assert "def combine" in result.new_source.split("class WidgetOps:")[1]


def test_split_class_refuses_missing_method(tmp_path) -> None:
    """Every requested method must exist on the class."""
    with pytest.raises(ValueError, match=r"has no method\(s\) \['nope'\]"):
        split_class(_MATH, "Math", ["nope"], "MathOps")


def test_split_class_refuses_existing_new_name(tmp_path) -> None:
    """The new class must not shadow anything already bound in the module."""
    source = _src(
        """
        class Math:
            def add(a, b):
                return a + b


        MathOps = 1
        """
    )

    with pytest.raises(ValueError, match="already bound"):
        split_class(source, "Math", ["add"], "MathOps")


def test_split_class_refuses_same_name(tmp_path) -> None:
    """Splitting a class onto itself is a no-op at best."""
    with pytest.raises(ValueError, match="must differ"):
        split_class(_MATH, "Math", ["add"], "Math")


def test_split_class_refuses_missing_class(tmp_path) -> None:
    """The source class must exist at the top level."""
    with pytest.raises(ValueError, match="no top-level class"):
        split_class(_MATH, "Missing", ["add"], "MathOps")


def test_split_class_refuses_nested_class(tmp_path) -> None:
    """Only top-level classes can be split."""
    source = _src(
        """
        class Outer:
            class Inner:
                def method(a):
                    return a
        """
    )

    with pytest.raises(ValueError, match="no top-level class"):
        split_class(source, "Inner", ["method"], "InnerOps")


def test_split_class_refuses_duplicate_method_request(tmp_path) -> None:
    """Requesting the same method twice is ambiguous."""
    with pytest.raises(ValueError, match="must not repeat"):
        split_class(_MATH, "Math", ["add", "add"], "MathOps")


def test_split_class_rejects_invalid_names(tmp_path) -> None:
    """Every name involved must be a real identifier."""
    with pytest.raises(ValueError, match="identifier"):
        split_class(_MATH, "Math", ["not a name"], "MathOps")
    with pytest.raises(ValueError, match="identifier"):
        split_class(_MATH, "Math", ["add"], "not a name")
    with pytest.raises(ValueError, match="identifier"):
        split_class(_MATH, "class", ["add"], "MathOps")
