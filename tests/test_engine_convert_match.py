"""Tests for the if/elif/else → match-case conversion."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from moray_surgeon.engine.convert_match import convert_to_match_case
from moray_surgeon.engine.dispatcher import execute_plan
from moray_surgeon.engine.scope_guard import snapshot_repository, verify_repository_scope
from moray_surgeon.schema import ConvertToMatchCase, RefactorPlan

_RUFF_TIMEOUT_SECONDS = 60


def _ruff_is_clean(source: str, tmp_path: Path) -> None:
    """Assert a generated module parses and passes a strict ``ruff check``.

    The fixtures are self-contained on purpose: a clean run then proves the
    conversion produced real, resolvable Python rather than a snippet that only
    parses.
    """
    ast.parse(source)  # the "must be valid Python" gate
    target = tmp_path / "generated.py"
    target.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-cache", "--isolated", str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_RUFF_TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _convert(source: str) -> str:
    """Convert the whole snippet, which must contain exactly one chain."""
    return convert_to_match_case(source, 1, len(source.splitlines())).new_source


def _convert_range(source: str, start_line: int, end_line: int) -> str:
    """Convert only the requested range."""
    return convert_to_match_case(source, start_line, end_line).new_source


# --------------------------------------------------------------------------- #
# Conversion matrix
# --------------------------------------------------------------------------- #

_VALUE_CHAIN = (
    "def describe(value):\n"
    "    if value == 1:\n"
    '        return "one"\n'
    "    elif value == 2:\n"
    '        return "two"\n'
    "    else:\n"
    '        return "many"\n'
)

_ISINSTANCE_CHAIN = (
    "def render(item):\n"
    "    if isinstance(item, str):\n"
    "        return item.strip()\n"
    "    elif isinstance(item, (int, float)):\n"
    "        return str(item)\n"
    "    else:\n"
    "        return repr(item)\n"
)

_OR_CHAIN = (
    "def bucket(code):\n"
    "    if code == 200 or code == 201:\n"
    '        return "ok"\n'
    "    else:\n"
    '        return "other"\n'
)

_CONSTANT_CHAIN = (
    "class Color:\n"
    "    RED = 1\n"
    "    BLUE = 2\n"
    "\n"
    "\n"
    "def paint():\n"
    "    return None\n"
    "\n"
    "\n"
    "def cool():\n"
    "    return None\n"
    "\n"
    "\n"
    "def apply(kind):\n"
    "    if kind == Color.RED:\n"
    "        paint()\n"
    "    elif kind == Color.BLUE:\n"
    "        cool()\n"
)


def test_converts_equality_chain(tmp_path):
    """``if value == 1`` becomes ``case 1`` and the else becomes the wildcard."""
    source = _convert(_VALUE_CHAIN)

    assert source == (
        "def describe(value):\n"
        "    match value:\n"
        "        case 1:\n"
        '            return "one"\n'
        "        case 2:\n"
        '            return "two"\n'
        "        case _:\n"
        '            return "many"\n'
    )
    _ruff_is_clean(source, tmp_path)


def test_converts_isinstance_chain(tmp_path):
    """A class test becomes a class pattern; a tuple becomes an or-pattern."""
    source = _convert(_ISINSTANCE_CHAIN)

    assert "    match item:\n" in source
    assert "        case str():\n" in source
    assert "        case int() | float():\n" in source
    assert "        case _:\n" in source
    _ruff_is_clean(source, tmp_path)


def test_converts_or_of_equalities(tmp_path):
    """``code == 200 or code == 201`` becomes ``case 200 | 201``."""
    source = _convert(_OR_CHAIN)

    assert "    match code:\n" in source
    assert "        case 200 | 201:\n" in source
    assert "        case _:\n" in source
    _ruff_is_clean(source, tmp_path)


def test_converts_dotted_constant_chain(tmp_path):
    """A dotted constant is a value pattern, not a capture pattern."""
    source = _convert(_CONSTANT_CHAIN)

    assert "    match kind:\n" in source
    assert "        case Color.RED:\n" in source
    assert "        case Color.BLUE:\n" in source
    _ruff_is_clean(source, tmp_path)


def test_converts_singletons_and_negative_numbers(tmp_path):
    """``None``/``True`` are singletons once unwrapped from ``==``."""
    source = _convert(
        "def flag(x):\n"
        "    if x == None:\n"
        "        return 0\n"
        "    elif x == True:\n"
        "        return 1\n"
        "    elif x == -1:\n"
        "        return 2\n"
    )

    assert "        case None:\n" in source
    assert "        case True:\n" in source
    assert "        case -1:\n" in source
    _ruff_is_clean(source, tmp_path)


def test_converts_mixed_value_and_class_patterns(tmp_path):
    """``x == 0 or isinstance(x, str)`` becomes a mixed or-pattern."""
    source = _convert(
        "def classify(x):\n"
        "    if x == 0 or isinstance(x, str):\n"
        '        return "special"\n'
        "    else:\n"
        '        return "plain"\n'
    )

    assert "        case 0 | str():\n" in source
    _ruff_is_clean(source, tmp_path)


def test_no_else_means_no_wildcard(tmp_path):
    """Without an else the match simply falls through, like the original."""
    source = _convert(
        "def f(x):\n"
        "    if x == 1:\n"
        "        print(x)\n"
        "    elif x == 2:\n"
        "        print(-x)\n"
    )

    assert "case _:" not in source
    assert source.count("case ") == 2
    _ruff_is_clean(source, tmp_path)


def test_comments_above_elif_survive(tmp_path):
    """A comment between branches moves onto the case it described."""
    source = _convert(
        "def f(x):\n"
        "    if x == 1:\n"
        "        print(1)\n"
        "    # explain two\n"
        "    elif x == 2:\n"
        "        print(2)\n"
        "    else:\n"
        "        print(3)\n"
    )

    assert "# explain two" in source
    assert source.index("# explain two") < source.index("case 2:")
    _ruff_is_clean(source, tmp_path)


def test_one_line_bodies_stay_one_line(tmp_path):
    """A suite written on one line keeps its shape inside the case."""
    source = _convert(
        "def f(x):\n"
        "    if x == 1: return 1\n"
        "    else: return 2\n"
    )

    assert "        case 1: return 1\n" in source
    assert "        case _: return 2\n" in source
    _ruff_is_clean(source, tmp_path)


def test_changes_counts_every_case():
    """``changes`` is the number of generated case clauses."""
    result = convert_to_match_case(_VALUE_CHAIN, 1, len(_VALUE_CHAIN.splitlines()))

    assert result.changes == 3
    assert result.new_source.count("case ") == 3


def test_only_the_target_chain_is_rewritten(tmp_path):
    """A neighbouring if and a nested if inside a branch stay as they are."""
    source = (
        "def f(x, flag):\n"
        "    if flag:\n"
        "        print('flag')\n"
        "    if x == 1:\n"
        "        if flag:\n"
        "            print('nested')\n"
        "        print('one')\n"
        "    elif x == 2:\n"
        "        print('two')\n"
    )

    converted = _convert_range(source, 4, 9)

    assert "    if flag:\n        print('flag')\n" in converted
    assert "    match x:\n" in converted
    # The nested if survives, one level deeper inside the case body.
    assert "        case 1:\n            if flag:\n                print('nested')\n" in converted
    assert converted.count("match ") == 1
    _ruff_is_clean(converted, tmp_path)


# --------------------------------------------------------------------------- #
# Refusals: never emit code that means something else
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("snippet", "message"),
    [
        ("if a and b:\n    one()\nelif c:\n    two()\n", "and"),
        ("if not a:\n    one()\n", "not"),
        ("if count != 3:\n    one()\n", "only '=='"),
        ("if count < 3:\n    one()\n", "only '=='"),
        ("if key in mapping:\n    one()\n", "only '=='"),
        ("if 1 < count < 5:\n    one()\n", "chained comparisons"),
        ("if state == FALLBACK:\n    one()\n", "capture pattern"),
        ("if x == 1:\n    one()\nelif y == 2:\n    two()\n", "same expression"),
        ("if fetch() == 1:\n    one()\n", "subject once"),
        ("if items[0] == 1:\n    one()\n", "subject once"),
        ("if x == 1:\n    x = 2\nelif x == 2:\n    two()\n", "rebinds"),
        (
            "if self.state == 1:\n    self.state = 2\nelif self.state == 2:\n    two()\n",
            "assigns to",
        ),
        ("if isinstance(x, classes):\n    one()\n", "may be a variable"),
        ("if isinstance(x, get_types()):\n    one()\n", "must name a class"),
        ("if isinstance(x):\n    one()\n", "two positional arguments"),
        ("if x == 1 + 2:\n    one()\n", "value pattern"),
        ("if x == [1, 2]:\n    one()\n", "value pattern"),
        ("if helper():\n    one()\n", "isinstance"),
    ],
)
def test_refuses_tests_that_cannot_become_patterns(snippet, message):
    with pytest.raises(ValueError, match=message):
        _convert(snippet)


def test_refuses_a_range_that_cuts_the_chain():
    source = "if x == 1:\n    a()\nelif x == 2:\n    b()\nelse:\n    c()\n"

    with pytest.raises(ValueError, match="whole statement"):
        convert_to_match_case(source, 1, 3)


def test_refuses_a_range_without_a_chain():
    with pytest.raises(ValueError, match="no complete if"):
        convert_to_match_case("value = 1\n", 1, 1)


def test_refuses_two_chains_in_one_range():
    source = "if a == 1:\n    one()\nif b == 2:\n    two()\n"

    with pytest.raises(ValueError, match="2 if chains"):
        convert_to_match_case(source, 1, 4)


@pytest.mark.parametrize(("start", "end"), [(0, 3), (3, 2), (-1, 1)])
def test_refuses_invalid_line_ranges(start, end):
    with pytest.raises(ValueError, match="line range"):
        convert_to_match_case("if x == 1:\n    a()\n", start, end)


def test_refuses_a_subject_that_is_rebound_by_a_for_loop():
    source = (
        "if x == 1:\n"
        "    for x in range(3):\n"
        "        touch(x)\n"
        "elif x == 2:\n"
        "    two()\n"
    )

    with pytest.raises(ValueError, match="assigns to 'x'"):
        _convert(source)


def test_refuses_a_subject_that_is_rebound_by_an_assignment():
    source = "if item == 1:\n    item = 3\nelif item == 2:\n    two()\n"

    with pytest.raises(ValueError, match="rebinds 'item'"):
        _convert(source)


# --------------------------------------------------------------------------- #
# More patterns and more ways a branch can rebind the subject
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("literal", "rendered"),
    [
        ("1.5", "case 1.5:"),
        ("2j", "case 2j:"),
        ('b"raw"', 'case b"raw":'),
        ("False", "case False:"),
        ('"a" "b"', 'case "a" "b":'),
    ],
)
def test_converts_every_literal_kind(tmp_path, literal, rendered):
    source = _convert(f"def f(x):\n    if x == {literal}:\n        print(1)\n")

    assert "    " + rendered in source
    _ruff_is_clean(source, tmp_path)


def test_converts_a_dotted_class_pattern(tmp_path):
    """``isinstance(x, errors.MyError)`` keeps the module path."""
    source = _convert(
        "from errors import MyError\n"
        "\n"
        "\n"
        "def f(x):\n"
        "    if x == 0:\n"
        "        print('zero')\n"
        "    elif isinstance(x, MyError):\n"
        "        print('error')\n"
    )

    assert "        case MyError():\n" in source
    _ruff_is_clean(source, tmp_path)


def test_flattens_nested_or_patterns(tmp_path):
    """A tuple class test inside an ``or`` yields one flat or-pattern."""
    source = _convert(
        "def f(x):\n"
        "    if isinstance(x, (int, float)) or x == 'auto':\n"
        "        return 1\n"
        "    else:\n"
        "        return 2\n"
    )

    assert "        case int() | float() | 'auto':\n" in source
    assert "case int() | (float() | 'auto')" not in source
    _ruff_is_clean(source, tmp_path)


def test_refuses_an_empty_class_tuple():
    with pytest.raises(ValueError, match="empty class tuple"):
        _convert("if isinstance(x, ()):\n    one()\n")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("        def x():\n            return 1\n", "rebinds 'x'"),
        ("        class x:\n            pass\n", "rebinds 'x'"),
        ("        import x\n", "rebinds 'x'"),
        ("        from mod import x\n", "rebinds 'x'"),
        ("        global x\n", "rebinds 'x'"),
        (
            "        try:\n            pass\n        except Exception as x:\n            pass\n",
            "rebinds 'x'",
        ),
        ("        x: int = 1\n", "rebinds 'x'"),
        ("        x += 1\n", "rebinds 'x'"),
        ("        for x in range(3):\n            pass\n", "assigns to 'x'"),
        ("        [x for x in range(3)]\n", "assigns to 'x'"),
        ("        (x := 1)\n", "assigns to 'x'"),
        ("        with open('f') as x:\n            pass\n", "assigns to 'x'"),
        ("        del x\n", "assigns to 'x'"),
        ("        x, y = 1, 2\n", "assigns to 'x'"),
    ],
)
def test_refuses_every_way_a_branch_rebinds_the_subject(body, message):
    """The original chain would re-read the subject; ``match`` would not."""
    source = f"def f():\n    x = 0\n    if x == 1:\n{body}    elif x == 2:\n        pass\n"

    with pytest.raises(ValueError, match=message):
        convert_to_match_case(source, 3, len(source.splitlines()))


def test_refuses_a_nonlocal_declaration():
    source = (
        "def outer():\n"
        "    x = 0\n"
        "    def inner():\n"
        "        if x == 1:\n"
        "            nonlocal x\n"
        "        elif x == 2:\n"
        "            pass\n"
    )

    with pytest.raises(ValueError, match="rebinds 'x'"):
        convert_to_match_case(source, 4, 7)


def test_refuses_a_star_import_in_a_branch():
    """``import *`` binds names the atom cannot see, so it cannot prove safety."""
    source = (
        "if x == 1:\n"
        "    from mod import *\n"
        "elif x == 2:\n"
        "    pass\n"
    )

    with pytest.raises(ValueError, match="import \\*"):
        _convert(source)


def test_refuses_an_or_expression_testing_two_subjects():
    with pytest.raises(ValueError, match="two different subjects"):
        _convert("if x == 1 or y == 2:\n    one()\n")


def test_converts_a_single_element_class_tuple():
    """``isinstance(x, (str,))`` still yields one class pattern."""
    source = _convert(
        "def f(x):\n"
        "    if isinstance(x, (str,)):\n"
        "        return 1\n"
        "    else:\n"
        "        return 2\n"
    )

    assert "        case str():\n" in source
    assert "|" not in source


def test_refuses_a_bare_truthiness_test():
    with pytest.raises(ValueError, match="only '<subject> == <literal>'"):
        _convert("if flag:\n    one()\n")


def test_converts_a_dotted_class_reference(tmp_path):
    """``case errors.MyError()`` keeps the qualified class path."""
    source = _convert(
        "class errors:\n"
        "    class MyError(Exception):\n"
        "        pass\n"
        "\n"
        "\n"
        "def f(x):\n"
        "    if isinstance(x, errors.MyError):\n"
        "        return 1\n"
        "    else:\n"
        "        return 2\n"
    )

    assert "        case errors.MyError():\n" in source
    _ruff_is_clean(source, tmp_path)


def test_refuses_a_non_constant_class_reference():
    with pytest.raises(ValueError, match="dotted constant"):
        _convert("if isinstance(x, registry().Cls):\n    one()\n")


def test_refuses_attribute_targets_through_annotated_and_augmented_assignment():
    annotated = (
        "def f(event):\n"
        "    if event.state == 1:\n"
        "        event.state: int = 2\n"
        "    elif event.state == 2:\n"
        "        pass\n"
    )
    augmented = annotated.replace("event.state: int = 2", "event.state += 1")

    for source in (annotated, augmented):
        with pytest.raises(ValueError, match=r"assigns to 'event\.state'"):
            _convert(source)


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #


def test_dispatcher_applies_the_instruction(tmp_path):
    target = tmp_path / "app.py"
    target.write_text(_VALUE_CHAIN, encoding="utf-8")
    plan = RefactorPlan(
        instructions=[
            ConvertToMatchCase(id="m1", target_file="app.py", start_line=2, end_line=7)
        ]
    )

    log = execute_plan(plan, tmp_path)

    assert log == ["[0] convert_to_match_case: 3 change(s) in app.py"]
    text = target.read_text(encoding="utf-8")
    assert "    match value:\n" in text
    assert "if value == 1" not in text


def test_scope_guard_authorises_the_conversion(tmp_path):
    """The scope gate must accept the rewritten function it was asked about."""
    target = tmp_path / "app.py"
    target.write_text(_VALUE_CHAIN, encoding="utf-8")
    plan = RefactorPlan(
        instructions=[
            ConvertToMatchCase(id="m1", target_file="app.py", start_line=2, end_line=7)
        ]
    )
    snapshot = snapshot_repository(tmp_path)

    execute_plan(plan, tmp_path)
    verify_repository_scope(snapshot, tmp_path, plan)  # must not raise


def test_dispatcher_reports_a_refusal_as_an_instruction_error(tmp_path):
    from moray_surgeon.exceptions import InstructionError

    target = tmp_path / "app.py"
    target.write_text("if a and b:\n    one()\n", encoding="utf-8")
    plan = RefactorPlan(
        instructions=[
            ConvertToMatchCase(id="m1", target_file="app.py", start_line=1, end_line=2)
        ]
    )

    with pytest.raises(InstructionError, match="and"):
        execute_plan(plan, tmp_path)
