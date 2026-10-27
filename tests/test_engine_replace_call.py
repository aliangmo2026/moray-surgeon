from __future__ import annotations

import textwrap

import pytest

from moray_surgeon.engine.replace_call import replace_call


def _src(text: str) -> str:
    """Dedent a triple-quoted fixture that starts on its own line."""
    return textwrap.dedent(text).lstrip("\n")


_REPLACE_CALL_GOLDEN_CASES = [
    pytest.param(
        "print(value)\n",
        "print",
        "log.info",
        "log.info(value)\n",
        1,
        id="simple-callee-swap",
    ),
    pytest.param(
        _src(
            """
            result = wrap(
                a,  # keep me
                b,
            )
            f = print
            print(value)
            """
        ),
        "wrap",
        "decorate",
        _src(
            """
            result = decorate(
                a,  # keep me
                b,
            )
            f = print
            print(value)
            """
        ),
        1,
        id="multiline-arguments-and-bare-name-untouched",
    ),
    pytest.param(
        "os.path.join(base, name)\n",
        "os.path.join",
        "pathlib.join",
        "pathlib.join(base, name)\n",
        1,
        id="dotted-callee",
    ),
    pytest.param(
        "print(len(items))\n",
        "len",
        "size",
        "print(size(items))\n",
        1,
        id="nested-call-only-inner-callee-changes",
    ),
    pytest.param(
        "print(a)\nprint(b)\n",
        "print",
        "log.info",
        "log.info(a)\nlog.info(b)\n",
        2,
        id="every-matching-call-site",
    ),
    pytest.param(
        "result = old(payload, key=1, flag=True)\n",
        "old",
        "new",
        "result = new(payload, key=1, flag=True)\n",
        1,
        id="keyword-arguments-are-copied-verbatim",
    ),
    pytest.param(
        "print (value)\n",
        "print",
        "log.info",
        "log.info (value)\n",
        1,
        id="whitespace-before-parenthesis-is-kept",
    ),
]


@pytest.mark.parametrize(
    ("source", "old_call", "new_call", "expected", "changes"),
    _REPLACE_CALL_GOLDEN_CASES,
)
def test_replace_call_golden(source, old_call, new_call, expected, changes) -> None:
    """Only the callee expression changes; arguments keep their exact source."""
    result = replace_call(source, old_call, new_call)

    assert result.new_source == expected
    assert result.changes == changes


def test_replace_call_ignores_non_call_occurrences() -> None:
    """A bare name that is never called is not a call site."""
    source = "alias = print\n"

    result = replace_call(source, "print", "log.info")

    assert result.changes == 0
    assert result.new_source == source


def test_replace_call_missing_callee_is_noop() -> None:
    """No matching callee means success with zero changes, not an error."""
    source = "print(value)\n"

    result = replace_call(source, "os.getcwd", "pathlib.Path.cwd")

    assert result.changes == 0
    assert result.new_source == source


@pytest.mark.parametrize(
    ("old_call", "new_call"),
    [
        ("", "log.info"),
        ("print", ""),
        ("   ", "log.info"),
        ("print", "   "),
        ("print", "def"),
        ("print", "a +"),
    ],
)
def test_replace_call_rejects_empty_or_invalid_expressions(old_call, new_call) -> None:
    """Empty call strings and unparsable replacements are malformed instructions."""
    with pytest.raises(ValueError):
        replace_call("print(value)\n", old_call, new_call)
