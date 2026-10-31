from __future__ import annotations

import textwrap

import pytest

from moray_surgeon.engine.inline_function import inline_function


def _src(text: str) -> str:
    """Dedent a triple-quoted fixture that starts on its own line."""
    return textwrap.dedent(text).lstrip("\n")


def _run(tmp_path, source: str, function_name: str = "helper", extra_files=()):
    (tmp_path / "app.py").write_text(source, encoding="utf-8")
    for name, content in extra_files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return inline_function(source, function_name, tmp_path, tmp_path / "app.py")


_INLINE_GOLDEN_CASES = [
    pytest.param(
        _src(
            """
            def add(a, b):
                return a + b


            total = add(1, 2)
            shout = add(total, 10)
            """
        ),
        "add",
        _src(
            """
            total = (1 + 2)
            shout = (total + 10)
            """
        ),
        3,
        id="simple-two-call-sites-definition-deleted",
    ),
    pytest.param(
        _src(
            """
            def power(base, exp=2):
                return base ** exp


            squared = power(3)
            cubed = power(3, exp=3)
            """
        ),
        "power",
        _src(
            """
            squared = (3 ** 2)
            cubed = (3 ** 3)
            """
        ),
        3,
        id="keywords-and-defaults-are-mapped",
    ),
    pytest.param(
        _src(
            """
            def double(value):
                return value * 2


            result = double(1 + 2) * 3
            """
        ),
        "double",
        _src(
            """
            result = ((1 + 2) * 2) * 3
            """
        ),
        2,
        id="compound-arguments-and-body-are-parenthesised",
    ),
    pytest.param(
        _src(
            """
            def tricky(a):
                return a + (lambda a: a * 0)(a)


            value = tricky(5)
            """
        ),
        "tricky",
        _src(
            """
            value = (5 + (lambda a: a * 0)(5))
            """
        ),
        2,
        id="shadowed-parameter-keeps-its-local-meaning",
    ),
    pytest.param(
        "def inc(x): return x + 1\n\n\nn = inc(4)\n",
        "inc",
        "n = (4 + 1)\n",
        2,
        id="one-line-def-is-supported",
    ),
    pytest.param(
        _src(
            """
            def gone():
                return 1


            untouched = 2
            """
        ),
        "gone",
        "untouched = 2\n",
        1,
        id="dead-function-is-just-deleted",
    ),
    pytest.param(
        _src(
            """
            def add_dup(a):
                return a + a


            value = add_dup(3)
            """
        ),
        "add_dup",
        "value = (3 + 3)\n",
        2,
        id="parameter-used-twice-with-pure-argument",
    ),
]


@pytest.mark.parametrize(
    ("source", "function_name", "expected", "changes"),
    _INLINE_GOLDEN_CASES,
)
def test_inline_function_golden(tmp_path, source, function_name, expected, changes) -> None:
    """Call sites become the substituted body; the definition disappears."""
    result = _run(tmp_path, source, function_name)

    assert result.new_source == expected
    assert result.changes == changes


def test_inline_function_transfers_header_comments(tmp_path) -> None:
    """Comments above a first-statement definition survive the deletion."""
    source = "# tax helpers\n\ndef tax(value):\n    return value * 2\n\n\nx = tax(3)\n"

    result = _run(tmp_path, source, "tax")

    assert result.new_source == "# tax helpers\nx = (3 * 2)\n"


@pytest.mark.parametrize(
    ("source", "extra_files", "match"),
    [
        pytest.param(
            _src(
                """
                def helper(a):
                    gone = a
                    return gone
                """
            ),
            (),
            "exactly one return statement",
            id="multi-statement-body",
        ),
        pytest.param(
            _src(
                """
                def helper(a):
                    print(a)
                """
            ),
            (),
            "exactly one return statement",
            id="no-return-value",
        ),
        pytest.param(
            _src(
                """
                def helper(n):
                    return 1 if helper(n - 1) else 2
                """
            ),
            (),
            "calls itself",
            id="recursion",
        ),
        pytest.param(
            _src(
                """
                def helper(a):
                    return (bound := a) + 1
                """
            ),
            (),
            "named expression",
            id="walrus-in-body",
        ),
        pytest.param(
            _src(
                """
                def helper():
                    return (yield 1)
                """
            ),
            (),
            "generator",
            id="yield-in-body",
        ),
        pytest.param(
            _src(
                """
                def helper():
                    return log()


                first = helper()
                second = helper()
                """
            ),
            (),
            "side effect per site",
            id="impure-body-with-several-sites",
        ),
        pytest.param(
            _src(
                """
                def helper():
                    return 1


                HANDLERS = [helper]
                """
            ),
            (),
            "referenced without being called",
            id="alias-in-container",
        ),
        pytest.param(
            _src(
                """
                def helper():
                    return 1


                @helper
                def other():
                    return 2
                """
            ),
            (),
            "referenced without being called",
            id="used-as-decorator",
        ),
        pytest.param(
            "def helper(a):\n    return a\n\n\nx = helper(1)\n",
            (("other.py", "from app import helper\nprint(helper(2))\n"),),
            "referenced at other.py",
            id="cross-file-caller",
        ),
        pytest.param(
            "def helper(a):\n    return a\n\n\nx = helper(1)\n",
            (("other.py", "from app import *\nprint(helper(2))\n"),),
            "referenced at other.py",
            id="star-imported-elsewhere",
        ),
        pytest.param(
            "def helper(a):\n    return a\n\n\nx = helper(1)\nkey = 'helper'\n",
            (),
            "dynamic reference",
            id="dynamic-getattr-string",
        ),
        pytest.param(
            "import functools\n"
            "\n\n@functools.cache\ndef helper(a):\n    return a\n\n\nx = helper(1)\n",
            (),
            "has decorators",
            id="decorated-definition",
        ),
        pytest.param(
            "async def helper(a):\n    return a\n\n\nx = helper(1)\n",
            (),
            "async",
            id="async-definition",
        ),
        pytest.param(
            "def helper(*args):\n    return 1\n\n\nx = helper(1)\n",
            (),
            r"\*args/\*\*kwargs signature",
            id="star-args-signature",
        ),
        pytest.param(
            "def helper(a):\n    return a\n\n\nargs = [1]\nx = helper(*args)\n",
            (),
            "star-arguments",
            id="star-args-call-site",
        ),
        pytest.param(
            "def helper(a, b):\n    return a + b\n\n\nx = helper(1)\n",
            (),
            "does not pass the parameter 'b'",
            id="missing-argument",
        ),
        pytest.param(
            "def helper(a):\n    return a\n\n\nx = helper(b=1)\n",
            (),
            "unknown keyword argument",
            id="unknown-keyword",
        ),
        pytest.param(
            "def helper(a):\n    return a\n\n\nx = helper(1, a=2)\n",
            (),
            "twice",
            id="duplicated-binding",
        ),
        pytest.param(
            _src(
                """
                def helper(a):
                    return a + a


                value = helper(load())
                """
            ),
            (),
            "evaluate it more than once",
            id="parameter-used-twice-with-impure-argument",
        ),
        pytest.param(
            _src(
                """
                class Thing:
                    def helper(self):
                        return 1
                """
            ),
            (),
            "no top-level function named",
            id="method-is-not-a-function",
        ),
        pytest.param(
            "def helper():\n    return 1\n\n\ndef helper():\n    return 2\n",
            (),
            "defined 2 times",
            id="duplicate-definitions",
        ),
    ],
)


def test_inline_function_refuses_unsafe_inlines(tmp_path, source, extra_files, match) -> None:
    """Every shadow of doubt refuses the inline; the file stays untouched."""
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, source, extra_files=extra_files)


def test_inline_function_allows_shadowed_name_in_other_scope(tmp_path) -> None:
    """A same-named binding in a nested scope is a different variable."""
    source = _src(
        """
        def helper(a):
            return a


        def shadow():
            helper = 5
            return helper
        """
    )

    result = _run(tmp_path, source)

    assert result.new_source == _src(
        """
        def shadow():
            helper = 5
            return helper
        """
    )
    # No call sites: the deletion alone is the change.
    assert result.changes == 1


def test_inline_function_rejects_invalid_name(tmp_path) -> None:
    """The lookup target must be a real identifier."""
    with pytest.raises(ValueError, match="function_name"):
        _run(tmp_path, "x = 1\n", "not a name")


def test_inline_function_empty_after_inline(tmp_path) -> None:
    """Inlining the only statement leaves an empty module."""
    result = _run(tmp_path, "def only():\n    return 7\n\n\nvalue = only()\n", "only")

    assert "only" not in result.new_source
    assert "7" in result.new_source
