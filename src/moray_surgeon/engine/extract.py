"""Extract module-level statements into a new parameterless function.

The move is expressed as a CST body edit, never as string slicing or re-parsing
of the rewritten file. libcst re-indents the moved statements (including nested
blocks and leading comments) according to the module's own indent unit, so the
extracted function keeps the original formatting style.

Deliberately out of scope for this MVP:

* Only statements in ``Module.body`` are considered. **Nested scopes are never
  extracted**: a range that lies inside a function, class or compound statement
  simply matches nothing.
* Only ``SimpleStatementLine``, ``ClassDef`` and ``FunctionDef`` are
  extractable. Other compound statements (``if`` / ``for`` / ``while`` /
  ``with`` / ``try``) are left in place; if one of them sits between two
  selected statements the extraction is refused instead of silently reordering
  code.
* The generated function takes no parameters and returns nothing. Selecting a
  block that reads names only available where it used to sit is the plan
  author's responsibility, not something this atom guesses at.
"""

from __future__ import annotations

import keyword

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from moray_surgeon.engine.base import EditResult

_EXTRACTABLE = (cst.SimpleStatementLine, cst.ClassDef, cst.FunctionDef)


def extract_function(
    source: str,
    function_name: str,
    start_line: int,
    end_line: int,
) -> EditResult:
    """Move the top-level statements fully inside ``[start_line, end_line]``.

    A statement is selected only when its whole span (as reported by
    ``PositionProvider``) is contained in the inclusive line range. The selected
    statements are replaced, at their original position, by a ``def`` holding
    them followed by a call to that new function::

        x = 1          ->    def helper():
        y = 2                    x = 1
                                 y = 2

                             helper()

    Leading comments / blank lines of the first moved statement move above the
    new ``def`` so the surrounding layout stays readable; comments inside the
    block travel with their statements.

    Args:
        source: Python module source. Re-parsed once, never re-generated.
        function_name: Name of the new function; must be a valid identifier.
        start_line: First line of the block, 1-based and inclusive.
        end_line: Last line of the block, 1-based and inclusive.

    Returns:
        An :class:`EditResult` whose ``changes`` counts the moved statements.

    Raises:
        ValueError: If the line range is invalid, no extractable top-level
            statement is fully contained in it, the selected statements are
            separated by a non-extractable compound statement, or
            ``function_name`` is not a valid identifier.
    """
    _validate_function_name(function_name)
    if start_line < 1 or end_line < 1:
        raise ValueError("start_line and end_line are 1-based and must be >= 1")
    if start_line > end_line:
        raise ValueError("start_line must be <= end_line")

    wrapper = MetadataWrapper(cst.parse_module(source))
    positions = wrapper.resolve(PositionProvider)
    body = wrapper.module.body

    selected: list[int] = []
    for index, statement in enumerate(body):
        if not isinstance(statement, _EXTRACTABLE):
            continue
        position = positions[statement]
        if position.start.line >= start_line and position.end.line <= end_line:
            selected.append(index)

    if not selected:
        raise ValueError(
            f"no module-level statement is fully inside lines {start_line}-{end_line}"
        )

    first, last = selected[0], selected[-1]
    if last - first + 1 != len(selected):
        raise ValueError(
            "selected statements are separated by a non-extractable module-level "
            "statement; only SimpleStatementLine, ClassDef and FunctionDef can move"
        )

    moved = list(body[first : last + 1])
    # Comments/blank lines before the block belong to the module layout, so they
    # stay above the new def instead of becoming dead space inside the body.
    leading_lines = moved[0].leading_lines
    moved[0] = moved[0].with_changes(leading_lines=())

    new_function = cst.FunctionDef(
        name=cst.Name(function_name),
        params=cst.Parameters(),
        body=cst.IndentedBlock(body=moved),
        leading_lines=leading_lines,
    )
    call = cst.SimpleStatementLine(
        body=[cst.Expr(cst.Call(func=cst.Name(function_name)))],
        leading_lines=[cst.EmptyLine()],
    )
    new_body = [*body[:first], new_function, call, *body[last + 1 :]]

    return EditResult(
        new_source=wrapper.module.with_changes(body=new_body).code,
        changes=len(moved),
    )


def _validate_function_name(function_name: str) -> None:
    """Reject names that would make the generated def syntactically invalid."""
    if not function_name:
        raise ValueError("function_name must not be empty")
    if not function_name.isidentifier() or keyword.iskeyword(function_name):
        raise ValueError(f"function_name must be a valid Python identifier, got {function_name!r}")
