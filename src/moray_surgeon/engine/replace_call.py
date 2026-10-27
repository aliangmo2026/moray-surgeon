"""Replace call targets (callee only) across a module.

Only the callee expression is swapped; arguments, their formatting, inline
comments and the whitespace between callee and ``(`` are preserved because they
live on the ``Call`` node, not on its ``func`` child.
"""

from __future__ import annotations

import libcst as cst

from moray_surgeon.engine.base import EditResult


class _ReplaceCalleeTransformer(cst.CSTTransformer):
    """Swap ``Call.func`` whenever its rendered source matches ``old_call``."""

    def __init__(self, old_call: str, new_func: cst.BaseExpression) -> None:
        self._old_call = old_call
        self._new_func = new_func
        self._renderer = cst.Module(body=())
        self.changes = 0

    # libcst dispatches by this exact CamelCase hook name.
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:  # noqa: N802
        """Match the original callee text and keep the argument list untouched."""
        if self._renderer.code_for_node(original_node.func).strip() != self._old_call:
            return updated_node
        self.changes += 1
        return updated_node.with_changes(func=self._new_func)


def replace_call(source: str, old_call: str, new_call: str) -> EditResult:
    """Replace every call whose callee renders as ``old_call`` with ``new_call``.

    Matching is textual on the callee's rendered source (``"os.path.join"``,
    ``"print"``, ...), as specified by the instruction protocol. ``new_call`` is
    parsed with ``cst.parse_expression`` and substituted for the callee only::

        print(payload, indent=2)  ->  log.info(payload, indent=2)

    Only the callee changes; the arguments are copied verbatim from the original
    tree. A non-call occurrence of the same name (``alias = print``) is not
    matched.

    Args:
        source: Python module source. Re-parsed once, never re-generated.
        old_call: Callee expression text to look for, e.g. ``"os.getcwd"``.
        new_call: Replacement callee expression, e.g. ``"pathlib.Path.cwd"``.

    Returns:
        An :class:`EditResult` whose ``changes`` counts replaced call sites. If
        no callee matches, ``changes`` is 0 and the source is unchanged.

    Raises:
        ValueError: If either argument is empty/whitespace or ``new_call`` is
            not a valid Python expression.
    """
    old = old_call.strip()
    new = new_call.strip()
    if not old:
        raise ValueError("old_call must not be empty")
    if not new:
        raise ValueError("new_call must not be empty")

    new_func = _parse_callee(new)
    transformer = _ReplaceCalleeTransformer(old, new_func)
    new_module = cst.parse_module(source).visit(transformer)
    return EditResult(new_source=new_module.code, changes=transformer.changes)


def _parse_callee(expression: str) -> cst.BaseExpression:
    """Parse the replacement expression, surfacing syntax errors as ValueError.

    ``from None`` is used because libcst's ``ParserSyntaxError.__str__`` can
    raise while formatting certain malformed inputs; our own message is enough
    for the caller to fix the plan.
    """
    try:
        return cst.parse_expression(expression)
    except cst.ParserSyntaxError:
        raise ValueError(f"new_call is not a valid Python expression: {expression!r}") from None
