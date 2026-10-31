"""Extract one expression into a new local variable, at an exact source range.

Given the span of an expression, the atom replaces that expression with a name
and inserts ``variable_name = <expression>`` on its own line immediately before
the statement containing the span. libcst keeps every other byte of the module
intact: comments around the target line, the block's indentation and the
expression's own internal formatting all survive untouched.

Coordinate convention (libcst's ``PositionProvider``): lines are 1-based,
columns are 0-based, and the end coordinates point *one past* the expression's
last character (exclusive, like editor selection ranges). The range must match
exactly one expression node;
anything looser is refused rather than guessed.

Deliberately refused (``ValueError``) instead of rewritten:

* Ranges matching zero or several expression nodes.
* Spans whose match is a binding position: assignment / for / comprehension /
  named-expression targets, keyword-argument names, attribute names, parameter
  and definition names, import names, ``global`` / ``nonlocal`` / ``del``
  operands, match captures. Replacing those would rename or relocate a binding,
  not extract a value.
* Spans whose evaluation the hoist would move across a scope or iteration
  boundary: pieces inside a ``lambda`` or comprehension body (their names would
  resolve differently once hoisted), inside a ``while`` test or ``except`` type
  (re-evaluated or lazily evaluated), inside a ``match`` pattern, or inside a
  one-line suite (hoisting a repeated ``while`` body expression would run it
  once).
* Expressions that bind or suspend when evaluated: named expressions (``:=``)
  and ``yield`` / ``yield from``.

``await`` is allowed: hoisting it to the line immediately above is temporally
equivalent because no other statement can run in between.
"""

from __future__ import annotations

import keyword
from collections.abc import Mapping
from typing import TypeVar, cast

import libcst as cst
from libcst.metadata import (
    CodePosition,
    CodeRange,
    MetadataWrapper,
    ParentNodeProvider,
    PositionProvider,
)

from moray_surgeon.engine.base import EditResult

# Block bodies hold exactly these two statement shapes; both carry leading_lines.
_BlockT = TypeVar("_BlockT", cst.Module, cst.IndentedBlock)
_NodeT = TypeVar("_NodeT", bound=cst.CSTNode)

# Expressions whose evaluation binds a name or suspends the frame; hoisting the
# assignment would move that binding to the wrong place or run it more often.
# libcst models both ``yield`` and ``yield from`` as ``Yield``.
_UNHOISTABLE_INNER = (cst.NamedExpr, cst.Yield)

# Ancestors that put the expression behind a new scope: hoisted out, the names
# inside would resolve against the enclosing scope instead.
_SCOPE_BOUNDARY_ANCESTORS = (
    cst.Lambda,
    cst.ListComp,
    cst.SetComp,
    cst.DictComp,
    cst.GeneratorExp,
)

# Match patterns evaluate lazily, while the subject is being matched.
_PATTERN_ANCESTORS = (
    cst.MatchValue,
    cst.MatchClass,
    cst.MatchSequence,
    cst.MatchList,
    cst.MatchTuple,
    cst.MatchMapping,
    cst.MatchAs,
    cst.MatchOr,
    cst.MatchStar,
    cst.MatchSingleton,
)


def extract_variable(
    source: str,
    start_line: int,
    start_column: int,
    end_line: int,
    end_column: int,
    variable_name: str,
) -> EditResult:
    """Extract the expression at ``[start, end]`` into ``variable_name``.

    The new assignment is inserted before the innermost block statement that
    contains the range, at that block's own indentation::

        # total price
        print(price * count)   ->   # total price
                                    total = price * count
                                    print(total)

    Args:
        source: Python module source. Re-parsed once, never re-generated.
        start_line: 1-based line of the expression's first character.
        start_column: 0-based column of the expression's first character.
        end_line: 1-based line one past the expression's last character.
        end_column: 0-based column one past the expression's last character.
        variable_name: Name of the new variable; must be a valid identifier.

    Returns:
        An :class:`EditResult` whose ``changes`` counts the replaced expression
        plus the inserted assignment (so a successful extraction is 2).

    Raises:
        ValueError: If the name is not an identifier, the coordinates are not
            ordered, the range does not match exactly one expression node, the
            match is a binding position, the hoist would cross a scope or
            iteration boundary, or the expression binds or suspends.
    """
    _validate_variable_name(variable_name)
    start = CodePosition(line=start_line, column=start_column)
    end = CodePosition(line=end_line, column=end_column)
    if (start_line, start_column) > (end_line, end_column):
        raise ValueError(
            f"start ({start_line}:{start_column}) must not come after end "
            f"({end_line}:{end_column})"
        )

    wrapper = MetadataWrapper(cst.parse_module(source))
    target = _find_unique_expression(wrapper, start, end)
    _assert_hoistable(wrapper.resolve(ParentNodeProvider), target)

    transformer = _ExtractVariableTransformer(
        target,
        start,
        end,
        wrapper.resolve(PositionProvider),
        variable_name,
    )
    new_module = wrapper.visit(transformer)
    if not transformer.inserted:
        raise ValueError(
            "the expression does not live inside a statement list; refusing to extract"
        )

    return EditResult(new_source=new_module.code, changes=transformer.changes)


def _validate_variable_name(variable_name: str) -> None:
    """Reject names that would make the generated assignment syntactically invalid."""
    if not variable_name.isidentifier() or keyword.iskeyword(variable_name):
        raise ValueError(
            f"variable_name must be a valid Python identifier, got {variable_name!r}"
        )


def _find_unique_expression(
    wrapper: MetadataWrapper,
    start: CodePosition,
    end: CodePosition,
) -> cst.BaseExpression:
    """Return the one expression node whose span equals ``start``..``end``."""
    matches: list[cst.BaseExpression] = []

    class _SpanScanner(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def on_visit(self, node: cst.CSTNode) -> bool:
            if not isinstance(node, cst.BaseExpression):
                return True
            span = self.get_metadata(PositionProvider, node)
            if span.start == start and span.end == end:
                matches.append(node)
            return True

    wrapper.visit(_SpanScanner())

    if not matches:
        raise ValueError(
            f"no expression occupies exactly lines {start.line}-{end.line}, columns "
            f"{start.column}-{end.column}; the range must match one whole expression "
            "(end coordinates point one past its last character)"
        )
    if len(matches) > 1:
        raise ValueError(
            f"the range matches {len(matches)} expression nodes; refusing to guess"
        )
    return matches[0]


def _assert_hoistable(
    parents: Mapping[cst.CSTNode, cst.CSTNode],
    target: cst.BaseExpression,
) -> None:
    """Refuse every position where hoisting would rewrite a binding or move work."""
    for inner in _walk(target):
        if isinstance(inner, _UNHOISTABLE_INNER):
            raise ValueError(
                f"the expression contains {type(inner).__name__}, which binds or "
                "suspends when evaluated; refusing to hoist it"
            )

    chain: list[cst.CSTNode] = []
    current: cst.CSTNode | None = parents.get(target)
    while current is not None:
        chain.append(current)
        current = parents.get(current)
    # The target itself counts: ``while.target`` *is* the target expression
    # when the whole test is being extracted.
    chain_ids = {id(target), *(id(node) for node in chain)}

    for ancestor in chain:
        if isinstance(ancestor, cst.SimpleStatementSuite):
            raise ValueError(
                "the expression sits inside a one-line suite; hoisting would change "
                "how often it runs"
            )
        if isinstance(ancestor, _SCOPE_BOUNDARY_ANCESTORS):
            raise ValueError(
                "the expression sits inside a lambda or comprehension body, where its "
                "names would resolve differently once hoisted"
            )
        if isinstance(ancestor, _PATTERN_ANCESTORS):
            raise ValueError("the expression sits inside a match pattern")
        if isinstance(ancestor, cst.While) and id(ancestor.test) in chain_ids:
            raise ValueError("the expression sits inside a while test, which re-evaluates")
        if isinstance(ancestor, cst.ExceptHandler) and id(ancestor.type) in chain_ids:
            raise ValueError("the expression sits inside an except type, evaluated lazily")
        reason = _binding_position_refusal(ancestor, target)
        if reason is not None:
            raise ValueError(f"cannot extract this expression: {reason}")


def _binding_position_refusal(parent: cst.CSTNode, node: cst.CSTNode) -> str | None:
    """Return a refusal reason when ``node`` is a binding position of ``parent``."""
    if isinstance(parent, cst.AssignTarget):
        return "it is an assignment target"
    if isinstance(parent, cst.AugAssign) and parent.target is node:
        return "it is an augmented-assignment target"
    if isinstance(parent, cst.AnnAssign) and parent.target is node:
        return "it is an annotated-assignment target"
    if isinstance(parent, cst.For) and parent.target is node:
        return "it is a for-loop target"
    if isinstance(parent, cst.NamedExpr) and parent.target is node:
        return "it is a named-expression target"
    if isinstance(parent, cst.CompFor) and parent.target is node:
        return "it is a comprehension target"
    if isinstance(parent, cst.Arg) and parent.keyword is node:
        return "it is a keyword-argument name"
    if isinstance(parent, cst.Attribute) and parent.attr is node:
        return "it is an attribute name"
    if isinstance(parent, cst.Param) and parent.name is node:
        return "it is a parameter name"
    if isinstance(parent, (cst.FunctionDef, cst.ClassDef)) and parent.name is node:
        return "it is a definition name"
    if isinstance(parent, cst.ImportAlias):
        return "it is an import name"
    if isinstance(parent, cst.AsName):
        # ``except E as name`` / ``with expr as name``: libcst wraps the bound
        # name in an AsName node.
        return "it is an as-binding name"
    if isinstance(parent, (cst.Global, cst.Nonlocal)):
        return "it is a global/nonlocal declaration name"
    if isinstance(parent, cst.Del):
        return "it is a del operand"
    return None


def _walk(node: cst.CSTNode) -> list[cst.CSTNode]:
    """Collect ``node`` and every descendant without the visitor ceremony."""
    nodes: list[cst.CSTNode] = []
    stack: list[cst.CSTNode] = [node]
    while stack:
        current = stack.pop()
        nodes.append(current)
        stack.extend(current.children)
    return nodes


def _key(position: CodePosition) -> tuple[int, int]:
    """Make a position orderable; ``CodePosition`` itself only supports equality."""
    return (position.line, position.column)


class _ExtractVariableTransformer(cst.CSTTransformer):
    """Swap the matched expression for a name and hoist its assignment.

    ``leave_IndentedBlock``/``leave_Module`` run post-order, so the first block
    that still contains the target's statement is the innermost one; that is
    where the assignment is inserted and every outer block skips.
    """

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        target: cst.BaseExpression,
        target_start: CodePosition,
        target_end: CodePosition,
        positions: Mapping[cst.CSTNode, CodeRange],
        variable_name: str,
    ) -> None:
        self._target = target
        self._target_start = target_start
        self._target_end = target_end
        self._positions = positions
        self._variable_name = variable_name
        self.changes = 0
        self.inserted = False

    def on_leave(
        self,
        original_node: _NodeT,
        updated_node: _NodeT,
    ) -> _NodeT | cst.RemovalSentinel | cst.FlattenSentinel[_NodeT]:
        """Replace the matched expression after its children have been rebuilt.

        The declared return type mirrors ``CSTTransformer.on_leave`` so the
        override stays compatible; the one deliberate node swap below is the
        whole point of this transformer.
        """
        rebuilt = super().on_leave(original_node, updated_node)
        if original_node is self._target and isinstance(rebuilt, cst.CSTNode):
            self.changes += 1
            return cast("_NodeT", cst.Name(self._variable_name))
        return rebuilt

    def leave_Module(  # noqa: N802
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        return self._maybe_insert(original_node, updated_node)

    def leave_IndentedBlock(  # noqa: N802
        self, original_node: cst.IndentedBlock, updated_node: cst.IndentedBlock
    ) -> cst.IndentedBlock:
        return self._maybe_insert(original_node, updated_node)

    def _maybe_insert(
        self,
        original_block: _BlockT,
        updated_block: _BlockT,
    ) -> _BlockT:
        """Insert ``name = <expr>`` directly above the statement holding the target.

        The target statement's leading blank/comment lines travel to the new
        assignment, so a comment above the target line keeps its position and
        the assignment lands immediately above the line it feeds.
        """
        if self.inserted:
            return updated_block
        for index, original_statement in enumerate(original_block.body):
            span = self._positions[original_statement]
            if _key(span.start) <= _key(self._target_start) and _key(self._target_end) <= _key(
                span.end
            ):
                assignment = cst.SimpleStatementLine(
                    body=[
                        cst.Assign(
                            targets=[cst.AssignTarget(cst.Name(self._variable_name))],
                            value=self._target,
                        )
                    ]
                )
                body = list(updated_block.body)
                following = cast(
                    "cst.SimpleStatementLine | cst.BaseCompoundStatement", body[index]
                )
                body[index] = following.with_changes(leading_lines=())
                body.insert(index, assignment.with_changes(leading_lines=following.leading_lines))
                self.inserted = True
                self.changes += 1
                return updated_block.with_changes(body=body)
        return updated_block
