"""Rewrite a consecutive ``if``/``elif``/``else`` chain into a ``match`` statement.

The transformation is only offered where it is *provably* equivalent, because a
match statement is not a cosmetic rewrite: ``case y:`` captures a new name instead
of comparing against ``y``, and the subject is evaluated once rather than once per
branch. The atom therefore accepts exactly three test shapes and refuses
everything else rather than guessing:

* ``subject == <literal or dotted constant>`` → ``case <value>:``
* ``isinstance(subject, <class or tuple of classes>)`` → ``case <class>():``
  (a tuple becomes an or-pattern: ``case str() | bytes():``)
* ``<those> or <those>`` → ``case a | b:``; ``and``/``not`` are refused because a
  pattern cannot express a conjunction of two different tests.

A bare name on the right of ``==`` is refused: ``if x == MODE`` can only become a
capture pattern, which would silently rebind instead of compare. ``case MODE`` is
correct only for a dotted constant (``Config.MODE``), and that is what the atom
accepts.

Two further safety rules come from evaluation semantics:

* the subject must be a plain name or attribute chain (``x``, ``self.state``) — a
  call or subscript would change from "evaluated per branch" to "evaluated once";
* no branch may rebind the subject (``x = ...`` for a name, ``self.state = ...``
  for an attribute), because the original chain would re-read it in the next test
  while ``match`` keeps the first value.

The generated code targets Python 3.10+, which is where ``match`` exists; the
project itself already requires 3.11+.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import libcst as cst
from libcst.metadata import (
    CodeRange,
    MetadataWrapper,
    ParentNodeProvider,
    PositionProvider,
)

from moray_surgeon.engine.base import EditResult

# ``isinstance(x, str)`` and ``case str():`` agree for these builtins, so a
# lowercase name is only accepted when it is one of them. Any other bare name
# could hold a tuple of classes, which ``isinstance`` accepts but a class pattern
# rejects at runtime with TypeError.
_BUILTIN_TYPES = frozenset(
    {
        "bool",
        "bytearray",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "memoryview",
        "object",
        "range",
        "set",
        "slice",
        "str",
        "tuple",
        "type",
    }
)


@dataclass(frozen=True)
class _Branch:
    """One arm of the chain: its test (None for ``else``) and its body."""

    test: cst.BaseExpression | None
    body: cst.BaseSuite
    leading_lines: Sequence[cst.EmptyLine]


def convert_to_match_case(source: str, start_line: int, end_line: int) -> EditResult:
    """Rewrite the if/elif/else chain covered by the line range into ``match``.

    The range must fully contain exactly one chain; a range that cuts a branch in
    half is refused rather than silently extending the edit.

    Args:
        source: Python source of the target file.
        start_line: First line of the range (1-based, inclusive).
        end_line: Last line of the range (1-based, inclusive).

    Returns:
        An :class:`EditResult` whose ``changes`` counts the generated ``case``
        clauses, the wildcard ``case _`` included.

    Raises:
        ValueError: The range is empty or invalid, contains no complete chain or
            more than one, a branch is not expressible as a pattern, the branches
            do not test the same subject, or a branch rebinds the subject.
    """
    if start_line < 1 or end_line < start_line:
        raise ValueError(
            f"line range must satisfy 1 <= start_line <= end_line, "
            f"got {start_line}..{end_line}"
        )

    wrapper = MetadataWrapper(cst.parse_module(source))
    positions = wrapper.resolve(PositionProvider)
    parents = wrapper.resolve(ParentNodeProvider)
    head = _find_chain(wrapper.module, positions, parents, start_line, end_line)
    cases, subject = _build_cases(wrapper.module, head)
    _assert_subject_stable(wrapper.module, subject, head)

    match_node = cst.Match(
        subject=subject,
        cases=cases,
        leading_lines=head.leading_lines,
    )
    new_module = wrapper.module.visit(_IfToMatch(head, match_node))
    return EditResult(new_source=new_module.code, changes=len(cases))


class _ChainHeadCollector(cst.CSTVisitor):
    """Collect every ``if`` statement, outermost chain heads included."""

    def __init__(self) -> None:
        self.nodes: list[cst.If] = []

    # libcst dispatches by this exact CamelCase hook name.
    def visit_If(self, node: cst.If) -> None:  # noqa: N802
        """Remember the node; filtering happens with the parent metadata."""
        self.nodes.append(node)


def _find_chain(
    module: cst.Module,
    positions: Mapping[cst.CSTNode, CodeRange],
    parents: Mapping[cst.CSTNode, cst.CSTNode],
    start_line: int,
    end_line: int,
) -> cst.If:
    """Return the single chain head fully contained in the line range."""
    collector = _ChainHeadCollector()
    module.visit(collector)

    contained: list[cst.If] = []
    for node in collector.nodes:
        parent = parents.get(node)
        if isinstance(parent, cst.If) and parent.orelse is node:
            continue  # an ``elif`` arm, not a chain head
        span = positions[node]
        if start_line <= span.start.line and span.end.line <= end_line:
            contained.append(node)

    if not contained:
        raise ValueError(
            f"no complete if/elif/else chain is contained in lines "
            f"{start_line}..{end_line}; the range must cover the whole statement"
        )
    # A branch body may hold another if statement; only the outermost chain in the
    # range is the one the caller asked about.
    outermost = [
        node
        for node in contained
        if not any(
            other is not node and _encloses(positions[other], positions[node])
            for other in contained
        )
    ]
    if len(outermost) > 1:
        raise ValueError(
            f"lines {start_line}..{end_line} contain {len(outermost)} if chains; "
            "narrow the range to the one you want to convert"
        )
    return outermost[0]


def _encloses(outer: CodeRange, inner: CodeRange) -> bool:
    """True when ``outer``'s line span contains ``inner``'s span."""
    return outer.start.line <= inner.start.line and inner.end.line <= outer.end.line


def _build_cases(
    module: cst.Module, head: cst.If
) -> tuple[list[cst.MatchCase], cst.BaseExpression]:
    """Convert every branch into a ``MatchCase`` sharing one subject."""
    subject: cst.BaseExpression | None = None
    cases: list[cst.MatchCase] = []

    for branch in _branches(head):
        if branch.test is None:
            pattern: cst.MatchPattern = cst.MatchAs()
        else:
            candidate, pattern = _convert_test(module, branch.test)
            if subject is None:
                subject = candidate
            elif module.code_for_node(subject) != module.code_for_node(candidate):
                raise ValueError(
                    "every branch must test the same expression to share one "
                    f"match subject; found {module.code_for_node(subject)!r} and "
                    f"{module.code_for_node(candidate)!r}"
                )
        cases.append(
            cst.MatchCase(
                pattern=pattern,
                body=branch.body,
                leading_lines=_comment_lines(branch.leading_lines),
            )
        )

    if subject is None:  # pragma: no cover - the head always carries a test
        raise ValueError("the chain has no conditional branch to convert")
    return cases, subject


def _branches(head: cst.If) -> Iterator[_Branch]:
    """Yield the chain arms in source order, ``else`` last."""
    node: cst.If | None = head
    while node is not None:
        yield _Branch(node.test, node.body, node.leading_lines)
        orelse = node.orelse
        if isinstance(orelse, cst.If):
            node = orelse
        elif isinstance(orelse, cst.Else):
            yield _Branch(None, orelse.body, orelse.leading_lines)
            node = None
        else:
            node = None


def _convert_test(
    module: cst.Module, test: cst.BaseExpression
) -> tuple[cst.BaseExpression, cst.MatchPattern]:
    """Turn one branch test into ``(subject, pattern)``."""
    if isinstance(test, cst.BooleanOperation):
        if not isinstance(test.operator, cst.Or):
            raise ValueError(
                f"'and' cannot become a match pattern: "
                f"{module.code_for_node(test)!r}; split the branch by hand"
            )
        operands = list(_flatten_or(test))
        subject: cst.BaseExpression | None = None
        patterns: list[cst.MatchPattern] = []
        for operand in operands:
            candidate, pattern = _convert_comparison(module, operand)
            if subject is None:
                subject = candidate
            elif module.code_for_node(subject) != module.code_for_node(candidate):
                raise ValueError(
                    f"one 'or' expression tests two different subjects: "
                    f"{module.code_for_node(test)!r}"
                )
            patterns.append(pattern)
        if subject is None:  # pragma: no cover - _flatten_or always yields a node
            raise ValueError("empty 'or' expression")
        return subject, _combine_or(patterns)

    if isinstance(test, cst.UnaryOperation) and isinstance(test.operator, cst.Not):
        raise ValueError(
            f"'not' cannot become a match pattern: {module.code_for_node(test)!r}"
        )
    return _convert_comparison(module, test)


def _combine_or(patterns: Sequence[cst.MatchPattern]) -> cst.MatchPattern:
    """Join patterns with ``|``, flattening an or-pattern that is already there.

    ``isinstance(x, (A, B)) or x == 3`` produces a class or-pattern and a value
    pattern; nesting ``MatchOr`` inside ``MatchOr`` is not valid syntax, so the
    elements are merged into one list. libcst stores or-elements in a
    ``MatchOrElement`` wrapper, and a raw pattern in its place renders fine in
    isolated tests but crashes code generation, so every element is wrapped.
    """
    if len(patterns) == 1:
        return patterns[0]
    elements: list[cst.MatchOrElement] = []
    for pattern in patterns:
        if isinstance(pattern, cst.MatchOr):
            elements.extend(pattern.patterns)
        else:
            elements.append(cst.MatchOrElement(pattern=pattern))
    return cst.MatchOr(patterns=elements)


def _flatten_or(node: cst.BaseExpression) -> Iterator[cst.BaseExpression]:
    """Yield the operands of a chain of ``or`` expressions, left to right."""
    if isinstance(node, cst.BooleanOperation) and isinstance(node.operator, cst.Or):
        yield from _flatten_or(node.left)
        yield from _flatten_or(node.right)
    else:
        yield node


def _convert_comparison(
    module: cst.Module, test: cst.BaseExpression
) -> tuple[cst.BaseExpression, cst.MatchPattern]:
    """Convert ``subject == value`` or ``isinstance(subject, Class)``."""
    if isinstance(test, cst.Comparison):
        if len(test.comparisons) != 1:
            raise ValueError(
                f"chained comparisons cannot become a match pattern: "
                f"{module.code_for_node(test)!r}"
            )
        comparison = test.comparisons[0]
        operator = comparison.operator
        if not isinstance(operator, cst.Equal):
            raise ValueError(
                f"only '==' can become a case pattern, got "
                f"{module.code_for_node(test)!r}"
            )
        subject = test.left
        _assert_simple_subject(module, subject)
        return subject, _value_pattern(module, comparison.comparator)
    if isinstance(test, cst.Call):
        return _isinstance_pattern(module, test)
    raise ValueError(
        "only '<subject> == <literal>', 'isinstance(<subject>, <class>)' and "
        f"'or' combinations of those can become case patterns, got "
        f"{module.code_for_node(test)!r}"
    )


def _value_pattern(module: cst.Module, expression: cst.BaseExpression) -> cst.MatchPattern:
    """Build a value pattern, refusing anything that would capture a name."""
    if isinstance(expression, cst.Name):
        if expression.value == "None":
            return cst.MatchSingleton(value=cst.Name("None"))
        if expression.value in {"True", "False"}:
            return cst.MatchSingleton(value=cst.Name(expression.value))
        raise ValueError(
            f"{expression.value!r} would become a capture pattern that binds a new "
            "name instead of comparing; use a literal or a dotted constant such as "
            "Config.MODE"
        )
    if isinstance(expression, cst.Attribute):
        _assert_dotted_constant(module, expression)
        return cst.MatchValue(value=expression)
    if isinstance(expression, (cst.Integer, cst.Float, cst.Imaginary)):
        return cst.MatchValue(value=expression)
    if isinstance(expression, (cst.SimpleString, cst.ConcatenatedString)):
        return cst.MatchValue(value=expression)
    if (
        isinstance(expression, cst.UnaryOperation)
        and isinstance(expression.operator, cst.Minus)
        and isinstance(expression.expression, (cst.Integer, cst.Float))
    ):
        return cst.MatchValue(value=expression)
    raise ValueError(
        f"cannot express {module.code_for_node(expression)!r} as a value pattern; "
        "pattern values must be literals or dotted constants"
    )


def _isinstance_pattern(
    module: cst.Module, call: cst.Call
) -> tuple[cst.BaseExpression, cst.MatchPattern]:
    """Convert ``isinstance(subject, Class)`` into a class pattern."""
    func = call.func
    if not (isinstance(func, cst.Name) and func.value == "isinstance"):
        raise ValueError(
            f"only isinstance(...) calls can become class patterns, got "
            f"{module.code_for_node(call)!r}"
        )
    if len(call.args) != 2 or any(argument.keyword is not None for argument in call.args):
        raise ValueError(
            f"isinstance() must be called with exactly two positional arguments, got "
            f"{module.code_for_node(call)!r}"
        )
    subject = call.args[0].value
    _assert_simple_subject(module, subject)
    classes = call.args[1].value
    if isinstance(classes, cst.Tuple):
        elements = [element.value for element in classes.elements]
        if not elements:
            raise ValueError("isinstance() with an empty class tuple matches nothing")
        patterns = [_class_pattern(module, element) for element in elements]
        return subject, _combine_or(patterns)
    return subject, _class_pattern(module, classes)


def _class_pattern(module: cst.Module, node: cst.BaseExpression) -> cst.MatchPattern:
    """Build ``ClassName()`` from a class reference, refusing ambiguous values."""
    if isinstance(node, cst.Name):
        if node.value not in _BUILTIN_TYPES and not node.value[:1].isupper():
            raise ValueError(
                f"isinstance() is given {node.value!r}, which may be a variable "
                "holding a tuple of classes; a class pattern needs a class. Use a "
                "dotted constant or a conventionally named class"
            )
        return cst.MatchClass(cls=node, patterns=[])
    if isinstance(node, cst.Attribute):
        _assert_dotted_constant(module, node)
        return cst.MatchClass(cls=node, patterns=[])
    raise ValueError(
        f"isinstance() must name a class, got {module.code_for_node(node)!r}"
    )


def _assert_simple_subject(module: cst.Module, subject: cst.BaseExpression) -> None:
    """The subject must be a plain name or attribute chain.

    ``match`` evaluates its subject once, while the original chain re-evaluated it
    per branch; a call or subscript could therefore run a different number of
    times, so only pure loads are accepted.
    """
    root = _root_name(subject)
    if root is None:
        raise ValueError(
            f"the tested expression {module.code_for_node(subject)!r} is not a plain "
            "name or attribute chain; match evaluates its subject once, so calls "
            "and subscripts cannot be converted safely"
        )


def _assert_dotted_constant(module: cst.Module, node: cst.Attribute) -> None:
    """Only ``Name.attr.attr`` chains are constants; calls inside are not."""
    if _root_name(node) is None:
        raise ValueError(
            f"{module.code_for_node(node)!r} is not a dotted constant; only names "
            "such as Config.MODE can appear in a pattern"
        )


def _root_name(node: cst.BaseExpression) -> str | None:
    """The leftmost name of a name/attribute chain, or None for anything else."""
    current: cst.BaseExpression = node
    while isinstance(current, cst.Attribute):
        current = current.value
    if isinstance(current, cst.Name) and current.value != "":
        return current.value
    return None


class _BoundNames(cst.CSTVisitor):
    """Names a subtree rebinds in the enclosing scope."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.star_import = False

    # libcst dispatches by this exact CamelCase hook name.
    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
        """A nested def binds its own name, and hoists arguments inside itself."""
        self.names.add(node.name.value)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
        """A nested class binds its own name."""
        self.names.add(node.name.value)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_Import(self, node: cst.Import) -> None:  # noqa: N802
        """``import a.b`` binds ``a``; ``import a as b`` binds ``b``."""
        for alias in node.names:
            binding = alias.evaluated_alias or alias.evaluated_name.split(".")[0]
            self.names.add(binding)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:  # noqa: N802
        """From-imports bind the alias or the imported name."""
        if isinstance(node.names, cst.ImportStar):
            # ``from mod import *`` binds an unknown set of names, which may
            # include the subject; the caller refuses rather than assume.
            self.star_import = True
            return
        for alias in node.names:
            self.names.add(alias.evaluated_alias or alias.evaluated_name)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_Global(self, node: cst.Global) -> None:  # noqa: N802
        """A global declaration names a binding the branch writes."""
        self.names.update(item.name.value for item in node.names)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_Nonlocal(self, node: cst.Nonlocal) -> None:  # noqa: N802
        """A nonlocal declaration names a binding the branch writes."""
        self.names.update(item.name.value for item in node.names)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_ExceptHandler(self, node: cst.ExceptHandler) -> None:  # noqa: N802
        """``except E as err`` binds ``err``."""
        if node.name is not None and isinstance(node.name.name, cst.Name):
            self.names.add(node.name.name.value)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_AssignTarget(self, node: cst.AssignTarget) -> None:  # noqa: N802
        """Assignment targets are bindings when they are names."""
        if isinstance(node.target, cst.Name):
            self.names.add(node.target.value)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:  # noqa: N802
        """``x: int = 1`` binds ``x``."""
        if isinstance(node.target, cst.Name):
            self.names.add(node.target.value)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_AugAssign(self, node: cst.AugAssign) -> None:  # noqa: N802
        """``x += 1`` rebinds ``x``."""
        if isinstance(node.target, cst.Name):
            self.names.add(node.target.value)


class _AssignmentTargets(cst.CSTVisitor):
    """Attribute expressions that a subtree assigns to."""

    def __init__(self) -> None:
        self.targets: list[cst.BaseExpression] = []

    # libcst dispatches by this exact CamelCase hook name.
    def visit_AssignTarget(self, node: cst.AssignTarget) -> None:  # noqa: N802
        """Record the assigned expression (a name, attribute, tuple, ...)."""
        self.targets.append(node.target)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:  # noqa: N802
        """Annotated assignment targets are bindings too."""
        self.targets.append(node.target)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_AugAssign(self, node: cst.AugAssign) -> None:  # noqa: N802
        """Augmented assignment targets are bindings too."""
        self.targets.append(node.target)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_For(self, node: cst.For) -> None:  # noqa: N802
        """The loop variable is rebound on every iteration."""
        self.targets.append(node.target)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_CompFor(self, node: cst.CompFor) -> None:  # noqa: N802
        """A comprehension rebinds its own target."""
        self.targets.append(node.target)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_NamedExpr(self, node: cst.NamedExpr) -> None:  # noqa: N802
        """A walrus rebinds its target."""
        self.targets.append(node.target)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_WithItem(self, node: cst.WithItem) -> None:  # noqa: N802
        """``with ... as target`` rebinds the target."""
        if node.asname is not None:
            self.targets.append(node.asname.name)

    # libcst dispatches by this exact CamelCase hook name.
    def visit_Del(self, node: cst.Del) -> None:  # noqa: N802
        """``del x`` removes the binding the chain is matching."""
        self.targets.append(node.target)


def _assert_subject_stable(
    module: cst.Module, subject: cst.BaseExpression, head: cst.If
) -> None:
    """Refuse a chain whose branches rebind the expression being matched.

    The original chain re-reads the subject for every test; ``match`` reads it
    once. A branch that assigns the subject would therefore change behaviour, so
    the conversion is refused instead of producing code that looks equivalent.
    """
    subject_code = module.code_for_node(subject)
    root = _root_name(subject)
    if root is None:  # pragma: no cover - _assert_simple_subject already refused
        return

    for branch in _branches(head):
        bound = _BoundNames()
        branch.body.visit(bound)
        if bound.star_import:
            raise ValueError(
                "a branch contains 'from ... import *', which can rebind the "
                "matched name without naming it; the conversion cannot prove the "
                "chain equivalent"
            )
        if root in bound.names:
            raise ValueError(
                f"a branch of this chain rebinds {root!r}, so the remaining tests "
                "would see a different value; match evaluates its subject once and "
                "cannot reproduce that"
            )
        targets = _AssignmentTargets()
        branch.body.visit(targets)
        for target in targets.targets:
            for assigned in _flatten_target(target):
                code = module.code_for_node(assigned)
                if subject_code == code or subject_code.startswith(f"{code}."):
                    raise ValueError(
                        f"a branch assigns to {code!r} while the chain matches "
                        f"{subject_code!r}; the conversion would change what later "
                        "cases compare against"
                    )


def _flatten_target(target: cst.BaseExpression) -> Iterator[cst.BaseExpression]:
    """Yield a target and, for tuples/lists, every element inside it."""
    if isinstance(target, (cst.Tuple, cst.List)):
        for element in target.elements:
            if element is not None:
                yield from _flatten_target(element.value)
    else:
        yield target


def _comment_lines(lines: Sequence[cst.EmptyLine]) -> list[cst.EmptyLine]:
    """Keep comments from ``elif``/``else`` leading lines, drop blank spacing."""
    return [line for line in lines if line.comment is not None]


class _IfToMatch(cst.CSTTransformer):
    """Replace exactly one ``if`` chain with the generated ``match`` statement."""

    def __init__(self, target: cst.If, replacement: cst.Match) -> None:
        self._target = target
        self._replacement = replacement

    # libcst dispatches by this exact CamelCase hook name. ``Match`` is a
    # BaseCompoundStatement, so the surrounding body accepts it; libcst's type
    # hint for this hook only mentions If nodes, hence the targeted ignore.
    def leave_If(  # noqa: N802
        self, original_node: cst.If, updated_node: cst.If
    ) -> cst.If:
        """Swap the chain head; every other if statement is left untouched."""
        if original_node is self._target:
            return self._replacement  # type: ignore[return-value]
        return updated_node
