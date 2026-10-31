"""Inline a single-return function into every same-module call site.

``def add(a, b): return a + b`` disappears and each ``add(x, y)`` in the same
file becomes ``(x + y)``: the body expression is rebuilt per call site with the
parameters substituted by the caller's arguments, then the definition is
deleted. The substitution is scope-aware — a parameter shadowed by a nested
lambda or comprehension keeps its local meaning and is left alone — and every
grouping hazard is handled with parentheses, never with string splicing.

Refusal rules (each is a way a naive inline would silently change behaviour):

* the name must be one unique top-level ``def`` — methods, nested functions,
  duplicated definitions and ``async def`` refuse;
* the body must be exactly one ``return <expr>`` with a value (so there are no
  statements to reorder and no local bindings to lose);
* a recursive body, and any ``yield`` in the body (inlining would silently
  turn the caller into a generator), refuse outright;
* a named expression (``:=``) in the body refuses: inlining would move the
  binding into the caller's scope;
* a body that calls or awaits is only inlinable when there is exactly one call
  site — with several, the side effect would run once per site;
* a parameter used more than once in the body refuses impure arguments
  (``double(g())`` would evaluate ``g`` twice);
* ``*args`` / ``**kwargs`` signatures and call-site star-arguments refuse;
* the name must be bound exactly once in the module, must never be referenced
  without being called (passed as a value, used as a decorator), and must have
  no callers outside the file, no star-imported re-export and no string-literal
  (dynamic ``getattr``) reference — Jedi and an independent candidate scan
  both have to agree, failing closed.
"""

from __future__ import annotations

import keyword
from collections.abc import Mapping, Sequence
from pathlib import Path

import libcst as cst
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider, ScopeProvider
from libcst.metadata.scope_provider import Assignment

from moray_surgeon.engine.base import EditResult
from moray_surgeon.engine.symbol_guards import (
    assert_no_dynamic_string_refs,
    assert_no_star_import,
    assert_not_referenced_from_other_modules,
)

# Atoms that can stand in any expression position without changing grouping;
# every other substituted shape is wrapped in fresh parentheses, which is
# semantically neutral and never a syntax error.
_PAREN_FREE = (
    cst.Name,
    cst.Integer,
    cst.Float,
    cst.Imaginary,
    cst.SimpleString,
    cst.Call,
    cst.Attribute,
    cst.Subscript,
)

# Body nodes whose evaluation is not obviously side-effect free; they may only
# be duplicated when there is exactly one call site to rewrite.
_IMPURE_INNER = (cst.Call, cst.Await)


def inline_function(
    source: str,
    function_name: str,
    repo_root: Path,
    target_path: Path,
) -> EditResult:
    """Inline the single-return ``function_name`` into all same-file call sites.

    Args:
        source: Python module source of ``target_path``.
        function_name: Name of the top-level function to inline.
        repo_root: Repository root, used for the project-wide reference proof.
        target_path: Absolute path of the file being edited.

    Returns:
        An :class:`EditResult` whose ``changes`` counts the rewritten call
        sites plus the deleted definition.

    Raises:
        ValueError: Under any of the refusal rules in the module docstring; the
            source is left untouched.
    """
    if not function_name.isidentifier() or keyword.iskeyword(function_name):
        raise ValueError(
            f"function_name must be a valid Python identifier, got {function_name!r}"
        )

    wrapper = MetadataWrapper(cst.parse_module(source))
    positions = wrapper.resolve(PositionProvider)
    scopes = wrapper.resolve(ScopeProvider)

    node = _find_unique_function(wrapper, target_path, function_name)
    span = positions[node]
    name_start = positions[node.name].start
    definition_start = (name_start.line, name_start.column)

    if node.decorators:
        raise ValueError(
            f"{function_name!r} has decorators; a decorated function may cache or "
            "register its result, which inlining would lose"
        )
    if node.asynchronous is not None:
        raise ValueError(f"{function_name!r} is async; async functions cannot be inlined")

    body_expr = _single_return_expression(node, function_name)
    _assert_single_binding(scopes, wrapper.module, node, function_name)
    _assert_not_recursive(body_expr, scopes, node, function_name)
    _assert_no_named_expression(body_expr)
    _assert_no_yield(body_expr)
    call_sites = _find_call_sites(wrapper, positions, scopes, node, span, function_name)
    if len(call_sites) > 1 and _contains_impure(body_expr):
        raise ValueError(
            f"the body of {function_name!r} calls or awaits, and there are "
            f"{len(call_sites)} call sites; inlining would run the side effect per site"
        )
    _assert_not_referenced_without_call(wrapper, positions, scopes, node, span, call_sites)
    _assert_no_duplicated_evaluation(node, body_expr, call_sites, function_name)

    assert_not_referenced_from_other_modules(
        repo_root,
        target_path,
        function_name,
        definition_start,
        span,
        "inline",
        include_same_file=False,
    )
    assert_no_dynamic_string_refs(repo_root, target_path, function_name, span, "inline")
    assert_no_star_import(repo_root, target_path, "inline")

    # The function scope (keyed by the body node) holds one binding per
    # parameter; libcst gives parameter assignments their own Name nodes, so
    # referents are matched by *assignment identity*, never by node identity.
    function_scope = scopes[node.body]
    param_bindings = {
        binding.name: binding
        for binding in function_scope.assignments  # type: ignore[union-attr]
        if binding.name in {param.name.value for param in node.params.params}
    }
    substitution_ids = {
        id(inner)
        for inner in _walk(body_expr)
        if isinstance(inner, cst.Name)
        and inner.value in param_bindings
        and any(
            referent is param_bindings[inner.value]
            for referent in _referents_of(scopes, inner)
        )
    }

    transformer = _InlineCallTransformer(
        function_name=function_name,
        span=span,
        body_expr=body_expr,
        substitution_ids=substitution_ids,
        params=node.params,
        positions=positions,
        definition=node,
    )
    new_module = wrapper.visit(transformer)

    return EditResult(new_source=new_module.code, changes=transformer.changes)


def _find_unique_function(
    wrapper: MetadataWrapper,
    target_path: Path,
    function_name: str,
) -> cst.FunctionDef:
    """Return the one top-level ``def`` spelled ``function_name``."""
    candidates = [
        statement
        for statement in wrapper.module.body
        if isinstance(statement, cst.FunctionDef) and statement.name.value == function_name
    ]
    if not candidates:
        raise ValueError(
            f"no top-level function named {function_name!r} was found in "
            f"{target_path.name}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"{function_name!r} is defined {len(candidates)} times at the top level; "
            "refusing to guess which one to inline"
        )
    return candidates[0]


def _single_return_expression(node: cst.FunctionDef, function_name: str) -> cst.BaseExpression:
    """The returned expression of a function whose body is one ``return``.

    Both body shapes are accepted: the usual ``IndentedBlock`` and the one-line
    ``def f(): return <expr>`` suite.
    """
    if isinstance(node.body, cst.SimpleStatementSuite):
        smalls: Sequence[cst.BaseSmallStatement] = node.body.body
        if len(smalls) == 1 and isinstance(smalls[0], cst.Return):
            value = smalls[0].value
            if value is not None:
                return value
        raise ValueError(
            f"the body of {function_name!r} must be exactly one return statement "
            "with a value"
        )
    statements = node.body.body
    if len(statements) != 1 or not isinstance(statements[0], cst.SimpleStatementLine):
        raise ValueError(
            f"the body of {function_name!r} must be exactly one return statement; "
            "functions with several statements cannot be inlined"
        )
    smalls = statements[0].body
    if len(smalls) != 1 or not isinstance(smalls[0], cst.Return):
        raise ValueError(
            f"the body of {function_name!r} must be exactly one return statement"
        )
    value = smalls[0].value
    if value is None:
        raise ValueError(f"{function_name!r} returns no value; there is nothing to inline")
    return value


def _assert_single_binding(
    scopes: Mapping[cst.CSTNode, object],
    module: cst.Module,
    node: cst.FunctionDef,
    function_name: str,
) -> None:
    """The name must have exactly one module-level binding: this definition."""
    scope = scopes.get(module)
    if scope is None:  # pragma: no cover - every module has a scope
        raise ValueError(f"could not determine the scope of {function_name!r}")
    bindings = [
        binding
        for binding in scope.assignments  # type: ignore[attr-defined]
        if binding.name == function_name
    ]
    if len(bindings) != 1 or not (
        isinstance(bindings[0], Assignment) and bindings[0].node is node
    ):
        raise ValueError(
            f"{function_name!r} is bound more than once in the module (reassigned, "
            "imported or shadowed); inlining would be ambiguous"
        )


def _assert_not_recursive(
    body_expr: cst.BaseExpression,
    scopes: Mapping[cst.CSTNode, object],
    node: cst.FunctionDef,
    function_name: str,
) -> None:
    """A body that calls the function itself cannot be inlined."""
    for inner in _walk(body_expr):
        if not isinstance(inner, cst.Call) or not isinstance(inner.func, cst.Name):
            continue
        if inner.func.value != function_name:
            continue
        if _resolves_to(inner.func, node, scopes):
            raise ValueError(
                f"{function_name!r} calls itself; a recursive function cannot be inlined"
            )


def _assert_no_named_expression(body_expr: cst.BaseExpression) -> None:
    """``:=`` inside the body would bind in the caller's scope after inlining."""
    for inner in _walk(body_expr):
        if isinstance(inner, cst.NamedExpr):
            raise ValueError(
                "the body contains a named expression (:=); inlining would move "
                "the binding into the caller's scope"
            )


def _assert_no_yield(body_expr: cst.BaseExpression) -> None:
    """A ``yield`` in the body would silently make the caller a generator."""
    for inner in _walk(body_expr):
        if isinstance(inner, cst.Yield):
            raise ValueError(
                "the body contains a yield; inlining would turn the caller into "
                "a generator function"
            )


def _contains_impure(body_expr: cst.BaseExpression) -> bool:
    """True when the expression calls or awaits."""
    return any(isinstance(inner, _IMPURE_INNER) for inner in _walk(body_expr))


def _resolves_to(
    name_node: cst.Name,
    definition_node: cst.CSTNode,
    scopes: Mapping[cst.CSTNode, object],
) -> bool:
    """True when this name resolves to the given definition binding."""
    return any(
        isinstance(referent, Assignment) and referent.node is definition_node
        for referent in _referents_of(scopes, name_node)
    )


def _referents_of(
    scopes: Mapping[cst.CSTNode, object],
    name_node: cst.Name,
) -> list[object]:
    """The assignments the name resolves to, from the scope analysis accesses."""
    for scope in scopes.values():
        if scope is None:
            continue
        for access in scope.accesses:  # type: ignore[attr-defined]
            if access.node is name_node:
                return list(access.referents)
    return []


def _find_call_sites(
    wrapper: MetadataWrapper,
    positions: Mapping[cst.CSTNode, CodeRange],
    scopes: Mapping[cst.CSTNode, object],
    node: cst.FunctionDef,
    span: CodeRange,
    function_name: str,
) -> list[cst.Call]:
    """Every same-module call of the function, outside its own span.

    A call counts only when its callee name resolves to this function's
    definition — a same-named binding in a nested scope is left alone.
    """
    call_sites: list[cst.Call] = []
    for inner in _walk(wrapper.module):
        if not isinstance(inner, cst.Call) or not isinstance(inner.func, cst.Name):
            continue
        if inner.func.value != function_name:
            continue
        position = positions[inner]
        if span.start.line <= position.start.line <= span.end.line:
            continue  # inside the deleted definition (e.g. a default argument)
        if _resolves_to(inner.func, node, scopes):
            call_sites.append(inner)
    return call_sites


def _assert_no_duplicated_evaluation(
    node: cst.FunctionDef,
    body_expr: cst.BaseExpression,
    call_sites: Sequence[cst.Call],
    function_name: str,
) -> None:
    """A parameter used twice in the body refuses side-effecting arguments.

    ``def add_dup(a): return a + a`` inlined at ``add_dup(get())`` becomes
    ``get() + get()``: the argument would run twice. Pure arguments are fine —
    the value is the same, only the computation repeats.
    """
    use_counts: dict[str, int] = {}
    for inner in _walk(body_expr):
        if isinstance(inner, cst.Name):
            use_counts[inner.value] = use_counts.get(inner.value, 0) + 1
    multi_use = {name for name, count in use_counts.items() if count > 1}
    for call in call_sites:
        binding = _bind_arguments(node.params, call, function_name)
        for name in sorted(multi_use & binding.keys()):
            if _contains_impure(binding[name]):
                raise ValueError(
                    f"the parameter {name!r} is used {use_counts[name]} times in "
                    f"the body of {function_name!r} and the call site passes a "
                    "side-effecting argument; inlining would evaluate it more "
                    "than once"
                )


def _assert_not_referenced_without_call(
    wrapper: MetadataWrapper,
    positions: Mapping[cst.CSTNode, CodeRange],
    scopes: Mapping[cst.CSTNode, object],
    node: cst.FunctionDef,
    span: CodeRange,
    call_sites: Sequence[cst.Call],
) -> None:
    """A bare reference (alias, decorator, container entry) cannot be inlined."""
    call_func_ids = {id(call.func) for call in call_sites}
    for inner in _walk(wrapper.module):
        if not isinstance(inner, cst.Name) or inner.value != node.name.value:
            continue
        position = positions[inner]
        if span.start.line <= position.start.line <= span.end.line:
            continue  # inside the deleted definition
        if id(inner) in call_func_ids:
            continue  # a call site; rewritten by the transformer
        if _resolves_to(inner, node, scopes):
            raise ValueError(
                f"{node.name.value!r} is referenced without being called (passed "
                "as a value, aliased or used as an attribute target); inlining "
                "would leave that reference dangling"
            )


def _walk(node: cst.CSTNode) -> list[cst.CSTNode]:
    """Collect ``node`` and every descendant without the visitor ceremony."""
    nodes: list[cst.CSTNode] = []
    stack: list[cst.CSTNode] = [node]
    while stack:
        current = stack.pop()
        nodes.append(current)
        stack.extend(current.children)
    return nodes


class _InlineCallTransformer(cst.CSTTransformer):
    """Rewrite every call site, then delete the definition.

    The deletion happens here — inside ``leave_Module`` where original and
    updated statements can be paired by identity — because a post-visit filter
    cannot rely on node identity surviving the transformation.
    """

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        function_name: str,
        span: CodeRange,
        body_expr: cst.BaseExpression,
        substitution_ids: set[int],
        params: cst.Parameters,
        positions: Mapping[cst.CSTNode, CodeRange],
        definition: cst.FunctionDef,
    ) -> None:
        self._function_name = function_name
        self._span = span
        self._body_expr = body_expr
        self._substitution_ids = substitution_ids
        self._params = params
        self._positions = positions
        self._definition = definition
        self.changes = 0

    def leave_Call(  # noqa: N802
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        """Replace a call of the function with its substituted body expression."""
        if not isinstance(original_node.func, cst.Name):
            return updated_node
        if original_node.func.value != self._function_name:
            return updated_node
        position = self._positions[original_node]
        if self._span.start.line <= position.start.line <= self._span.end.line:
            return updated_node  # inside the deleted definition
        substituted = _substitute_call(
            self._body_expr,
            self._substitution_ids,
            self._params,
            original_node,
            self._function_name,
        )
        self.changes += 1
        return substituted

    def leave_Module(  # noqa: N802
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        """Delete the inlined definition, keeping its header comments readable."""
        for index, statement in enumerate(original_node.body):
            if statement is not self._definition:
                continue
            self.changes += 1
            body = list(updated_node.body)
            body.pop(index)
            if not body:
                return updated_node.with_changes(body=(), header=(), footer=())
            if index == 0:
                # Comments above the first statement live in Module.header;
                # hand them to the next surviving statement instead of
                # destroying them, and drop the now-dangling blank lines.
                header_comments = [
                    empty_line for empty_line in updated_node.header if empty_line.comment
                ]
                following = body[0]
                kept_leading = list(following.leading_lines)
                while kept_leading and not kept_leading[0].comment:
                    kept_leading.pop(0)
                body[0] = following.with_changes(leading_lines=[*header_comments, *kept_leading])
                return updated_node.with_changes(body=body, header=())
            self.changes += 1
            return updated_node.with_changes(body=body)
        return updated_node  # pragma: no cover - the definition is always found


def _substitute_call(
    body_expr: cst.BaseExpression,
    substitution_ids: set[int],
    params: cst.Parameters,
    call: cst.Call,
    function_name: str,
) -> cst.BaseExpression:
    """Rebuild the body expression with this call's arguments substituted in."""
    arguments = _bind_arguments(params, call, function_name)
    replacer = _ArgumentSubstitution(substitution_ids, arguments)
    substituted = body_expr.visit(replacer)
    if not isinstance(substituted, cst.BaseExpression):  # pragma: no cover - defensive
        raise ValueError("substitution produced a non-expression node")
    return _parenthesise_body(substituted)


class _ArgumentSubstitution(cst.CSTTransformer):
    """Swap parameter references for the caller's argument expressions."""

    def __init__(
        self,
        substitution_ids: set[int],
        arguments: Mapping[str, cst.BaseExpression],
    ) -> None:
        self._substitution_ids = substitution_ids
        self._arguments = arguments

    def leave_Name(  # noqa: N802
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.BaseExpression:
        if id(original_node) not in self._substitution_ids:
            return updated_node
        return _parenthesise_argument(self._arguments[original_node.value])


def _parenthesise_argument(argument: cst.BaseExpression) -> cst.BaseExpression:
    """Wrap a non-atom argument so grouping survives the substitution."""
    lpar = getattr(argument, "lpar", None)
    if isinstance(argument, _PAREN_FREE) or lpar:
        return argument
    return argument.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])


def _parenthesise_body(substituted: cst.BaseExpression) -> cst.BaseExpression:
    """Wrap the whole substituted body when precedence requires it."""
    lpar = getattr(substituted, "lpar", None)
    if isinstance(substituted, _PAREN_FREE) or lpar:
        return substituted
    return substituted.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])


def _bind_arguments(
    params: cst.Parameters,
    call: cst.Call,
    function_name: str,
) -> Mapping[str, cst.BaseExpression]:
    """Map every parameter to this call's argument expression.

    Positional arguments fill the parameters in order, keyword arguments fill
    by name, and defaults stand in for anything the caller left out. Star
    arguments on either side refuse: their expansion cannot be expressed as a
    per-parameter substitution.
    """
    if isinstance(params.star_arg, cst.Param) or isinstance(params.star_kwarg, cst.Param):
        raise ValueError(
            f"{function_name!r} has a *args/**kwargs signature; its parameters "
            "cannot be mapped onto call-site arguments"
        )
    if params.kwonly_params:
        raise ValueError(
            f"{function_name!r} has keyword-only parameters, which the inline "
            "mapping does not support"
        )

    names = [param.name.value for param in params.params]
    binding: dict[str, cst.BaseExpression] = {}
    positional: list[cst.Arg] = []
    for argument in call.args:
        if argument.star:
            raise ValueError(
                "the call site uses star-arguments, which cannot be mapped onto "
                f"the parameters of {function_name!r}"
            )
        if argument.keyword is None:
            positional.append(argument)
        elif isinstance(argument.keyword, cst.Name):
            keyword_name = argument.keyword.value
            if keyword_name not in names:
                raise ValueError(
                    f"a call site passes unknown keyword argument {keyword_name!r}; "
                    f"{function_name!r} has no such parameter"
                )
            if keyword_name in binding:
                raise ValueError(
                    f"a call site binds the parameter {keyword_name!r} twice; the "
                    "call would not execute as written"
                )
            binding[keyword_name] = argument.value
        else:
            raise ValueError(
                "the call site passes a non-name keyword argument; cannot map it "
                f"onto the parameters of {function_name!r}"
            )

    if len(positional) > len(names):
        raise ValueError(
            f"a call site passes {len(positional)} positional arguments but "
            f"{function_name!r} only accepts {len(names)}"
        )
    for param, argument in zip(
        params.params[: len(positional)], positional, strict=True
    ):
        # Positional binding happens after the keyword sweep so a call like
        # ``f(1, a=2)`` is caught as a duplicate, not silently overwritten.
        if param.name.value in binding:
            raise ValueError(
                f"a call site binds the parameter {param.name.value!r} twice; the "
                "call would not execute as written"
            )
        binding[param.name.value] = argument.value

    for param in params.params:
        if param.name.value in binding:
            continue
        # A parsed parameter without a default carries ``None`` here (this
        # libcst build), while a hand-built one carries MaybeSentinel.DEFAULT;
        # both mean "the caller must supply this argument".
        if param.default is None or isinstance(param.default, cst.MaybeSentinel):
            raise ValueError(
                f"a call site does not pass the parameter {param.name.value!r} and "
                "it has no default; the call would not execute as written"
            )
        binding[param.name.value] = param.default
    return binding
