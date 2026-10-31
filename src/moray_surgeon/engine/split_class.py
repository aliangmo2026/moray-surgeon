"""Split one class into two by moving selected methods to a new sibling.

``SplitClass(Person, ["greet"], PersonGreeter)`` removes ``greet`` from
``Person`` and appends a ``PersonGreeter`` class right after it, holding the
moved methods verbatim — decorators, docstrings and leading comments travel
with them.

This iteration deliberately refuses anything instance-shaped. A moved method
that references its receiver (``self`` / ``cls`` — by any spelling the method's
first parameter uses) or calls ``super()`` depends on the class it leaves
behind: the new class's instances would lack the state and the MRO the method
expects. Only receiver-free methods — static helpers or pure parameter
functions — may move. Wiring the two classes back together (delegation,
inheritance, ``__init__`` synthesis) is future work.
"""

from __future__ import annotations

import keyword
from collections.abc import Sequence

import libcst as cst
from libcst.metadata import MetadataWrapper, ScopeProvider

from moray_surgeon.engine.base import EditResult


def split_class(
    source: str,
    class_name: str,
    methods_to_move: Sequence[str],
    new_class_name: str,
) -> EditResult:
    """Move ``methods_to_move`` out of ``class_name`` into a new sibling class.

    Args:
        source: Python module source.
        class_name: Name of the top-level class to split.
        methods_to_move: Names of the methods to move; every one must exist as
            a direct method of the class.
        new_class_name: Name of the class that receives the methods; it must
            not already be bound in the module and must differ from
            ``class_name``.

    Returns:
        An :class:`EditResult` whose ``changes`` counts the moved methods plus
        the created class.

    Raises:
        ValueError: Under any of the refusal rules in the module docstring, or
            when a name is not an identifier; the source is left untouched.
    """
    names = list(methods_to_move)
    for name in (*names, class_name, new_class_name):
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"expected a Python identifier, got {name!r}")
    if new_class_name == class_name:
        raise ValueError("new_class_name must differ from class_name")
    if len(set(names)) != len(names):
        raise ValueError("methods_to_move must not repeat a method name")

    wrapper = MetadataWrapper(cst.parse_module(source))
    scopes = wrapper.resolve(ScopeProvider)

    node = _find_unique_class(wrapper, class_name)
    _assert_new_name_is_free(scopes, wrapper.module, new_class_name)

    moved, missing = _collect_methods(node, names)
    if missing:
        raise ValueError(
            f"{class_name!r} has no method(s) {missing}; only its own methods can move"
        )
    for method in moved:
        _assert_method_is_receiver_free(method, class_name, new_class_name)

    remaining = _without_methods(node, moved)
    new_class = _build_new_class(new_class_name, moved)

    body = list(wrapper.module.body)
    index = body.index(node)
    body[index] = node.with_changes(body=remaining)
    body.insert(index + 1, new_class)
    return EditResult(
        new_source=wrapper.module.with_changes(body=body).code,
        changes=len(moved) + 1,
    )


def _find_unique_class(wrapper: MetadataWrapper, class_name: str) -> cst.ClassDef:
    """Return the one top-level ``class`` spelled ``class_name``."""
    candidates = [
        statement
        for statement in wrapper.module.body
        if isinstance(statement, cst.ClassDef) and statement.name.value == class_name
    ]
    if not candidates:
        raise ValueError(f"no top-level class named {class_name!r} was found")
    if len(candidates) > 1:
        raise ValueError(
            f"{class_name!r} is defined {len(candidates)} times at the top level; "
            "refusing to guess which one to split"
        )
    return candidates[0]


def _assert_new_name_is_free(
    scopes: object,
    module: cst.Module,
    new_class_name: str,
) -> None:
    """The new class name must not collide with any module-level binding."""
    scope = scopes.get(module)  # type: ignore[attr-defined]
    if scope is None:  # pragma: no cover - every module has a scope
        raise ValueError(f"could not analyse the module scope for {new_class_name!r}")
    if any(binding.name == new_class_name for binding in scope.assignments):  # type: ignore[attr-defined]
        raise ValueError(
            f"{new_class_name!r} is already bound in this module; the new class "
            "would shadow it"
        )


def _collect_methods(
    node: cst.ClassDef,
    names: Sequence[str],
) -> tuple[list[cst.FunctionDef], list[str]]:
    """Pair the requested names with the class's own methods, in class order."""
    wanted = set(names)
    methods = [
        statement
        for statement in node.body.body
        if isinstance(statement, cst.FunctionDef) and statement.name.value in wanted
    ]
    found = {method.name.value for method in methods}
    missing = [name for name in names if name not in found]
    return methods, missing


def _assert_method_is_receiver_free(
    method: cst.FunctionDef,
    class_name: str,
    new_class_name: str,
) -> None:
    """A method that touches its receiver cannot move to a stateless class."""
    if any(
        isinstance(inner, cst.Call)
        and isinstance(inner.func, cst.Name)
        and inner.func.value == "super"
        for inner in _walk(method)
    ):
        raise ValueError(
            f"{method.name.value!r} calls super(); the new class would resolve a "
            "different MRO"
        )
    if not method.params.params:
        return  # no receiver at all: nothing to depend on
    receiver_param = method.params.params[0]
    receiver = receiver_param.name.value
    if receiver not in ("self", "cls"):
        # A receiver-shaped parameter is the only instance dependency this
        # iteration reasons about; plain parameters just travel with the
        # method ("methods that only take arguments" are allowed to move).
        return
    receiver_node_id = id(receiver_param.name)
    uses = any(
        isinstance(inner, cst.Name)
        and inner.value == receiver
        and id(inner) != receiver_node_id
        for inner in _walk(method)
    )
    if uses:
        raise ValueError(
            f"{method.name.value!r} uses its receiver {receiver!r}; instance state "
            f"cannot move to {new_class_name!r} in this iteration"
        )


def _without_methods(
    node: cst.ClassDef,
    moved: Sequence[cst.FunctionDef],
) -> cst.BaseSuite:
    """The class body without the moved methods; ``pass`` if it empties."""
    moved_ids = {id(method) for method in moved}
    body = [
        statement for statement in node.body.body if id(statement) not in moved_ids
    ]
    if not body:
        body = [cst.SimpleStatementLine(body=[cst.Pass()])]
    return node.body.with_changes(body=body)


def _build_new_class(
    new_class_name: str,
    moved: Sequence[cst.FunctionDef],
) -> cst.ClassDef:
    """Assemble the sibling class holding the moved methods verbatim."""
    body = list(moved)
    first = body[0]
    kept_leading = list(first.leading_lines)
    while kept_leading and not kept_leading[0].comment:
        kept_leading.pop(0)
    body[0] = first.with_changes(leading_lines=kept_leading)
    return cst.ClassDef(
        name=cst.Name(new_class_name),
        body=cst.IndentedBlock(body=body),
        leading_lines=[cst.EmptyLine(), cst.EmptyLine()],
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
