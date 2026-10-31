"""Delete a top-level function or class that nothing in the project uses.

Deadness is decided by two independent analysers that must *both* agree, plus
a set of structural guards for the references Jedi structurally cannot see:

* Jedi ``get_references`` (project scope, budget raised to the repository size)
  must report no reference outside the symbol's own span — self-recursion dies
  with the symbol and is ignored.
* A libcst candidate scan of every in-repo module (same file included) walks
  every name/attribute/import spelled like the symbol and makes Jedi
  ``goto(follow_imports=True)`` resolve it somewhere else. Anything that
  resolves back to the definition, or that cannot be resolved at all, vetoes
  the deletion.
* ``__all__`` entries, any ``__all__`` manipulation, a decorated symbol, a
  star import of the module, and any string literal equal to the symbol name
  (a potential ``getattr`` / ``setattr`` / monkeypatch key) all veto.

Only when every guard passes is the node removed from the module body — with
the comments above it, because they describe the deleted code.
"""

from __future__ import annotations

import keyword
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from moray_surgeon.engine.base import EditResult
from moray_surgeon.engine.symbol_guards import (
    assert_no_dynamic_string_refs,
    assert_no_star_import,
    assert_not_referenced_from_other_modules,
    plain_string_value,
)


def remove_dead_code(
    source: str,
    symbol_name: str,
    repo_root: Path,
    target_path: Path,
) -> EditResult:
    """Delete the unreferenced top-level ``symbol_name`` from ``source``.

    Args:
        source: Python module source of ``target_path``.
        symbol_name: Name of the top-level function or class to delete.
        repo_root: Repository root; the reference search covers it.
        target_path: Absolute path of the file being edited.

    Returns:
        An :class:`EditResult` whose ``changes`` is 1 (the deleted definition,
        comments above it included).

    Raises:
        ValueError: Under any of the refusal rules in the module docstring, or
            when the name is not an identifier, no/multiple matching top-level
            definitions exist. The source is left untouched.
    """
    if not symbol_name.isidentifier() or keyword.iskeyword(symbol_name):
        raise ValueError(
            f"symbol_name must be a valid Python identifier, got {symbol_name!r}"
        )

    wrapper = MetadataWrapper(cst.parse_module(source))
    positions = wrapper.resolve(PositionProvider)

    candidates = [
        statement
        for statement in wrapper.module.body
        if isinstance(statement, (cst.FunctionDef, cst.ClassDef))
        and statement.name.value == symbol_name
    ]
    if not candidates:
        raise ValueError(
            f"no top-level function or class named {symbol_name!r} was found in "
            f"{target_path.name}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"{symbol_name!r} is defined {len(candidates)} times at the top level; "
            "refusing to guess which one is dead"
        )
    node = candidates[0]
    span = positions[node]
    # Jedi resolves references from the *name's* position; the node's own span
    # starts at the ``def``/``class`` keyword, which Jedi does not understand.
    name_start = positions[node.name].start
    definition_start = (name_start.line, name_start.column)

    if node.decorators:
        raise ValueError(
            f"{symbol_name!r} has decorators; the decorator may register it "
            "dynamically, so it cannot be proven dead"
        )

    _assert_not_exported(wrapper.module, symbol_name)
    assert_no_dynamic_string_refs(repo_root, target_path, symbol_name, span, "delete")
    assert_not_referenced_from_other_modules(
        repo_root,
        target_path,
        symbol_name,
        definition_start,
        span,
        "delete",
        include_same_file=True,
    )
    assert_no_star_import(repo_root, target_path, "delete")

    body = [statement for statement in wrapper.module.body if statement is not node]
    if not body:
        # The whole module was the dead symbol; nothing survives, comments
        # included.
        empty = wrapper.module.with_changes(body=(), header=(), footer=())
        return EditResult(new_source=empty.code, changes=1)

    if node is wrapper.module.body[0]:
        # Comments above the first statement live in Module.header, not in the
        # node's leading_lines; hand them to the next surviving statement
        # instead of destroying them, and drop the blank lines that used to
        # separate the deleted block from the rest.
        header_comments = [
            empty_line for empty_line in wrapper.module.header if empty_line.comment
        ]
        following = body[0]
        kept_leading = list(following.leading_lines)
        while kept_leading and not kept_leading[0].comment:
            kept_leading.pop(0)
        body[0] = following.with_changes(leading_lines=[*header_comments, *kept_leading])
        new_module = wrapper.module.with_changes(
            body=body,
            header=[
                empty_line for empty_line in wrapper.module.header if not empty_line.comment
            ],
        )
    else:
        new_module = wrapper.module.with_changes(body=body)
    return EditResult(new_source=new_module.code, changes=1)


def _assert_not_exported(module: cst.Module, symbol_name: str) -> None:
    """Refuse when ``__all__`` mentions the symbol or is manipulated at all."""
    for statement in module.body:
        if isinstance(statement, (cst.SimpleStatementLine, cst.SimpleStatementSuite)):
            for small in statement.body:
                _audit_all_assignment(small, symbol_name)


def _audit_all_assignment(small: cst.BaseSmallStatement, symbol_name: str) -> None:
    """Check one small statement for ``__all__`` exports or manipulation."""
    # ``__all__.append(...)`` and friends build the export list dynamically:
    # the engine cannot know whether the symbol ends up in it.
    if isinstance(small, cst.Expr) and isinstance(small.value, cst.Call):
        func = small.value.func
        if (
            isinstance(func, cst.Attribute)
            and isinstance(func.value, cst.Name)
            and func.value.value == "__all__"
        ):
            raise ValueError(
                "__all__ is manipulated dynamically (__all__.append/extend); the "
                "export list cannot be proven not to contain the symbol"
            )
    if isinstance(small, cst.Assign):
        if any(
            isinstance(target, cst.AssignTarget)
            and isinstance(target.target, cst.Name)
            and target.target.value == "__all__"
            for target in small.targets
        ):
            _refuse_exported_string(small.value, symbol_name)
    elif (
        isinstance(small, cst.AugAssign)
        and isinstance(small.target, cst.Name)
        and small.target.value == "__all__"
    ):
        _refuse_exported_string(small.value, symbol_name)


def _refuse_exported_string(value: cst.BaseExpression, symbol_name: str) -> None:
    """Refuse when an ``__all__`` value contains the symbol's name."""
    for inner in _walk(value):
        if isinstance(inner, cst.SimpleString) and plain_string_value(inner) == symbol_name:
            raise ValueError(
                f"{symbol_name!r} is exported by __all__; deleting it would "
                "break the module's public contract"
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
