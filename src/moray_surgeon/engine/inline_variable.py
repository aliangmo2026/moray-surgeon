"""Inline a single-assignment (SSA) variable into every reference.

``x = <expr>`` disappears and every later read of ``x`` in the same scope is
rewritten to ``<expr>``. libcst's scope analysis answers *which* reads belong
to the assignment (scope-aware, unlike text search); Jedi answers whether any
*other module* reads it. Only when both agree the variable is safely
replaceable does the edit happen — anything ambiguous is refused with
``ValueError``, never rewritten approximately.

Refusal rules (each is a way a naive inline would silently change behaviour):

* the name has no simple ``x = <expr>`` assignment, or has more than one
  candidate assignment (rebinding, or the same name bound in several scopes);
* anything else binds the name in the assignment's scope: augmented or
  annotated assignment, ``for`` / ``with`` / ``except`` / comprehension
  targets, imports, ``def`` / ``class``;
* a read exists before the assignment, or the right-hand side reads the
  variable itself;
* a read lives behind a function, class or lambda boundary (the inlined
  expression would start evaluating in a scope where its free names may
  resolve differently);
* a read lives inside a comprehension whose bound names collide with the free
  names of ``<expr>``;
* a free name of ``<expr>`` is rebound between the assignment and any read
  (later reads would pick up the new value instead of the captured one);
* ``<expr>`` calls, awaits or assigns and there is more than one read (the
  side effect would run once per read);
* ``del x`` or a ``global`` / ``nonlocal x`` declaration touches the variable
  in its scope;
* the variable is referenced from another module, or Jedi could not prove it
  is not (failing closed);
* the assignment sits in a one-line suite, where deleting it would leave an
  empty body.
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import jedi
import libcst as cst
from libcst.metadata import (
    BuiltinScope,
    ComprehensionScope,
    MetadataWrapper,
    ParentNodeProvider,
    PositionProvider,
    Scope,
    ScopeProvider,
)
from libcst.metadata.scope_provider import Assignment

from moray_surgeon.engine.base import EditResult
from moray_surgeon.engine.rename import _SKIP_DIR_NAMES

# Block bodies hold exactly these two statement shapes; both carry leading_lines.
_BlockT = TypeVar("_BlockT", cst.Module, cst.IndentedBlock)

# Right-hand sides with these nodes are not obviously side-effect free; they
# may only be duplicated when there is exactly one read to rewrite.
_IMPURE_INNER = (cst.Call, cst.Await, cst.NamedExpr, cst.Yield)

# Atoms that can be dropped into any expression position without changing
# grouping; everything else is wrapped in parentheses at every read site.
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


def inline_variable(
    source: str,
    variable_name: str,
    repo_root: Path,
    target_file: Path,
) -> EditResult:
    """Inline the only assignment of ``variable_name`` into all of its reads.

    The assignment statement is removed (comments above it move to the next
    statement) and every read is replaced by the assignment's right-hand side,
    parenthesised when precedence requires it::

        total = price * count     ->  (line deleted)
        print(total)                  print((price * count))

    Args:
        source: Python module source of ``target_file``.
        variable_name: The variable to inline; must be a valid identifier.
        repo_root: Repository root, used for the project-wide Jedi reference
            search that proves no other module reads the variable.
        target_file: Absolute path of the file being edited (already contained
            in ``repo_root`` by the dispatcher).

    Returns:
        An :class:`EditResult` whose ``changes`` counts the rewritten reads
        plus the removed assignment.

    Raises:
        ValueError: Under any of the refusal rules listed in the module
            docstring; the source is left untouched.
    """
    if not variable_name.isidentifier() or keyword.iskeyword(variable_name):
        raise ValueError(
            f"variable_name must be a valid Python identifier, got {variable_name!r}"
        )

    wrapper = MetadataWrapper(cst.parse_module(source))
    scan = _scan(wrapper, variable_name)

    replacement = _inline_expression(scan.assign.value)
    free_names = _all_name_values(scan.assign.value)

    _assert_reads_are_inlinable(wrapper, scan, free_names)
    _assert_free_names_are_stable(wrapper, scan, free_names)
    _assert_not_referenced_from_other_modules(
        repo_root, target_file, variable_name, scan.assign_start
    )

    transformer = _InlineVariableTransformer(
        scan.assign_start,
        scan.read_starts,
        replacement,
    )
    new_module = wrapper.visit(transformer)
    return EditResult(new_source=new_module.code, changes=transformer.changes)


@dataclass(frozen=True)
class _Scan:
    """Everything the guards and the transformer need about one candidate."""

    assign: cst.Assign
    target_name: cst.Name
    assign_start: tuple[int, int]
    assign_end: tuple[int, int]
    reads: list[cst.Name]
    read_starts: set[tuple[int, int]]


def _scan(wrapper: MetadataWrapper, variable_name: str) -> _Scan:
    """Locate the candidate assignment and prove it is the name's only binding.

    The scope analysis is the source of truth for what counts as a binding: a
    plain ``Assign`` is the only form this atom can delete cleanly, and the
    assignment's scope must contain exactly one binding of the name (ours).
    """
    positions = wrapper.resolve(PositionProvider)
    scopes = wrapper.resolve(ScopeProvider)
    parents = wrapper.resolve(ParentNodeProvider)

    candidates: list[cst.Assign] = []

    class _CandidateScan(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (ParentNodeProvider,)

        def visit_Assign(self, node: cst.Assign) -> bool:  # noqa: N802
            if _is_simple_assign_of(node, variable_name):
                if isinstance(parents.get(node), cst.SimpleStatementSuite):
                    raise ValueError(
                        f"{variable_name!r} is assigned inside a one-line suite; "
                        "deleting the assignment would leave an empty body"
                    )
                candidates.append(node)
            return True

    wrapper.visit(_CandidateScan())

    if not candidates:
        raise ValueError(
            f"no simple assignment {variable_name!r} = <expr> was found; "
            "only single-target plain assignments can be inlined"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"found {len(candidates)} assignments of {variable_name!r}; inlining "
            "requires exactly one (SSA form)"
        )

    assign = candidates[0]
    target_name = assign.targets[0].target
    if not isinstance(target_name, cst.Name):  # guaranteed by _is_simple_assign_of
        raise ValueError(  # pragma: no cover - defensive
            f"{variable_name!r} is not assigned to a plain name"
        )
    scope = scopes[target_name]
    if scope is None:  # pragma: no cover - every expression node has a scope
        raise ValueError(f"could not determine the scope of {variable_name!r}")

    bindings = [
        binding
        for binding in scope.assignments
        if binding.name == variable_name and isinstance(binding, Assignment)
    ]
    if len(bindings) != 1 or bindings[0].node is not target_name:
        raise ValueError(
            f"{variable_name!r} is bound more than once in its scope "
            "(reassigned, imported, defined or loop-bound); not in SSA form"
        )

    assign_start = positions[target_name].start
    assign_end = positions[assign].end
    reads: list[cst.Name] = []
    for reference in bindings[0].references:
        node = reference.node
        if isinstance(node, cst.Name):
            reads.append(node)
    return _Scan(
        assign=assign,
        target_name=target_name,
        assign_start=(assign_start.line, assign_start.column),
        assign_end=(assign_end.line, assign_end.column),
        reads=reads,
        read_starts={(positions[read].start.line, positions[read].start.column) for read in reads},
    )


def _is_simple_assign_of(node: cst.Assign, variable_name: str) -> bool:
    """True for ``variable_name = <expr>`` with exactly one plain-name target."""
    if len(node.targets) != 1:
        return False
    target = node.targets[0].target
    return isinstance(target, cst.Name) and target.value == variable_name


def _assert_reads_are_inlinable(
    wrapper: MetadataWrapper,
    scan: _Scan,
    free_names: set[str],
) -> None:
    """Every read must sit in the assignment's scope, after it, and be safe."""
    positions = wrapper.resolve(PositionProvider)
    scopes = wrapper.resolve(ScopeProvider)
    parents = wrapper.resolve(ParentNodeProvider)
    scope = scopes[scan.target_name]
    if scope is None:  # pragma: no cover - every expression node has a scope
        raise ValueError(f"could not determine the scope of {scan.target_name.value!r}")

    # libcst cannot resolve a read that happens before any visible binding:
    # a read before the assignment, or the variable read inside its own
    # right-hand side. Both would change meaning under inlining.
    for access in scope.accesses:
        node = access.node
        if not isinstance(node, cst.Name):
            continue
        if node.value == scan.target_name.value and not access.referents:
            start = positions[node].start
            raise ValueError(
                f"{scan.target_name.value!r} has a read libcst cannot resolve to any "
                f"binding (line {start.line}); the code reads the variable before it "
                "is assigned or inside its own assignment"
            )

    if not scan.reads:
        return

    for read in scan.reads:
        read_start = positions[read].start
        if (read_start.line, read_start.column) <= scan.assign_end:
            raise ValueError(
                f"{scan.target_name.value!r} is read before its assignment finishes "
                f"(line {read_start.line}); inlining would change the program"
            )
        parent = parents.get(read)
        if isinstance(parent, cst.Del):
            raise ValueError(f"{scan.target_name.value!r} is deleted somewhere; refusing")
        read_scope = scopes[read]
        if read_scope is None or not _reaches_without_closure(read_scope, scope):
            raise ValueError(
                f"{scan.target_name.value!r} is read from a nested function or class; "
                "inlining would capture the expression instead of the value"
            )
        _assert_no_shadowing_in_comprehension_or_lambda(parents, read, free_names)

    if len(scan.reads) > 1 and _contains_impure(scan.assign.value):
        raise ValueError(
            "the right-hand side calls, awaits or assigns, and the variable is read "
            f"{len(scan.reads)} times; inlining would run the side effect per read"
        )


def _reaches_without_closure(read_scope: Scope, assign_scope: Scope) -> bool:
    """True when ``read_scope`` is the assignment scope or a comprehension of it.

    Comprehension bodies resolve free names exactly like their enclosing scope,
    so a read there can be replaced safely (modulo shadowing, checked separately).
    Function and class boundaries are refused: the replacement would evaluate
    inside another scope whose bindings may shadow the expression's names.
    """
    current: Scope | None = read_scope
    while current is not None:
        if current is assign_scope:
            return True
        if not isinstance(current, ComprehensionScope):
            return False
        current = getattr(current, "parent", None)
    return False


def _assert_no_shadowing_in_comprehension_or_lambda(
    parents: Mapping[cst.CSTNode, cst.CSTNode],
    read: cst.Name,
    free_names: set[str],
) -> None:
    """Refuse when the read sits behind bindings that would capture the value's names.

    Every lambda parameter and comprehension target between the read and the
    assignment's scope is collected, so a read nested in several comprehensions
    is checked against all of their bound names at once.
    """
    bound: set[str] = set()
    current: cst.CSTNode | None = parents.get(read)
    while current is not None:
        if isinstance(current, cst.Lambda):
            bound.update(_lambda_param_names(current))
        elif isinstance(current, (cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp)):
            bound.update(_comprehension_bound_names(current))
        current = parents.get(current)
    if bound & free_names:
        raise ValueError(
            "the read sits inside a lambda or comprehension that binds names also "
            f"used by the expression: {sorted(bound & free_names)}"
        )


def _assert_free_names_are_stable(
    wrapper: MetadataWrapper,
    scan: _Scan,
    free_names: set[str],
) -> None:
    """No free name of the value may be rebound between assignment and last read.

    ``a = 1; x = a + 1; print(x); a = 2; print(x)`` must not inline: the second
    print would evaluate ``a + 1`` against the new ``a``. The guard walks every
    binding visible from the assignment's scope (its own and its ancestors')
    and refuses any store inside the risky window.
    """
    if not scan.reads:
        return
    positions = wrapper.resolve(PositionProvider)
    scopes = wrapper.resolve(ScopeProvider)

    scope: Scope | None = scopes[scan.target_name]
    last_read = max(scan.read_starts)
    visited: set[int] = set()
    while scope is not None and id(scope) not in visited:
        visited.add(id(scope))
        # Builtins can never be rebound by user code; their scope also forms a
        # self-parented tail on every scope chain.
        if isinstance(scope, BuiltinScope):
            break
        for binding in scope.assignments:
            if binding.name not in free_names:
                continue
            if not isinstance(binding, Assignment):  # pragma: no cover - defensive
                continue
            node = binding.node
            start = positions[node].start
            if scan.assign_start < (start.line, start.column) <= last_read:
                raise ValueError(
                    f"name {binding.name!r} is rebound between the assignment and a "
                    f"read (line {start.line}); inlining would change what the "
                    "expression evaluates to"
                )
        scope = getattr(scope, "parent", None)


def _assert_not_referenced_from_other_modules(
    repo_root: Path,
    target_file: Path,
    variable_name: str,
    target_start: tuple[int, int],
) -> None:
    """Prove no other in-repo module reads the variable; fail closed.

    Jedi's ``get_references`` cannot trace a *plain variable assignment* across
    modules (verified: class and function definitions resolve across files, a
    module-level ``x = ...`` does not), so the check runs the other way: libcst
    collects every place in another in-repo file that could read this variable
    (a ``from <module> import x`` binding, or a name/attribute spelled like the
    variable) and Jedi ``goto(follow_imports=True)`` must resolve each one
    somewhere else. Anything that resolves back to the assignment — or that
    cannot be resolved at all — refuses the inline.
    """
    project = jedi.Project(path=str(repo_root))
    for candidate in sorted(_python_files(repo_root)):
        if candidate == target_file:
            continue
        text = candidate.read_text(encoding="utf-8")
        if not re.search(rf"\b{re.escape(variable_name)}\b", text):
            continue
        _audit_candidate_file(project, candidate, target_file, variable_name, target_start)


def _audit_candidate_file(
    project: jedi.Project,
    candidate: Path,
    target_file: Path,
    variable_name: str,
    target_start: tuple[int, int],
) -> None:
    """Refuse when any candidate read site in ``candidate`` resolves back to us."""
    wrapper = MetadataWrapper(cst.parse_module(candidate.read_text(encoding="utf-8")))
    positions = wrapper.resolve(PositionProvider)
    parents = wrapper.resolve(ParentNodeProvider)
    sites: list[tuple[int, int]] = []

    class _SiteScan(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider)

        def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:  # noqa: N802
            if isinstance(node.names, cst.ImportStar):
                return True
            for alias in node.names:
                if not isinstance(alias.name, cst.Name):
                    continue
                if alias.name.value != variable_name:
                    continue
                if node.module is None:
                    # ``from . import x``: binds straight from a package
                    # namespace that may re-export anything.
                    raise ValueError(
                        f"{variable_name!r} is bound by a relative import in "
                        f"{candidate.name}; inlining would break that module"
                    )
                start = positions[node.module].start
                sites.append((start.line, start.column))
            return True

        def visit_Name(self, node: cst.Name) -> bool:  # noqa: N802
            if node.value != variable_name:
                return True
            parent = parents.get(node)
            if isinstance(parent, cst.ImportAlias):
                # The imported spelling is handled by visit_ImportFrom.
                return True
            start = positions[node].start
            sites.append((start.line, start.column))
            return True

    wrapper.visit(_SiteScan())

    script = jedi.Script(path=str(candidate), project=project)
    for line, column in sites:
        try:
            definitions = script.goto(line, column, follow_imports=True)
        except Exception as exc:
            raise ValueError(
                f"could not verify a use of {variable_name!r} in {candidate.name} ({exc}); "
                "refusing to inline"
            ) from None
        for definition in definitions:
            module_path = definition.module_path
            if module_path is None:
                continue
            if (
                Path(module_path).resolve() == target_file.resolve()
                and (definition.line, definition.column) == target_start
            ):
                raise ValueError(
                    f"{variable_name!r} is referenced from {candidate.name}; inlining "
                    "would break that module"
                )
        if not definitions:
            raise ValueError(
                f"a use of {variable_name!r} in {candidate.name} (line {line}) cannot "
                "be resolved; refusing to inline rather than guess"
            )


def _python_files(repo_root: Path) -> list[Path]:
    """Every ``.py`` file under the root, skipping dependency/build folders."""
    files: list[Path] = []
    for path in repo_root.rglob("*.py"):
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        files.append(path)
    return files


def _inline_expression(value: cst.BaseExpression) -> cst.BaseExpression:
    """Return the value as it should appear at a read site.

    libcst stores parentheses in the node's own ``lpar``/``rpar`` fields, so a
    value that was parenthesised in the source already renders with them at
    every read site. Atoms (names, literals, calls, attribute and subscript
    chains) can go anywhere without changing grouping; every other bare shape
    gets fresh parentheses, which is semantically neutral and never a syntax
    error.
    """
    lpar = getattr(value, "lpar", None)
    if isinstance(value, _PAREN_FREE) or lpar:
        return value
    return value.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])


def _contains_impure(node: cst.CSTNode) -> bool:
    """True when the expression calls, awaits, assigns or suspends."""
    return any(isinstance(inner, _IMPURE_INNER) for inner in _walk(node))


def _all_name_values(node: cst.CSTNode) -> set[str]:
    """Every identifier appearing in the subtree; a conservative free-name set."""
    return {
        inner.value
        for inner in _walk(node)
        if isinstance(inner, cst.Name)
    }


def _lambda_param_names(node: cst.Lambda) -> set[str]:
    """All parameter names of a lambda, star args included."""
    params = node.params
    candidates = [*params.params, *params.kwonly_params]
    if isinstance(params.star_arg, cst.Param):
        candidates.append(params.star_arg)
    if isinstance(params.star_kwarg, cst.Param):
        candidates.append(params.star_kwarg)
    return {param.name.value for param in candidates}


def _comprehension_bound_names(node: cst.BaseComp) -> set[str]:
    """Every name bound by the comprehension's ``for`` clauses, nested included."""
    names: set[str] = set()
    for_in: cst.CompFor | None = node.for_in
    while for_in is not None:
        names.update(
            inner.value for inner in _walk(for_in.target) if isinstance(inner, cst.Name)
        )
        for_in = for_in.inner_for_in
    return names


def _walk(node: cst.CSTNode) -> list[cst.CSTNode]:
    """Collect ``node`` and every descendant without the visitor ceremony."""
    nodes: list[cst.CSTNode] = []
    stack: list[cst.CSTNode] = [node]
    while stack:
        current = stack.pop()
        nodes.append(current)
        stack.extend(current.children)
    return nodes


def _pop_and_transfer_leading_lines(body: list[cst.BaseStatement], index: int) -> None:
    """Drop ``body[index]``, moving its leading comments onto the next statement.

    When the removed statement is the block's last, its comment lines have
    nothing left to describe and are dropped with it.
    """
    removed = body.pop(index)
    if index >= len(body):
        return
    following = body[index]
    if isinstance(removed, cst.SimpleStatementLine) and isinstance(
        following, (cst.SimpleStatementLine, cst.BaseCompoundStatement)
    ):
        body[index] = following.with_changes(
            leading_lines=[*removed.leading_lines, *following.leading_lines]
        )


class _InlineVariableTransformer(cst.CSTTransformer):
    """Rewrite reads and delete the assignment, one edit per counted change."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        assign_start: tuple[int, int],
        read_starts: set[tuple[int, int]],
        replacement: cst.BaseExpression,
    ) -> None:
        self._assign_start = assign_start
        self._read_starts = read_starts
        self._replacement = replacement
        self._line_removed = False
        self.changes = 0

    def leave_Name(  # noqa: N802
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.BaseExpression:
        """Swap a read of the variable for the (possibly parenthesised) value."""
        start = self.get_metadata(PositionProvider, original_node).start
        if (start.line, start.column) not in self._read_starts:
            return updated_node
        self.changes += 1
        return self._replacement

    def leave_SimpleStatementLine(  # noqa: N802
        self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        """Drop the assignment from a line that carries other statements too."""
        if self._line_removed:
            return updated_node
        for original_small in original_node.body:
            start = self.get_metadata(PositionProvider, original_small).start
            if (start.line, start.column) == self._assign_start and len(updated_node.body) > 1:
                kept = [
                    updated_small
                    for updated_small, original_candidate in zip(
                        updated_node.body, original_node.body, strict=True
                    )
                    if original_candidate is not original_small
                ]
                self.changes += 1
                self._line_removed = True
                return updated_node.with_changes(body=kept)
        return updated_node

    def leave_Module(  # noqa: N802
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        return self._maybe_remove_line(original_node, updated_node)

    def leave_IndentedBlock(  # noqa: N802
        self, original_node: cst.IndentedBlock, updated_node: cst.IndentedBlock
    ) -> cst.IndentedBlock:
        return self._maybe_remove_line(original_node, updated_node)

    def _maybe_remove_line(
        self,
        original_block: _BlockT,
        updated_block: _BlockT,
    ) -> _BlockT:
        """Remove the whole assignment line, keeping its comments readable.

        Leading comments travel to the next statement; when the assignment is
        the block's last statement its comment lines are dropped, because there
        is nothing left in the block for them to describe.
        """
        if self._line_removed:
            return updated_block
        for index, original_statement in enumerate(original_block.body):
            if not isinstance(original_statement, cst.SimpleStatementLine):
                continue
            if len(original_statement.body) != 1:
                continue
            start = self.get_metadata(PositionProvider, original_statement.body[0]).start
            if (start.line, start.column) != self._assign_start:
                continue
            body: list[cst.BaseStatement] = list(updated_block.body)
            _pop_and_transfer_leading_lines(body, index)
            self.changes += 1
            self._line_removed = True
            return updated_block.with_changes(body=body)
        return updated_block
