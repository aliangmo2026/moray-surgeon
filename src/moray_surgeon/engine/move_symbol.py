"""Move a top-level function or class from one module to another.

The node (leading comments and docstring included) is appended to the target
module and deleted from the source; ``from <source> import <symbol>``
statements in other in-repo modules are rewritten to import from the target
module. Everything the engine cannot prove safe refuses the whole move before
any file is touched:

* the symbol must be a unique top-level ``def``/``class`` in the source;
* the target must not already bind the name;
* the source must not still reference the symbol after the deletion (move the
  callers too, or add the import back manually);
* every reference Jedi finds outside the two files must sit on a rewritable
  ``from``-import: attribute-style usage (``import mod; mod.symbol``) refuses;
* absolute imports are rewritten to the target's dotted module; relative
  imports only when source and target share a directory, because only then do
  the dots keep pointing at the right package;
* the moved node's module-level dependencies must exist in the target (or be
  builtins) — ``os`` imported in the source but not the target is a refusal,
  not a broken move.

All rewrites are computed first and only then written, so a refusal anywhere
leaves every file byte-for-byte untouched.
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import jedi
import libcst as cst
from libcst.metadata import (
    CodeRange,
    MetadataWrapper,
    ParentNodeProvider,
    PositionProvider,
    ScopeProvider,
)
from libcst.metadata.scope_provider import Assignment

from moray_surgeon.engine.base import EditResult
from moray_surgeon.engine.import_manager import add_import
from moray_surgeon.engine.inline_variable import _python_files
from moray_surgeon.engine.rename import _jedi_reference_budget


def move_symbol(
    repo_root: Path,
    source_path: Path,
    target_path: Path,
    symbol_name: str,
) -> EditResult:
    """Move the top-level ``symbol_name`` from ``source_path`` to ``target_path``.

    Args:
        repo_root: Repository root; every rewrite stays inside it.
        source_path: Absolute path of the file the symbol lives in.
        target_path: Absolute path of the file the symbol moves to; it must
            already exist.
        symbol_name: Name of the top-level function or class to move.

    Returns:
        An :class:`EditResult` for ``source_path`` whose ``changes`` counts the
        deletion, the insertion and every rewritten importer file.

    Raises:
        ValueError: Under any of the refusal rules in the module docstring.
    """
    if not symbol_name.isidentifier() or keyword.iskeyword(symbol_name):
        raise ValueError(
            f"symbol_name must be a valid Python identifier, got {symbol_name!r}"
        )
    if source_path.resolve() == target_path.resolve():
        raise ValueError("source_file and target_file must be different files")

    source = source_path.read_text(encoding="utf-8")
    target = target_path.read_text(encoding="utf-8")

    source_wrapper = MetadataWrapper(cst.parse_module(source))
    source_positions = source_wrapper.resolve(PositionProvider)
    node = _find_unique_symbol(source_wrapper, source_path, symbol_name)
    # Jedi resolves references from the *name's* position; the node's own span
    # starts at the ``def``/``class`` keyword.
    name_span = source_positions[node.name]
    source_module_name = _module_name(source_path, repo_root)
    target_module_name = _module_name(target_path, repo_root)

    _assert_not_exported(source_wrapper.module, symbol_name, source_path)
    _assert_target_does_not_bind(target, target_path, symbol_name)
    _assert_source_has_no_leftovers(
        source_wrapper, source_positions, node, symbol_name, source_path
    )
    free_names = _moved_free_names(source_wrapper, node, source_path, symbol_name)
    _assert_dependencies_available(target, target_path, free_names, symbol_name, source_path)

    moved = _prepare_moved_node(node)
    new_source = _delete_from_source(source_wrapper, node)
    new_target = _append_to_target(target, moved)

    rewrites = _plan_importer_rewrites(
        repo_root,
        source_path,
        target_path,
        source_module_name,
        target_module_name,
        symbol_name,
        name_span,
    )

    source_path.write_text(new_source, encoding="utf-8")
    target_path.write_text(new_target, encoding="utf-8")
    for path, text in rewrites.items():
        path.write_text(text, encoding="utf-8")

    return EditResult(new_source=new_source, changes=2 + len(rewrites))


def _find_unique_symbol(
    wrapper: MetadataWrapper,
    source_path: Path,
    symbol_name: str,
) -> cst.FunctionDef | cst.ClassDef:
    """Return the one top-level ``def``/``class`` spelled ``symbol_name``."""
    candidates = [
        statement
        for statement in wrapper.module.body
        if isinstance(statement, (cst.FunctionDef, cst.ClassDef))
        and statement.name.value == symbol_name
    ]
    if not candidates:
        raise ValueError(
            f"no top-level function or class named {symbol_name!r} was found in "
            f"{source_path.name}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"{symbol_name!r} is defined {len(candidates)} times at the top level "
            f"of {source_path.name}; refusing to guess which one to move"
        )
    return candidates[0]


def _module_name(path: Path, repo_root: Path) -> str:
    """The dotted module name of ``path`` inside ``repo_root``.

    ``pkg/mod.py`` becomes ``pkg.mod``; an ``__init__.py`` names its package.
    """
    relative = path.resolve().relative_to(repo_root.resolve())
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        raise ValueError(f"cannot compute a module name for {path.name}")
    return ".".join(parts)


def _assert_not_exported(module: cst.Module, symbol_name: str, source_path: Path) -> None:
    """Refuse when the source's ``__all__`` mentions the symbol."""
    for statement in module.body:
        if not isinstance(statement, (cst.SimpleStatementLine, cst.SimpleStatementSuite)):
            continue
        for small in statement.body:
            if not isinstance(small, cst.Assign):
                continue
            if not any(
                isinstance(target, cst.AssignTarget)
                and isinstance(target.target, cst.Name)
                and target.target.value == "__all__"
                for target in small.targets
            ):
                continue
            for inner in _walk(small.value):
                if isinstance(inner, cst.SimpleString) and _string_value(inner) == symbol_name:
                    raise ValueError(
                        f"{symbol_name!r} is exported by __all__ in "
                        f"{source_path.name}; update the exports manually"
                    )


def _string_value(node: cst.SimpleString) -> str | None:
    """The unquoted value of a simple string, or None for exotic spellings."""
    match = re.match(r"^[rbuRBU]{0,2}(['\"])(.*?)\1$", node.value, re.DOTALL)
    return match.group(2) if match else None


def _assert_target_does_not_bind(target: str, target_path: Path, symbol_name: str) -> None:
    """The destination must not already bind the name in any way."""
    wrapper = MetadataWrapper(cst.parse_module(target))
    scopes = wrapper.resolve(ScopeProvider)
    global_scope = scopes[wrapper.module]
    if global_scope is None:  # pragma: no cover - every module has a scope
        raise ValueError(f"could not analyse {target_path.name}")
    if any(binding.name == symbol_name for binding in global_scope.assignments):
        raise ValueError(
            f"{target_path.name} already binds {symbol_name!r}; the move would "
            "create a conflict"
        )


def _assert_source_has_no_leftovers(
    wrapper: MetadataWrapper,
    positions: Mapping[cst.CSTNode, CodeRange],
    node: cst.FunctionDef | cst.ClassDef,
    symbol_name: str,
    source_path: Path,
) -> None:
    """Every reference in the source must die with the node."""
    span_start = positions[node].start
    span_end = positions[node].end

    class _LeftoverScan(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def visit_Name(self, current: cst.Name) -> bool:  # noqa: N802
            if current.value != symbol_name:
                return True
            start = positions[current].start
            if span_start.line <= start.line <= span_end.line:
                return True  # inside the moved node itself
            raise ValueError(
                f"{symbol_name!r} is still referenced at {source_path.name}:"
                f"{start.line} outside the moved definition; move the callers "
                "too or add an import back manually"
            )

    wrapper.visit(_LeftoverScan())


def _moved_free_names(
    wrapper: MetadataWrapper,
    node: cst.FunctionDef | cst.ClassDef,
    source_path: Path,
    symbol_name: str,
) -> set[str]:
    """Module-level names the moved node reads: it needs them at the target.

    An access inside the node counts when it resolves to a binding in the
    source module's global scope — parameters, locals and nested bindings
    travel with the node; builtins are handled by the caller.
    """
    scopes = wrapper.resolve(ScopeProvider)
    parents = wrapper.resolve(ParentNodeProvider)
    global_scope = scopes[wrapper.module]
    if global_scope is None:  # pragma: no cover - every module has a scope
        raise ValueError(f"could not analyse {source_path.name}")

    def _inside_moved(current: cst.CSTNode | None) -> bool:
        while current is not None:
            if current is node:
                return True
            current = parents.get(current)
        return False

    free: set[str] = set()
    for scope in scopes.values():
        if scope is None:
            continue
        for access in scope.accesses:
            if not isinstance(access.node, cst.Name):
                continue
            if access.node.value == symbol_name or not _inside_moved(access.node):
                continue
            if any(
                isinstance(referent, Assignment) and scopes.get(referent.node) is global_scope
                for referent in access.referents
            ):
                free.add(access.node.value)
    return free


def _assert_dependencies_available(
    target: str,
    target_path: Path,
    free_names: set[str],
    symbol_name: str,
    source_path: Path,
) -> None:
    """Every module-level dependency must exist at the destination or be a builtin."""
    if not free_names:
        return
    wrapper = MetadataWrapper(cst.parse_module(target))
    scopes = wrapper.resolve(ScopeProvider)
    global_scope = scopes[wrapper.module]
    if global_scope is None:  # pragma: no cover - every module has a scope
        raise ValueError(f"could not analyse {target_path.name}")
    target_names = {binding.name for binding in global_scope.assignments}
    builtin_scope = getattr(global_scope, "parent", None)
    builtin_names = (
        {binding.name for binding in builtin_scope.assignments}
        if builtin_scope is not None
        else set()
    )
    missing = sorted(
        name for name in free_names if name not in target_names and name not in builtin_names
    )
    if missing:
        raise ValueError(
            f"{symbol_name!r} uses {missing}, which are defined in "
            f"{source_path.name} but not available in {target_path.name}; move "
            "the dependencies too or add imports to the target first"
        )


def _prepare_moved_node(node: cst.FunctionDef | cst.ClassDef) -> cst.FunctionDef | cst.ClassDef:
    """Strip blank leading lines; comments travel, spacing is re-created."""
    kept_leading = [
        empty_line for empty_line in node.leading_lines if empty_line.comment is not None
    ]
    return node.with_changes(leading_lines=kept_leading)


def _delete_from_source(wrapper: MetadataWrapper, node: cst.FunctionDef | cst.ClassDef) -> str:
    """Remove the node from the source module body."""
    body = [statement for statement in wrapper.module.body if statement is not node]
    return wrapper.module.with_changes(body=body).code


def _append_to_target(target: str, moved: cst.FunctionDef | cst.ClassDef) -> str:
    """Append the node at the end of the target, comments attached."""
    wrapper = MetadataWrapper(cst.parse_module(target))
    body = list(wrapper.module.body)
    leading = (
        [cst.EmptyLine(), cst.EmptyLine(), *moved.leading_lines]
        if body
        else list(moved.leading_lines)
    )
    appended = moved.with_changes(leading_lines=leading)
    return wrapper.module.with_changes(body=[*body, appended]).code


def _plan_importer_rewrites(
    repo_root: Path,
    source_path: Path,
    target_path: Path,
    source_module_name: str,
    target_module_name: str,
    symbol_name: str,
    name_span: CodeRange,
) -> dict[Path, str]:
    """Rewrite every ``from <source> import <symbol>`` in the repository.

    Files whose references are not import statements (attribute-style usage)
    refuse the move, computed from Jedi's reference set before anything is
    written.
    """
    referencing_files = _referencing_files(
        repo_root, source_path, target_path, symbol_name, name_span
    )
    rewrites: dict[Path, str] = {}
    for path in _python_files(repo_root):
        if path.resolve() in (source_path.resolve(), target_path.resolve()):
            continue
        text = path.read_text(encoding="utf-8")
        if symbol_name not in text:
            continue
        rewritten = _rewrite_imports(
            text,
            repo_root,
            path,
            source_path,
            target_path,
            source_module_name,
            target_module_name,
            symbol_name,
        )
        if rewritten is not None:
            rewrites[path] = rewritten
            referencing_files.discard(path)
    if referencing_files:
        listed = ", ".join(sorted(path.name for path in referencing_files))
        raise ValueError(
            f"{symbol_name!r} is used without a rewritable "
            f"'from {source_module_name} import' statement in: {listed}; "
            "update its imports manually"
        )
    return rewrites


def _referencing_files(
    repo_root: Path,
    source_path: Path,
    target_path: Path,
    symbol_name: str,
    name_span: CodeRange,
) -> set[Path]:
    """Files Jedi sees using the symbol, source and target excluded."""
    try:
        with _jedi_reference_budget(repo_root):
            project = jedi.Project(path=str(repo_root))
            script = jedi.Script(path=str(source_path), project=project)
            references = script.get_references(
                name_span.start.line, name_span.start.column, scope="project"
            )
    except Exception as exc:
        raise ValueError(
            f"could not verify references for {symbol_name!r} ({exc}); refusing to move"
        ) from None

    files: set[Path] = set()
    for reference in references:
        if reference.module_path is None:
            continue
        path = Path(reference.module_path).resolve()
        if path in (source_path.resolve(), target_path.resolve()):
            continue
        files.add(path)
    return files


def _rewrite_imports(
    text: str,
    repo_root: Path,
    importer: Path,
    source_path: Path,
    target_path: Path,
    source_module_name: str,
    target_module_name: str,
    symbol_name: str,
) -> str | None:
    """Rewrite matching ``from``-imports; None when nothing matched.

    A single-name import swaps its module in place; a multi-name import drops
    the symbol and the new import is added afterwards with ``add_import`` so it
    lands sorted in the import block.
    """
    wrapper = MetadataWrapper(cst.parse_module(text))
    outcome = _RewriteOutcome()

    class _ImportRewriter(cst.CSTTransformer):
        def leave_ImportFrom(  # noqa: N802
            self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
        ) -> cst.ImportFrom:
            if isinstance(updated_node.names, cst.ImportStar):
                return updated_node
            if _resolve_import_module(original_node, importer, repo_root) != source_module_name:
                return updated_node
            kept = [
                alias
                for alias in updated_node.names
                if not (isinstance(alias.name, cst.Name) and alias.name.value == symbol_name)
            ]
            if len(kept) == len(updated_node.names):
                return updated_node
            outcome.matched = True
            if not kept:  # the statement imported only the symbol: swap module
                new_module_node, new_relative = _new_module_node(
                    original_node, importer, repo_root, source_path, target_path, target_module_name
                )
                return updated_node.with_changes(
                    module=new_module_node, relative=new_relative
                )
            kept[-1] = kept[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
            outcome.needs_new_import = True
            return updated_node.with_changes(names=kept)

    new_module = wrapper.visit(_ImportRewriter())
    if not outcome.matched:
        return None
    new_text = new_module.code
    if outcome.needs_new_import:
        new_text = add_import(new_text, target_module_name, symbol_name).new_source
    return new_text


class _RewriteOutcome:
    """Mutable flags a closure can update while libcst walks the tree."""

    def __init__(self) -> None:
        self.matched = False
        self.needs_new_import = False


def _new_module_node(
    original_node: cst.ImportFrom,
    importer: Path,
    repo_root: Path,
    source_path: Path,
    target_path: Path,
    target_module_name: str,
) -> tuple[cst.BaseExpression, Sequence[cst.Dot]]:
    """The replacement module node and dots for a rewritten from-import.

    Absolute imports point at the target's dotted module. Relative imports
    keep their dots and swap the module tail — allowed only when the rewritten
    statement still resolves to the target module from the importer's own
    package, which is what makes the dots meaningful. A package-namespace
    import (``from . import x``) cannot be redirected at all.
    """
    if not original_node.relative:
        return _dotted_module(target_module_name), ()
    if original_node.module is None:
        raise ValueError(
            "a 'from . import' statement cannot be redirected to "
            f"{target_path.name}; update it manually"
        )
    rewritten = cst.ImportFrom(
        module=cst.Name(target_path.stem),
        relative=original_node.relative,
        names=[cst.ImportAlias(cst.Name("_"))],
    )
    resolved = _resolve_import_module(rewritten, importer, repo_root)
    if resolved != target_module_name:
        raise ValueError(
            f"a relative import in {importer.name} would not reach "
            f"{target_path.name} after the move ({resolved!r}); move within the "
            "same package or rewrite the import as absolute first"
        )
    return cst.Name(target_path.stem), original_node.relative


def _resolve_import_module(
    node: cst.ImportFrom,
    importer: Path,
    repo_root: Path,
) -> str | None:
    """The absolute module a from-import names, relative imports included."""
    module = node.module
    tail = _dotted_name(module) if module is not None else ""
    level = len(node.relative)
    if level == 0:
        return tail
    package = _package_of(importer, repo_root)
    package_parts = package.split(".") if package else []
    up = level - 1
    if up > len(package_parts):
        return None
    base = package_parts[: len(package_parts) - up] if up else package_parts
    parts = [*base, *([tail] if tail else [])]
    return ".".join(parts) if parts else None


def _package_of(path: Path, repo_root: Path) -> str:
    """The dotted package containing ``path``; ``__init__.py`` owns its package."""
    relative_parent = path.resolve().parent.relative_to(repo_root.resolve())
    parts = list(relative_parent.parts)
    if path.name != "__init__.py":
        return ".".join(parts) if parts else ""
    return ".".join(parts[:-1]) if len(parts) > 1 else ""


def _dotted_name(node: cst.BaseExpression | None) -> str:
    """Render a Name/Attribute chain as a dotted string."""
    if node is None:
        return ""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr.value}" if prefix else node.attr.value
    return ""


def _dotted_module(module: str) -> cst.BaseExpression:
    """Build the Name/Attribute chain for a dotted module string."""
    parts = module.split(".")
    expression: cst.BaseExpression = cst.Name(parts[0])
    for part in parts[1:]:
        expression = cst.Attribute(value=expression, attr=cst.Name(part))
    return expression


def _walk(node: cst.CSTNode) -> list[cst.CSTNode]:
    """Collect ``node`` and every descendant without the visitor ceremony."""
    nodes: list[cst.CSTNode] = []
    stack: list[cst.CSTNode] = [node]
    while stack:
        current = stack.pop()
        nodes.append(current)
        stack.extend(current.children)
    return nodes
