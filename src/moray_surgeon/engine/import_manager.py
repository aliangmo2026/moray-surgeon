"""Add and remove module-level imports without disturbing the rest of the file.

Both atoms edit ``Module.body`` only, so imports nested in ``try``/``if``
blocks are never touched and never satisfy an ``add``. Every statement the
operation does not rewrite round-trips byte-for-byte, including its leading
blank lines and comments.

Placement follows the conventions ``isort`` uses by default:

* ``from __future__`` imports stay first, then straight imports, then
  ``from`` imports, each group sorted case-insensitively by module.
* A new import is inserted into the top import block, never after executable
  code; a module with no imports gets one right after its docstring.
* When a name is merged into an existing ``from`` statement, its alias list is
  re-sorted and commas are normalized, so no trailing comma or double space is
  introduced.
* Removing the last name of a statement removes the whole line; comments that
  sat above it are re-attached to the next statement instead of being deleted.
"""

from __future__ import annotations

import keyword
from collections.abc import Sequence
from dataclasses import dataclass

import libcst as cst

from moray_surgeon.engine.base import EditResult

_ImportNode = cst.Import | cst.ImportFrom

# libcst types Module.body as exactly these two statement kinds, and only they
# carry a leading_lines list.
_Statement = cst.SimpleStatementLine | cst.BaseCompoundStatement

# ruff's default line length. The engine cannot read the target project's lint
# config, so it wraps at the strictest common default; wrapping early is valid
# Python and isort leaves a manually wrapped block alone.
_MAX_IMPORT_LINE = 88
_INDENT_UNIT = "    "


def add_import(
    source: str,
    module: str,
    name: str | None = None,
    alias: str | None = None,
) -> EditResult:
    """Add one module-level import, keeping the import block sorted.

    Args:
        source: Python module source. Re-parsed once, never re-generated.
        module: Absolute dotted module path, e.g. ``"collections.abc"``.
        name: Imported name for a ``from module import name`` statement. When
            ``None``, a plain ``import module`` statement is written.
        alias: Optional ``as`` name. Requires ``module``; with ``name`` set the
            statement becomes ``from module import name as alias``.

    Returns:
        An :class:`EditResult` whose ``changes`` is 1 when a statement was
        inserted or a name was merged into an existing ``from`` statement, and 0
        when the exact import already exists (source unchanged).

    Raises:
        ValueError: ``module`` is not a dotted module path, or ``name``/``alias``
            is not a valid identifier.
        libcst.ParserSyntaxError: ``source`` is not valid Python.
    """
    _validate_module(module)
    if name is not None:
        _validate_identifier("name", name)
    if alias is not None:
        _validate_identifier("alias", alias)

    module_node = cst.parse_module(source)
    body: list[_Statement] = list(module_node.body)

    if name is None:
        if _has_straight_import(body, module, alias):
            return EditResult(new_source=source, changes=0)
        statement = _parse_import_statement(_straight_import_text(module, alias))
        key = (module.lower(), 0, "")
    else:
        existing = _find_from_import(body, module)
        if existing is not None:
            updated, added = _merge_name(_as_line(body[existing]), module, name, alias)
            if not added:
                return EditResult(new_source=source, changes=0)
            body[existing] = _as_statement(_maybe_wrap(updated))
            return EditResult(new_source=module_node.with_changes(body=body).code, changes=1)
        statement = _parse_import_statement(_from_import_text(module, name, alias))
        key = (module.lower(), 1, name)

    _insert_statement(body, _maybe_wrap(statement), key)
    had_body = bool(module_node.body)
    return EditResult(
        new_source=module_node.with_changes(
            body=body,
            # An empty module carries no trailing newline; the first statement we
            # add must still end its line like any normal file.
            has_trailing_newline=module_node.has_trailing_newline or not had_body,
        ).code,
        changes=1,
    )


def remove_import(source: str, module: str, name: str | None = None) -> EditResult:
    """Remove module-level imports of ``module`` from one file.

    Args:
        source: Python module source. Re-parsed once, never re-generated.
        module: Absolute dotted module path.
        name: When set, only that name is dropped from ``from module import ...``
            statements. When ``None``, every straight ``import module`` alias and
            every ``from module import ...`` statement is removed entirely.

    Returns:
        An :class:`EditResult` whose ``changes`` counts statements that were
        removed or rewritten. Zero means nothing matched and the source is
        returned unchanged.

    Raises:
        ValueError: ``module`` is not a dotted module path, or ``name`` is not a
            valid identifier.
        libcst.ParserSyntaxError: ``source`` is not valid Python.
    """
    _validate_module(module)
    if name is not None:
        _validate_identifier("name", name)

    module_node = cst.parse_module(source)
    new_body: list[_Statement] = []
    pending_comments: list[cst.EmptyLine] = []
    changes = 0
    removed_first = False

    for index, statement in enumerate(module_node.body):
        if not _is_plain_import(statement):
            if pending_comments:
                statement = statement.with_changes(
                    leading_lines=(*pending_comments, *statement.leading_lines)
                )
                pending_comments = []
            new_body.append(statement)
            continue

        replacement, removed, comments = _remove_from_statement(statement, module, name)
        if not removed:
            new_body.append(statement)
            continue

        changes += 1
        removed_first = removed_first or index == 0
        pending_comments.extend(comments)
        if replacement is None:
            continue
        replacement = _as_statement(replacement)
        if pending_comments:
            replacement = replacement.with_changes(
                leading_lines=(*pending_comments, *replacement.leading_lines)
            )
            pending_comments = []
        new_body.append(replacement)

    if changes == 0:
        return EditResult(new_source=source, changes=0)

    header = module_node.header
    if removed_first:
        # Blank lines that separated the file start from the deleted import are
        # layout debris; comments (shebang, encoding) always stay.
        if _only_blank_lines(header):
            header = ()
        new_body = _strip_leading_blank_lines(new_body)

    if not new_body and not pending_comments and not header:
        return EditResult(new_source="", changes=changes)

    footer = (*module_node.footer, *pending_comments)
    new_module = module_node.with_changes(body=new_body, header=header, footer=footer)
    return EditResult(new_source=new_module.code, changes=changes)


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #


def _insert_statement(
    body: list[_Statement],
    statement: cst.BaseStatement,
    key: tuple[str, int, str],
) -> None:
    """Insert ``statement`` into the top import block at its sorted position.

    Blank lines and comments above the import that is pushed down are moved to
    the new statement, so a gap that used to precede the block still precedes
    the block rather than falling between two imports. Comment-bearing lines are
    never moved: the comment stays attached to the import it describes.
    """
    line = _as_line(statement)
    start, end = _import_region(body)
    index = end
    for candidate in range(start, end):
        if _statement_sort_key(body[candidate]) > key:
            index = candidate
            break

    leading_lines: Sequence[cst.EmptyLine] = ()
    if index < end and index == start:
        # Only the top of the block can carry preamble spacing (a blank line
        # after the docstring, for example). A gap between two import groups
        # must stay where it is, so nothing is moved for later insertions.
        target = body[index]
        if _only_blank_lines(target.leading_lines):
            leading_lines = target.leading_lines
            body[index] = target.with_changes(leading_lines=())
    elif index == 1 and _is_docstring(body[0]):
        # A module docstring is normally followed by one blank line before the
        # first import; add it only when this is the very first import.
        leading_lines = (cst.EmptyLine(),)

    body.insert(index, line.with_changes(leading_lines=leading_lines))


def _import_region(body: Sequence[_Statement]) -> tuple[int, int]:
    """Return ``[start, end)``: the docstring/future-aware top import block."""
    start = 1 if body and _is_docstring(body[0]) else 0
    while start < len(body) and _is_future_import(body[start]):
        start += 1
    end = start
    while end < len(body) and _is_plain_import(body[end]):
        end += 1
    return start, end


def _statement_sort_key(statement: cst.BaseStatement) -> tuple[str, int, str]:
    """Sort straight imports before ``from`` imports, then by module name."""
    node = _single_import_node(statement)
    if isinstance(node, cst.Import):
        modules = [_dotted_name(alias.name).lower() for alias in node.names]
        return (min(modules), 0, "")
    assert isinstance(node, cst.ImportFrom)  # narrowed by _single_import_node
    return (_dotted_name(node.module).lower(), 1, "")


def _find_from_import(body: Sequence[cst.BaseStatement], module: str) -> int | None:
    """Index of the first module-level ``from module import ...`` statement."""
    for index, statement in enumerate(body):
        node = _single_import_node(statement)
        if not isinstance(node, cst.ImportFrom) or not _is_absolute_import(node):
            continue
        if node.module is None or _dotted_name(node.module) != module:
            continue
        if isinstance(node.names, cst.ImportStar):
            continue
        return index
    return None


def _merge_name(
    statement: cst.SimpleStatementLine,
    module: str,
    name: str,
    alias: str | None,
) -> tuple[cst.BaseStatement, bool]:
    """Add ``name`` to an existing ``from module import ...`` statement."""
    node = _single_import_node(statement)
    assert isinstance(node, cst.ImportFrom)
    assert not isinstance(node.names, cst.ImportStar)  # filtered by _find_from_import
    names = list(node.names)
    if any(_alias_name(item) == name and _alias_identifier(item) == alias for item in names):
        return statement, False

    fresh_statement = _parse_import_statement(_from_import_text(module, name, alias))
    fresh = _single_import_node(fresh_statement)
    assert isinstance(fresh, cst.ImportFrom)
    assert not isinstance(fresh.names, cst.ImportStar)
    merged = _sorted_aliases(names, fresh.names[0])
    normalized = _rebuild_commas(node, names, merged)
    return statement.with_changes(body=[node.with_changes(names=normalized)]), True


# --------------------------------------------------------------------------- #
# Removal
# --------------------------------------------------------------------------- #


def _remove_from_statement(
    statement: _Statement,
    module: str,
    name: str | None,
) -> tuple[cst.BaseStatement | None, int, list[cst.EmptyLine]]:
    """Remove matching imports from one statement.

    Returns:
        ``(replacement, removed, comments)``: the rewritten statement (None when
        the whole line goes away), how many names/statements were removed, and
        the comment-only leading lines that must survive the deletion.
    """
    node = _single_import_node(statement)
    assert node is not None  # caller checked _is_plain_import

    if isinstance(node, cst.Import):
        if name is not None:
            return statement, 0, []
        kept = [alias for alias in node.names if _dotted_name(alias.name) != module]
        removed = len(node.names) - len(kept)
        if removed == 0:
            return statement, 0, []
        if not kept:
            return None, removed, _comment_lines(statement.leading_lines)
        normalized = _rebuild_commas(node, node.names, kept)
        return statement.with_changes(body=[node.with_changes(names=normalized)]), removed, []

    # ImportFrom
    if not _is_absolute_import(node) or node.module is None:
        return statement, 0, []
    if _dotted_name(node.module) != module:
        return statement, 0, []
    if isinstance(node.names, cst.ImportStar):
        # A star import has no individual names to match.
        if name is not None:
            return statement, 0, []
        return None, 1, _comment_lines(statement.leading_lines)

    if name is None:
        return None, len(node.names), _comment_lines(statement.leading_lines)

    kept = [alias for alias in node.names if _alias_name(alias) != name]
    removed = len(node.names) - len(kept)
    if removed == 0:
        return statement, 0, []
    if not kept:
        return None, removed, _comment_lines(statement.leading_lines)
    normalized = _rebuild_commas(node, node.names, kept)
    return statement.with_changes(body=[node.with_changes(names=normalized)]), removed, []


@dataclass(frozen=True)
class _AliasLayout:
    """Where the aliases of one statement sit, so commas can be re-numbered.

    Attributes:
        multiline: True for a parenthesized import broken across lines.
        item_indent: Whitespace before each alias on its own line.
        close_indent: Whitespace before the closing paren after a trailing comma.
        trailing: The statement's trailing comma, or None when it has none.
    """

    multiline: bool
    item_indent: str
    close_indent: str
    trailing: cst.Comma | None


def _alias_layout(
    node: cst.Import | cst.ImportFrom,
    original: Sequence[cst.ImportAlias],
) -> _AliasLayout:
    """Describe the statement's current layout once, before aliases move."""
    trailing = _explicit_comma(original[-1])
    if isinstance(node, cst.ImportFrom) and isinstance(node.lpar, cst.LeftParen):
        after_lpar = node.lpar.whitespace_after
        if isinstance(after_lpar, cst.ParenthesizedWhitespace):
            close = trailing.whitespace_after if trailing is not None else after_lpar
            close_indent = (
                close.last_line.value
                if isinstance(close, cst.ParenthesizedWhitespace)
                else after_lpar.last_line.value
            )
            return _AliasLayout(True, after_lpar.last_line.value, close_indent, trailing)
    return _AliasLayout(False, "", "", trailing)


def _rebuild_commas(
    node: cst.Import | cst.ImportFrom,
    original: Sequence[cst.ImportAlias],
    items: Sequence[cst.ImportAlias],
) -> list[cst.ImportAlias]:
    """Re-number commas for the aliases that survived an edit.

    Existing commas are reused whenever they already end at the right place, so
    per-alias comments and unusual spacing survive. A comma that used to end at
    the closing paren is retargeted when its alias is no longer last, and the
    new last alias gets a trailing comma only when the author had one.
    """
    layout = _alias_layout(node, original)
    result: list[cst.ImportAlias] = []
    for position, alias in enumerate(items):
        is_last = position == len(items) - 1
        if is_last and layout.trailing is None:
            comma: cst.Comma | cst.MaybeSentinel = cst.MaybeSentinel.DEFAULT
        elif is_last:
            comma = _comma_at(
                layout,
                _explicit_comma(alias) or layout.trailing,
                layout.close_indent,
            )
        else:
            comma = _comma_at(layout, _explicit_comma(alias), layout.item_indent)
        result.append(alias.with_changes(comma=comma))
    return result


def _comma_at(layout: _AliasLayout, existing: cst.Comma | None, indent: str) -> cst.Comma:
    """Return a comma whose trailing whitespace ends at ``indent``."""
    if layout.multiline:
        if existing is not None and isinstance(
            existing.whitespace_after, cst.ParenthesizedWhitespace
        ):
            whitespace = existing.whitespace_after
            if whitespace.last_line.value == indent:
                return existing
            return existing.with_changes(
                whitespace_after=whitespace.with_changes(
                    last_line=cst.SimpleWhitespace(indent)
                )
            )
        return cst.Comma(
            whitespace_after=cst.ParenthesizedWhitespace(
                first_line=cst.TrailingWhitespace(newline=cst.Newline()),
                indent=True,
                last_line=cst.SimpleWhitespace(indent),
            )
        )
    if existing is not None and isinstance(existing.whitespace_after, cst.SimpleWhitespace):
        return existing
    return _fallback_comma()


def _sorted_aliases(
    names: Sequence[cst.ImportAlias],
    alias: cst.ImportAlias,
) -> list[cst.ImportAlias]:
    """Insert ``alias`` into a case-insensitively sorted alias list."""
    items = list(names)
    key = _alias_sort_key(alias)
    index = len(items)
    for position, item in enumerate(items):
        if _alias_sort_key(item) > key:
            index = position
            break
    items.insert(index, alias)
    return items


def _explicit_comma(alias: cst.ImportAlias) -> cst.Comma | None:
    """Return the alias's own comma node, or None when it has none."""
    if isinstance(alias.comma, cst.Comma):
        return alias.comma
    return None


def _fallback_comma() -> cst.Comma:
    """Comma for a one-line list that had no comma to copy."""
    return cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))


def _strip_leading_blank_lines(
    body: list[_Statement],
) -> list[_Statement]:
    """Drop blank lines left at the top of the file by a removed first import.

    Comment lines (shebangs, encoding declarations) are never dropped, so a
    file that starts with one is returned untouched.
    """
    if body and _only_blank_lines(body[0].leading_lines) and body[0].leading_lines:
        return [body[0].with_changes(leading_lines=()), *body[1:]]
    return body


# --------------------------------------------------------------------------- #
# Statement helpers
# --------------------------------------------------------------------------- #


def _maybe_wrap(statement: cst.BaseStatement) -> cst.BaseStatement:
    """Parenthesize a from-import whose rendered line exceeds the safe width.

    The engine has no access to the target project's configured line length, so
    it wraps at ruff's default (88). A wrapped block is valid Python, stays
    isort-clean, and keeps a long merge from producing an E501 failure.

    Comments inside the alias list are never rewritten: if any alias carries one,
    the statement is returned untouched rather than risk deleting it.
    """
    if not isinstance(statement, cst.SimpleStatementLine):
        return statement
    node = _single_import_node(statement)
    if not isinstance(node, cst.ImportFrom):
        return statement
    if isinstance(node.lpar, cst.LeftParen) or isinstance(node.names, cst.ImportStar):
        # Already parenthesized (comments and layout inside it are preserved) or
        # a star import, which has nothing to wrap.
        return statement
    rendered = cst.Module(body=()).code_for_node(statement)
    if max((len(line) for line in rendered.splitlines()), default=0) <= _MAX_IMPORT_LINE:
        return statement
    return statement.with_changes(body=[_parenthesize(node)])


def _parenthesize(node: cst.ImportFrom) -> cst.ImportFrom:
    """Render one from-import as a one-name-per-line parenthesized block."""
    assert not isinstance(node.names, cst.ImportStar)
    names = list(node.names)
    rewritten: list[cst.ImportAlias] = []
    for position, alias in enumerate(names):
        is_last = position == len(names) - 1
        indent = "" if is_last else _INDENT_UNIT
        comma = cst.Comma(whitespace_after=_newline_with_indent(indent))
        rewritten.append(alias.with_changes(comma=comma))
    return node.with_changes(
        names=rewritten,
        lpar=cst.LeftParen(whitespace_after=_newline_with_indent(_INDENT_UNIT)),
        rpar=cst.RightParen(whitespace_before=cst.SimpleWhitespace("")),
    )


def _newline_with_indent(indent: str) -> cst.ParenthesizedWhitespace:
    """Whitespace that breaks the line and indents the next token."""
    return cst.ParenthesizedWhitespace(
        first_line=cst.TrailingWhitespace(newline=cst.Newline()),
        indent=True,
        last_line=cst.SimpleWhitespace(indent),
    )


def _as_line(statement: cst.BaseStatement) -> cst.SimpleStatementLine:
    """Narrow to a single-line statement (the only kind that can hold an import)."""
    assert isinstance(statement, cst.SimpleStatementLine)
    return statement


def _as_statement(statement: cst.BaseStatement) -> _Statement:
    """Narrow to the two statement kinds libcst gives a ``leading_lines`` list."""
    assert isinstance(statement, (cst.SimpleStatementLine, cst.BaseCompoundStatement))
    return statement


def _single_import_node(statement: cst.BaseStatement) -> _ImportNode | None:
    """Return the import node when ``statement`` is one lone import statement."""
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return None
    node = statement.body[0]
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return node
    return None


def _is_plain_import(statement: cst.BaseStatement) -> bool:
    """True when the statement is exactly one top-level import (no ``;``)."""
    return _single_import_node(statement) is not None


def _is_future_import(statement: cst.BaseStatement) -> bool:
    """True for ``from __future__ import ...`` statements."""
    node = _single_import_node(statement)
    return (
        isinstance(node, cst.ImportFrom)
        and node.module is not None
        and _dotted_name(node.module) == "__future__"
    )


def _is_docstring(statement: cst.BaseStatement) -> bool:
    """True when the statement is a bare string expression (module docstring)."""
    return (
        isinstance(statement, cst.SimpleStatementLine)
        and len(statement.body) == 1
        and isinstance(statement.body[0], cst.Expr)
        and isinstance(statement.body[0].value, cst.SimpleString)
    )


def _is_absolute_import(node: cst.ImportFrom) -> bool:
    """True for ``from x import y``; false for ``from . import y``."""
    return len(node.relative) == 0


def _has_straight_import(
    body: Sequence[cst.BaseStatement],
    module: str,
    alias: str | None,
) -> bool:
    """True when ``import module [as alias]`` already exists at module level."""
    for statement in body:
        node = _single_import_node(statement)
        if not isinstance(node, cst.Import):
            continue
        for item in node.names:
            if _dotted_name(item.name) == module and _alias_identifier(item) == alias:
                return True
    return False


def _only_blank_lines(lines: Sequence[cst.EmptyLine]) -> bool:
    """True when every leading line is blank (no comment anywhere)."""
    return all(line.comment is None for line in lines)


def _comment_lines(lines: Sequence[cst.EmptyLine]) -> list[cst.EmptyLine]:
    """Return the comment-bearing leading lines that must outlive a deletion."""
    return [line for line in lines if line.comment is not None]


def _alias_name(alias: cst.ImportAlias) -> str:
    """Imported name of one alias (``a`` in ``from x import a``)."""
    return alias.name.value if isinstance(alias.name, cst.Name) else ""


def _alias_identifier(alias: cst.ImportAlias) -> str | None:
    """The ``as`` name of one alias, or None when it has no alias."""
    if isinstance(alias.asname, cst.AsName) and isinstance(alias.asname.name, cst.Name):
        return alias.asname.name.value
    return None


def _alias_sort_key(alias: cst.ImportAlias) -> tuple[int, str]:
    """Sort key matching isort's default ``order-by-type`` grouping.

    Ruff/isort place constants (``LIMIT``) first, then classes (``Counter``),
    then lowercase names (``defaultdict``), each group case-insensitively. A
    single uppercase letter is a class, not a constant — verified against
    ``ruff check --select I``.
    """
    name = _alias_name(alias)
    if len(name) > 1 and name.isupper():
        return (0, name.casefold())
    if name[:1].isupper():
        return (1, name.casefold())
    return (2, name.casefold())


def _dotted_name(node: cst.BaseExpression | None) -> str:
    """Render a ``Name``/``Attribute`` module reference without formatting."""
    if node is None:
        return ""
    return cst.Module(body=()).code_for_node(node).replace(" ", "")


# --------------------------------------------------------------------------- #
# Building and validation
# --------------------------------------------------------------------------- #


def _straight_import_text(module: str, alias: str | None) -> str:
    """Source text for a plain import statement."""
    return f"import {module}" + (f" as {alias}" if alias else "")


def _from_import_text(module: str, name: str, alias: str | None) -> str:
    """Source text for a from-import statement."""
    return f"from {module} import {name}" + (f" as {alias}" if alias else "")


def _parse_import_statement(text: str) -> cst.BaseStatement:
    """Parse generated import text into a ``SimpleStatementLine``.

    The text is built exclusively from validated identifiers, so a parse error
    here would be an engine bug rather than bad user input.
    """
    return cst.parse_statement(text)


def _validate_module(module: str) -> None:
    """Reject paths that are not absolute dotted module names."""
    parts = module.split(".")
    if not module or not all(part.isidentifier() for part in parts):
        raise ValueError(f"module must be a dotted Python module path, got {module!r}")
    if any(keyword.iskeyword(part) for part in parts):
        raise ValueError(f"module must not contain a Python keyword, got {module!r}")


def _validate_identifier(role: str, value: str) -> None:
    """Reject empty / non-identifier / keyword names before touching the CST."""
    if not value:
        raise ValueError(f"{role} must not be empty")
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(f"{role} must be a valid Python identifier, got {value!r}")
