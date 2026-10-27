"""Generate a compact code map so the model can plan against real symbols.

The map lists module-level classes and functions (plus methods of those classes)
with their signatures, line numbers and the first docstring line. Signatures are
whitespace-collapsed so every class/function occupies exactly one line, and each
file block is capped to keep the prompt bounded. It is built with libcst, never
regexes, so decorators, async defs and annotations are rendered correctly.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import libcst as cst
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider

from moray_surgeon.exceptions import PathTraversalError

_MAX_FILE_MAP_CHARS = 2000
_TRUNCATION_MARKER = "\n... (truncated)"


def generate_repo_map(repo_root: Path, target_files: list[str]) -> str:
    """Render a Markdown-like map for the given repository files.

    Only ``target_files`` are read. Each block starts with ``# File: <rel>`` and
    lists top-level classes/functions; class methods follow their class. Nested
    functions inside functions are not included because they are not part of the
    public surface a plan should target.

    Args:
        repo_root: Repository root; paths outside it are rejected.
        target_files: Repository-relative Python files to map (order preserved,
            duplicates ignored).

    Returns:
        One block per file, separated by a blank line; an empty string when
        ``target_files`` is empty.

    Raises:
        PathTraversalError: A target resolves outside ``repo_root``.
        FileNotFoundError: A target does not exist.
        OSError: A target could not be read.
    """
    root = repo_root.resolve()
    blocks: list[str] = []
    seen: set[Path] = set()

    for target_file in target_files:
        path = (root / target_file).resolve()
        if not path.is_relative_to(root):
            raise PathTraversalError(
                f"target_file {target_file!r} resolves outside the repository"
            )
        if path in seen:
            continue
        seen.add(path)
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        block = f"# File: {relative}\n{_file_entries(source)}"
        blocks.append(_truncate_block(block))

    return "\n\n".join(blocks)


def _file_entries(source: str) -> str:
    """Render the class/function entries of one module."""
    try:
        wrapper = MetadataWrapper(cst.parse_module(source))
        positions = wrapper.resolve(PositionProvider)
    except cst.ParserSyntaxError:
        return "(unable to parse: invalid Python syntax)"

    renderer = cst.Module(body=())
    lines: list[str] = []

    for statement in wrapper.module.body:
        if isinstance(statement, cst.ClassDef):
            lines.append(_entry(statement, positions, renderer))
            for member in statement.body.body:
                if isinstance(member, cst.FunctionDef):
                    lines.append(_entry(member, positions, renderer))
        elif isinstance(statement, cst.FunctionDef):
            lines.append(_entry(statement, positions, renderer))

    return "\n".join(lines)


def _entry(
    node: cst.ClassDef | cst.FunctionDef,
    positions: Mapping[cst.CSTNode, CodeRange],
    renderer: cst.Module,
) -> str:
    """Render one signature line with its def/class line and docstring summary."""
    line = positions[node.name].start.line
    entry = f"{_signature(node, renderer)} (line {line})"
    docstring = node.get_docstring()
    if docstring:
        first_line = _compact(docstring.strip().splitlines()[0])
        if first_line:
            entry += f" # {first_line}"
    return entry


def _signature(node: cst.ClassDef | cst.FunctionDef, renderer: cst.Module) -> str:
    """Rebuild a one-line signature from the CST (decorators intentionally omitted)."""
    if isinstance(node, cst.ClassDef):
        arguments = [
            _strip_arg(renderer.code_for_node(argument))
            for argument in (*node.bases, *node.keywords)
        ]
        head = f"class {node.name.value}"
        if arguments:
            head += f"({', '.join(arguments)})"
        return f"{head}:"

    prefix = "async def " if node.asynchronous is not None else "def "
    params = _compact(renderer.code_for_node(node.params)).rstrip(",").strip()
    returns = ""
    if node.returns is not None:
        returns = f" -> {_compact(renderer.code_for_node(node.returns.annotation))}"
    return f"{prefix}{node.name.value}({params}){returns}:"


def _strip_arg(rendered: str) -> str:
    """Compact an argument and drop libcst's formatter comma."""
    return _compact(rendered).rstrip(",").strip()


def _compact(rendered: str) -> str:
    """Collapse source newlines and indentation into single spaces."""
    return " ".join(rendered.split())


def _truncate_block(block: str) -> str:
    """Cap one file block at ``_MAX_FILE_MAP_CHARS``, marker included.

    Cuts on a line boundary when possible so the map never ends with half a
    signature; a single over-long entry still falls back to a character cut.
    """
    if len(block) <= _MAX_FILE_MAP_CHARS:
        return block
    limit = _MAX_FILE_MAP_CHARS - len(_TRUNCATION_MARKER)
    window = block[:limit]
    cut = window.rfind("\n")
    if cut > 0:
        window = window[:cut]
    return window.rstrip() + _TRUNCATION_MARKER
