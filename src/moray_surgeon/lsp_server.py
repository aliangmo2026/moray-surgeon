"""Language Server Protocol layer: moray-surgeon refactors as editor code actions.

Editors (VSCode among them) launch this process over stdio and get:

- ``textDocument/codeAction``: cheap, synchronous suggestions computed from the
  current selection/cursor (extract, inline, match-case conversion, ...).
- ``codeAction/resolve``: the heavy half. The picked action becomes a
  ``RefactorPlan`` that the real engine applies to the files on disk
  (``execute_plan``); the result is captured and the working tree is restored
  byte-for-byte. The captured before/after texts travel back as a
  ``WorkspaceEdit``, so the user reviews and saves the change themselves.

Architecture notes:

- The pipeline is deliberately git-free. Editors keep unsaved buffers, and a
  stash/reset cycle running underneath them would race with keystrokes;
  instead every ``*.py`` file is snapshotted before surgery and restored
  afterwards, and nothing is ever committed.
- If an open document inside the repository diverges from disk, the action is
  refused: the engine reads from disk, so a stale buffer would receive edits
  computed against the wrong base text. Saving first is a small, predictable
  price for correctness.
- stdout belongs to JSON-RPC. Diagnostics go through ``logging`` to a file
  (``MORAY_LSP_LOG``, default ``~/.moray-surgeon/lsp.log``), mirroring the MCP
  server.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from lsprotocol import types
from pygls.lsp.server import LanguageServer
from pygls.uris import from_fs_path, to_fs_path
from pygls.workspace import TextDocument

from moray_surgeon import __version__
from moray_surgeon.engine.dispatcher import execute_plan
from moray_surgeon.exceptions import MorayError
from moray_surgeon.intent import IntentParser, IntentParserError
from moray_surgeon.schema import RefactorPlan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.propagate = False

# pytest and IDE hosts attach their own handlers to module loggers; only our
# named handler decides whether the file sink is already configured.
_LOG_HANDLER_NAME = "moray-surgeon-lsp-file"

# Never snapshotted, never refactored: tooling noise around the user's code.
_SKIP_DIRS = frozenset(
    {".git", ".moray-cache", "__pycache__", ".venv", "venv", ".tox", "node_modules"}
)

_DEF_LINE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)")
_CLASS_LINE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)")

_NO_CHANGES_MESSAGE = (
    "The refactor produced no changes (the model may have returned a no-op plan)."
)
_MODEL_HELP = (
    "Could not reach the configured model. For a local Ollama: run `ollama serve`, "
    "pull the model (e.g. `ollama pull qwen2.5-coder:7b`), then retry. For a cloud "
    "API: launch the server with --api-type openai|anthropic and set MORAY_API_KEY.\n\n"
    "Details: "
)

# One refactor at a time: the engine shares a process-wide code index and the
# snapshot/restore window must stay atomic.
_REFACTOR_LOCK = threading.Lock()


class _ActionRejectedError(Exception):
    """A code action cannot run; its message is shown to the user."""


@dataclass(frozen=True)
class LspServerConfig:
    """Provider settings for the AI code action; mirrors the CLI's flags."""

    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434"
    api_type: str = "ollama"
    api_key: str | None = None


class MorayLanguageServer(LanguageServer):
    """LanguageServer variant that carries the provider configuration."""

    def __init__(self, config: LspServerConfig) -> None:
        super().__init__("moray-surgeon", __version__)
        self.moray_config = config


server = MorayLanguageServer(LspServerConfig())


@dataclass(frozen=True)
class ActionSpec:
    """Payload round-tripped through ``CodeAction.data`` into ``codeAction/resolve``.

    Exactly one of ``instruction`` (a schema-shaped direct action) or ``intent``
    (natural language for the AI action) is populated.
    """

    doc_uri: str
    instruction: dict[str, object] | None = None
    intent: str = ""


@server.feature(types.TEXT_DOCUMENT_CODE_ACTION, types.CodeActionOptions(resolve_provider=True))
def code_action(
    ls: MorayLanguageServer, params: types.CodeActionParams
) -> list[types.CodeAction]:
    """Offer moray-surgeon refactors for the current selection or cursor.

    Suggestion computation is deliberately cheap and synchronous; the engine
    only runs when the user picks an action (``codeAction/resolve``).
    """
    try:
        doc = ls.workspace.get_text_document(params.text_document.uri)
        actions = _suggest_actions(ls, doc, params)
    except (_ActionRejectedError, KeyError, OSError):
        return []  # unknown/untitled documents simply offer nothing

    only = params.context.only
    if only:
        actions = [
            action
            for action in actions
            if action.kind is not None
            and any(o and action.kind.startswith(o) for o in only)
        ]
    return actions


@server.feature(types.CODE_ACTION_RESOLVE)
async def code_action_resolve(
    ls: MorayLanguageServer, params: types.CodeAction
) -> types.CodeAction:
    """Run the picked refactor and attach the resulting ``WorkspaceEdit``.

    Every failure becomes a ``window/showMessage`` notification; the action is
    returned unchanged so the protocol loop never sees an exception.
    """
    spec = _spec_from_data(params.data)
    if spec is None:
        return params
    try:
        edit = await asyncio.to_thread(_execute_action, ls, spec)
    except _ActionRejectedError as exc:
        _show_message(ls, types.MessageType.Warning, str(exc))
        return params
    except MorayError as exc:
        _show_message(ls, types.MessageType.Error, f"moray-surgeon refused the edit: {exc}")
        return params
    except OSError as exc:
        _show_message(ls, types.MessageType.Error, f"I/O error while refactoring: {exc}")
        return params
    except Exception:
        logger.exception("moray-surgeon code action failed unexpectedly")
        _show_message(
            ls, types.MessageType.Error, "moray-surgeon failed; see the LSP log for details."
        )
        return params
    return types.CodeAction(title=params.title, kind=params.kind, data=params.data, edit=edit)


def _suggest_actions(
    ls: MorayLanguageServer, doc: TextDocument, params: types.CodeActionParams
) -> list[types.CodeAction]:
    """Build the concrete action list for one ``codeAction`` request."""
    text = doc.source
    lines = text.splitlines()
    if not lines:
        return []

    uri = doc.uri
    doc_path = _doc_path(uri)
    root = _repo_root(doc_path, ls)
    target = _relative_target(doc_path, root)
    start, end = params.range.start, params.range.end
    has_selection = doc.offset_at_position(start) != doc.offset_at_position(end)
    word = "" if has_selection else doc.word_at_position(start)

    specs: list[tuple[str, types.CodeActionKind, ActionSpec]] = []

    if has_selection:
        specs.extend(_selection_specs(doc, target, uri, start, end))

    header = next(
        (
            i
            for i in range(start.line, min(end.line, len(lines) - 1) + 1)
            if _is_branch_header(lines[i])
        ),
        None,
    )
    if header is not None:
        bounds = _if_chain_bounds(lines, header)
        if bounds is not None:
            specs.append(
                (
                    "moray: Convert if/elif chain to match-case",
                    types.CodeActionKind.RefactorRewrite,
                    ActionSpec(
                        doc_uri=uri,
                        instruction=_instruction_dict(
                            "convert_to_match_case",
                            target,
                            start_line=bounds[0] + 1,
                            end_line=bounds[1] + 1,
                        ),
                    ),
                )
            )

    if word:
        specs.append(
            (
                f"moray: Inline variable '{word}'",
                types.CodeActionKind.RefactorInline,
                ActionSpec(
                    doc_uri=uri,
                    instruction=_instruction_dict("inline_variable", target, variable_name=word),
                ),
            )
        )
        specs.append(
            (
                f"moray: Inline function '{word}'",
                types.CodeActionKind.RefactorInline,
                ActionSpec(
                    doc_uri=uri,
                    instruction=_instruction_dict("inline_function", target, function_name=word),
                ),
            )
        )

    if not has_selection and start.line < len(lines):
        specs.extend(_cursor_specs(lines[start.line], target, uri))

    intent = _intent_text(doc, text, target, start, end, has_selection, word)
    specs.append(
        (
            "moray: Refactor with AI (moray-surgeon)",
            types.CodeActionKind.Refactor,
            ActionSpec(doc_uri=uri, intent=intent),
        )
    )
    return [_code_action(title, kind, spec) for title, kind, spec in specs]


def _selection_specs(
    doc: TextDocument, target: str, uri: str, start: types.Position, end: types.Position
) -> list[tuple[str, types.CodeActionKind, ActionSpec]]:
    """Actions that operate on the selected text (extract family)."""
    last_line = end.line
    if end.character == 0 and last_line > start.line:
        last_line -= 1  # the selection ends before any character of that line
    extract_function = ActionSpec(
        doc_uri=uri,
        instruction=_instruction_dict(
            "extract_function",
            target,
            function_name="extracted_function",
            start_line=start.line + 1,
            end_line=last_line + 1,
        ),
    )
    extract_variable = ActionSpec(
        doc_uri=uri,
        instruction=_instruction_dict(
            "extract_variable",
            target,
            start_line=start.line + 1,
            start_column=_char_column(doc, start),
            end_line=end.line + 1,
            end_column=_char_column(doc, end),
            variable_name="extracted_value",
        ),
    )
    return [
        (
            "moray: Extract selection to function",
            types.CodeActionKind.RefactorExtract,
            extract_function,
        ),
        (
            "moray: Extract expression to variable",
            types.CodeActionKind.RefactorExtract,
            extract_variable,
        ),
    ]


def _cursor_specs(
    line: str, target: str, uri: str
) -> list[tuple[str, types.CodeActionKind, ActionSpec]]:
    """Actions that depend on which definition line the cursor sits on."""
    function_match = _DEF_LINE.match(line)
    class_match = _CLASS_LINE.match(line)
    specs: list[tuple[str, types.CodeActionKind, ActionSpec]] = []
    if function_match is not None:
        name = function_match.group(1)
        specs.append(
            (
                f"moray: Add type hints to '{name}'",
                types.CodeActionKind.RefactorRewrite,
                ActionSpec(
                    doc_uri=uri,
                    instruction=_instruction_dict(
                        "add_type_hints", target, function_name=name
                    ),
                ),
            )
        )
    if function_match is not None:
        name = function_match.group(1)
        specs.append(
            (
                f"moray: Remove dead code '{name}'",
                types.CodeActionKind.Refactor,
                ActionSpec(
                    doc_uri=uri,
                    instruction=_instruction_dict(
                        "remove_dead_code", target, function_name=name
                    ),
                ),
            )
        )
        return specs
    if class_match is not None:
        name = class_match.group(1)
        specs.append(
            (
                f"moray: Remove dead code '{name}'",
                types.CodeActionKind.Refactor,
                ActionSpec(
                    doc_uri=uri,
                    instruction=_instruction_dict(
                        "remove_dead_code", target, class_name=name
                    ),
                ),
            )
        )
        return specs
    specs.append(
        (
            "moray: Add return type hints to this file",
            types.CodeActionKind.RefactorRewrite,
            ActionSpec(
                doc_uri=uri,
                instruction=_instruction_dict("add_type_hints", target, function_name=None),
            ),
        )
    )
    return specs


def _intent_text(
    doc: TextDocument,
    text: str,
    target: str,
    start: types.Position,
    end: types.Position,
    has_selection: bool,
    word: str,
) -> str:
    """Compose the natural-language intent for the AI code action."""
    if has_selection:
        selected = text[doc.offset_at_position(start) : doc.offset_at_position(end)]
        return (
            f"Refactor this code from {target} (lines {start.line + 1}-{end.line + 1}):\n\n"
            f"{selected}"
        )
    if word:
        return f"Refactor the symbol '{word}' in {target} at line {start.line + 1}."
    return f"Refactor {target}."


def _code_action(title: str, kind: types.CodeActionKind, spec: ActionSpec) -> types.CodeAction:
    """Wrap one suggestion; the payload rides in ``data`` until resolve."""
    return types.CodeAction(
        title=title,
        kind=kind,
        data={
            "moray": True,
            "doc_uri": spec.doc_uri,
            "instruction": spec.instruction,
            "intent": spec.intent,
        },
    )


def _spec_from_data(data: object) -> ActionSpec | None:
    """Rebuild the payload the client echoed back; non-moray actions pass through."""
    if not isinstance(data, dict) or data.get("moray") is not True:
        return None
    try:
        instruction = data["instruction"]
        if instruction is not None and not isinstance(instruction, dict):
            return None
        return ActionSpec(
            doc_uri=str(data["doc_uri"]),
            instruction=instruction,
            intent=str(data.get("intent", "")),
        )
    except (KeyError, TypeError):
        return None


def _instruction_dict(action: str, target_file: str, **fields: object) -> dict[str, object]:
    """One schema-shaped instruction; ids only need uniqueness inside the plan."""
    payload: dict[str, object] = {
        "id": f"lsp-{uuid.uuid4().hex[:8]}",
        "action": action,
        "target_file": target_file,
    }
    payload.update(fields)
    return payload


def _execute_action(ls: MorayLanguageServer, spec: ActionSpec) -> types.WorkspaceEdit:
    """Apply the plan for real, capture the result, restore the tree, build the edit."""
    doc_path = _doc_path(spec.doc_uri)
    root = _repo_root(doc_path, ls)
    target = _relative_target(doc_path, root)
    _require_saved_buffers(ls, root)
    plan = _plan_for(ls, spec, root, target)

    with _REFACTOR_LOCK:
        snapshot = _snapshot_python_sources(root)
        try:
            execute_plan(plan, root)
            changes = _collect_changes(root, snapshot)
        finally:
            _restore_python_sources(root, snapshot)

    # Someone may have typed while the engine ran in its worker thread: edits
    # computed against the pre-keystroke text would land at shifted ranges.
    _require_saved_buffers(ls, root)

    if not changes:
        raise _ActionRejectedError(_NO_CHANGES_MESSAGE)
    return _build_workspace_edit(ls, changes)


def _plan_for(
    ls: MorayLanguageServer, spec: ActionSpec, root: Path, target: str
) -> RefactorPlan:
    """Direct actions validate locally; the AI action asks the configured model."""
    if spec.instruction is not None:
        try:
            return RefactorPlan.model_validate(
                {"instructions": [spec.instruction], "summary": "moray-surgeon code action"}
            )
        except ValueError as exc:
            raise _ActionRejectedError(f"The code action payload failed validation: {exc}") from exc

    config = ls.moray_config
    parser = IntentParser(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        api_type=config.api_type,
    )
    try:
        return parser.parse(spec.intent, root, f"Target file: {target}")
    except IntentParserError as exc:
        raise _ActionRejectedError(f"{_MODEL_HELP}{exc}") from exc


def _doc_path(uri: str) -> Path:
    """URI to a resolved filesystem path; untitled buffers cannot be refactored."""
    path = to_fs_path(uri)
    if path is None:
        raise _ActionRejectedError(
            "Only real files on disk can be refactored (no untitled buffers)."
        )
    return Path(path).resolve()


def _repo_root(doc_path: Path, ls: MorayLanguageServer) -> Path:
    """Nearest enclosing ``.git`` directory, else the workspace root, else the file's."""
    for parent in doc_path.parents:
        if (parent / ".git").exists():
            return parent.resolve()
    root_uri = ls.workspace.root_uri
    fs_root = to_fs_path(root_uri) if root_uri else None
    if fs_root and _within(doc_path, Path(fs_root)):
        return Path(fs_root).resolve()
    return doc_path.parent.resolve()


def _relative_target(doc_path: Path, root: Path) -> str:
    """Repository-relative POSIX path; the engine's own containment gate backs this up."""
    try:
        return doc_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise _ActionRejectedError("The document is outside the detected repository root.") from exc


def _within(path: Path, root: Path) -> bool:
    """True when ``path`` resolves inside ``root`` (path-traversal gate)."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _require_saved_buffers(ls: MorayLanguageServer, root: Path) -> None:
    """Refuse to run while any open Python document inside ``root`` diverges from disk.

    The engine reads from disk, so the WorkspaceEdit ranges would be computed
    against a base the editor no longer shows.
    """
    for uri, doc in ls.workspace.text_documents.items():
        path = to_fs_path(uri)
        if path is None or Path(path).suffix != ".py" or not _within(Path(path), root):
            continue
        try:
            disk = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        if disk != doc.source:
            raise _ActionRejectedError(
                f"'{Path(path).name}' has unsaved changes in the editor. "
                "Save your Python files, then retry the code action."
            )


def _snapshot_python_sources(root: Path) -> dict[Path, bytes]:
    """Byte snapshot of every tracked ``*.py`` file; restore() rewrites them."""
    snapshot: dict[Path, bytes] = {}
    for path in root.rglob("*.py"):
        if not path.is_file():
            continue
        if _SKIP_DIRS & set(path.relative_to(root).parts):
            continue
        snapshot[path] = path.read_bytes()
    return snapshot


def _collect_changes(root: Path, snapshot: dict[Path, bytes]) -> list[tuple[Path, str, str]]:
    """(path, before, after) triples for every file surgery touched."""
    current = {
        path: path.read_bytes()
        for path in root.rglob("*.py")
        if path.is_file() and not _SKIP_DIRS & set(path.relative_to(root).parts)
    }
    changes: list[tuple[Path, str, str]] = []
    for path, before in snapshot.items():
        after = current.get(path)
        if after is None:
            changes.append((path, before.decode("utf-8"), ""))
        elif after != before:
            before_text = before.decode("utf-8")
            changes.append(
                (path, before_text, _restyle_newlines(after.decode("utf-8"), before_text))
            )
    for path, after in current.items():
        if path not in snapshot:
            changes.append((path, "", _restyle_newlines(after.decode("utf-8"), "")))
    return changes


def _restyle_newlines(text: str, reference: str) -> str:
    """Re-express ``text`` in the reference document's newline style.

    The engine writes through ``Path.write_text``, whose universal-newline mode
    translates ``\\n`` to ``os.linesep`` — an LF file comes back CRLF on Windows
    even on lines libcst did not touch. Normalize everything first, then apply
    the reference style, so untouched lines diff as equal again.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\r\n" in reference:
        return normalized.replace("\n", "\r\n")
    if "\r" in reference:
        return normalized.replace("\n", "\r")
    return normalized


def _restore_python_sources(root: Path, snapshot: dict[Path, bytes]) -> None:
    """Undo the surgery byte-for-byte; created files are removed again.

    Called from ``finally``: whatever the engine did, the editor must keep
    seeing the tree it started with, because the edit travels via WorkspaceEdit.
    """
    for path, data in snapshot.items():
        try:
            if path.exists() and path.read_bytes() == data:
                continue
            path.write_bytes(data)
        except OSError:
            logger.exception("failed to restore %s", path)
    for path in root.rglob("*.py"):
        if path in snapshot or _SKIP_DIRS & set(path.relative_to(root).parts):
            continue
        try:
            path.unlink()
        except OSError:
            logger.exception("failed to remove file created by the refactor: %s", path)


def _build_workspace_edit(
    ls: MorayLanguageServer, changes: list[tuple[Path, str, str]]
) -> types.WorkspaceEdit:
    """Captured before/after texts -> per-document TextEdits in client units."""
    encoding = ls.workspace.position_encoding
    edits_by_uri: dict[str, list[types.TextEdit]] = {}
    for path, before, after in sorted(changes, key=lambda item: item[0].as_posix()):
        uri = from_fs_path(str(path))
        if uri is None:
            continue
        edits = _diff_to_edits(before, after, encoding)
        if edits:
            edits_by_uri[uri] = edits
    return types.WorkspaceEdit(changes=edits_by_uri)


def _diff_to_edits(
    before: str, after: str, encoding: types.PositionEncodingKind | str | None
) -> list[types.TextEdit]:
    """Line-granular edits covering every changed region, in ascending order."""
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    edits: list[types.TextEdit] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        edits.append(
            types.TextEdit(
                range=types.Range(
                    start=_anchor(before_lines, i1, encoding),
                    end=_anchor(before_lines, i2, encoding),
                ),
                new_text="".join(after_lines[j1:j2]),
            )
        )
    return edits


def _anchor(
    lines: list[str], index: int, encoding: types.PositionEncodingKind | str | None
) -> types.Position:
    """Start-of-line position; past-the-end positions clamp precisely to EOF.

    A trailing newline means the document has a virtual final line, so
    ``Position(len(lines), 0)`` is exact there; without one, the character
    offset must land at the end of the last real line.
    """
    if index < len(lines):
        return types.Position(line=index, character=0)
    if not lines:
        return types.Position(line=0, character=0)
    last = lines[-1]
    if last.endswith("\n"):
        return types.Position(line=len(lines), character=0)
    return types.Position(line=len(lines) - 1, character=_client_units(last, encoding))


def _client_units(text: str, encoding: types.PositionEncodingKind | str | None) -> int:
    """Length of ``text`` in the negotiated LSP position encoding's units."""
    name = getattr(encoding, "value", encoding)
    if name == "utf-32":
        return len(text)
    if name == "utf-8":
        return len(text.encode("utf-8"))
    # utf-16 is the protocol default: one unit per BMP code point, two for astral.
    return len(text) + sum(1 for char in text if ord(char) > 0xFFFF)


def _char_column(doc: TextDocument, position: types.Position) -> int:
    """0-based character (code point) column of a client position within its line."""
    line_start = types.Position(line=position.line, character=0)
    return doc.offset_at_position(position) - doc.offset_at_position(line_start)


def _is_branch_header(line: str) -> bool:
    """True for ``if``/``elif``/``else`` header lines."""
    stripped = line.strip()
    return stripped.startswith(("if ", "elif ", "else:")) or stripped == "else:"


def _indent(line: str) -> int:
    """Visual indent width in columns."""
    return len(line) - len(line.lstrip())


def _if_chain_bounds(lines: list[str], header: int) -> tuple[int, int] | None:
    """Expand the if/elif/else chain around ``header`` to 0-based inclusive bounds.

    Purely textual and conservative: unusual continuation formatting can
    truncate the chain, and the engine then refuses the range instead of
    rewriting something wrong. A bare ``if`` without ``elif``/``else`` has no
    match-case equivalent and yields None.
    """
    count = len(lines)
    if not (0 <= header < count) or not _is_branch_header(lines[header]):
        return None
    indent = _indent(lines[header])
    start = header

    if lines[header].strip().startswith(("elif ", "else:")):
        cursor = header - 1
        while cursor >= 0:
            stripped = lines[cursor].strip()
            if stripped == "":
                cursor -= 1
                continue
            line_indent = _indent(lines[cursor])
            if line_indent > indent:
                cursor -= 1  # the previous branch's body
                continue
            if line_indent < indent:
                return None
            if stripped.startswith(("elif ", "else:")):
                start = cursor
                cursor -= 1
                continue
            if stripped.startswith("if "):
                start = cursor
                break
            return None  # another statement sits between: not one chain

    end = start
    cursor = start + 1
    while cursor < count:
        stripped = lines[cursor].strip()
        if stripped == "":
            cursor += 1
            continue
        line_indent = _indent(lines[cursor])
        if line_indent > indent:
            end = cursor
            cursor += 1
            continue
        if line_indent == indent and stripped.startswith(("elif ", "else:")):
            end = cursor
            cursor += 1
            continue
        break  # the next sibling statement: the chain ended

    if not any(
        i != start and _indent(lines[i]) == indent and _is_branch_header(lines[i])
        for i in range(start, end + 1)
    ):
        return None
    return start, end


def _show_message(ls: MorayLanguageServer, level: types.MessageType, text: str) -> None:
    """Forward a user-facing diagnostic to the editor."""
    ls.window_show_message(types.ShowMessageParams(type=level, message=text))


def _configure_file_logging() -> None:
    """Attach one named file handler; stdout/stderr stay clean for JSON-RPC."""
    if any(handler.get_name() == _LOG_HANDLER_NAME for handler in logger.handlers):
        return
    log_path = Path(
        os.environ.get("MORAY_LSP_LOG") or Path.home() / ".moray-surgeon" / "lsp.log"
    )
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError:
        logger.warning("cannot open %s; LSP logging stays disabled", log_path)
        return
    handler.set_name(_LOG_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main(config: LspServerConfig | None = None) -> None:
    """Run the language server over stdio (editors launch this as the LSP command)."""
    if config is not None:
        server.moray_config = config
    _configure_file_logging()
    logger.info("moray-surgeon LSP server starting")
    try:
        server.start_io()
    finally:
        logger.info("moray-surgeon LSP server stopped")
