"""Tests for the LSP server layer: features driven directly, no stdio, no network.

The resolve tests run the real libcst engine against a temporary directory, then
verify that the produced WorkspaceEdit, applied the way an editor would, yields
the expected text while the disk returns to its original bytes.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest
from lsprotocol import types
from pygls.uris import from_fs_path
from pygls.workspace import Workspace
from typer.testing import CliRunner

from moray_surgeon import lsp_server
from moray_surgeon.exceptions import NoOpEditError
from moray_surgeon.lsp_server import ActionSpec, LspServerConfig, MorayLanguageServer
from moray_surgeon.schema import AddTypeHints, RefactorPlan

_CHAIN_SOURCE = (
    "def grade(x):\n"
    "    if x == 1:\n"
    "        label = 'one'\n"
    "    elif x == 2:\n"
    "        label = 'two'\n"
    "    else:\n"
    "        label = 'many'\n"
    "    return label\n"
)

_HINT_SOURCE = "def a():\n    pass\n"
_NESTED_CHAIN = [
    "def f():",
    "    if a:",
    "        if b:",
    "            x = 1",
    "        else:",
    "            x = 2",
    "    else:",
    "        x = 3",
]


def make_server(root: Path | None, config: LspServerConfig | None = None) -> MorayLanguageServer:
    """A MorayLanguageServer with its workspace pre-initialized (no stdio loop)."""
    server = MorayLanguageServer(config or LspServerConfig())
    root_uri = from_fs_path(str(root)) if root is not None else None
    server.protocol._workspace = Workspace(root_uri)
    return server


def put_doc(server: MorayLanguageServer, root: Path, name: str, text: str) -> str:
    """Open ``name`` inside the server workspace and return its URI."""
    uri = from_fs_path(str(root / name))
    server.workspace.put_text_document(
        types.TextDocumentItem(uri=uri, language_id="python", version=1, text=text)
    )
    return uri


def action_params(
    uri: str,
    line: int,
    char: int,
    end_line: int | None = None,
    end_char: int | None = None,
    only: list[str] | None = None,
) -> types.CodeActionParams:
    """Build a codeAction request for a cursor point (or a selection)."""
    end = types.Position(
        line=end_line if end_line is not None else line,
        character=end_char if end_char is not None else char,
    )
    return types.CodeActionParams(
        text_document=types.TextDocumentIdentifier(uri=uri),
        range=types.Range(start=types.Position(line=line, character=char), end=end),
        context=types.CodeActionContext(diagnostics=[], only=only),
    )


def by_title(actions: list[types.CodeAction], needle: str) -> types.CodeAction:
    """Fetch one suggested action by title fragment."""
    for action in actions:
        if needle in action.title:
            return action
    raise AssertionError(f"missing action {needle!r}; got {[a.title for a in actions]}")


def record_messages(
    ls: MorayLanguageServer, monkeypatch: pytest.MonkeyPatch
) -> list[types.ShowMessageParams]:
    """Capture window/showMessage notifications sent to the client."""
    seen: list[types.ShowMessageParams] = []
    monkeypatch.setattr(ls, "window_show_message", seen.append)
    return seen


def direct_action(uri: str, instruction: dict[str, object], title: str = "t") -> types.CodeAction:
    """A resolve-ready CodeAction carrying a hand-built payload."""
    return types.CodeAction(
        title=title, data={"moray": True, "doc_uri": uri, "instruction": instruction, "intent": ""}
    )


def hints_action(uri: str) -> types.CodeAction:
    """A resolve-ready add_type_hints action shared by the failure tests."""
    return direct_action(
        uri,
        {"id": "a1", "action": "add_type_hints", "target_file": "app.py", "function_name": "a"},
    )


def apply_edits(text: str, edits: list[types.TextEdit]) -> str:
    """Apply TextEdits like an editor would: client units, ascending positions."""
    lines = text.splitlines(keepends=True)

    def offset(position: types.Position) -> int:
        if position.line >= len(lines):
            return len(text)
        base = sum(len(line) for line in lines[: position.line])
        content = lines[position.line].rstrip("\r\n")
        units = 0
        consumed = 0
        for char in content:
            if units >= position.character:
                break
            units += 1 + (ord(char) > 0xFFFF)
            consumed += 1
        else:
            consumed = len(content)
        return base + consumed

    spans = sorted(
        ((offset(edit.range.start), offset(edit.range.end), edit.new_text) for edit in edits),
        key=lambda span: span[0],
    )
    pieces: list[str] = []
    previous = 0
    for start, stop, new_text in spans:
        pieces.append(text[previous:start])
        pieces.append(new_text)
        previous = stop
    pieces.append(text[previous:])
    return "".join(pieces)


# --------------------------------------------------------------------------- suggestions


def test_code_action_offers_match_case_on_chain_header(tmp_path: Path) -> None:
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _CHAIN_SOURCE)
    actions = lsp_server.code_action(ls, action_params(uri, 1, 8))

    convert = by_title(actions, "Convert if/elif chain to match-case")
    assert convert.kind == types.CodeActionKind.RefactorRewrite
    instruction = convert.data["instruction"]
    assert instruction["action"] == "convert_to_match_case"
    assert instruction["start_line"] == 2  # engine lines are 1-based
    assert instruction["end_line"] == 7
    assert instruction["target_file"] == "app.py"
    assert convert.data["doc_uri"] == uri
    assert by_title(actions, "Refactor with AI").data["intent"]


def test_code_action_no_match_case_for_bare_if(tmp_path: Path) -> None:
    source = "def f(x):\n    if x == 1:\n        y = 1\n    return y\n"
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", source)
    actions = lsp_server.code_action(ls, action_params(uri, 1, 8))
    assert not any("match-case" in action.title for action in actions)


def test_code_action_extract_function_lines_from_selection(tmp_path: Path) -> None:
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _CHAIN_SOURCE)
    # select lines 2..4 (0-based), ending at column 0 of line 5: that line is not selected
    actions = lsp_server.code_action(ls, action_params(uri, 2, 0, end_line=5, end_char=0))
    extract = by_title(actions, "Extract selection to function")
    assert extract.kind == types.CodeActionKind.RefactorExtract
    instruction = extract.data["instruction"]
    assert instruction["start_line"] == 3
    assert instruction["end_line"] == 5


def test_code_action_extract_variable_converts_utf16_columns(tmp_path: Path) -> None:
    source = 'text = "\U0001F600" + x\n'  # astral emoji: two UTF-16 units, one code point
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", source)
    # The client column of `x` is 14 (emoji counts twice); the engine wants 13.
    actions = lsp_server.code_action(ls, action_params(uri, 0, 14, end_line=0, end_char=15))
    extract = by_title(actions, "Extract expression to variable")
    instruction = extract.data["instruction"]
    assert instruction["start_column"] == 13  # engine columns are code points
    assert instruction["end_column"] == 14


def test_code_action_cursor_word_offers_inline_actions(tmp_path: Path) -> None:
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _CHAIN_SOURCE)
    actions = lsp_server.code_action(ls, action_params(uri, 2, 10))  # cursor inside 'label'
    inline_var = by_title(actions, "Inline variable 'label'")
    inline_fn = by_title(actions, "Inline function 'label'")
    assert inline_var.data["instruction"]["variable_name"] == "label"
    assert inline_fn.data["instruction"]["function_name"] == "label"


def test_code_action_def_line_offers_hints_and_dead_code(tmp_path: Path) -> None:
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _CHAIN_SOURCE)
    actions = lsp_server.code_action(ls, action_params(uri, 0, 5))
    hints = by_title(actions, "Add type hints to 'grade'")
    assert hints.data["instruction"]["function_name"] == "grade"
    dead = by_title(actions, "Remove dead code 'grade'")
    assert dead.data["instruction"]["function_name"] == "grade"


def test_code_action_class_line_offers_dead_code_only(tmp_path: Path) -> None:
    source = "class Widget:\n    pass\n"
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", source)
    actions = lsp_server.code_action(ls, action_params(uri, 0, 6))
    dead = by_title(actions, "Remove dead code 'Widget'")
    assert dead.data["instruction"]["class_name"] == "Widget"
    assert not any("Add type hints" in action.title for action in actions)


def test_code_action_file_wide_type_hints_elsewhere(tmp_path: Path) -> None:
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _CHAIN_SOURCE)
    actions = lsp_server.code_action(ls, action_params(uri, 7, 4))
    hints = by_title(actions, "Add return type hints to this file")
    assert hints.data["instruction"]["function_name"] is None


def test_code_action_respects_only_filter(tmp_path: Path) -> None:
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _CHAIN_SOURCE)
    quickfix = lsp_server.code_action(
        ls, action_params(uri, 2, 0, end_line=2, end_char=12, only=["quickfix"])
    )
    assert quickfix == []


def test_code_action_only_extract_filter(tmp_path: Path) -> None:
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _CHAIN_SOURCE)
    actions = lsp_server.code_action(
        ls, action_params(uri, 2, 0, end_line=2, end_char=12, only=["refactor.extract"])
    )
    assert actions
    assert all(action.kind == types.CodeActionKind.RefactorExtract for action in actions)


def test_code_action_unknown_uri(tmp_path: Path) -> None:
    ls = make_server(tmp_path)
    assert lsp_server.code_action(ls, action_params("file:///nowhere/x.py", 0, 0)) == []


def test_code_action_untitled_buffer(tmp_path: Path) -> None:
    ls = make_server(tmp_path)
    ls.workspace.put_text_document(
        types.TextDocumentItem(
            uri="untitled:Untitled-1", language_id="python", version=1, text="x = 1\n"
        )
    )
    assert lsp_server.code_action(ls, action_params("untitled:Untitled-1", 0, 0)) == []


def test_code_action_empty_document(tmp_path: Path) -> None:
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", "")
    assert lsp_server.code_action(ls, action_params(uri, 0, 0)) == []


# ---------------------------------------------------------------- resolve: direct actions


def test_resolve_direct_action_runs_real_engine_and_restores_disk(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(_CHAIN_SOURCE, encoding="utf-8")
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _CHAIN_SOURCE)
    suggested = by_title(lsp_server.code_action(ls, action_params(uri, 1, 8)), "Convert if/elif")

    resolved = asyncio.run(lsp_server.code_action_resolve(ls, suggested))

    assert resolved.edit is not None
    edits = resolved.edit.changes[uri]
    applied = apply_edits(_CHAIN_SOURCE, edits)
    assert "match x:" in applied
    assert "elif" not in applied
    assert "case 1:" in applied
    # The surgery happened on disk but the tree was restored byte-for-byte.
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == _CHAIN_SOURCE


def test_resolve_ai_action_with_real_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "app.py").write_text(_HINT_SOURCE, encoding="utf-8")
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _HINT_SOURCE)
    suggested = by_title(
        lsp_server.code_action(ls, action_params(uri, 0, 0)), "Refactor with AI"
    )
    plan = RefactorPlan(
        instructions=[AddTypeHints(id="a1", target_file="app.py", function_name=None)]
    )

    class FakeParser:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def parse(self, intent: str, repo_root: Path, extra_context: str = "") -> RefactorPlan:
            assert "Target file: app.py" in extra_context
            return plan

    monkeypatch.setattr(lsp_server, "IntentParser", FakeParser)
    resolved = asyncio.run(lsp_server.code_action_resolve(ls, suggested))

    assert resolved.edit is not None
    applied = apply_edits(_HINT_SOURCE, resolved.edit.changes[uri])
    assert "def a() -> None:" in applied
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == _HINT_SOURCE


def test_resolve_ai_action_unreachable_model_prompts_ollama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.py").write_text(_HINT_SOURCE, encoding="utf-8")
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _HINT_SOURCE)
    suggested = by_title(lsp_server.code_action(ls, action_params(uri, 0, 0)), "Refactor with AI")
    seen = record_messages(ls, monkeypatch)

    from moray_surgeon.intent import IntentParserError

    class UnreachableParser:
        def __init__(self, **kwargs: object) -> None:
            pass

        def parse(self, intent: str, repo_root: Path, extra_context: str = "") -> RefactorPlan:
            raise IntentParserError("connection refused")

    monkeypatch.setattr(lsp_server, "IntentParser", UnreachableParser)
    resolved = asyncio.run(lsp_server.code_action_resolve(ls, suggested))

    assert resolved.edit is None
    assert seen and "Ollama" in seen[0].message


def test_resolve_engine_failure_restores_disk_and_notifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.py").write_text(_HINT_SOURCE, encoding="utf-8")
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _HINT_SOURCE)
    action = hints_action(uri)
    seen = record_messages(ls, monkeypatch)

    def failing_plan(plan: object, root: Path, indexer: object = None) -> list[str]:
        (tmp_path / "app.py").write_text("GARBAGE", encoding="utf-8")
        raise NoOpEditError("matched nothing")

    monkeypatch.setattr(lsp_server, "execute_plan", failing_plan)
    resolved = asyncio.run(lsp_server.code_action_resolve(ls, action))

    assert resolved.edit is None
    assert seen and "refused the edit" in seen[0].message
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == _HINT_SOURCE


def test_resolve_unexpected_failure_notifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.py").write_text(_HINT_SOURCE, encoding="utf-8")
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _HINT_SOURCE)
    action = hints_action(uri)
    seen = record_messages(ls, monkeypatch)
    def exploding_plan(plan: object, root: Path, indexer: object = None) -> list[str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(lsp_server, "execute_plan", exploding_plan)

    resolved = asyncio.run(lsp_server.code_action_resolve(ls, action))

    assert resolved.edit is None
    assert seen and "failed" in seen[0].message


def test_resolve_files_created_by_plan_are_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.py").write_text(_HINT_SOURCE, encoding="utf-8")
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _HINT_SOURCE)
    action = hints_action(uri)

    def messy_plan(plan: object, root: Path, indexer: object = None) -> list[str]:
        (root / "app.py").write_text("def a() -> None:\n    pass\n", encoding="utf-8")
        (root / "created.py").write_text("x = 1\n", encoding="utf-8")
        return ["log"]

    monkeypatch.setattr(lsp_server, "execute_plan", messy_plan)
    resolved = asyncio.run(lsp_server.code_action_resolve(ls, action))

    assert resolved.edit is not None
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == _HINT_SOURCE
    assert not (tmp_path / "created.py").exists()
    assert from_fs_path(str(tmp_path / "app.py")) in resolved.edit.changes


def test_resolve_refuses_unsaved_buffers(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(_HINT_SOURCE, encoding="utf-8")
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _HINT_SOURCE + "# buffer edit\n")
    action = hints_action(uri)
    seen: list[types.ShowMessageParams] = []
    ls.window_show_message = seen.append  # type: ignore[method-assign]

    resolved = asyncio.run(lsp_server.code_action_resolve(ls, action))

    assert resolved.edit is None
    assert seen and "unsaved" in seen[0].message


def test_resolve_ignores_open_doc_missing_on_disk(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(_HINT_SOURCE, encoding="utf-8")
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _HINT_SOURCE)
    ghost = from_fs_path(str(tmp_path / "ghost.py"))
    ls.workspace.put_text_document(
        types.TextDocumentItem(uri=ghost, language_id="python", version=1, text="x = 1\n")
    )
    action = hints_action(uri)

    resolved = asyncio.run(lsp_server.code_action_resolve(ls, action))
    assert resolved.edit is not None


def test_resolve_invalid_payload_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(_HINT_SOURCE, encoding="utf-8")
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _HINT_SOURCE)
    action = direct_action(uri, {"id": "a1", "action": "no_such_action", "target_file": "app.py"})
    seen: list[types.ShowMessageParams] = []
    ls.window_show_message = seen.append  # type: ignore[method-assign]

    resolved = asyncio.run(lsp_server.code_action_resolve(ls, action))

    assert resolved.edit is None
    assert seen and "validation" in seen[0].message


def test_resolve_passes_through_foreign_actions() -> None:
    ls = make_server(None)
    foreign = types.CodeAction(title="other", data={"something": "else"})
    assert asyncio.run(lsp_server.code_action_resolve(ls, foreign)) is foreign
    bare = types.CodeAction(title="no-data")
    assert asyncio.run(lsp_server.code_action_resolve(ls, bare)) is bare


def test_resolve_corrupt_payload_passes_through() -> None:
    ls = make_server(None)
    broken = types.CodeAction(title="t", data={"moray": True, "doc_uri": "file:///x.py"})
    assert asyncio.run(lsp_server.code_action_resolve(ls, broken)) is broken


# --------------------------------------------------------------------------- suggestions helpers


def test_spec_from_data_variants() -> None:
    assert lsp_server._spec_from_data(None) is None
    assert lsp_server._spec_from_data({"x": 1}) is None
    assert lsp_server._spec_from_data({"moray": False, "doc_uri": "u"}) is None
    assert lsp_server._spec_from_data({"moray": True}) is None  # doc_uri missing
    assert lsp_server._spec_from_data({"moray": True, "doc_uri": "u", "instruction": 5}) is None
    spec = lsp_server._spec_from_data(
        {"moray": True, "doc_uri": "u", "instruction": None, "intent": "i"}
    )
    assert spec == ActionSpec(doc_uri="u", instruction=None, intent="i")


@pytest.mark.parametrize(
    ("lines", "header", "expected"),
    [
        (["if a:", "    x = 1", "elif b:", "    x = 2", "else:", "    x = 3", "y = 9"], 0, (0, 5)),
        (["if a:", "    x = 1", "elif b:", "    x = 2", "else:", "    x = 3"], 2, (0, 5)),
        (["if a:", "    x = 1", "elif b:", "    x = 2"], 2, (0, 3)),
        (_NESTED_CHAIN, 4, (2, 5)),
        (_NESTED_CHAIN, 6, (1, 7)),
        (["if a:", "    x = 1", "", "elif b:", "    x = 2"], 0, (0, 4)),
        (["if a:", "    x = 1", "y = 2"], 0, None),  # bare if
        (["if a:", "    if b:", "    x = 1", "y = 2"], 0, None),  # nested if is not a branch header
        (["if a:", "    x = 1"], 1, None),  # cursor not on a header line
    ],
)
def test_if_chain_bounds(lines: list[str], header: int, expected: tuple[int, int] | None) -> None:
    assert lsp_server._if_chain_bounds(lines, header) == expected


@pytest.mark.parametrize(
    ("encoding", "text", "expected"),
    [
        (types.PositionEncodingKind.Utf16, "\U0001F600a", 3),  # astral: two units
        (types.PositionEncodingKind.Utf32, "\U0001F600a", 2),
        (types.PositionEncodingKind.Utf8, "\U0001F600a", 5),
        (types.PositionEncodingKind.Utf16, "海a", 2),  # BMP char: one unit
        (types.PositionEncodingKind.Utf8, "海a", 4),
        (None, "\U0001F600a", 3),
        ("utf-8", "abc", 3),
    ],
)
def test_client_units(encoding: object, text: str, expected: int) -> None:
    assert lsp_server._client_units(text, encoding) == expected


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("a\nb\nc\n", "a\nX\nc\n"),
        ("a\nb\nc\n", "a\nc\n"),
        ("a\nc\n", "a\nb\nc\n"),
        ("a\nb", "a\nb\nc\n"),
        ("a\nb", "a\n"),
        ("海\n尾\n", "海\n尾\n扎\n"),
        ("海\n尾", "海\n"),
        ("", "hello\n"),
        ("x = 1\n", ""),
    ],
)
def test_diff_to_edits_round_trip(before: str, after: str) -> None:
    edits = lsp_server._diff_to_edits(before, after, types.PositionEncodingKind.Utf16)
    assert apply_edits(before, edits) == after


@pytest.mark.parametrize(
    ("text", "reference", "expected"),
    [
        ("a\r\nb\r\n", "x\n", "a\nb\n"),  # engine CRLF'd an LF document -> back to LF
        ("a\r\nb\r\n", "x\r\n", "a\r\nb\r\n"),  # CRLF document stays CRLF
        ("a\nb\n", "x\r\n", "a\r\nb\r\n"),  # LF engine output in a CRLF document
        ("a\r\nb\r\n", "x\r", "a\rb\r"),  # exotic CR document
        ("a\r\n", "", "a\n"),  # no reference (created file) -> LF
    ],
)
def test_restyle_newlines(text: str, reference: str, expected: str) -> None:
    assert lsp_server._restyle_newlines(text, reference) == expected


def test_collect_changes_restyles_lf_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An LF file the engine rewrote with CRLF endings diffs only on real changes."""
    target = tmp_path / "app.py"
    target.write_bytes(_HINT_SOURCE.encode("utf-8"))  # force LF on disk

    def crlf_engine(plan: object, root: Path, indexer: object = None) -> list[str]:
        target.write_text(_HINT_SOURCE.replace("def a():", "def a() -> None:"), encoding="utf-8")
        return ["log"]

    monkeypatch.setattr(lsp_server, "execute_plan", crlf_engine)
    ls = make_server(tmp_path)
    uri = put_doc(ls, tmp_path, "app.py", _HINT_SOURCE)
    resolved = asyncio.run(
        lsp_server.code_action_resolve(
            ls, hints_action(uri)
        )
    )

    assert resolved.edit is not None
    edits = resolved.edit.changes[uri]
    applied = apply_edits(_HINT_SOURCE, edits)
    assert applied == "def a() -> None:\n    pass\n"  # LF preserved, no CRLF leaked
    assert all("\r" not in edit.new_text for edit in edits)
    assert target.read_bytes() == _HINT_SOURCE.encode("utf-8")  # disk restored to LF


# --------------------------------------------------------------------------- repo root / targets


def test_repo_root_prefers_git_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    package = repo / "pkg"
    package.mkdir()
    doc = package / "m.py"
    doc.write_text("x = 1\n", encoding="utf-8")
    assert lsp_server._repo_root(doc, make_server(tmp_path)) == repo.resolve()


def test_repo_root_falls_back_to_workspace(tmp_path: Path) -> None:
    doc = tmp_path / "m.py"
    doc.write_text("x = 1\n", encoding="utf-8")
    assert lsp_server._repo_root(doc, make_server(tmp_path)) == tmp_path.resolve()


def test_repo_root_falls_back_to_parent(tmp_path: Path) -> None:
    doc = tmp_path / "m.py"
    doc.write_text("x = 1\n", encoding="utf-8")
    ls = make_server(None)
    assert lsp_server._repo_root(doc, ls) == tmp_path.resolve()


def test_relative_target_outside_root_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "definitely-outside.py"
    with pytest.raises(lsp_server._ActionRejectedError):
        lsp_server._relative_target(outside, tmp_path)


# --------------------------------------------------------------------------- process entry points


def test_main_attaches_config_starts_io_and_logs_to_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[int] = []
    monkeypatch.setattr(lsp_server.server, "start_io", lambda: started.append(1))
    monkeypatch.setenv("MORAY_LSP_LOG", str(tmp_path / "lsp.log"))
    config = LspServerConfig(model="m2")

    lsp_server.main(config)
    lsp_server.main(config)  # second call must not duplicate the file handler

    assert started == [1, 1]
    assert lsp_server.server.moray_config is config
    assert (tmp_path / "lsp.log").exists()
    named = [
        handler
        for handler in lsp_server.logger.handlers
        if handler.get_name() == lsp_server._LOG_HANDLER_NAME
    ]
    assert len(named) == 1
    lsp_server.logger.handlers.remove(named[0])


def test_cli_lsp_command_builds_config_from_flags_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from moray_surgeon.cli import app

    recorded: dict[str, LspServerConfig] = {}
    monkeypatch.setattr(
        "moray_surgeon.lsp_server.main",
        lambda config=None: recorded.update(config=config),
    )
    monkeypatch.setenv("MORAY_API_KEY", "env-key")

    result = CliRunner().invoke(app, ["lsp", "--model", "m1", "--api-type", "openai", "--base-url", "https://api.example.com/v1"])

    assert result.exit_code == 0
    config = recorded["config"]
    assert config.model == "m1"
    assert config.api_type == "openai"
    assert config.base_url == "https://api.example.com/v1"
    assert config.api_key == "env-key"


def test_cli_lsp_flag_beats_env_for_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from moray_surgeon.cli import app

    recorded: dict[str, LspServerConfig] = {}
    monkeypatch.setattr(
        "moray_surgeon.lsp_server.main",
        lambda config=None: recorded.update(config=config),
    )
    monkeypatch.setenv("MORAY_API_KEY", "env-key")

    result = CliRunner().invoke(app, ["lsp", "--api-key", "flag-key"])

    assert result.exit_code == 0
    assert recorded["config"].api_key == "flag-key"


def test_importing_lsp_server_writes_nothing_to_stdout() -> None:
    process = subprocess.run(
        [sys.executable, "-I", "-c", "import moray_surgeon.lsp_server"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert process.stdout == ""
