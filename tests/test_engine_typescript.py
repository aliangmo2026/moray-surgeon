"""Tests for the TypeScript backend and its Python/Node JSON-RPC protocol.

Two layers are covered independently:

* The protocol layer runs against ``tests/fixtures/fake_ts_bridge.py`` — a Python
  executable that speaks the exact same JSON-RPC contract. It needs neither Node
  nor ts-morph, so framing, error mapping, timeouts and crash handling are
  always tested.
* The language layer runs against the real ``node bridge.mjs`` and ts-morph when
  they are installed (``scripts/setup_ts_backend.sh``). Those tests validate the
  actual TypeScript semantics: syntax checks, cross-file rename, extraction,
  call replacement and imports. They skip with an actionable reason otherwise.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from moray_surgeon.engine.backends import TypeScriptBackend, backend_for
from moray_surgeon.engine.backends import base as backends_base
from moray_surgeon.engine.backends.typescript_backend import TypeScriptBridgeClient
from moray_surgeon.engine.dispatcher import execute_plan
from moray_surgeon.exceptions import (
    PathTraversalError,
    TypeScriptBridgeError,
    UnsupportedOperationError,
)
from moray_surgeon.schema import RefactorPlan, RenameSymbol, ReplaceCall

_FIXTURES = Path(__file__).parent / "fixtures" / "typescript"
_FAKE_BRIDGE = Path(__file__).parent / "fixtures" / "fake_ts_bridge.py"
_BRIDGE_DIR = (
    Path(__file__).parent.parent
    / "src"
    / "moray_surgeon"
    / "engine"
    / "backends"
    / "ts_bridge"
)
_HAS_NODE = shutil.which("node") is not None
_HAS_TS_MORPH = (_BRIDGE_DIR / "node_modules" / "ts-morph").is_dir()

requires_node = pytest.mark.skipif(
    not (_HAS_NODE and _HAS_TS_MORPH),
    reason=(
        "real TypeScript tests need Node.js 18+ and ts-morph; "
        "run scripts/setup_ts_backend.sh (Windows: scripts/setup_ts_backend.ps1)"
    ),
)


def _fake_command(*args: str) -> list[str]:
    """Command line that runs the Python fake bridge in UTF-8 mode."""
    return [sys.executable, "-X", "utf8", str(_FAKE_BRIDGE), *args]


def _fake_backend(*args: str) -> TypeScriptBackend:
    """A backend wired to the fake bridge."""
    return TypeScriptBackend(command=_fake_command(*args))


def _fake_client(*args: str, timeout: float = 30.0) -> TypeScriptBridgeClient:
    """A protocol client wired to the fake bridge."""
    return TypeScriptBridgeClient(command=_fake_command(*args), timeout=timeout)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _copy_fixtures(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(_FIXTURES, repo)
    return repo


def _source(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# JSON-RPC protocol (fake bridge, no Node required)
# --------------------------------------------------------------------------- #


def test_protocol_ping_round_trip():
    assert _fake_client().request("ping") == {
        "pong": True,
        "node": "fake",
        "typeScript": "fake",
    }


def test_protocol_application_error_is_a_value_error():
    with pytest.raises(ValueError, match="no definition"):
        _fake_client().request("fail-application")


def test_protocol_internal_error_is_a_bridge_error():
    with pytest.raises(TypeScriptBridgeError, match="bridge exploded"):
        _fake_client().request("fail-internal")


def test_protocol_malformed_json_is_rejected():
    with pytest.raises(TypeScriptBridgeError, match="invalid JSON"):
        _fake_client().request("bad-json")


def test_protocol_two_response_lines_are_rejected():
    with pytest.raises(TypeScriptBridgeError, match="2 response line"):
        _fake_client().request("two-lines")


def test_protocol_mismatched_request_id_is_rejected():
    with pytest.raises(TypeScriptBridgeError, match="answered request 999"):
        _fake_client().request("wrong-id")


def test_protocol_exit_error_includes_stderr():
    with pytest.raises(TypeScriptBridgeError, match="bridge crashed"):
        _fake_client().request("exit-error")


def test_protocol_missing_dependencies_gets_the_setup_hint():
    with pytest.raises(TypeScriptBridgeError) as excinfo:
        _fake_client().request("missing-deps")

    assert "scripts/setup_ts_backend.sh" in str(excinfo.value)


def test_protocol_timeout_is_reported():
    with pytest.raises(TypeScriptBridgeError, match="timed out"):
        _fake_client(timeout=0.2).request("sleep")


def test_protocol_missing_executable_is_reported():
    client = TypeScriptBridgeClient(command=["definitely-not-a-node-binary"])

    with pytest.raises(TypeScriptBridgeError, match=r"Node\.js 18\+"):
        client.request("ping")


def test_protocol_missing_bridge_script_is_reported(tmp_path):
    client = TypeScriptBridgeClient(script=tmp_path / "missing.mjs")

    with pytest.raises(TypeScriptBridgeError, match="bridge script not found"):
        client.request("ping")


def test_protocol_rejects_a_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout must be positive"):
        TypeScriptBridgeClient(command=_fake_command(), timeout=0)


def test_protocol_rejects_an_empty_method():
    with pytest.raises(ValueError, match="method must not be empty"):
        _fake_client().request("")


@pytest.mark.parametrize(
    "method,message",
    [
        ("not-object", "must be a JSON object"),
        ("wrong-version", "not JSON-RPC 2.0"),
        ("error-not-object", "error must be an object"),
        ("no-result", "has no object result"),
    ],
)
def test_protocol_rejects_malformed_envelopes(method, message):
    with pytest.raises(TypeScriptBridgeError) as excinfo:
        _fake_client().request(method)

    assert message in str(excinfo.value)


def test_node_override_is_used(monkeypatch):
    monkeypatch.setenv("MORAY_NODE", sys.executable)
    client = TypeScriptBridgeClient(script=_FAKE_BRIDGE)

    assert client.request("ping")["pong"] is True


def test_invalid_node_override_is_reported(monkeypatch):
    monkeypatch.setenv("MORAY_NODE", "definitely-not-a-node-binary")
    client = TypeScriptBridgeClient(script=_FAKE_BRIDGE)

    with pytest.raises(TypeScriptBridgeError, match=r"not an executable Node\.js binary"):
        client.request("ping")


def test_missing_node_is_reported(monkeypatch):
    monkeypatch.delenv("MORAY_NODE", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    client = TypeScriptBridgeClient(script=_FAKE_BRIDGE)

    with pytest.raises(TypeScriptBridgeError) as excinfo:
        client.request("ping")

    assert "Node.js 18+" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Backend against the fake bridge (dispatcher plumbing, no Node required)
# --------------------------------------------------------------------------- #


def test_fake_verify_and_parse():
    backend = _fake_backend()

    assert backend.verify_syntax("const x = 1;") is True
    assert backend.verify_syntax("BROKEN") is False
    module = backend.parse("const x = 1;")
    assert module.statements == 1
    assert module.diagnostics == ()


def test_fake_extract_variable_returns_the_bridge_source():
    result = _fake_backend().extract_variable("const x = 1;\n", 1, 10, 1, 11, "value")

    assert result.new_source == "const x = 1;\n/* fake */"
    assert result.changes == 1


def test_fake_rename_writes_the_returned_files(tmp_path):
    target = _write(tmp_path / "app.ts", "const x = old(1);\n")

    result = _fake_backend().rename_symbol(tmp_path, "app.ts", "old", "new")

    assert result.new_source == "// renamed\n"
    assert result.changes == 1
    assert target.read_text(encoding="utf-8") == "// renamed\n"


def test_fake_rename_rejects_a_path_outside_the_repository(tmp_path):
    target = _write(tmp_path / "app.ts", "const x = old(1);\n")

    with pytest.raises(PathTraversalError):
        _fake_backend("escape").rename_symbol(tmp_path, "app.ts", "old", "new")

    assert target.read_text(encoding="utf-8") == "const x = old(1);\n"


def test_fake_rename_requires_an_existing_target(tmp_path):
    with pytest.raises(FileNotFoundError):
        _fake_backend().rename_symbol(tmp_path, "missing.ts", "old", "new")


@pytest.mark.parametrize(
    "mode,message",
    [
        ("rename-no-files", "no 'files' list"),
        ("rename-bad-entry", "file edit must be an object"),
        ("rename-no-path", "has no path"),
        ("rename-no-text", "has no text"),
        ("rename-empty", "no file edits"),
    ],
)
def test_rename_rejects_malformed_bridge_results(tmp_path, mode, message):
    _write(tmp_path / "app.ts", "const x = 1;\n")

    with pytest.raises(TypeScriptBridgeError) as excinfo:
        _fake_backend(mode).rename_symbol(tmp_path, "app.ts", "old", "new")

    assert message in str(excinfo.value)


def test_rename_requires_the_target_in_the_response(tmp_path):
    _write(tmp_path / "app.ts", "const x = 1;\n")
    _write(tmp_path / "other.ts", "const y = 2;\n")

    with pytest.raises(TypeScriptBridgeError, match="did not return a rewrite"):
        _fake_backend("rename-wrong-target").rename_symbol(
            tmp_path, "app.ts", "old", "new"
        )


@pytest.mark.parametrize(
    "mode,message",
    [
        ("parse-bad-statements", "must be an integer"),
        ("parse-bad-diagnostics", "no 'diagnostics' list"),
        ("parse-bad-diagnostic", "diagnostic must be an object"),
        ("parse-bad-message", "must be a string"),
    ],
)
def test_parse_rejects_malformed_bridge_results(mode, message):
    with pytest.raises(TypeScriptBridgeError) as excinfo:
        _fake_backend(mode).parse("const x = 1;\n")

    assert message in str(excinfo.value)


def test_verify_rejects_a_non_boolean_result():
    with pytest.raises(TypeScriptBridgeError, match="must be a boolean"):
        _fake_backend("verify-bad-ok").verify_syntax("const x = 1;\n")


def test_replace_call_rejects_an_empty_callee():
    with pytest.raises(ValueError, match="must not be empty"):
        _fake_backend().replace_call("const x = old(1);\n", " ", "new")


def test_add_import_rejects_an_empty_module():
    with pytest.raises(ValueError, match="module must not be empty"):
        _fake_backend().add_import("const x = 1;\n", "  ")


def test_add_import_rejects_an_alias_without_a_name():
    with pytest.raises(ValueError, match="alias requires name"):
        _fake_backend().add_import("const x = 1;\n", "react", alias="r")


def test_remove_import_rejects_an_empty_module():
    with pytest.raises(ValueError, match="module must not be empty"):
        _fake_backend().remove_import("const x = 1;\n", "")


def test_extract_function_rejects_bad_ranges():
    backend = _fake_backend()

    with pytest.raises(ValueError, match="1-based"):
        backend.extract_function("const x = 1;\n", "f", 0, 1)
    with pytest.raises(ValueError, match="start_line"):
        backend.extract_function("const x = 1;\n", "f", 2, 1)


def test_extract_variable_rejects_bad_ranges():
    backend = _fake_backend()

    with pytest.raises(ValueError, match="1-based"):
        backend.extract_variable("const x = 1;\n", 0, 0, 1, 1, "v")
    with pytest.raises(ValueError, match="must not be negative"):
        backend.extract_variable("const x = 1;\n", 1, -1, 1, 1, "v")
    with pytest.raises(ValueError, match="empty or reversed"):
        backend.extract_variable("const x = 1;\n", 2, 0, 1, 0, "v")


@pytest.mark.parametrize("old_name,new_name", [("old-name", "new"), ("old", "1new")])
def test_rename_rejects_invalid_identifiers(tmp_path, old_name, new_name):
    _write(tmp_path / "app.ts", "const x = 1;\n")

    with pytest.raises(ValueError, match="TypeScript identifier"):
        _fake_backend().rename_symbol(tmp_path, "app.ts", old_name, new_name)


def test_rename_rejects_an_unchanged_name(tmp_path):
    _write(tmp_path / "app.ts", "const x = 1;\n")

    with pytest.raises(ValueError, match="must differ"):
        _fake_backend().rename_symbol(tmp_path, "app.ts", "same", "same")


@pytest.mark.parametrize(
    "call",
    [
        lambda backend: backend.add_type_hints("const x = 1;\n"),
        lambda backend: backend.convert_to_match_case("const x = 1;\n", 1, 1),
        lambda backend: backend.inline_variable("x", "v", Path("."), Path("a.ts")),
        lambda backend: backend.inline_function("x", "f", Path("."), Path("a.ts")),
        lambda backend: backend.remove_dead_code("x", "f", Path("."), Path("a.ts")),
        lambda backend: backend.split_class("x", "A", ["m"], "B"),
        lambda backend: backend.move_symbol(Path("."), Path("a.ts"), Path("b.ts"), "f"),
    ],
)
def test_unported_atoms_fail_loudly(call):
    with pytest.raises(UnsupportedOperationError, match="does not implement"):
        call(_fake_backend())


def test_registry_claims_typescript_suffixes():
    assert isinstance(backend_for(Path("app.ts")), TypeScriptBackend)
    assert isinstance(backend_for(Path("app.tsx")), TypeScriptBackend)
    assert isinstance(backend_for(Path("app.TS")), TypeScriptBackend)


def test_backend_is_constructible_without_touching_node():
    backend = TypeScriptBackend()

    assert backend.language == "typescript"
    assert backend.suffixes == (".ts", ".tsx")


def test_dispatcher_routes_a_typescript_instruction(tmp_path, monkeypatch):
    fake = _fake_backend()
    monkeypatch.setitem(backends_base._BACKENDS, ".ts", fake)
    monkeypatch.setitem(backends_base._BACKENDS, ".tsx", fake)
    target = _write(tmp_path / "app.ts", "const value = old(1);\n")
    plan = RefactorPlan(
        instructions=[
            ReplaceCall(id="s1", target_file="app.ts", old_call="old", new_call="new")
        ]
    )

    log = execute_plan(plan, tmp_path, max_workers=1)

    assert log == ["[0] replace_call: 1 change(s) in app.ts"]
    assert target.read_text(encoding="utf-8") == "const value = old(1);\n/* fake */"


# --------------------------------------------------------------------------- #
# Real Node.js + ts-morph tests
# --------------------------------------------------------------------------- #


@requires_node
def test_node_bridge_ping_reports_versions():
    result = TypeScriptBridgeClient().request("ping")

    assert result["pong"] is True
    assert isinstance(result["node"], str) and result["node"].startswith("v")
    assert isinstance(result["typeScript"], str) and result["typeScript"]


@requires_node
def test_node_verify_syntax():
    backend = TypeScriptBackend()

    assert backend.verify_syntax("const x: number = 1;\n") is True
    assert backend.verify_syntax("function broken( {\n") is False


@requires_node
def test_node_parse_reports_positioned_diagnostics():
    module = TypeScriptBackend().parse("function broken( {\n")

    assert module.statements == 1
    assert module.diagnostics
    diagnostic = module.diagnostics[0]
    assert diagnostic.code == 1005
    assert diagnostic.line >= 1
    assert diagnostic.column >= 0


@requires_node
def test_node_rename_updates_definition_import_and_call(tmp_path):
    repo = _copy_fixtures(tmp_path)

    result = TypeScriptBackend().rename_symbol(repo, "math.ts", "oldName", "newName")

    assert result.changes == 3
    assert (repo / "math.ts").read_text(encoding="utf-8") == (
        "export function newName(value: number): number {\n"
        "  return value + 1;\n"
        "}\n"
    )
    assert (repo / "consumer.ts").read_text(encoding="utf-8") == (
        'import { newName } from "./math";\n'
        "\n"
        "export const answer = newName(41);\n"
    )


@requires_node
def test_node_rename_unknown_symbol_is_a_value_error(tmp_path):
    repo = _copy_fixtures(tmp_path)

    with pytest.raises(ValueError, match="no definition"):
        TypeScriptBackend().rename_symbol(repo, "math.ts", "missing", "newName")


@requires_node
def test_node_extract_variable(tmp_path):
    source = _source("extract.ts")
    start = source.index("a + b")
    line = source.count("\n", 0, start) + 1
    column = start - source.rfind("\n", 0, start) - 1

    result = TypeScriptBackend().extract_variable(
        source, line, column, line, column + len("a + b"), "sum"
    )

    assert result.changes == 1
    assert result.new_source == (
        "function add(a: number, b: number): number {\n"
        "  const sum = a + b;\n"
        "  const total = sum;\n"
        "  return total;\n"
        "}\n"
    )


@requires_node
def test_node_extract_function():
    source = 'console.log("a");\nconsole.log("b");\nconsole.log("c");\n'

    result = TypeScriptBackend().extract_function(source, "extracted", 1, 2)

    assert result.changes == 2
    assert result.new_source == (
        "function extracted() {\n"
        '    console.log("a");\n'
        '    console.log("b");\n'
        "}\n"
        "extracted();\n"
        'console.log("c");\n'
    )


@requires_node
def test_node_extract_function_rejects_an_empty_range():
    with pytest.raises(ValueError, match="no statements"):
        TypeScriptBackend().extract_function("const x = 1;\n", "empty", 5, 6)


@requires_node
def test_node_replace_call_keeps_arguments_and_unicode():
    result = TypeScriptBackend().replace_call(
        'const label = old(41, "你好");\n', "old", "fresh"
    )

    assert result.changes == 1
    assert result.new_source == 'const label = fresh(41, "你好");\n'


@requires_node
def test_node_replace_call_rejects_an_invalid_expression():
    with pytest.raises(ValueError, match="not a valid expression"):
        TypeScriptBackend().replace_call("const x = old(1);\n", "old", "not valid )(")


@requires_node
def test_node_add_import_merges_into_the_existing_declaration():
    result = TypeScriptBackend().add_import(_source("imports.ts"), "react", "useEffect")

    assert result.changes == 1
    assert result.new_source.startswith(
        'import { useState, useEffect } from "react";\n'
    )
    assert "setCount" in result.new_source


@requires_node
def test_node_add_import_appends_a_scoped_style_side_effect_import():
    result = TypeScriptBackend().add_import("const x = 1;\n", "reflect-metadata")

    assert result.changes == 1
    assert result.new_source.startswith('import "reflect-metadata";\n')


@requires_node
def test_node_remove_import_narrows_one_name():
    result = TypeScriptBackend().remove_import(
        'import { useState, useEffect } from "react";\n', "react", "useState"
    )

    assert result.changes == 1
    assert result.new_source == 'import { useEffect } from "react";\n'


@requires_node
def test_node_remove_import_drops_the_whole_declaration():
    result = TypeScriptBackend().remove_import(
        'import { useState } from "react";\nconsole.log(useState);\n', "react"
    )

    assert result.changes == 1
    assert result.new_source == "console.log(useState);\n"


@requires_node
def test_node_dispatch_replaces_a_call(tmp_path):
    target = _write(tmp_path / "app.ts", "const value = old(1);\n")
    plan = RefactorPlan(
        instructions=[
            ReplaceCall(id="s1", target_file="app.ts", old_call="old", new_call="fresh")
        ]
    )

    log = execute_plan(plan, tmp_path, max_workers=1)

    assert log == ["[0] replace_call: 1 change(s) in app.ts"]
    assert target.read_text(encoding="utf-8") == "const value = fresh(1);\n"


@requires_node
def test_node_dispatch_renames_across_files(tmp_path):
    repo = _copy_fixtures(tmp_path)
    plan = RefactorPlan(
        instructions=[
            RenameSymbol(
                id="s1", target_file="math.ts", old_name="oldName", new_name="newName"
            )
        ]
    )

    log = execute_plan(plan, repo, max_workers=1)

    assert log == ["[0] rename_symbol: 3 change(s) in math.ts"]
    assert "function newName" in (repo / "math.ts").read_text(encoding="utf-8")
    assert "newName(41)" in (repo / "consumer.ts").read_text(encoding="utf-8")
