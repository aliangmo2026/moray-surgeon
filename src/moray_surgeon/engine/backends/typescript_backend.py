"""TypeScript support through ``ts-morph`` behind a JSON-RPC process boundary.

Python cannot parse TypeScript, so this backend delegates every AST operation to
a Node.js script (:mod:`bridge.mjs <moray_surgeon.engine.backends.ts_bridge>`)
that runs the TypeScript compiler through ``ts-morph``. The Python side owns the
protocol, path validation and writes; the Node side owns parsing and rewriting.

Protocol (one request per process, UTF-8, one JSON line in and one out)::

    -> {"jsonrpc":"2.0","id":1,"method":"rename",
        "params":{"repoRoot":"/repo","targetFile":"src/a.ts",
                  "oldName":"old","newName":"new"}}
    <- {"jsonrpc":"2.0","id":1,"result":{"changes":3,
        "files":[{"path":"src/a.ts","text":"..."}, ...]}}

    error -> {"jsonrpc":"2.0","id":1,
              "error":{"code":-32603,"message":"..."}}

Error codes are part of the contract: ``-32602`` (invalid params) and ``-32000``
(operation failed) become :class:`ValueError`, which the dispatcher classifies as
an instruction rejection; every other code, a timeout, a crash, a malformed
response or a missing ``node``/``ts-morph`` installation becomes
:class:`~moray_surgeon.exceptions.TypeScriptBridgeError`.

Supported atoms: ``parse``, ``verify_syntax``, ``rename_symbol`` (ts-morph
LanguageService across files), ``extract_function``, ``extract_variable``,
``replace_call``, ``add_import`` and ``remove_import``. The Python-specific and
still-unported atoms raise
:class:`~moray_surgeon.exceptions.UnsupportedOperationError` instead of silently
doing nothing.

Limitations worth knowing before planning TS work:

* The schema's ``module`` field is still a dotted path, so ``add_import`` /
  ``remove_import`` can express bare packages (``react``) but not relative
  (``./util``) or scoped (``@org/pkg``) specifiers until the protocol widens.
* Columns in ``extract_variable`` are JavaScript string indices (UTF-16 code
  units), not Unicode code points. For ASCII sources the two coincide.
* ``extract_function`` moves top-level statements, mirroring the Python atom;
  leading comments above the first selected statement stay where they are.

Node.js 18+ and the bridge dependencies are required at *call* time, not at
import time, so this module can be imported and registered on a machine that has
no Node. Install them with ``scripts/setup_ts_backend.sh`` (or
``scripts/setup_ts_backend.ps1`` on Windows); override discovery with
``MORAY_NODE`` and ``MORAY_TS_BRIDGE``.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast

from moray_surgeon.engine.backends.base import LanguageBackend
from moray_surgeon.engine.base import EditResult
from moray_surgeon.exceptions import (
    PathTraversalError,
    TypeScriptBridgeError,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from moray_surgeon.engine.indexer import Indexer

_DEFAULT_TIMEOUT_SECONDS = 120.0
_NODE_ENV = "MORAY_NODE"
_BRIDGE_ENV = "MORAY_TS_BRIDGE"
_SCRIPT_PATH = Path(__file__).with_name("ts_bridge") / "bridge.mjs"
_TS_IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_INVALID_PARAMS = -32602
_APPLICATION_ERROR = -32000
_ERROR_DETAIL_LIMIT = 2000
_SETUP_HINT = (
    "install the Node dependencies with scripts/setup_ts_backend.sh "
    "(Windows: scripts/setup_ts_backend.ps1)"
)
_NODE_HINT = f"Node.js 18+ is required for the TypeScript backend; {_SETUP_HINT}"


@dataclass(frozen=True)
class TsDiagnostic:
    """One TypeScript diagnostic, positioned for a human or a model.

    Attributes:
        code: Compiler diagnostic code (for example 1005).
        message: Flattened compiler message.
        line: 1-based line number.
        column: 0-based column in UTF-16 code units.
    """

    code: int
    message: str
    line: int
    column: int


@dataclass(frozen=True)
class TsModule:
    """Opaque parse handle returned by :meth:`TypeScriptBackend.parse`.

    The real AST lives inside the Node bridge and is discarded with the process;
    this handle carries what callers can use without another round-trip.
    """

    source: str
    statements: int
    diagnostics: tuple[TsDiagnostic, ...]


class TypeScriptBridgeClient:
    """One-request-per-process JSON-RPC client for the Node bridge.

    A fresh Node process per request keeps timeouts, crashes and memory state
    isolated; the price is process startup on every atom (tens to hundreds of
    milliseconds, plus project loading for a rename). A persistent pool can be
    added later without changing this protocol.
    """

    def __init__(
        self,
        command: Sequence[str] | None = None,
        script: Path | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Configure the transport.

        Args:
            command: Complete command line to run. Tests use this to point at a
                fake bridge; ``None`` builds ``[node, script]`` lazily.
            script: Bridge entry point. ``None`` uses ``MORAY_TS_BRIDGE`` or the
                packaged ``ts_bridge/bridge.mjs``.
            timeout: Hard per-request limit in seconds.

        Raises:
            ValueError: ``timeout`` is not positive.
        """
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._command = list(command) if command is not None else None
        override = os.environ.get(_BRIDGE_ENV, "").strip()
        self._script = script or (Path(override) if override else _SCRIPT_PATH)
        self._timeout = timeout
        self._ids = itertools.count(1)
        self._resolved_command: list[str] | None = None

    def request(
        self,
        method: str,
        params: dict[str, object] | None = None,
        *,
        cwd: Path | None = None,
    ) -> dict[str, object]:
        """Send one JSON-RPC request and return its result object.

        Args:
            method: Bridge method name.
            params: JSON-serialisable parameters.
            cwd: Working directory for the Node process. Rename passes the
                repository root so a ``tsconfig.json`` and relative globs resolve
                against it.

        Returns:
            The ``result`` object of a successful response.

        Raises:
            ValueError: The bridge rejected the parameters or the operation
                (JSON-RPC ``-32602`` / ``-32000``).
            TypeScriptBridgeError: Node is missing, the process failed, the
                response is malformed, the request timed out, or the bridge
                reported an internal error.
        """
        if not method:
            raise ValueError("method must not be empty")
        request_id = next(self._ids)
        payload = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        try:
            completed = subprocess.run(
                self._command_line(),
                input=payload,
                cwd=None if cwd is None else str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise TypeScriptBridgeError(f"{_NODE_HINT} ({exc})") from exc
        except subprocess.TimeoutExpired as exc:
            raise TypeScriptBridgeError(
                f"TypeScript bridge timed out after {self._timeout:.0f}s "
                f"while running {method!r}"
            ) from exc

        if completed.returncode != 0:
            detail = _truncate(
                (completed.stderr or completed.stdout).strip()
                or f"exit code {completed.returncode}"
            )
            raise TypeScriptBridgeError(
                f"TypeScript bridge failed for {method!r}: {detail}"
                f"{_install_hint(detail)}"
            )
        return self._parse_response(completed.stdout, request_id, method)

    def _parse_response(
        self, stdout: str, request_id: int, method: str
    ) -> dict[str, object]:
        """Validate the one-line JSON-RPC envelope and unwrap its result."""
        lines = [line for line in stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise TypeScriptBridgeError(
                f"TypeScript bridge returned {len(lines)} response line(s) for "
                f"{method!r}; expected exactly one"
            )
        try:
            response = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise TypeScriptBridgeError(
                f"TypeScript bridge returned invalid JSON for {method!r}: "
                f"{_truncate(lines[0])}"
            ) from exc
        if not isinstance(response, dict):
            raise TypeScriptBridgeError("TypeScript bridge response must be a JSON object")
        envelope = cast("dict[str, object]", response)
        if envelope.get("jsonrpc") != "2.0":
            raise TypeScriptBridgeError("TypeScript bridge response is not JSON-RPC 2.0")
        if envelope.get("id") != request_id:
            raise TypeScriptBridgeError(
                f"TypeScript bridge answered request {envelope.get('id')!r}, "
                f"expected {request_id}"
            )
        error = envelope.get("error")
        if error is not None:
            if not isinstance(error, dict):
                raise TypeScriptBridgeError("TypeScript bridge error must be an object")
            failure = cast("dict[str, object]", error)
            code = failure.get("code")
            message = failure.get("message")
            text = message if isinstance(message, str) else "unknown bridge error"
            if code in (_INVALID_PARAMS, _APPLICATION_ERROR):
                raise ValueError(text)
            raise TypeScriptBridgeError(f"TypeScript bridge error {code!r}: {text}")
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise TypeScriptBridgeError(
                f"TypeScript bridge response for {method!r} has no object result"
            )
        return cast("dict[str, object]", result)

    def _command_line(self) -> list[str]:
        """Resolve ``[node, script]`` once, or return the injected command."""
        if self._resolved_command is not None:
            return self._resolved_command
        if self._command is not None:
            self._resolved_command = self._command
        else:
            if not self._script.is_file():
                raise TypeScriptBridgeError(
                    f"TypeScript bridge script not found at {self._script}; "
                    "reinstall moray-surgeon or set MORAY_TS_BRIDGE"
                )
            self._resolved_command = [_resolve_node_executable(), str(self._script)]
        return self._resolved_command


class TypeScriptBackend(LanguageBackend):
    """TypeScript atoms backed by ``ts-morph`` in a Node.js subprocess.

    ``rename_symbol`` is the project-aware atom: it loads the repository (the
    ``tsconfig.json`` when present, otherwise a ``**/*.ts``/``**/*.tsx`` glob),
    resolves the definition with the LanguageService, collects every reference,
    deduplicates the overlapping symbol/alias groups and returns the rewritten
    text of each changed file. This process writes those files atomically after
    validating every path again, so a bridge bug cannot escape the repository.
    """

    language = "typescript"
    suffixes = (".ts", ".tsx")

    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        script: Path | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        client: TypeScriptBridgeClient | None = None,
    ) -> None:
        """Build the backend without touching Node (registration is import-time).

        Args:
            command: Injected command line, used by tests and by hosts with a
                non-standard Node launcher.
            script: Bridge entry point override.
            timeout: Per-request limit in seconds.
            client: Fully built client; wins over the other arguments.
        """
        self._client = client or TypeScriptBridgeClient(
            command=command, script=script, timeout=timeout
        )

    def parse(self, source: str) -> TsModule:
        """Parse with the TypeScript compiler and return its diagnostics."""
        result = self._client.request(
            "parse", {"source": source, "fileName": "input.ts"}
        )
        return TsModule(
            source=source,
            statements=_require_int(result, "statements"),
            diagnostics=_read_diagnostics(result),
        )

    def verify_syntax(self, source: str) -> bool:
        """Report whether ``source`` parses; syntax errors are ``False``.

        Infrastructure failures (no Node, missing ``ts-morph``) raise
        :class:`TypeScriptBridgeError` instead of pretending the source is
        broken.
        """
        result = self._client.request(
            "verify", {"source": source, "fileName": "input.ts"}
        )
        return _require_bool(result, "ok")

    def rename_symbol(
        self,
        repo_root: Path,
        target_file: str,
        old_name: str,
        new_name: str,
        indexer: Indexer | None = None,
    ) -> EditResult:
        """Rename every project reference to the symbol defined in ``target_file``.

        Args:
            repo_root: Repository root; the bridge runs with this cwd.
            target_file: Repository-relative path of the file holding the
                definition.
            old_name: Current identifier.
            new_name: Replacement identifier.
            indexer: Unused — the Python SQLite index only knows Python symbols.

        Returns:
            The target file's rewritten source and the number of renamed
            occurrences. Every changed file is already written atomically.

        Raises:
            ValueError: The identifier is invalid, names are equal, or the
                bridge found no definition/references.
            PathTraversalError: The target or a bridge-returned path escapes the
                repository.
            FileNotFoundError: The target does not exist.
            TypeScriptBridgeError: The Node bridge failed.
        """
        _validate_identifier(old_name, "old_name")
        _validate_identifier(new_name, "new_name")
        if old_name == new_name:
            raise ValueError("old_name and new_name must differ")
        root = repo_root.resolve()
        target = _resolve_repo_file(root, target_file)
        result = self._client.request(
            "rename",
            {
                "repoRoot": str(root),
                "targetFile": target.relative_to(root).as_posix(),
                "oldName": old_name,
                "newName": new_name,
            },
            cwd=root,
        )
        edits = _read_file_edits(result)
        changes = _require_int(result, "changes")
        # Validate every returned path before writing a single byte.
        resolved = [(_resolve_repo_file(root, relative), text) for relative, text in edits]
        target_text = next((text for path, text in resolved if path == target), None)
        if target_text is None:
            raise TypeScriptBridgeError(
                f"the bridge did not return a rewrite for {target_file!r}"
            )
        # Imported here: concurrency imports this package, so a module-level
        # import would close the cycle.
        from moray_surgeon.engine.concurrency import atomic_write_text

        for path, text in resolved:
            atomic_write_text(path, text)
        return EditResult(new_source=target_text, changes=changes)

    def extract_function(
        self, source: str, function_name: str, start_line: int, end_line: int
    ) -> EditResult:
        """Move top-level statements inside the line range into a new function.

        Raises:
            ValueError: The name is invalid or the range is reversed/empty.
            TypeScriptBridgeError: The Node bridge failed.
        """
        _validate_identifier(function_name, "function_name")
        _validate_line_range(start_line, end_line)
        return _edit_result(
            self._client.request(
                "extract_function",
                {
                    "source": source,
                    "functionName": function_name,
                    "startLine": start_line,
                    "endLine": end_line,
                },
            )
        )

    def extract_variable(
        self,
        source: str,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
        variable_name: str,
    ) -> EditResult:
        """Extract the expression at ``[start, end)`` into a ``const``.

        Columns are 0-based UTF-16 code units, matching the TypeScript compiler.
        """
        _validate_identifier(variable_name, "variable_name")
        _validate_expression_range(
            start_line, start_column, end_line, end_column
        )
        return _edit_result(
            self._client.request(
                "extract_variable",
                {
                    "source": source,
                    "variableName": variable_name,
                    "startLine": start_line,
                    "startColumn": start_column,
                    "endLine": end_line,
                    "endColumn": end_column,
                },
            )
        )

    def replace_call(self, source: str, old_call: str, new_call: str) -> EditResult:
        """Replace every call whose callee renders as ``old_call``."""
        old = old_call.strip()
        new = new_call.strip()
        if not old or not new:
            raise ValueError("old_call and new_call must not be empty")
        return _edit_result(
            self._client.request(
                "replace_call",
                {"source": source, "oldCall": old, "newCall": new},
            )
        )

    def add_import(
        self,
        source: str,
        module: str,
        name: str | None = None,
        alias: str | None = None,
    ) -> EditResult:
        """Add ``import { name } from "module"`` (or a side-effect import)."""
        if not module.strip():
            raise ValueError("module must not be empty")
        if name is None and alias is not None:
            raise ValueError("alias requires name")
        return _edit_result(
            self._client.request(
                "add_import",
                {"source": source, "module": module, "name": name, "alias": alias},
            )
        )

    def remove_import(
        self, source: str, module: str, name: str | None = None
    ) -> EditResult:
        """Remove imports of ``module``; ``name`` narrows it to one named import."""
        if not module.strip():
            raise ValueError("module must not be empty")
        return _edit_result(
            self._client.request(
                "remove_import",
                {"source": source, "module": module, "name": name},
            )
        )

    def inline_variable(
        self, source: str, variable_name: str, repo_root: Path, target_file: Path
    ) -> EditResult:
        """Not implemented for TypeScript."""
        self._unsupported("inline_variable")

    def inline_function(
        self, source: str, function_name: str, repo_root: Path, target_path: Path
    ) -> EditResult:
        """Not implemented for TypeScript."""
        self._unsupported("inline_function")

    def remove_dead_code(
        self, source: str, symbol_name: str, repo_root: Path, target_path: Path
    ) -> EditResult:
        """Not implemented for TypeScript."""
        self._unsupported("remove_dead_code")

    def split_class(
        self,
        source: str,
        class_name: str,
        methods_to_move: Sequence[str],
        new_class_name: str,
    ) -> EditResult:
        """Not implemented for TypeScript."""
        self._unsupported("split_class")

    def add_type_hints(
        self, source: str, function_name: str | None = None
    ) -> EditResult:
        """Not implemented for TypeScript: the syntax has no ``-> None`` analogue."""
        self._unsupported("add_type_hints")

    def convert_to_match_case(
        self, source: str, start_line: int, end_line: int
    ) -> EditResult:
        """Not implemented for TypeScript: ``match`` is a Python statement."""
        self._unsupported("convert_to_match_case")

    def move_symbol(
        self,
        repo_root: Path,
        source_path: Path,
        target_path: Path,
        symbol_name: str,
        indexer: Indexer | None = None,
    ) -> EditResult:
        """Not implemented for TypeScript."""
        self._unsupported("move_symbol")

    def _unsupported(self, action: str) -> NoReturn:
        """Fail loudly for an atom this backend intentionally does not port."""
        raise UnsupportedOperationError(
            f"the typescript backend does not implement {action!r}; "
            "no file was modified"
        )


def _resolve_node_executable() -> str:
    """Find the Node executable, honouring ``MORAY_NODE``."""
    override = os.environ.get(_NODE_ENV, "").strip()
    if override:
        candidate = shutil.which(override)
        if candidate is None and Path(override).is_file():
            candidate = override
        if candidate is None:
            raise TypeScriptBridgeError(
                f"{_NODE_ENV}={override!r} is not an executable Node.js binary"
            )
        return candidate
    for name in ("node", "nodejs"):
        found = shutil.which(name)
        if found is not None:
            return found
    raise TypeScriptBridgeError(_NODE_HINT)


def _install_hint(detail: str) -> str:
    """Append the setup command when Node could not load ``ts-morph``."""
    lowered = detail.lower()
    if "cannot find package" in lowered or "err_module_not_found" in lowered:
        return f" ({_SETUP_HINT})"
    return ""


def _truncate(text: str) -> str:
    """Bound bridge diagnostics that reach an exception message."""
    if len(text) <= _ERROR_DETAIL_LIMIT:
        return text
    return text[:_ERROR_DETAIL_LIMIT] + "..."


def _validate_identifier(value: str, field: str) -> None:
    """Reject names the TypeScript parser could not accept."""
    if not _TS_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field} must be a TypeScript identifier, got {value!r}")


def _validate_line_range(start_line: int, end_line: int) -> None:
    """Reject a line range the bridge would have to guess about."""
    if start_line < 1 or end_line < 1:
        raise ValueError("line numbers are 1-based and must be positive")
    if start_line > end_line:
        raise ValueError("start_line must be <= end_line")


def _validate_expression_range(
    start_line: int, start_column: int, end_line: int, end_column: int
) -> None:
    """Reject empty, reversed or negative expression coordinates."""
    if start_line < 1 or end_line < 1:
        raise ValueError("line numbers are 1-based and must be positive")
    if start_column < 0 or end_column < 0:
        raise ValueError("columns are 0-based and must not be negative")
    if (start_line, start_column) >= (end_line, end_column):
        raise ValueError("the expression range must not be empty or reversed")


def _resolve_repo_file(repo_root: Path, relative: str) -> Path:
    """Resolve a repository-relative path and prove it stays inside the repo."""
    target = (repo_root / relative).resolve()
    if not target.is_relative_to(repo_root):
        raise PathTraversalError(
            f"path {relative!r} resolves outside the repository"
        )
    if not target.is_file():
        raise FileNotFoundError(
            f"path {relative!r} does not exist or is not a regular file"
        )
    return target


def _read_file_edits(result: dict[str, object]) -> list[tuple[str, str]]:
    """Extract ``[(relative_path, text)]`` from a rename result.

    Newlines are normalised to ``\\n``. The bridge preserves the file's own line
    endings (ts-morph reads bytes), but the engine's canonical in-memory form is
    LF and the writer applies the platform convention once; writing CRLF text
    through a text-mode file would double the carriage returns on Windows.
    """
    raw = result.get("files")
    if not isinstance(raw, list):
        raise TypeScriptBridgeError("bridge rename result has no 'files' list")
    edits: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeScriptBridgeError("bridge file edit must be an object")
        entry = cast("dict[str, object]", item)
        path = entry.get("path")
        text = entry.get("text")
        if not isinstance(path, str) or not path:
            raise TypeScriptBridgeError("bridge file edit has no path")
        if not isinstance(text, str):
            raise TypeScriptBridgeError("bridge file edit has no text")
        edits.append((path, _normalize_newlines(text)))
    if not edits:
        raise TypeScriptBridgeError("bridge rename returned no file edits")
    return edits


def _read_diagnostics(result: dict[str, object]) -> tuple[TsDiagnostic, ...]:
    """Extract the diagnostics list from a parse result."""
    raw = result.get("diagnostics")
    if not isinstance(raw, list):
        raise TypeScriptBridgeError("bridge parse result has no 'diagnostics' list")
    diagnostics: list[TsDiagnostic] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeScriptBridgeError("bridge diagnostic must be an object")
        entry = cast("dict[str, object]", item)
        diagnostics.append(
            TsDiagnostic(
                code=_require_int(entry, "code"),
                message=_require_str(entry, "message"),
                line=_require_int(entry, "line"),
                column=_require_int(entry, "column"),
            )
        )
    return tuple(diagnostics)


def _edit_result(result: dict[str, object]) -> EditResult:
    """Build an :class:`EditResult` from a source-in/source-out bridge result."""
    return EditResult(
        new_source=_normalize_newlines(_require_str(result, "source")),
        changes=_require_int(result, "changes"),
    )


def _normalize_newlines(text: str) -> str:
    """Return ``text`` in the engine's canonical LF form.

    The writer translates ``\\n`` to the platform separator when it writes, so
    passing text that already contains ``\\r\\n`` would double the carriage
    returns on Windows. Normalising here also keeps the returned
    ``EditResult.new_source`` byte-stable when the dispatcher writes it again.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _require_str(mapping: dict[str, object], key: str) -> str:
    """Read a required string field, rejecting a malformed bridge response."""
    value = mapping.get(key)
    if not isinstance(value, str):
        raise TypeScriptBridgeError(f"bridge response field {key!r} must be a string")
    return value


def _require_int(mapping: dict[str, object], key: str) -> int:
    """Read a required integer field (booleans are not integers here)."""
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeScriptBridgeError(f"bridge response field {key!r} must be an integer")
    return value


def _require_bool(mapping: dict[str, object], key: str) -> bool:
    """Read a required boolean field, rejecting a malformed bridge response."""
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise TypeScriptBridgeError(f"bridge response field {key!r} must be a boolean")
    return value
