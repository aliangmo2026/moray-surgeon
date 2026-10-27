from __future__ import annotations

import copy
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from moray_surgeon.schema.exceptions import CircularSchemaError, SchemaInlineError

_REF_PREFIX = "#/$defs/"


class _BaseInstruction(BaseModel):
    """Shared contract for every instruction: frozen, closed, auditable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning: str = ""


class _FileInstruction(_BaseInstruction):
    """An instruction bound to one target Python file inside the repo."""

    target_file: str

    @field_validator("target_file")
    @classmethod
    def _validate_target_file(cls, value: str) -> str:
        """Reject non-.py, absolute and traversal paths; return POSIX form.

        `Path.anchor` is checked instead of only `is_absolute()` because on
        Windows a rooted path like "/abs/x.py" is not "absolute" but must
        still be rejected.
        """
        path = Path(value)
        if path.suffix != ".py":
            raise ValueError("target_file must end with .py")
        if ".." in path.parts:
            raise ValueError("target_file must not contain '..'")
        if path.anchor:
            raise ValueError("target_file must be relative and drive-less")
        return path.as_posix()


class RenameSymbol(_FileInstruction):
    """Rename a symbol (function/class/variable) within one file."""

    action: Literal["rename_symbol"] = "rename_symbol"
    old_name: str
    new_name: str


class ExtractFunction(_FileInstruction):
    """Extract lines [start_line, end_line] into a new named function.

    NOTE: start_line <= end_line is enforced by the engine layer, not here.
    """

    action: Literal["extract_function"] = "extract_function"
    function_name: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class AddTypeHints(_FileInstruction):
    """Add type hints to one function, or to the whole file if function_name is None."""

    action: Literal["add_type_hints"] = "add_type_hints"
    function_name: str | None = None


class ReplaceCall(_FileInstruction):
    """Replace one call expression with another within one file."""

    action: Literal["replace_call"] = "replace_call"
    old_call: str
    new_call: str


class Noop(_BaseInstruction):
    """Explicit no-operation carrying a human-readable message."""

    action: Literal["noop"] = "noop"
    message: str = "Nothing to do."


Instruction = Annotated[
    RenameSymbol | ExtractFunction | AddTypeHints | ReplaceCall | Noop,
    Field(discriminator="action"),
]


class RefactorPlan(BaseModel):
    """An ordered batch of refactor instructions plus a human summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instructions: list[Instruction] = Field(default_factory=list)
    summary: str = ""

    @classmethod
    def export_schema(cls) -> dict[str, Any]:
        """Return an Ollama-safe JSON Schema: every $ref inlined, discriminator kept.

        Ollama's `format` converts JSON Schema to GBNF grammar and mishandles
        $ref/$defs, so the default pydantic output is normalized here into a
        self-contained oneOf tree before it ever reaches the model.
        """
        document = _inline_document(cls.model_json_schema())
        _assert_no_ref_keys(document)
        return document


def _expand_ref(
    name: str,
    defs: dict[str, Any],
    visiting: set[str],
    memo: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Expand one $defs entry into a fully inlined copy, memoized per name.

    `visiting` holds the names on the current expansion stack so a cycle is
    detected (and reported) instead of recursing forever.
    """
    cached = memo.get(name)
    if cached is not None:
        return copy.deepcopy(cached)
    if name in visiting:
        raise CircularSchemaError(f"circular $ref while expanding {name!r}")
    definition = defs.get(name)
    if definition is None:
        raise SchemaInlineError(f"dangling $ref target {name!r}")
    visiting.add(name)
    try:
        expanded = _inline_node(definition, defs, visiting, memo)
    finally:
        visiting.discard(name)
    if not isinstance(expanded, dict):
        raise SchemaInlineError(f"$ref target {name!r} must inline to an object schema")
    memo[name] = expanded
    # Deep copies keep later sibling merges from mutating the shared memo entry.
    return copy.deepcopy(expanded)


def _inline_discriminator(
    discriminator: dict[str, object],
    defs: dict[str, Any],
    visiting: set[str],
    memo: dict[str, dict[str, Any]],
) -> dict[str, object]:
    """Replace discriminator mapping pointers with the inlined branch schemas.

    Mapping values are "#/$defs/X" strings in pydantic output; after inlining
    those pointers would dangle, so each value becomes the expanded branch.
    """
    mapping = discriminator.get("mapping")
    if not isinstance(mapping, dict):
        return discriminator
    new_mapping: dict[str, object] = {}
    for tag, target in mapping.items():
        if isinstance(target, str) and target.startswith(_REF_PREFIX):
            new_mapping[tag] = _expand_ref(target[len(_REF_PREFIX) :], defs, visiting, memo)
        else:
            new_mapping[tag] = target
    return {**discriminator, "mapping": new_mapping}


def _inline_node(
    node: object,
    defs: dict[str, Any],
    visiting: set[str],
    memo: dict[str, dict[str, Any]],
) -> object:
    """Recursively inline every $ref inside an arbitrary JSON Schema fragment."""
    if isinstance(node, dict):
        new_node: dict[str, object] = {}
        for key, value in node.items():
            if key == "$ref":
                continue
            new_node[key] = _inline_node(value, defs, visiting, memo)
        ref = node.get("$ref")
        if ref is not None:
            if not isinstance(ref, str) or not ref.startswith(_REF_PREFIX):
                raise SchemaInlineError(f"unsupported $ref form: {ref!r}")
            # Sibling keywords next to $ref override the target (JSON Schema 2020-12).
            new_node = {**_expand_ref(ref[len(_REF_PREFIX) :], defs, visiting, memo), **new_node}
        discriminator = new_node.get("discriminator")
        if isinstance(discriminator, dict):
            new_node["discriminator"] = _inline_discriminator(discriminator, defs, visiting, memo)
        return new_node
    if isinstance(node, list):
        return [_inline_node(item, defs, visiting, memo) for item in node]
    return node


def _inline_document(document: dict[str, Any]) -> dict[str, Any]:
    """Inline every $ref of a JSON Schema document and drop the $defs section."""
    defs = document.get("$defs", {})
    if not isinstance(defs, dict):
        raise SchemaInlineError("$defs must be an object when present")
    root = {key: value for key, value in document.items() if key != "$defs"}
    return cast("dict[str, Any]", _inline_node(root, defs, set(), {}))


def _assert_no_ref_keys(node: object) -> int:
    """Raise SchemaInlineError if any "$ref"/"$defs" key survives; return node count.

    This is the production-side invariant check: an inlined schema handed to
    Ollama must be fully self-contained at every depth, not just at the top.
    """
    count = 1
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("$ref", "$defs"):
                raise SchemaInlineError(f"uninlined schema key survived: {key!r}")
            count += _assert_no_ref_keys(value)
    elif isinstance(node, list):
        for item in node:
            count += _assert_no_ref_keys(item)
    return count
