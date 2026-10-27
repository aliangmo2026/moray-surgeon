from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from rich.console import Console

from moray_surgeon.schema import (
    AddTypeHints,
    CircularSchemaError,
    ExtractFunction,
    Instruction,
    MorayError,
    Noop,
    RefactorPlan,
    RenameSymbol,
    ReplaceCall,
    SchemaInlineError,
)
from moray_surgeon.schema.instructions import _assert_no_ref_keys, _inline_document

console = Console()

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "refactor_plan.schema.json"

INSTRUCTION_ADAPTER = TypeAdapter(Instruction)

ACTION_SAMPLES: list[tuple[dict[str, Any], type[Any]]] = [
    (
        {
            "action": "rename_symbol",
            "target_file": "pkg/mod.py",
            "old_name": "foo",
            "new_name": "bar",
        },
        RenameSymbol,
    ),
    (
        {
            "action": "extract_function",
            "target_file": "pkg/mod.py",
            "function_name": "helper",
            "start_line": 10,
            "end_line": 20,
        },
        ExtractFunction,
    ),
    ({"action": "add_type_hints", "target_file": "pkg/mod.py"}, AddTypeHints),
    (
        {
            "action": "replace_call",
            "target_file": "pkg/mod.py",
            "old_call": "f(x)",
            "new_call": "g(x)",
        },
        ReplaceCall,
    ),
    ({"action": "noop"}, Noop),
]

# REGENERATE: REGEN_SNAPSHOT=1 pytest tests/test_schema.py
def _generate_fixture() -> None:
    """Write the schema snapshot only when REGEN_SNAPSHOT is requested."""
    if not os.environ.get("REGEN_SNAPSHOT"):
        return
    _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(RefactorPlan.export_schema(), ensure_ascii=False, indent=2, sort_keys=True)
    _FIXTURE_PATH.write_text(payload + "\n", encoding="utf-8")
    console.print("snapshot regenerated")


def _walk(node: object) -> tuple[int, list[str]]:
    """Independently walk the schema; count nodes and collect $ref/$defs keys.

    This deliberately duplicates the production-side assertion so a bug in
    _assert_no_ref_keys cannot mask an uninlined schema in the export.
    """
    count = 1
    violations: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("$ref", "$defs"):
                violations.append(str(key))
            sub_count, sub_violations = _walk(value)
            count += sub_count
            violations.extend(sub_violations)
    elif isinstance(node, list):
        for item in node:
            sub_count, sub_violations = _walk(item)
            count += sub_count
            violations.extend(sub_violations)
    return count, violations


@pytest.mark.parametrize(("payload", "expected_type"), ACTION_SAMPLES)
def test_each_action_parses_to_its_model(payload: dict[str, Any], expected_type: type[Any]) -> None:
    """Every action tag must route to its dedicated model via the discriminator."""
    instruction = INSTRUCTION_ADAPTER.validate_python(payload)
    assert isinstance(instruction, expected_type)


@pytest.mark.parametrize(("payload", "_expected_type"), ACTION_SAMPLES)
def test_round_trip_is_stable(payload: dict[str, Any], _expected_type: type[Any]) -> None:
    """serialize -> deserialize must reproduce an equal instruction."""
    first = INSTRUCTION_ADAPTER.validate_python(payload)
    second = INSTRUCTION_ADAPTER.validate_python(first.model_dump())
    assert first == second


def test_reasoning_defaults_to_empty() -> None:
    """reasoning is audit-only and must default to '' when omitted."""
    instruction = INSTRUCTION_ADAPTER.validate_python({"action": "noop"})
    assert instruction.reasoning == ""


def test_noop_message_default() -> None:
    """noop must carry the documented default message."""
    assert Noop().message == "Nothing to do."


def test_add_type_hints_function_name_is_optional() -> None:
    """Omitting function_name means whole-file scope and must parse to None."""
    instruction = INSTRUCTION_ADAPTER.validate_python(
        {"action": "add_type_hints", "target_file": "pkg/mod.py"}
    )
    assert isinstance(instruction, AddTypeHints)
    assert instruction.function_name is None


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"action": "explode", "target_file": "a.py"}, "Input tag"),
        (
            {
                "action": "rename_symbol",
                "target_file": "../evil.py",
                "old_name": "a",
                "new_name": "b",
            },
            "not contain",
        ),
        (
            {
                "action": "rename_symbol",
                "target_file": "/abs/path.py",
                "old_name": "a",
                "new_name": "b",
            },
            "relative",
        ),
        (
            {"action": "rename_symbol", "target_file": "a.txt", "old_name": "a", "new_name": "b"},
            r"end with \.py",
        ),
        ({"action": "noop", "unexpected": 1}, "unexpected"),
        ({"action": "rename_symbol", "target_file": "a.py", "new_name": "b"}, "old_name"),
        (
            {
                "action": "extract_function",
                "target_file": "a.py",
                "function_name": "f",
                "start_line": 0,
                "end_line": 2,
            },
            "greater than or equal to 1",
        ),
    ],
)
def test_invalid_payload_raises_validation_error(payload: dict[str, Any], match: str) -> None:
    """Bad actions, paths, extra keys, missing fields and bad ranges must all fail."""
    with pytest.raises(ValidationError, match=match):
        INSTRUCTION_ADAPTER.validate_python(payload)


def test_frozen_instance_rejects_assignment() -> None:
    """Frozen models must reject attribute writes after construction."""
    instance = RenameSymbol(target_file="a.py", old_name="A", new_name="B")
    # pydantic v2 frozen models raise ValidationError (frozen_instance) rather
    # than the stdlib FrozenInstanceError; accept any of the three.
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        instance.new_name = "C"  # type: ignore[misc]


def test_plan_defaults() -> None:
    """RefactorPlan must default to an empty instruction list and empty summary."""
    plan = RefactorPlan()
    assert plan.instructions == []
    assert plan.summary == ""


def test_plan_round_trip_and_extra_forbidden() -> None:
    """A populated plan must round-trip; extra plan-level keys must be rejected."""
    payload = {
        "instructions": [ACTION_SAMPLES[0][0], ACTION_SAMPLES[4][0]],
        "summary": "rename foo and stop",
    }
    plan = RefactorPlan.model_validate(payload)
    assert RefactorPlan.model_validate(plan.model_dump()) == plan
    with pytest.raises(ValidationError, match="unexpected_plan_key"):
        RefactorPlan.model_validate({**payload, "unexpected_plan_key": 1})


def test_exception_hierarchy() -> None:
    """Both schema errors must sit under the MorayError base."""
    assert issubclass(CircularSchemaError, MorayError)
    assert issubclass(SchemaInlineError, MorayError)


def test_circular_ref_raises() -> None:
    """A hand-built cyclic $defs graph must raise, never silently truncate."""
    cyclic = {
        "properties": {"root": {"$ref": "#/$defs/A"}},
        "$defs": {
            "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
            "B": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}},
        },
    }
    with pytest.raises(CircularSchemaError):
        _inline_document(cyclic)


def test_dangling_ref_raises() -> None:
    """A $ref pointing outside $defs must be reported, not dropped."""
    with pytest.raises(SchemaInlineError, match="dangling"):
        _inline_document({"properties": {"x": {"$ref": "#/$defs/NoSuch"}}})


def test_export_schema_propagates_inline_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """export_schema (public API) must surface inline failures, not swallow them."""
    cyclic = {
        "properties": {"root": {"$ref": "#/$defs/A"}},
        "$defs": {
            "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
            "B": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}},
        },
    }
    monkeypatch.setattr(RefactorPlan, "model_json_schema", staticmethod(lambda: cyclic))
    with pytest.raises(CircularSchemaError):
        RefactorPlan.export_schema()


def test_assert_no_ref_keys_detects_survivors() -> None:
    """The production walker itself must raise when handed an uninlined schema."""
    with pytest.raises(SchemaInlineError):
        _assert_no_ref_keys({"oneOf": [{"$ref": "#/$defs/A"}]})


def test_export_schema_shape() -> None:
    """Export must keep oneOf + discriminator and drop $defs entirely."""
    schema = RefactorPlan.export_schema()
    items = schema["properties"]["instructions"]["items"]
    assert "oneOf" in items
    assert "discriminator" in items
    assert "$defs" not in schema


def test_discriminator_mapping_values_are_inlined_dicts() -> None:
    """Every mapping value must be the inlined branch dict, never a $ref string."""
    items = RefactorPlan.export_schema()["properties"]["instructions"]["items"]
    mapping = items["discriminator"]["mapping"]
    expected_tags = {"rename_symbol", "extract_function", "add_type_hints", "replace_call", "noop"}
    assert set(mapping) == expected_tags
    assert all(isinstance(branch, dict) for branch in mapping.values())
    assert all("$ref" not in branch for branch in mapping.values())


def test_export_schema_has_no_ref_keys_at_any_depth() -> None:
    """An independent walker must find zero $ref/$defs keys below the root."""
    _count, violations = _walk(RefactorPlan.export_schema())
    assert violations == []


def test_export_schema_walker_visits_enough_nodes() -> None:
    """Node count >= 50 proves the walker really recursed into every branch."""
    count, violations = _walk(RefactorPlan.export_schema())
    assert count >= 50
    assert violations == []
    # Also exercise the production walker on the same tree.
    assert _assert_no_ref_keys(RefactorPlan.export_schema()) >= 50


def test_export_schema_is_json_serializable() -> None:
    """Ollama consumes the export through JSON, so serialization must never fail."""
    json.dumps(RefactorPlan.export_schema())


def test_export_schema_matches_snapshot() -> None:
    """The export must stay byte-for-byte dict-identical to the committed fixture."""
    _generate_fixture()
    assert _FIXTURE_PATH.exists(), (
        "schema snapshot missing; regenerate with: REGEN_SNAPSHOT=1 pytest tests/test_schema.py"
    )
    snapshot = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert RefactorPlan.export_schema() == snapshot
