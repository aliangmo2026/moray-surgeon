from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from moray_surgeon.intent import IntentParser, IntentParserError
from moray_surgeon.intent.parser import _extract_json
from moray_surgeon.schema import RefactorPlan


def _plan_payload() -> dict[str, object]:
    return {
        "instructions": [
            {
                "action": "rename_symbol",
                "target_file": "app.py",
                "old_name": "old_name",
                "new_name": "new_name",
                "reasoning": "rename for clarity",
            }
        ],
        "summary": "rename old_name",
    }


def _response(content: str, status_code: int = 200) -> httpx.Response:
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return httpx.Response(
        status_code,
        json={"message": {"role": "assistant", "content": content}},
        request=request,
    )


class _FakeOllama:
    """Replays canned ``message.content`` values and records every request."""

    def __init__(self, contents: list[str], status_code: int = 200) -> None:
        self._contents = list(contents)
        self._status_code = status_code
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append((url, kwargs))
        return _response(self._contents.pop(0), self._status_code)


def _install(monkeypatch, fake: _FakeOllama) -> None:
    monkeypatch.setattr(httpx.Client, "post", fake)


def test_valid_json_returns_plan(monkeypatch):
    fake = _FakeOllama([json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    plan = IntentParser().parse("rename old_name to new_name", Path("repo"))

    assert isinstance(plan, RefactorPlan)
    assert plan.summary == "rename old_name"
    assert len(plan.instructions) == 1
    instruction = plan.instructions[0]
    assert instruction.action == "rename_symbol"
    assert instruction.old_name == "old_name"
    assert instruction.new_name == "new_name"
    assert len(fake.calls) == 1


def test_markdown_wrapped_json_is_extracted(monkeypatch):
    content = (
        "Sure, here is the plan:\n"
        "```json\n"
        f"{json.dumps(_plan_payload())}\n"
        "```\n"
        "Let me know if you need anything else."
    )
    _install(monkeypatch, _FakeOllama([content]))

    plan = IntentParser().parse("rename it", Path("repo"))

    assert plan.instructions[0].new_name == "new_name"


def test_invalid_json_is_retried_then_fails(monkeypatch):
    fake = _FakeOllama(["not json at all", '{"instructions": "not-a-list"}'])
    _install(monkeypatch, fake)

    with pytest.raises(IntentParserError, match="2 attempts"):
        IntentParser().parse("rename it", Path("repo"))

    assert len(fake.calls) == 2


def test_request_payload_shape(monkeypatch):
    fake = _FakeOllama([json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    IntentParser(model="test-model", base_url="http://example.test:1234").parse(
        "extract the parser", Path("repo"), extra_context="symbols: parse, load"
    )

    url, kwargs = fake.calls[0]
    assert url == "http://example.test:1234/api/chat"
    payload = kwargs["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "test-model"
    assert payload["stream"] is False

    schema = payload["format"]
    assert isinstance(schema, dict)
    assert schema == RefactorPlan.export_schema()
    assert "oneOf" in json.dumps(schema)
    assert "oneOf" in schema["properties"]["instructions"]["items"]

    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert json.dumps(RefactorPlan.export_schema(), ensure_ascii=False) in messages[0]["content"]
    assert "oneOf" in messages[0]["content"]
    assert "不许输出源代码" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "extract the parser" in messages[1]["content"]
    assert "symbols: parse, load" in messages[1]["content"]
    assert "Repository root: repo" in messages[1]["content"]


def test_schema_failure_then_success_retries_once(monkeypatch):
    invalid = json.dumps({"action": "unknown"})
    fake = _FakeOllama([invalid, json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    plan = IntentParser().parse("rename it", Path("repo"))

    assert plan.instructions[0].old_name == "old_name"
    assert len(fake.calls) == 2
    messages = fake.calls[1][1]["json"]["messages"]
    assert len(messages) == 4
    assert messages[2] == {"role": "assistant", "content": invalid}
    assert messages[3]["role"] == "user"
    assert "Validation error" in messages[3]["content"]


def test_http_error_is_not_retried(monkeypatch):
    fake = _FakeOllama(["ignored"], status_code=500)
    _install(monkeypatch, fake)

    with pytest.raises(IntentParserError, match="Ollama request failed"):
        IntentParser().parse("rename it", Path("repo"))

    assert len(fake.calls) == 1


def test_missing_message_content_raises(monkeypatch):
    def fake_post(self, url, **kwargs: object):
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"message": {}}, request=request)

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    with pytest.raises(IntentParserError, match=r"message\.content"):
        IntentParser().parse("rename it", Path("repo"))


def test_non_http_base_url_is_rejected():
    with pytest.raises(IntentParserError, match="http"):
        IntentParser(base_url="file:///tmp/ollama")


def test_extra_context_is_forwarded_verbatim(monkeypatch):
    fake = _FakeOllama([json.dumps(_plan_payload())])
    _install(monkeypatch, fake)
    context = (
        "Previous attempt failed verification.\n"
        "Code changes (git diff):\n-def old_name():\n+def new_name():\n\n"
        "Verification error:\nassertion failed at line 5"
    )

    IntentParser().parse("rename it", Path("repo"), extra_context=context)

    messages = fake.calls[0][1]["json"]["messages"]
    assert context in messages[1]["content"]
    assert "previous attempt" in messages[1]["content"].lower()
    assert "previous attempt" in messages[0]["content"].lower()


def test_extract_json_uses_outermost_braces():
    assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _extract_json('prefix {"a": {"b": 2}} suffix') == '{"a": {"b": 2}}'

    with pytest.raises(ValueError, match="JSON object"):
        _extract_json("no json here")
