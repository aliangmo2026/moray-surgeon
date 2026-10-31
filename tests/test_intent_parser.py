from __future__ import annotations

import json
import secrets
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
                "id": "step1",
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


def _cyclic_payload() -> str:
    return json.dumps(
        {
            "instructions": [
                {"id": "a", "action": "noop", "depends_on": ["b"]},
                {"id": "b", "action": "noop", "depends_on": ["a"]},
            ]
        }
    )


def test_system_prompt_documents_ids_and_dependencies(monkeypatch):
    fake = _FakeOllama([json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    IntentParser().parse("rename it", Path("repo"))

    system = fake.calls[0][1]["json"]["messages"][0]["content"]
    assert "depends_on" in system
    assert "Put ordering knowledge" in system
    assert "Never create a dependency cycle" in system
    assert "unique \"id\"" in system


def test_system_prompt_permits_and_urges_test_directory_edits(monkeypatch):
    """Test modules are legitimate targets when a refactor breaks them."""
    fake = _FakeOllama([json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    IntentParser().parse("rename it", Path("repo"))

    system = fake.calls[0][1]["json"]["messages"][0]["content"]
    assert "Tests are callers too" in system
    assert "under tests/" in system
    # Both actions the repair loop relies on are named explicitly.
    assert "rename_symbol" in system
    assert "replace_call" in system
    # The edits stay surgical: assertions' meaning must survive.
    assert "never change what the tests assert" in system


def test_cyclic_plan_is_repaired_in_the_next_round(monkeypatch):
    fake = _FakeOllama([_cyclic_payload(), json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    plan = IntentParser().parse("rename it", Path("repo"))

    assert plan.instructions[0].old_name == "old_name"
    assert len(fake.calls) == 2
    repair = fake.calls[1][1]["json"]["messages"][-1]["content"]
    assert "circular dependency" in repair


def test_persistent_cycle_fails_after_two_attempts(monkeypatch):
    fake = _FakeOllama([_cyclic_payload(), _cyclic_payload()])
    _install(monkeypatch, fake)

    with pytest.raises(IntentParserError, match="circular dependency"):
        IntentParser().parse("rename it", Path("repo"))

    assert len(fake.calls) == 2


# --------------------------------------------------------------------------- #
# OpenAI-compatible cloud APIs
# --------------------------------------------------------------------------- #


def _openai_response(content: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]},
        request=httpx.Request("POST", "https://api.example.test/v1/chat/completions"),
    )


class _FakeOpenAI:
    """Replays OpenAI-shaped ``choices`` payloads and records every request."""

    def __init__(self, contents: list[str], status_code: int = 200) -> None:
        self._contents = list(contents)
        self._status_code = status_code
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append((url, kwargs))
        return _openai_response(self._contents.pop(0), self._status_code)


def test_openai_request_uses_chat_completions_bearer_and_json_mode(monkeypatch):
    token = secrets.token_urlsafe(24)
    fake = _FakeOpenAI([json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    plan = IntentParser(
        model="cloud-model",
        base_url="https://api.example.test/v1/",
        api_type="openai",
        api_key=token,
    ).parse("rename it", Path("repo"))

    assert plan.instructions[0].new_name == "new_name"
    url, kwargs = fake.calls[0]
    assert url == "https://api.example.test/v1/chat/completions"
    assert kwargs["headers"] == {"Authorization": f"Bearer {token}"}

    payload = kwargs["json"]
    assert payload["model"] == "cloud-model"
    assert payload["temperature"] == 0.0
    assert payload["response_format"] == {"type": "json_object"}
    assert "format" not in payload
    assert "stream" not in payload

    # JSON mode has no grammar, so the schema must still travel in the prompt.
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert json.dumps(RefactorPlan.export_schema(), ensure_ascii=False) in messages[0]["content"]
    assert messages[1]["role"] == "user"


def test_openai_without_key_sends_no_authorization_header(monkeypatch):
    fake = _FakeOpenAI([json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    IntentParser(api_type="openai", base_url="http://localhost:8000/v1").parse(
        "rename it", Path("repo")
    )

    _url, kwargs = fake.calls[0]
    assert kwargs["headers"] == {}


def test_ollama_request_shape_is_unchanged(monkeypatch):
    fake = _FakeOllama([json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    IntentParser().parse("rename it", Path("repo"))

    url, kwargs = fake.calls[0]
    assert url == "http://localhost:11434/api/chat"
    payload = kwargs["json"]
    assert payload["format"] == RefactorPlan.export_schema()
    assert payload["stream"] is False
    assert "response_format" not in payload
    assert kwargs["headers"] == {}


def test_openai_validation_error_is_retried(monkeypatch):
    fake = _FakeOpenAI(["not json at all", json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    plan = IntentParser(api_type="openai", base_url="https://api.example.test/v1").parse(
        "rename it", Path("repo")
    )

    assert plan.summary == "rename old_name"
    assert len(fake.calls) == 2


def test_openai_malformed_response_is_reported(monkeypatch):
    def fake_post(self, url, **kwargs: object):
        return httpx.Response(200, json={"choices": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    with pytest.raises(IntentParserError, match="choices"):
        IntentParser(api_type="openai", base_url="https://api.example.test/v1").parse(
            "rename it", Path("repo")
        )


def test_non_json_http_body_is_reported(monkeypatch):
    def fake_post(self, url, **kwargs: object):
        request = httpx.Request("POST", url)
        return httpx.Response(200, content=b"<html>not json</html>", request=request)

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    with pytest.raises(IntentParserError, match="non-JSON response"):
        IntentParser().parse("rename it", Path("repo"))


def test_transport_error_names_the_endpoint_but_never_the_key(monkeypatch):
    token = secrets.token_urlsafe(24)

    def failing_post(self, url, **kwargs: object):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.Client, "post", failing_post)

    with pytest.raises(IntentParserError) as excinfo:
        IntentParser(
            api_type="openai", base_url="https://api.example.test/v1", api_key=token
        ).parse("rename it", Path("repo"))

    message = str(excinfo.value)
    assert "OpenAI request failed" in message
    assert "https://api.example.test/v1/chat/completions" in message
    assert token not in message


def test_unknown_api_type_is_rejected():
    with pytest.raises(IntentParserError, match="unsupported api_type"):
        IntentParser(api_type="gemini")


def test_base_url_without_a_host_is_rejected():
    with pytest.raises(IntentParserError, match="include a host"):
        IntentParser(base_url="http:///v1")


def test_base_url_with_embedded_credentials_is_rejected():
    token = secrets.token_urlsafe(16)

    with pytest.raises(IntentParserError, match="credentials"):
        IntentParser(base_url=f"https://user:{token}@api.example.test/v1")


def test_base_url_with_a_query_string_is_rejected():
    token = secrets.token_urlsafe(16)

    with pytest.raises(IntentParserError, match="query string"):
        IntentParser(base_url=f"https://api.example.test/v1?api-key={token}")


def test_base_url_with_a_fragment_is_rejected():
    with pytest.raises(IntentParserError, match="fragment"):
        IntentParser(base_url="https://api.example.test/v1#frag")


# --------------------------------------------------------------------------- #
# Anthropic Messages API (Claude, Xiaomi Token Plan, ...)
# --------------------------------------------------------------------------- #


def _anthropic_response(content: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": content}],
        },
        request=httpx.Request("POST", "https://api.example.test/v1/messages"),
    )


class _FakeAnthropic:
    """Replays Anthropic Messages payloads and records every request."""

    def __init__(self, contents: list[str], status_code: int = 200) -> None:
        self._contents = list(contents)
        self._status_code = status_code
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append((url, kwargs))
        return _anthropic_response(self._contents.pop(0), self._status_code)


def test_anthropic_request_extracts_system_and_parses_content(monkeypatch):
    token = secrets.token_urlsafe(24)
    fake = _FakeAnthropic([json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    plan = IntentParser(
        model="token-plan-model",
        base_url="https://api.example.test/v1",
        api_type="anthropic",
        api_key=token,
    ).parse("rename it", Path("repo"))

    assert plan.instructions[0].new_name == "new_name"
    url, kwargs = fake.calls[0]
    assert url == "https://api.example.test/v1/messages"
    headers = kwargs["headers"]
    assert headers["x-api-key"] == token
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"

    payload = kwargs["json"]
    assert payload["model"] == "token-plan-model"
    assert payload["max_tokens"] == 4096
    assert payload["temperature"] == 0.0
    assert "format" not in payload
    assert "schema" not in payload

    # The system prompt is a top-level field and still carries the schema.
    assert payload["system"].startswith("You are moray-surgeon's plan compiler.")
    assert json.dumps(RefactorPlan.export_schema(), ensure_ascii=False) in payload["system"]
    assert [message["role"] for message in payload["messages"]] == ["user"]


def test_anthropic_repair_round_keeps_the_system_field(monkeypatch):
    fake = _FakeAnthropic(["not json at all", json.dumps(_plan_payload())])
    _install(monkeypatch, fake)

    IntentParser(api_type="anthropic", base_url="https://api.example.test").parse(
        "rename it", Path("repo")
    )

    assert len(fake.calls) == 2
    payload = fake.calls[1][1]["json"]
    assert "system" in payload
    assert [message["role"] for message in payload["messages"]] == [
        "user",
        "assistant",
        "user",
    ]


def test_anthropic_parser_skips_reasoning_blocks(monkeypatch):
    """DeepSeek-class models prepend a thinking block before the text block."""

    def fake_post(self, url, **kwargs: object):
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "thinking", "thinking": "the intent asks for a rename"},
                    {"type": "text", "text": json.dumps(_plan_payload())},
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    plan = IntentParser(api_type="anthropic", base_url="https://api.example.test").parse(
        "rename it", Path("repo")
    )

    assert plan.instructions[0].new_name == "new_name"


def test_anthropic_malformed_response_is_reported(monkeypatch):
    def fake_post(self, url, **kwargs: object):
        return httpx.Response(200, json={"content": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    with pytest.raises(IntentParserError, match="content"):
        IntentParser(api_type="anthropic", base_url="https://api.example.test/v1").parse(
            "rename it", Path("repo")
        )
