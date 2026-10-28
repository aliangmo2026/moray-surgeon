from __future__ import annotations

import secrets

import pytest

from moray_surgeon.exceptions import MorayError
from moray_surgeon.intent.providers import (
    AnthropicProvider,
    ChatRequest,
    OllamaProvider,
    OpenAIProvider,
    ProviderResponseError,
    build_provider,
)

_SCHEMA = {"type": "object", "properties": {"instructions": {"type": "array"}}}
_MESSAGES = [{"role": "user", "content": "rename it"}]


def _token() -> str:
    """A throwaway token generated per test; no credential literal in the suite."""
    return secrets.token_urlsafe(24)


def _request(provider, base_url: str = "https://api.example.test/v1") -> ChatRequest:
    return provider.build_request(
        base_url=base_url, model="test-model", messages=list(_MESSAGES), schema=_SCHEMA
    )


# --------------------------------------------------------------------------- #
# Ollama dialect
# --------------------------------------------------------------------------- #


def test_ollama_request_uses_the_native_chat_endpoint():
    request = _request(OllamaProvider())

    assert request.url == "https://api.example.test/v1/api/chat"
    assert request.headers == {}
    assert request.payload == {
        "model": "test-model",
        "messages": _MESSAGES,
        "stream": False,
        "format": _SCHEMA,
    }


def test_ollama_request_normalizes_a_trailing_slash():
    request = _request(OllamaProvider(), base_url="http://localhost:11434/")

    assert request.url == "http://localhost:11434/api/chat"


def test_ollama_parses_message_content():
    content = OllamaProvider().parse_content(
        {"message": {"role": "assistant", "content": '{"summary": "x"}'}}
    )

    assert content == '{"summary": "x"}'


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ("not a dict", "expected a JSON object"),
        ({}, "missing 'message' object"),
        ({"message": []}, "missing 'message' object"),
        ({"message": {"content": 1}}, "message.content"),
    ],
)
def test_ollama_rejects_malformed_responses(data, match):
    with pytest.raises(ProviderResponseError, match=match):
        OllamaProvider().parse_content(data)


# --------------------------------------------------------------------------- #
# OpenAI-compatible dialect
# --------------------------------------------------------------------------- #


def test_openai_request_uses_chat_completions_with_bearer_auth():
    token = _token()

    request = _request(OpenAIProvider(api_key=token))

    assert request.url == "https://api.example.test/v1/chat/completions"
    assert request.headers == {"Authorization": f"Bearer {token}"}
    assert request.payload["model"] == "test-model"
    assert request.payload["messages"] == _MESSAGES
    assert request.payload["temperature"] == 0.0
    assert request.payload["response_format"] == {"type": "json_object"}


def test_openai_request_normalizes_a_trailing_slash():
    request = _request(OpenAIProvider(api_key=_token()), base_url="https://api.x/v1///")

    assert request.url == "https://api.x/v1/chat/completions"


def test_openai_request_without_key_sends_no_authorization_header():
    request = _request(OpenAIProvider())

    assert request.headers == {}


def test_openai_request_never_sends_the_schema_as_a_grammar():
    request = _request(OpenAIProvider(api_key=_token()))

    assert "format" not in request.payload
    assert "schema" not in request.payload
    assert set(request.payload) == {"model", "messages", "temperature", "response_format"}


def test_openai_parses_choices_content():
    content = OpenAIProvider().parse_content(
        {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": '{"summary": "x"}'}}
            ]
        }
    )

    assert content == '{"summary": "x"}'


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ("not a dict", "expected a JSON object"),
        ({}, "missing 'choices' array"),
        ({"choices": []}, "missing 'choices' array"),
        ({"choices": "nope"}, "missing 'choices' array"),
        ({"choices": [1]}, "choices\\[0\\]"),
        ({"choices": [{}]}, "choices\\[0\\]\\.message"),
        ({"choices": [{"message": {"content": None}}]}, "message\\.content"),
    ],
)
def test_openai_rejects_malformed_responses(data, match):
    with pytest.raises(ProviderResponseError, match=match):
        OpenAIProvider().parse_content(data)


def test_openai_repr_never_shows_the_key():
    token = _token()
    provider = OpenAIProvider(api_key=token)

    assert token not in repr(provider)
    assert "OpenAIProvider" in repr(provider)


def test_openai_mismatch_includes_the_raw_payload():
    data = {
        "choices": [{"message": {"content": None}}],
        "vendor_error": "quota exhausted",
    }

    with pytest.raises(ProviderResponseError) as excinfo:
        OpenAIProvider().parse_content(data)

    message = str(excinfo.value)
    assert "response format mismatch" in message
    assert "Raw data:" in message
    assert "quota exhausted" in message


def test_openai_mismatch_includes_scalar_payloads():
    with pytest.raises(ProviderResponseError) as excinfo:
        OpenAIProvider().parse_content("rate limited")

    assert 'Raw data: "rate limited"' in str(excinfo.value)


def test_provider_response_error_is_a_domain_error():
    assert issubclass(ProviderResponseError, MorayError)


# --------------------------------------------------------------------------- #
# Anthropic Messages dialect
# --------------------------------------------------------------------------- #


def test_anthropic_request_uses_v1_messages_when_base_has_no_version():
    request = _request(AnthropicProvider(api_key=_token()), base_url="https://api.example.test")

    assert request.url == "https://api.example.test/v1/messages"


def test_anthropic_request_does_not_double_v1():
    request = _request(
        AnthropicProvider(api_key=_token()), base_url="https://api.example.test/v1"
    )

    assert request.url == "https://api.example.test/v1/messages"


def test_anthropic_request_normalizes_a_trailing_slash_after_v1():
    request = _request(
        AnthropicProvider(api_key=_token()), base_url="https://api.example.test/v1/"
    )

    assert request.url == "https://api.example.test/v1/messages"


def test_anthropic_request_headers_and_payload():
    token = _token()

    request = _request(AnthropicProvider(api_key=token))

    assert request.headers == {
        "x-api-key": token,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    assert request.payload["model"] == "test-model"
    assert request.payload["max_tokens"] == 4096
    assert request.payload["temperature"] == 0.0
    assert request.payload["messages"] == _MESSAGES


def test_anthropic_extracts_the_system_prompt_and_filters_messages():
    messages = [
        {"role": "system", "content": "You compile plans."},
        {"role": "user", "content": "rename it"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again"},
    ]

    request = AnthropicProvider(api_key=_token()).build_request(
        base_url="https://api.example.test",
        model="test-model",
        messages=messages,
        schema=_SCHEMA,
    )

    assert request.payload["system"] == "You compile plans."
    assert request.payload["messages"] == [
        {"role": "user", "content": "rename it"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again"},
    ]
    assert all(message["role"] != "system" for message in request.payload["messages"])


def test_anthropic_joins_multiple_system_messages():
    messages = [
        {"role": "system", "content": "first"},
        {"role": "system", "content": "second"},
        {"role": "user", "content": "rename it"},
    ]

    request = AnthropicProvider().build_request(
        base_url="https://api.example.test",
        model="test-model",
        messages=messages,
        schema=_SCHEMA,
    )

    assert request.payload["system"] == "first\n\nsecond"


def test_anthropic_omits_system_when_absent():
    request = _request(AnthropicProvider(api_key=_token()))

    assert "system" not in request.payload


def test_anthropic_request_without_key_omits_x_api_key():
    request = _request(AnthropicProvider())

    assert "x-api-key" not in request.headers
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["content-type"] == "application/json"


def test_anthropic_request_never_sends_the_schema():
    request = _request(AnthropicProvider(api_key=_token()))

    assert "format" not in request.payload
    assert "schema" not in request.payload
    assert set(request.payload) == {"model", "max_tokens", "messages", "temperature"}


def test_anthropic_parses_content_text():
    content = AnthropicProvider().parse_content(
        {"content": [{"type": "text", "text": '{"summary": "x"}'}]}
    )

    assert content == '{"summary": "x"}'


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ("not a dict", "expected a JSON object"),
        ({}, "missing 'content' array"),
        ({"content": []}, "missing 'content' array"),
        ({"content": "nope"}, "missing 'content' array"),
        ({"content": [1]}, "missing 'text' content block"),
        (
            {"content": [{"type": "thinking", "thinking": "hm"}]},
            "missing 'text' content block",
        ),
        ({"content": [{"type": "text"}]}, "content\\[0\\]\\.text"),
        ({"content": [{"type": "text", "text": None}]}, "content\\[0\\]\\.text"),
        ({"content": [{"text": None}]}, "missing 'text' content block"),
    ],
)
def test_anthropic_rejects_malformed_responses(data, match):
    with pytest.raises(ProviderResponseError, match=match):
        AnthropicProvider().parse_content(data)


def test_anthropic_parses_the_text_block_after_a_thinking_block():
    content = AnthropicProvider().parse_content(
        {
            "content": [
                {"type": "thinking", "thinking": "weighing the rename"},
                {"type": "text", "text": '{"summary": "x"}'},
            ]
        }
    )

    assert content == '{"summary": "x"}'


def test_anthropic_uses_the_first_valid_text_block_after_malformed_ones():
    content = AnthropicProvider().parse_content(
        {
            "content": [
                {"type": "text"},
                {"type": "thinking", "thinking": "still thinking"},
                {"type": "text", "text": '{"summary": "x"}'},
            ]
        }
    )

    assert content == '{"summary": "x"}'


def test_anthropic_rejects_a_response_without_a_text_block():
    data = {"content": [{"type": "thinking", "thinking": "still thinking"}]}

    with pytest.raises(ProviderResponseError) as excinfo:
        AnthropicProvider().parse_content(data)

    message = str(excinfo.value)
    assert "missing 'text' content block" in message
    assert "Raw data:" in message
    assert "still thinking" in message


def test_anthropic_repr_never_shows_the_key():
    token = _token()
    provider = AnthropicProvider(api_key=token)

    assert token not in repr(provider)
    assert "AnthropicProvider" in repr(provider)


def test_anthropic_mismatch_includes_the_raw_payload():
    data = {
        "content": [{"type": "thinking", "thinking": "hmm"}],
        "vendor_error": "upstream rejected the request",
    }

    with pytest.raises(ProviderResponseError) as excinfo:
        AnthropicProvider().parse_content(data)

    message = str(excinfo.value)
    assert "response format mismatch" in message
    assert "missing 'text' content block" in message
    assert "Raw data:" in message
    assert "upstream rejected the request" in message


def test_mismatch_raw_data_is_truncated_to_500_characters():
    data = {"content": [], "padding": "x" * 1000}

    with pytest.raises(ProviderResponseError) as excinfo:
        AnthropicProvider().parse_content(data)

    raw = str(excinfo.value).split("Raw data: ", 1)[1]
    assert len(raw) == 500
    assert "x" * 400 in raw


def test_mismatch_raw_data_keeps_non_ascii_readable():
    data = {"content": [{"text": None}], "error": "配额不足"}

    with pytest.raises(ProviderResponseError, match="配额不足"):
        AnthropicProvider().parse_content(data)


def test_mismatch_raw_data_falls_back_to_repr_when_not_json_serializable():
    with pytest.raises(ProviderResponseError) as excinfo:
        AnthropicProvider().parse_content({1, 2})

    assert "Raw data:" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("api_type", "expected"),
    [
        ("ollama", OllamaProvider),
        ("OLLAMA", OllamaProvider),
        ("openai", OpenAIProvider),
        ("  OpenAI  ", OpenAIProvider),
        ("anthropic", AnthropicProvider),
        ("  ANTHROPIC  ", AnthropicProvider),
    ],
)
def test_build_provider_selects_the_dialect(api_type, expected):
    assert isinstance(build_provider(api_type), expected)


def test_build_provider_passes_the_key_only_to_openai():
    token = _token()

    provider = build_provider("openai", token)

    assert isinstance(provider, OpenAIProvider)
    assert provider.api_key == token


def test_build_provider_passes_the_key_to_anthropic():
    token = _token()

    provider = build_provider("anthropic", token)

    assert isinstance(provider, AnthropicProvider)
    assert provider.api_key == token


def test_build_provider_treats_an_empty_key_as_absent():
    provider = build_provider("openai", "   ")

    assert isinstance(provider, OpenAIProvider)
    assert provider.api_key is None


def test_build_provider_strips_surrounding_whitespace_from_the_key():
    token = _token()

    provider = build_provider("openai", f"  {token}\n")

    assert isinstance(provider, OpenAIProvider)
    assert provider.api_key == token


def test_chat_request_repr_never_shows_headers():
    token = _token()

    request = _request(OpenAIProvider(api_key=token))

    assert token not in repr(request)
    assert request.url in repr(request)


def test_build_provider_rejects_unknown_types():
    with pytest.raises(ValueError, match="unsupported api_type"):
        build_provider("gemini")
