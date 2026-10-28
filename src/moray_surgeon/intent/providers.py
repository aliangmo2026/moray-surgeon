"""Transport dialects for chat models: one request shape per API family.

Ollama speaks its own JSON protocol and accepts the exported plan schema as a
decoding grammar (``format``), which is what keeps malformed JSON from ever
reaching the parser. OpenAI-compatible cloud APIs (DeepSeek, vLLM, LM Studio, ...)
use the ``/chat/completions`` shape, and providers such as Xiaomi's Token Plan or
Claude use the Anthropic Messages shape (``/v1/messages``). Neither cloud
protocol has a grammar field, so the constraint moves into the system prompt,
Pydantic validation and the parser's one repair round-trip.

Providers are frozen value objects that only build a request and read a response.
Retries, timeouts and plan validation stay in :class:`~moray_surgeon.intent.parser.IntentParser`,
so every dialect shares exactly one control flow.

A response that does not match its dialect raises ``ProviderResponseError`` with
the first 500 characters of the raw payload attached. "Compatible" endpoints do
occasionally return an error object, a streaming frame or a vendor-specific shape;
without the raw bytes the failure is impossible to diagnose from the CLI.

The API key is excluded from ``repr`` so a stray ``print(provider)`` or a logged
exception can never leak it, and it is only ever placed in the auth header the
dialect requires (``Authorization`` or ``x-api-key``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from moray_surgeon.exceptions import MorayError

#: One chat message; every dialect uses the same role/content object.
Message = dict[str, str]

#: How much of an unreadable response travels in the error message.
_RAW_DATA_LIMIT = 500


class ProviderResponseError(MorayError):
    """The endpoint answered with a payload its dialect cannot read."""


def _raw_data(data: object) -> str:
    """Render a response payload for diagnostics, capped at 500 characters.

    ``ensure_ascii=False`` keeps provider error text (often Chinese) readable, and
    the ``repr`` fallback guarantees the diagnostic itself can never fail on a
    payload that ``json.dumps`` rejects.
    """
    try:
        rendered = json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = repr(data)
    return rendered[:_RAW_DATA_LIMIT]


def _format_mismatch(label: str, reason: str, data: object) -> ProviderResponseError:
    """Build the error for an unreadable response, including a bounded payload."""
    return ProviderResponseError(
        f"{label} response format mismatch: {reason}. Raw data: {_raw_data(data)}"
    )


@dataclass(frozen=True)
class ChatRequest:
    """One fully built HTTP request: URL, headers and JSON payload.

    ``headers`` is excluded from ``repr`` because it may carry the bearer token.
    """

    url: str
    headers: dict[str, str] = field(repr=False)
    payload: dict[str, object]


class ChatProvider(Protocol):
    """A chat-completion dialect the parser can talk to."""

    @property
    def label(self) -> str:
        """Human-readable dialect name, used in transport error messages."""

    def build_request(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[Message],
        schema: dict[str, Any],
    ) -> ChatRequest:
        """Build the HTTP request for one chat turn."""

    def parse_content(self, data: object) -> str:
        """Extract the assistant message text, or raise ProviderResponseError."""


@dataclass(frozen=True)
class OllamaProvider:
    """Ollama's native ``/api/chat`` protocol, with schema-constrained decoding."""

    label: str = "Ollama"

    def build_request(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[Message],
        schema: dict[str, Any],
    ) -> ChatRequest:
        """Build the native request; ``format`` constrains decoding to the schema."""
        return ChatRequest(
            url=f"{base_url.rstrip('/')}/api/chat",
            headers={},
            payload={
                "model": model,
                "messages": messages,
                "stream": False,
                "format": schema,
            },
        )

    def parse_content(self, data: object) -> str:
        """Read ``message.content`` from Ollama's chat response."""
        if not isinstance(data, dict):
            raise ProviderResponseError("unexpected Ollama response: expected a JSON object")
        message = data.get("message")
        if not isinstance(message, dict):
            raise ProviderResponseError("unexpected Ollama response: missing 'message' object")
        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderResponseError(
                "unexpected Ollama response: 'message.content' is not a string"
            )
        return content


@dataclass(frozen=True)
class OpenAIProvider:
    """The OpenAI-compatible ``/chat/completions`` dialect used by cloud APIs.

    ``api_key`` is optional because local OpenAI-compatible servers often accept
    unauthenticated requests; when it is unset no ``Authorization`` header is
    sent. It is excluded from ``repr`` on purpose.
    """

    api_key: str | None = field(default=None, repr=False)
    label: str = "OpenAI"

    def build_request(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[Message],
        schema: dict[str, Any],
    ) -> ChatRequest:
        """Build an OpenAI-shaped request.

        ``schema`` is accepted for a uniform provider interface but deliberately
        not sent: the OpenAI protocol has no grammar field. The system prompt
        already embeds the schema, and validation still runs on the answer.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return ChatRequest(
            url=f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            payload={
                "model": model,
                "messages": messages,
                # Greedy decoding keeps plans reproducible across attempts.
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
        )

    def parse_content(self, data: object) -> str:
        """Read ``choices[0].message.content`` from an OpenAI chat response.

        Every shape mismatch carries the bounded raw payload, so a provider that
        is OpenAI-compatible in name but not in payload can be diagnosed without
        re-running under a debugger.
        """
        if not isinstance(data, dict):
            raise _format_mismatch("OpenAI", "expected a JSON object", data)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _format_mismatch("OpenAI", "missing 'choices' array", data)
        choice = choices[0]
        if not isinstance(choice, dict):
            raise _format_mismatch("OpenAI", "'choices[0]' is not an object", data)
        message = choice.get("message")
        if not isinstance(message, dict):
            raise _format_mismatch("OpenAI", "missing 'choices[0].message' object", data)
        content = message.get("content")
        if not isinstance(content, str):
            raise _format_mismatch(
                "OpenAI", "'choices[0].message.content' is not a string", data
            )
        return content


@dataclass(frozen=True)
class AnthropicProvider:
    """The Anthropic Messages dialect (``/v1/messages``), used by Claude and by
    providers such as Xiaomi's Token Plan.

    Two shape differences from the OpenAI dialect matter:

    * the system prompt is a top-level ``system`` field, not a message with
      ``role="system"``;
    * authentication travels in ``x-api-key`` plus a required
      ``anthropic-version`` header, not in ``Authorization``.

    ``max_tokens`` is mandatory in the Messages API; 4,096 is enough for a plan
    of a few dozen instructions and keeps a runaway response bounded.
    """

    api_key: str | None = field(default=None, repr=False)
    label: str = "Anthropic"

    def build_request(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[Message],
        schema: dict[str, Any],
    ) -> ChatRequest:
        """Build an Anthropic Messages request.

        ``schema`` is accepted for a uniform provider interface but not sent: the
        Messages API has no grammar field, so the system prompt carries the
        schema and the parser validates the answer.
        """
        system_parts = [
            message["content"] for message in messages if message.get("role") == "system"
        ]
        payload: dict[str, object] = {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {"role": message["role"], "content": message["content"]}
                for message in messages
                if message.get("role") in ("user", "assistant")
            ],
            "temperature": 0.0,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return ChatRequest(url=_anthropic_url(base_url), headers=headers, payload=payload)

    def parse_content(self, data: object) -> str:
        """Read the first ``type == "text"`` block from a Messages response.

        Reasoning models prepend a ``thinking`` block, so the plan is not
        necessarily ``content[0]``: DeepSeek's Anthropic-compatible endpoint (and
        Claude with extended thinking) returns
        ``[{"type": "thinking", ...}, {"type": "text", "text": ...}]``. Scan for
        the first text block with a string payload instead of assuming an index.

        When no usable text block exists, the error carries the first 500
        characters of the raw payload: an error object, a streaming frame and a
        vendor-specific block shape all look identical otherwise.
        """
        if not isinstance(data, dict):
            raise _format_mismatch("Anthropic", "expected a JSON object", data)
        content = data.get("content")
        if not isinstance(content, list) or not content:
            raise _format_mismatch("Anthropic", "missing 'content' array", data)

        malformed_index: int | None = None
        for index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str):
                return text
            if malformed_index is None:
                malformed_index = index
        if malformed_index is not None:
            raise _format_mismatch(
                "Anthropic", f"'content[{malformed_index}].text' is not a string", data
            )
        raise _format_mismatch("Anthropic", "missing 'text' content block", data)


def _anthropic_url(base_url: str) -> str:
    """Join the Messages path, tolerating a base URL that already ends in ``/v1``.

    ``https://api.anthropic.com`` and ``https://api.anthropic.com/v1`` must both
    produce ``.../v1/messages``; a trailing slash is already normalized away by
    the parser, and the extra ``rstrip`` keeps direct provider use safe too.
    """
    base = base_url.rstrip("/")
    suffix = "/messages" if base.endswith("/v1") else "/v1/messages"
    return f"{base}{suffix}"


def build_provider(api_type: str, api_key: str | None = None) -> ChatProvider:
    """Return the transport for ``api_type``.

    Args:
        api_type: ``"ollama"``, ``"openai"`` or ``"anthropic"``,
            case-insensitive and whitespace-tolerant.
        api_key: Credential for the cloud dialects: a Bearer token for OpenAI, an
            ``x-api-key`` for Anthropic. Empty or None disables the header
            (local servers, or an endpoint behind another auth layer).

    Raises:
        ValueError: ``api_type`` is not a supported dialect. The parser converts
            this into ``IntentParserError`` so callers see one error family.
    """
    normalized = api_type.strip().lower()
    if normalized == "ollama":
        return OllamaProvider()
    # A key pasted from an env file often carries surrounding whitespace; strip
    # it, and treat an all-whitespace key as absent.
    cleaned = api_key.strip() if api_key else None
    if normalized == "openai":
        return OpenAIProvider(api_key=cleaned or None)
    if normalized == "anthropic":
        return AnthropicProvider(api_key=cleaned or None)
    raise ValueError(
        f"unsupported api_type {api_type!r}; expected 'ollama', 'openai' or 'anthropic'"
    )
