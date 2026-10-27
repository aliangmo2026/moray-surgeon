"""Turn a natural-language refactor intent into a validated RefactorPlan.

The model never writes code here: it is asked for one JSON object matching the
schema exported by ``RefactorPlan``, and the answer is validated before it can
reach the engine. Repair is limited to one extra round-trip that sends the
validation error back to the model; transport failures are not retried.

The parser deliberately does not read repository files. Everything the model
needs beyond the intent must arrive through ``extra_context`` from the caller.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from moray_surgeon.exceptions import MorayError
from moray_surgeon.schema import RefactorPlan

_MAX_ATTEMPTS = 2

# Kept verbatim (including full-width punctuation) because the card requires this
# exact instruction; the prompt is user-facing model guidance, not source text.
_CHINESE_ONLY_JSON_RULE = (
    "你只能输出符合该 JSON Schema 的 JSON，不许输出 markdown 代码块"  # noqa: RUF001
    "（```json ... ```），不许输出源代码。"  # noqa: RUF001
)

_SYSTEM_PROMPT_TEMPLATE = """You are moray-surgeon's plan compiler.

Translate the user's refactoring intent into a RefactorPlan JSON object.
{chinese_rule}

Hard rules:
1. Output exactly one JSON object and nothing else: no markdown fences, no prose, no comments.
2. Never generate, rewrite, quote or guess source code. The plan is data only.
3. Every target_file is a repository-relative POSIX path ending in .py.
4. Use the "noop" action when the intent requires no edit.
5. If the user message carries a previous attempt's git diff and verification errors,
   fix those specific problems instead of repeating the same plan.

JSON Schema (the response must validate against it):
{schema}
"""


class IntentParserError(MorayError):
    """Raised when the model cannot produce a valid plan or the request fails."""


class IntentParser:
    """Ask a local Ollama model for a RefactorPlan and validate its answer."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        """Configure the model endpoint and cache the exported schema.

        Args:
            model: Ollama model tag, e.g. ``"qwen2.5-coder:7b"``.
            base_url: Ollama HTTP endpoint. Must be http(s); the default targets
                a local daemon, which is the normal deployment for this tool.
            timeout: Per-request timeout in seconds.

        Raises:
            IntentParserError: ``base_url`` is malformed or not http(s).
        """
        self._model = model
        self._timeout = timeout
        self._schema = RefactorPlan.export_schema()
        self._schema_json = json.dumps(self._schema, ensure_ascii=False)
        self._base_url = _validate_base_url(base_url)
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            schema=self._schema_json, chinese_rule=_CHINESE_ONLY_JSON_RULE
        )

    def parse(self, intent: str, repo_root: Path, extra_context: str = "") -> RefactorPlan:
        """Compile ``intent`` into a plan, retrying once on invalid model output.

        Args:
            intent: The user's refactoring request in natural language.
            repo_root: Repository the plan targets (only its path is mentioned in
                the prompt; no file is read here).
            extra_context: Caller-supplied context such as symbol lists. Appended
                verbatim when non-empty.

        Returns:
            A validated, frozen :class:`RefactorPlan`.

        Raises:
            IntentParserError: The HTTP call failed, the response shape was
                unexpected, or two consecutive answers failed validation.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": _user_prompt(intent, repo_root, extra_context)},
        ]
        last_error = ""

        with httpx.Client(timeout=self._timeout) as client:
            for attempt in range(_MAX_ATTEMPTS):
                content = self._chat(client, messages)
                try:
                    return RefactorPlan.model_validate_json(_extract_json(content))
                except ValueError as exc:
                    # ValidationError, JSON syntax errors and "no object at all"
                    # are all retryable model-output problems; transport errors
                    # already raised above and are never retried.
                    last_error = str(exc)
                    if attempt + 1 >= _MAX_ATTEMPTS:
                        break
                    messages = [
                        *messages,
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": _repair_prompt(last_error)},
                    ]

        raise IntentParserError(
            f"model did not return a valid RefactorPlan after {_MAX_ATTEMPTS} attempts: "
            f"{last_error}"
        )

    def _chat(self, client: httpx.Client, messages: list[dict[str, str]]) -> str:
        """POST one chat request and return ``message.content``.

        Raises:
            IntentParserError: Transport/status failure or an unexpected payload.
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "format": self._schema,
        }
        url = f"{self._base_url}/api/chat"
        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise IntentParserError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise IntentParserError(f"Ollama returned a non-JSON response: {exc}") from exc

        return _message_content(data)


def _message_content(data: object) -> str:
    """Pull ``message.content`` out of Ollama's chat response, or fail clearly."""
    if not isinstance(data, dict):
        raise IntentParserError("unexpected Ollama response: expected a JSON object")
    message = data.get("message")
    if not isinstance(message, dict):
        raise IntentParserError("unexpected Ollama response: missing 'message' object")
    content = message.get("content")
    if not isinstance(content, str):
        raise IntentParserError("unexpected Ollama response: 'message.content' is not a string")
    return content


def _extract_json(text: str) -> str:
    """Return the first ``{`` through the last ``}`` of a model response.

    Models sometimes wrap JSON in markdown fences or add prose around it; slicing
    between the outermost braces recovers the object without trusting the fence.
    A response with no braces raises ``ValueError`` so ``parse`` can retry with
    the error attached instead of guessing.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("response does not contain a JSON object")
    return text[start : end + 1]


def _validate_base_url(base_url: str) -> str:
    """Normalize the endpoint and reject anything that is not HTTP(S)."""
    try:
        url = httpx.URL(base_url)
    except (httpx.InvalidURL, ValueError) as exc:
        raise IntentParserError(f"invalid base_url: {base_url!r}") from exc
    if url.scheme not in ("http", "https"):
        raise IntentParserError(f"base_url must use http or https, got {url.scheme!r}")
    if not url.host:
        raise IntentParserError(f"base_url must include a host: {base_url!r}")
    return str(url).rstrip("/")


def _user_prompt(intent: str, repo_root: Path, extra_context: str) -> str:
    """Build the user message from the intent and caller-supplied context."""
    sections = [
        f"Repository root: {repo_root}",
        f"Refactoring intent:\n{intent.strip()}",
    ]
    if extra_context.strip():
        sections.append(
            "Additional context (may include the previous attempt's diff and "
            f"verification errors):\n{extra_context.strip()}"
        )
    sections.append("Return the RefactorPlan JSON object now.")
    return "\n\n".join(sections)


def _repair_prompt(error: str) -> str:
    """Tell the model exactly why its previous answer was rejected."""
    return (
        "Your previous response did not validate as a RefactorPlan.\n\n"
        f"Validation error:\n{error}\n\n"
        "Return a corrected JSON object only: no markdown fences, no prose, no source code."
    )
