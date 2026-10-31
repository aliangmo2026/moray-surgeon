"""Turn a natural-language refactor intent into a validated RefactorPlan.

The model never writes code here: it is asked for one JSON object matching the
schema exported by ``RefactorPlan``, and the answer is validated before it can
reach the engine. Validation covers both the instruction payloads and the plan's
dependency graph (duplicate ids, dangling ``depends_on`` targets, cycles), so a
malformed DAG is repaired in the model round rather than crashing the dispatcher.
Repair is limited to one extra round-trip that sends the validation error back to
the model; transport failures are not retried.

The HTTP dialect lives in :mod:`moray_surgeon.intent.providers`: Ollama's native
``/api/chat`` with schema-constrained decoding, the OpenAI-compatible
``/chat/completions`` shape (DeepSeek, vLLM, LM Studio, ...), or the Anthropic
Messages API (``/v1/messages``, Claude and Xiaomi's Token Plan). Retries, JSON
recovery, validation and repair are shared by all three, so swapping the brain
changes only the wire format.

The parser deliberately does not read repository files. Everything the model
needs beyond the intent must arrive through ``extra_context`` from the caller.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from moray_surgeon.engine.dag import topological_sort
from moray_surgeon.exceptions import DependencyError, MorayError
from moray_surgeon.intent.providers import ChatProvider, ProviderResponseError, build_provider
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
3. Every instruction has a unique "id" (for example "step1"); "depends_on" lists the ids
   of the steps that must run before it.
4. Put ordering knowledge in "depends_on" instead of relying on list position: add an
   import before the call that needs it, rename a symbol before rewriting its callers.
   Independent steps need no dependency and still run in list order.
5. Never create a dependency cycle and never reference an id that is not in the plan.
6. Every target_file is a repository-relative POSIX path ending in .py.
7. Use the "noop" action when the intent requires no edit.
8. If the user message carries a previous attempt's git diff and verification errors,
   fix those specific problems instead of repeating the same plan.
9. Tests are callers too. When the intent or the repair context shows failing test
   files under tests/, include rename_symbol or replace_call instructions whose
   target_file is that test module so the tests follow the refactored source. Keep
   test edits surgical: update only the symbols and callee names the tests
   reference; never change what the tests assert.

JSON Schema (the response must validate against it):
{schema}
"""


class IntentParserError(MorayError):
    """Raised when the model cannot produce a valid plan or the request fails."""


class IntentParser:
    """Ask a chat model — local Ollama or an OpenAI-compatible cloud API — for a plan."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        api_key: str | None = None,
        api_type: str = "ollama",
    ) -> None:
        """Configure the model endpoint and cache the exported schema.

        Args:
            model: Model tag or name, e.g. ``"qwen2.5-coder:7b"`` locally or
                ``"deepseek-chat"`` for a cloud API.
            base_url: Endpoint base. Ollama defaults to a local daemon; an
                OpenAI-compatible provider takes its API root (usually ending in
                ``/v1``). Must be http(s); a trailing slash is accepted. Embedded
                credentials and query strings are rejected so a key can never
                travel inside a URL that might be logged.
            timeout: Per-request timeout in seconds.
            api_key: Credential for the cloud dialects: a Bearer token for
                ``api_type="openai"``, an ``x-api-key`` for
                ``api_type="anthropic"``. None sends no auth header, which is
                what local OpenAI- or Anthropic-compatible servers expect. Never
                logged or included in error messages.
            api_type: Transport dialect: ``"ollama"`` (native, schema-constrained
                decoding), ``"openai"`` (``/chat/completions`` + JSON mode), or
                ``"anthropic"`` (Messages API: top-level ``system``, ``x-api-key``).

        Raises:
            IntentParserError: ``base_url`` is malformed, not http(s), or carries
                credentials/query data, or ``api_type`` is not supported.
        """
        self._model = model
        self._timeout = timeout
        self._schema = RefactorPlan.export_schema()
        self._schema_json = json.dumps(self._schema, ensure_ascii=False)
        self._base_url = _validate_base_url(base_url)
        try:
            self._provider: ChatProvider = build_provider(api_type, api_key)
        except ValueError as exc:
            raise IntentParserError(str(exc)) from exc
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

        # Redirects are never followed: a bearer token must not be forwarded to
        # a host the caller did not configure (httpx's default, pinned here).
        with httpx.Client(timeout=self._timeout, follow_redirects=False) as client:
            for attempt in range(_MAX_ATTEMPTS):
                content = self._chat(client, messages)
                try:
                    plan = RefactorPlan.model_validate_json(_extract_json(content))
                    # The graph is part of the plan contract: sorting validates
                    # duplicate ids, dangling dependencies and cycles without
                    # touching a file, and the repair round can fix all three.
                    topological_sort(plan.instructions)
                    return plan
                except (ValueError, DependencyError) as exc:
                    # ValidationError, JSON syntax errors, "no object at all" and
                    # an unexecutable dependency graph are all retryable model
                    # output problems; transport errors already raised above and
                    # are never retried.
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
        """Send one chat request through the configured provider.

        Raises:
            IntentParserError: Transport/status failure or an unexpected payload.
                The endpoint URL is included so a forgotten ``--base-url`` is
                obvious; it cannot contain credentials (rejected at construction).
        """
        request = self._provider.build_request(
            base_url=self._base_url,
            model=self._model,
            messages=messages,
            schema=self._schema,
        )
        try:
            response = client.post(request.url, json=request.payload, headers=request.headers)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise IntentParserError(
                f"{self._provider.label} request failed ({request.url}): {exc}"
            ) from exc
        except ValueError as exc:
            raise IntentParserError(
                f"{self._provider.label} returned a non-JSON response ({request.url}): {exc}"
            ) from exc

        try:
            return self._provider.parse_content(data)
        except ProviderResponseError as exc:
            raise IntentParserError(str(exc)) from exc


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
    """Normalize the endpoint; reject non-HTTP(S), credentials and URL extras.

    The transport appends a dialect path (``/api/chat`` or ``/chat/completions``),
    so credentials, query strings and fragments would either break that join or
    smuggle a token into a URL that ends up in logs and error messages. All three
    are rejected; the only supported credential channel is ``api_key``.

    Private and loopback hosts are deliberately allowed: the URL is supplied by
    the operator of a local-first developer tool (the default Ollama daemon is
    ``localhost``) and is never fetched on behalf of a third party.
    """
    try:
        url = httpx.URL(base_url)
    except (httpx.InvalidURL, ValueError) as exc:
        raise IntentParserError(f"invalid base_url: {base_url!r}") from exc
    if url.scheme not in ("http", "https"):
        raise IntentParserError(f"base_url must use http or https, got {url.scheme!r}")
    if not url.host:
        raise IntentParserError(f"base_url must include a host: {base_url!r}")
    if url.username or url.password:
        raise IntentParserError(
            "base_url must not embed credentials; pass the token as api_key instead"
        )
    if url.query:
        raise IntentParserError(
            "base_url must not contain a query string; pass the token as api_key instead"
        )
    if url.fragment:
        raise IntentParserError("base_url must not contain a fragment")
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
