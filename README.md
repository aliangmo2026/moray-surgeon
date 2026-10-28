<div align="center">

# moray-surgeon

**AST-based zero-hallucination code refactoring engine.**

*The model writes no code. It writes a plan. `libcst` does the surgery.*

[![Tests](https://img.shields.io/badge/Tests-441%20passed-brightgreen)](#engineering-standards)
[![Coverage](https://img.shields.io/badge/Coverage-97%25-brightgreen)](#engineering-standards)
[![Ruff](https://img.shields.io/badge/Ruff-0%20warnings-blue)](#engineering-standards)
[![Plans](https://img.shields.io/badge/Plans-DAG%20ordered-blueviolet)](#dependency-aware-plans-dag)
[![CI Gateway](https://img.shields.io/badge/CI-PR%20gateway-2088FF?logo=githubactions&logoColor=white)](#use-as-a-ci-gateway)
[![Stress Test](https://img.shields.io/badge/Stress-8%2C046%20LOC%20%C2%B7%2050%20instructions-critical)](./docs/STRESS_TEST_REPORT.md)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](#quick-start)
[![MCP](https://img.shields.io/badge/MCP-Ready-6f42c1)](#mcp-integration)
[![License](https://img.shields.io/badge/License-PolyForm%20NC%201.0.0-orange)](./LICENSE)

[Why](#why-this) · [Features](#features) · [DAG](#dependency-aware-plans-dag) · [CI gateway](#use-as-a-ci-gateway) · [Stress test](./docs/STRESS_TEST_REPORT.md) · [Quick start](#quick-start) · [MCP](#mcp-integration) · [Architecture](./docs/ARCHITECTURE.md) · [Standards](#engineering-standards) · [License](#license-and-contact)

</div>

---

> ### 💥 Proven under load
> A generated **8,046-line, 300-class, 50-module** project passed a single **50-instruction refactor** — 54 files changed, +453/−337 — with real `ruff`, `mypy`, `pytest` **and** the AST scope guard green. Two separate poisoning runs (one mypy-red, one pytest-red) were rolled back in **under 0.1 s** with the worktree restored byte-for-byte and the full diff captured as feedback.
> **[Read the full stress report →](./docs/STRESS_TEST_REPORT.md)**

---

## Why this?

Ask a general-purpose coding agent to rename one function and it usually rewrites the whole file. What comes back compiles often enough to feel safe: new formatting, a dropped import, an unrelated line quietly changed. Tests catch some of it. Code review catches the rest. The trust budget does not survive the third time.

moray-surgeon removes the model from the editing path. The LLM compiles natural language into a JSON plan of seven typed operations; a deterministic AST engine applies it. The model never returns a file body or a statement suite: even imports are expressed as a module path plus identifiers, and the only expression-shaped field is `replace_call.new_call`, which must parse as a single Python expression and can only replace a callee. A hallucinated file has no slot to arrive in.

That is what *zero-hallucination* means here: the model cannot invent code structure. It can still choose the wrong operation — which is why every attempt ends at **two** gates, not one: the verification gates (`ruff`/`mypy`/`pytest`) and an AST scope guard that proves no function, class or module-level statement outside the plan was touched. Either gate rolls the whole attempt back.

| | Chat-driven editing | moray-surgeon |
|---|---|---|
| What the model emits | source code | a JSON `RefactorPlan` |
| What touches your files | the model | `libcst` + Jedi |
| Formatting and comments | regenerated | preserved outside the edited node |
| Imports | freely rewritten | typed `add_import` / `remove_import` atoms |
| Invalid output | looks plausible | rejected by schema, scope guard or verification |
| Recovery | manual `git restore` | automatic reset to `HEAD` + retry |

The plan schema is closed (`extra="forbid"`, frozen models) and, on the local Ollama path, is also handed to the model as the decoding grammar, so even malformed JSON cannot be generated. Cloud APIs get the schema through the prompt (plus JSON mode where the provider supports it), with the same validation and repair.

<details>
<summary><b>中文速览（30 秒）</b></summary>

moray-surgeon 是一个「零幻觉」代码重构引擎：大模型只负责把自然语言需求编译成 **JSON 指令**，真正改代码的是 `libcst` 语法树手术刀。它不会删你的 import（增删 import 也必须走受控指令）、不会改你的引号、不会重排格式；每一轮修改都在 Git 事务里执行，`ruff` / `mypy` / `pytest` 任一失败立即回滚；并且在验证之前还有一道 **AST 范围门控**：计划之外的函数/类只要有改动，整轮立刻回滚，并把 diff 和报错喂回模型重试。

适合：大规模符号重命名、函数抽取、批量调用替换、补类型注解、受控增删 import。不适合：从零生成模块、跨文件重组目录结构。已在 **8,046 行 / 300 类 / 50 模块** 的合成项目上通过 **50 指令单事务压力测试**，详见 [docs/STRESS_TEST_REPORT.md](./docs/STRESS_TEST_REPORT.md)。

</details>

## Features

- 🛡️ **Zero-hallucination, schema-driven.** The model can only emit one of seven instructions, validated against a frozen Pydantic model; on the Ollama path the same JSON Schema is sent as the `format` decoding grammar, constraining generation rather than just checking afterwards.
- 🔪 **AST precision with `libcst`.** Edits are CST node transformations, not string replacement or `ast.unparse`. Comments, blank lines, quote style, indent width and untouched expressions round-trip byte-for-byte.
- 🔗 **Dependency-aware plans (DAG).** Every instruction carries a unique `id` and may list `depends_on`. Before the first file is read, `execute_plan` runs Kahn's algorithm over the plan: rename a class before rewriting its callers, add the import before the call that needs it. A plan with no edges keeps v0.2.0's authored-order semantics; duplicate ids, dangling references and cycles are rejected before any edit.
- 🕵️‍♂️ **Jedi cross-file resolution.** `rename_symbol` resolves references at project scope, so renaming a function updates its callers across files without touching same-named attributes or keyword arguments. The atom raises Jedi's built-in 30-file search budget to the repository size, so large projects are not silently half-renamed.
- 🛑 **Git transaction and rollback.** Every attempt enters a transaction: uncommitted work is snapshotted with `git stash -u`, the refactor is committed only on a green verification, and any failure runs `git reset --hard HEAD` plus `git clean -fd`, then restores your snapshot.
- 🧭 **AST scope guard (the second safety net).** Before the plan runs, every Python file is fingerprinted (per-class, per-function, per-module-level statement). After it runs, every changed node must be explained by the plan — a renamed symbol, a Jedi reference site, or a file-global atom. Anything else raises `ScopeViolationError`, rolls the attempt back and feeds the offending node names to the next prompt.
- 📦 **Controlled imports.** `add_import` and `remove_import` edit the import block through typed fields (module path + identifier), never free-form source. Imports are merged into the right statement, sorted with isort's `order-by-type` rule, wrapped when a line would exceed 88 characters, and never reorder unrelated statements.
- 🔄 **Auto-fix loop.** On a red gate (behavioural *or* structural) the worktree diff and the diagnostics are captured before rollback and fed into the next plan request. The next attempt re-plans from a clean tree, up to `--max-attempts` rounds (default 3).
- 🧠 **Bring your own model.** Run against a local Ollama daemon or any OpenAI-compatible or Anthropic-compatible cloud API (DeepSeek, Xiaomi Token Plan, Claude, vLLM, LM Studio, ...). Only the wire format changes: validation, repair, transactions and both gates are identical. The API key comes from `MORAY_API_KEY` (or `--api-key`), is never logged or echoed, and is never forwarded across a redirect.
- 🔌 **MCP ready.** A stdio MCP server exposes the engine to Cursor, Claude Desktop and any MCP host: list actions, execute a structured plan, or run the full natural-language loop.
- 🧪 **Dry run.** `--dry-run` applies the plan inside the transaction, prints a syntax-highlighted unified diff, and rolls back without verifying or committing.
- 🚦 **CI gateway.** `moray-surgeon audit` is the read-only counterpart of the refactor loop: it diffs the working tree against a base ref, re-fingerprints every changed Python file, applies the same scope gate when a plan is supplied, and runs `ruff`/`mypy`/`pytest` — then exits non-zero with a JSON report. No edit, no commit, no stash.
- 💥 **Proven under load.** An 8,046-line, 300-class synthetic project survived a 50-instruction single-transaction refactor; two poisoned-instruction runs rolled back in under 0.1 s. [Full report with diffs and timings](./docs/STRESS_TEST_REPORT.md).

### Instruction set

| Action | Effect | Scope |
|---|---|---|
| `rename_symbol` | Rename a symbol and every Jedi-resolved reference | Project-wide, cross-file |
| `extract_function` | Move a line range into a new parameterless function and insert the call | Module top level only |
| `replace_call` | Swap the callee of matching calls | Arguments and layout preserved |
| `add_type_hints` | Add a missing `-> None` return annotation | Conservative by design |
| `add_import` | Add `import x` / `from x import y`, sorted into the import block | Module level, absolute modules |
| `remove_import` | Drop an import, or one name from a from-import | Module level, absolute modules |
| `noop` | Explicitly declare that no edit is needed | Retried with feedback if the whole plan is `noop` |

Every instruction also carries a unique `id` and an optional `depends_on` list; see the next section.

### Dependency-aware plans (DAG)

List position is a weak ordering. A real multi-step refactor usually has a graph: the import has to exist before the call that uses it, and the rename has to land before its callers are rewritten. Each instruction therefore has a plan-local `id` and may name the steps it waits for:

```json
{
  "instructions": [
    {
      "id": "rename",
      "action": "rename_symbol",
      "target_file": "src/models.py",
      "old_name": "User",
      "new_name": "Account"
    },
    {
      "id": "rewrite-callers",
      "action": "replace_call",
      "target_file": "src/api.py",
      "old_call": "User.load",
      "new_call": "Account.load",
      "depends_on": ["rename"]
    },
    {
      "id": "import-account",
      "action": "add_import",
      "target_file": "src/api.py",
      "module": "src.models",
      "name": "Account",
      "depends_on": ["rewrite-callers"]
    }
  ],
  "summary": "rename User to Account and update the API layer"
}
```

`execute_plan` topologically sorts the plan with Kahn's algorithm before it reads the first file. Among steps that are ready at the same time, the one authored earlier runs first, which gives two useful properties: an all-independent plan executes exactly as written (so existing plans and existing tests keep their behaviour), and one plan always produces one deterministic order, on every run and interpreter.

`id` is required by the schema as of v0.3.0 — a plan written for v0.2.0 needs one id per instruction and nothing else.

Graph validity is part of plan ingestion. A duplicate `id`, a `depends_on` naming no step, and a cycle are rejected before anything is touched:

| Error | Meaning |
|---|---|
| `DuplicateInstructionIdError` | Two steps share an id, so `depends_on` would be ambiguous |
| `UnknownDependencyError` | `depends_on` names an id that is not in the plan |
| `CircularDependencyError` | No execution order exists; the message names one concrete cycle (`step1 -> step2 -> step1`) |

The intent parser treats all three as retryable model-output errors: the validation message goes back to the model in the single repair round-trip, so a cyclic first answer can come back fixed. The dispatcher never touches a file when the graph is invalid.

## Proof under load

The numbers below are generated by [`tests/stress_test/run_bombing.py`](./tests/stress_test/run_bombing.py) and refreshed in [docs/STRESS_TEST_REPORT.md](./docs/STRESS_TEST_REPORT.md). The test bed is a 50-module project with 300 classes, two-level inheritance, a runtime dependency ring and a pytest suite — not a fixture with fake tools.

| Metric | Result |
|---|---|
| Instructions in one plan | **50** (2 project-wide renames, 20 add-import + 20 replace-call pairs, 4 call rewrites, 4 re-export removals) |
| Plan applied + verified | **16.9 s** total (10.4 s apply, 1.6 s snapshot, 3.1 s scope check, 1.5 s gates) |
| Files changed by that plan | **54** (+453 / −337) |
| Gates | `ruff` ✅ `mypy` ✅ `pytest` ✅ |
| Scope guard | every changed node authorised ✅ |
| Poisoned instruction → mypy red | rolled back in **0.064 s**, HEAD restored, full diff + mypy errors captured |
| Poisoned instruction → pytest red | rolled back in **0.050 s**, HEAD restored, full diff + pytest traceback captured |

Reproduce it locally:

```bash
python tests/stress_test/run_bombing.py --keep
```

The project is regenerated from a checked-in template each run, so the evidence is not a one-off checkout.

## Quick start

**Requirements:** Python 3.11+, Git, and either a local [Ollama](https://ollama.com/) instance or a cloud API (OpenAI- or Anthropic-compatible). The default local path keeps the code and the model on your machine; nothing is sent anywhere.

```bash
# 1. Install from a checkout of this repository
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Pull and run a local code model (default endpoint: http://localhost:11434)
#    This opens a chat REPL — Ctrl+D exits it; the daemon keeps serving.
ollama run qwen2.5-coder:7b

# 3. Run from a Git repository with at least one commit — preview only
moray-surgeon refactor "Rename print to logging.info" --path . --dry-run
```

A dry run applies the plan inside a transaction, prints the diff, and rolls back. Nothing is committed.

> Imports are part of the plan too. The example above produces `logging.info(...)` calls, so the plan includes `{"id": "step1", "action": "add_import", "target_file": "app.py", "module": "logging"}`; if the import is still missing, verification goes red and the attempt rolls back — that is the intended contract, not a workaround. When one step needs another, the plan says so with `depends_on` instead of relying on list order.

### Local or cloud

The default provider is a local Ollama daemon. To use a stronger cloud model, point `--base-url` at any OpenAI-compatible API root and put the token in `MORAY_API_KEY`:

```bash
export MORAY_API_KEY="<your token>"   # read from the environment; never commit it

moray-surgeon refactor "Rename parse_args to parse_cli_options" \
  --path . \
  --api-type openai \
  --base-url https://api.deepseek.com/v1 \
  --model deepseek-chat \
  --dry-run
```

The same three flags work for any OpenAI-compatible endpoint (DeepSeek, vLLM, LM Studio, ...); for providers speaking the Anthropic Messages protocol, see the next section. `--api-key` also exists, but the environment variable is preferred — a flag is visible in shell history and process listings. The key is never logged, never included in error messages, and never forwarded across an HTTP redirect.

One difference matters. Ollama accepts the plan schema as a decoding grammar (`format`), so the model physically cannot emit a non-conforming plan. The OpenAI-compatible path offers JSON mode (`response_format={"type": "json_object"}`); the Anthropic path has no such field. In both cloud cases the schema is enforced by the system prompt, Pydantic validation and the one repair round-trip instead. A weaker cloud model may therefore need the repair round more often, and `--max-attempts` is the budget for that.

### Anthropic-compatible APIs (Xiaomi Token Plan, Claude, ...)

Some providers — including Xiaomi's Token Plan — expose the Anthropic Messages protocol (`/v1/messages`) rather than OpenAI's chat-completions protocol. Select it with `--api-type anthropic`:

```bash
export MORAY_API_KEY="<your token>"   # x-api-key, read from the environment

moray-surgeon refactor "Rename parse_args to parse_cli_options" \
  --path . \
  --api-type anthropic \
  --base-url https://api.your-provider.example \
  --model your-model-name \
  --dry-run
```

`--base-url` may be the API root (`https://api.anthropic.com`) or already include `/v1` (`https://api.anthropic.com/v1`); both resolve to `.../v1/messages` without doubling the segment. The token is sent as `x-api-key` alongside the required `anthropic-version: 2023-06-01` header. Because the Messages API has no `role="system"` message, the system prompt is moved to the top-level `system` field and the conversation keeps only `user`/`assistant` turns; `max_tokens` is fixed at 4,096 and `temperature` at 0.0. Reasoning models can prepend a `thinking` block to the `content` array, so the parser returns the first block whose `type` is `text` rather than assuming index 0. There is no grammar or JSON-mode switch on this protocol either, so the schema travels in the prompt and the answer goes through the same Pydantic validation and repair round as the OpenAI path.

Both cloud dialects attach the first 500 characters of the raw response to a format-mismatch error (for example `Anthropic response format mismatch: 'content[0].text' is not a string. Raw data: {...}`). A provider that is compatible in name but not in payload shape — an error object, a streaming frame, a vendor-specific block — can therefore be diagnosed from the CLI without a debugger.

`--path` must be a Git repository root; the CLI exits with code `2` otherwise. Any repository root works, including a nested checkout:

```bash
moray-surgeon refactor "Rename print to logging.info" --path ./src --dry-run
```

When the preview looks right, drop `--dry-run`:

```bash
moray-surgeon refactor "Rename parse_args to parse_cli_options" --path . --files src/app.py
```

`--files` controls which files are summarised into the 【Project Code Map】 context. Without it, the CLI maps up to 50 Python files under the repository root.

### CLI options

| Flag | Default | Purpose |
|---|---|---|
| `--path` | `.` | Repository root (must contain `.git` and a `HEAD` commit) |
| `--model` | `qwen2.5-coder:7b` | Ollama model tag, or cloud model name |
| `--base-url` / `--ollama` | `http://localhost:11434` | Ollama endpoint, or cloud API root (OpenAI `.../v1` or Anthropic Messages; `/v1` is optional for Anthropic) |
| `--api-type` | `ollama` | Transport dialect: `ollama` (local, schema grammar), `openai` (chat completions, JSON mode), or `anthropic` (Messages API) |
| `--api-key` | `$MORAY_API_KEY`, else none | Credential for `openai` (Bearer) or `anthropic` (`x-api-key`); prefer the environment variable |
| `--max-attempts` | `3` | Plan → apply → verify rounds |
| `--context` | empty | Extra context appended to the prompt |
| `--files` | auto-scan, max 50 | Comma-separated, repo-relative files for the code map |
| `--mypy-target` | `src` | Path passed to `mypy` |
| `--pytest-args` | `-q` | Arguments passed to `pytest` |
| `--dry-run` | off | Show the diff and roll back without committing |

Exit codes: `0` success, `1` refactor or verification failure, `2` not a Git repository. The read-only [`audit`](#use-as-a-ci-gateway) command has its own flags and exit codes.

## MCP integration

`moray-surgeon mcp` serves three tools over stdio. stdout carries JSON-RPC only; diagnostics go to `~/.moray-surgeon/mcp.log` (override with `MORAY_MCP_LOG`).

| Tool | What it does |
|---|---|
| `list_available_actions` | Returns the exact action names and fields the schema accepts |
| `execute_plan` | Validates and applies a `RefactorPlan` JSON inside a Git transaction; commits only if verification passes |
| `refactor_smart` | Natural language → plan → apply → verify → auto-fix, using the local Ollama model |

Add the server to Cursor — project scope `.cursor/mcp.json` or global scope `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "moray-surgeon": {
      "command": "moray-surgeon",
      "args": ["mcp"]
    }
  }
}
```

Then ask Cursor:

> Use moray-surgeon to rename `parse_args` to `parse_cli_options` in `/path/to/repo`.

The same `mcpServers` block works for Claude Desktop's `claude_desktop_config.json`. If the executable is not on `PATH` (common when Cursor does not inherit your virtualenv), point `command` at the venv script: `.venv/bin/moray-surgeon` on macOS/Linux, `.venv\Scripts\moray-surgeon.exe` on Windows.

Two operational notes: the MCP tools do not build a code map, so name the files and symbols in the intent; and they use the default verification profile (`mypy src`, `pytest -q`). For a repository without a `src/` directory, use the CLI, where `--mypy-target` and `--pytest-args` are configurable.

## Use as a CI Gateway

`moray-surgeon audit` is the read-only counterpart of the refactor loop. It never edits a file, commits or stashes, so it is safe to run against an untrusted pull request:

```bash
moray-surgeon audit --path . --base-ref origin/main --json-output moray-audit.json
```

The command diffs the working tree against `--base-ref`, fingerprints every changed Python file against its base-commit content, and runs the project's own `ruff` → `mypy` → `pytest` gates unchanged. The report lists the changed AST nodes per file, any scope violations, and the verification result.

- **Exit `0`** — prints `Audit passed`.
- **Exit `1`** — prints the JSON report on stdout (and writes `--json-output` when given). Use it to fail the check or to post the report as a comment.
- **Exit `2`** — the path is not a Git repository, the base ref cannot be resolved, or the plan is unreadable.

### Authorising a diff with a plan

Without `--plan`, the audit is an inventory: created files are reported as `added` but not treated as violations, because a human pull request is allowed to add files. Pass the `RefactorPlan` that produced the change and the same scope gate the orchestrator uses is enforced:

```bash
moray-surgeon audit --path . --base-ref origin/main --plan moray-plan.json
```

With a plan, every changed node must be explained by an instruction, and a created, deleted, renamed, non-Python or unparseable file is a violation — the engine can produce none of those. The plan's dependency graph is validated too, so a cyclic plan fails as a setup error rather than authorising anything.

### Report format

```json
{
  "ok": false,
  "repo_root": "/home/runner/work/app/app",
  "base_ref": "origin/main",
  "base_commit": "3f2c1a9…",
  "plan": "moray-plan.json",
  "files": [
    {
      "path": "src/api.py",
      "status": "modified",
      "origin": null,
      "changed_nodes": ["neighbour", "old_name", "new_name"],
      "parse_error": false
    }
  ],
  "scope_violations": ["src/api.py: neighbour"],
  "verification": {"ok": true, "stage": "all", "stdout": "…", "stderr": ""}
}
```

`status` is one of `added`, `modified`, `deleted` or `renamed`; `changed_nodes` holds the qualified AST node names whose fingerprint differs from the base commit. Verification output is bounded to the last 4,000 characters so the artefact stays small.

| Flag | Default | Purpose |
|---|---|---|
| `--path` | `.` | Repository root |
| `--base-ref` | `HEAD` | Git ref the working tree is compared against (`origin/main` in CI) |
| `--plan` | none | `RefactorPlan` JSON that authorises the diff; enables the scope gate |
| `--json-output` | none | Write the report to this file (it is also printed on failure) |
| `--mypy-target` | `src` | Path passed to mypy |
| `--pytest-args` | `-q` | Arguments passed to pytest |

### GitHub Actions

A ready-to-use workflow lives at [`.github/workflows/moray-audit.yml`](./.github/workflows/moray-audit.yml). It runs on every `pull_request`, checks out with full history so the base ref resolves, installs `.[dev]`, audits against `origin/${{ github.base_ref }}`, and uploads the JSON report as an artefact even when the audit fails:

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0

- run: pip install -e ".[dev]"

- run: moray-surgeon audit --path . --base-ref "origin/${{ github.base_ref }}"
```

To enforce an authorising plan in CI, add `--plan moray-plan.json` and commit that file with the pull request.

## Engineering standards

| Gate | Result |
|---|---|
| Test suite | **441 passed** |
| Coverage | **97%** (enforced gate: 90%) |
| Lint | **Ruff: 0 warnings** |
| Stress test | **PASS** — 50 instructions on 8,046 lines / 300 classes, 54 files changed |
| Runtime | Python 3.11+ (verified on CPython 3.12.2) |

The suite is fully offline: model calls (Ollama and OpenAI-compatible) are mocked at the `httpx.Client.post` boundary, while Git, `ruff`, `mypy` and `pytest` run for real in temporary repositories. The stress harness is intentionally *not* part of `pytest -q` (it takes tens of seconds and generates a project); run it explicitly.

Safety is part of the test surface, not an afterthought: path containment is checked after `resolve()`, every external command has a timeout, failed transactions restore the user's stash, both gates capture their evidence before rollback, and rollback errors are raised rather than swallowed. See [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) for the invariants behind each module.

## Known limitations

1. **Controlled imports have a deliberate scope.** `add_import`/`remove_import` only touch top-level statements; imports inside `try:` blocks or functions are left alone. `module` must be an absolute dotted path (relative imports are never generated or merged). Existing statements are never reordered — only a merged alias list is sorted. A rename that unsorts a from-import is the project's isort gate's concern, not a silent engine reformat.
2. **`rename_symbol` is best-effort outside Jedi's view.** The atom raises Jedi's project-search budget to the repository's file count, but if Jedi cannot analyse the project it falls back to a single-file textual rename and logs a warning.
3. **`extract_function` is top-level only.** Nested scopes are not extracted, and a compound statement between the selected lines makes the atom refuse rather than silently reorder code.
4. **`add_type_hints` only writes `-> None`.** Parameter hints and richer return types are not guessed: a wrong hint is worse than a missing one.
5. **The scope guard is structural, not semantic.** It allows every node the plan names (and, for a class rename, its members) and every file-global atom in its target file. Two same-named members in different scopes can therefore not be told apart; the verification gates still cover behaviour. It is on by default and can only be disabled programmatically (`Orchestrator(..., scope_guard=False)`), never by the model.
6. **Auto-fix is bounded.** Each round gets a fresh transaction; after `--max-attempts` failures the tree stays at `HEAD` and the last gate's feedback is reported.
7. **A Git repository with a `HEAD` commit is required.** Without one there is no rollback baseline, and the transaction refuses to start.
8. **No file creation or deletion.** The engine edits existing files; the scope guard treats a created or deleted file as a violation.
9. **DAG ordering is sequential, not parallel.** `depends_on` constrains order only; steps still run one at a time because every atom reads and writes the same worktree. Independent steps are never run concurrently.
10. **The audit enforces scope only when a plan is supplied.** Without `--plan`, an audit reduces to the verification gates plus a structural inventory; added files are reported but not rejected, because a human pull request may legitimately create them.
11. **Cloud models do not get grammar-constrained decoding.** The OpenAI-compatible path sends `response_format={"type": "json_object"}`; the Anthropic Messages path has neither a grammar nor a JSON-mode switch. Both rely on the system prompt plus validation; a model that ignores the schema costs a repair round or an attempt. Ollama's `format` grammar remains the strongest guarantee.
12. **Cloud authentication is header-based and single-credential.** `openai` sends `Authorization: Bearer`, `anthropic` sends `x-api-key` plus `anthropic-version`. Base URLs with embedded credentials or query strings are rejected (the token belongs in `api_key`), redirects are not followed, and endpoints that need an API version in the query string (Azure-style) are not supported out of the box.
13. **Provider flags are CLI-only.** The MCP tools keep the local Ollama defaults; a cloud provider is configured through `moray-surgeon refactor`.

## License and contact

This project is source-available under the **PolyForm Noncommercial License 1.0.0**: free for personal, research, educational and other noncommercial use. It is not an OSI-approved open-source license, because noncommercial-only terms restrict a field of use.

**This project is for non-commercial use only. For commercial licensing, please contact: 848908615@qq.com**

Commercial use — including internal business tooling, paid services, or distribution as part of a commercial product — requires a separate written license. See [LICENSE](./LICENSE) for the binding text.

---

<div align="center">

**If a model never writes your code, it can never break your code.**

</div>
