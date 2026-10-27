<div align="center">

# moray-surgeon

**AST-based zero-hallucination code refactoring engine.**

*The model writes no code. It writes a plan. `libcst` does the surgery.*

[![Tests](https://img.shields.io/badge/Tests-188%20passed-brightgreen)](#engineering-standards)
[![Coverage](https://img.shields.io/badge/Coverage-94%25-brightgreen)](#engineering-standards)
[![Ruff](https://img.shields.io/badge/Ruff-0%20warnings-blue)](#engineering-standards)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](#quick-start)
[![MCP](https://img.shields.io/badge/MCP-Ready-6f42c1)](#mcp-integration)
[![License](https://img.shields.io/badge/License-PolyForm%20NC%201.0.0-orange)](./LICENSE)

[Why](#why-this) · [Features](#features) · [Quick start](#quick-start) · [MCP](#mcp-integration) · [Architecture](./docs/ARCHITECTURE.md) · [Standards](#engineering-standards) · [License](#license-and-contact)

</div>

---

## Why this?

Ask a general-purpose coding agent to rename one function and it usually rewrites the whole file. What comes back compiles often enough to feel safe: new formatting, a dropped import, an unrelated line quietly changed. Tests catch some of it. Code review catches the rest. The trust budget does not survive the third time.

moray-surgeon removes the model from the editing path. The LLM compiles natural language into a JSON plan of five typed operations; a deterministic AST engine applies it. The model never returns a file body, a statement suite or an import — the only expression-shaped field is `replace_call.new_call`, which must parse as a single Python expression and can only replace a callee. A hallucinated file has no slot to arrive in.

That is what *zero-hallucination* means here: the model cannot invent code structure. It can still choose the wrong operation — which is why every attempt ends at the verification gates, and a red gate rolls back.

| | Chat-driven editing | moray-surgeon |
|---|---|---|
| What the model emits | source code | a JSON `RefactorPlan` |
| What touches your files | the model | `libcst` + Jedi |
| Formatting and comments | regenerated | preserved outside the edited node |
| Invalid output | looks plausible | rejected by schema or by verification |
| Recovery | manual `git restore` | automatic reset to `HEAD` + retry |

The plan schema is closed (`extra="forbid"`, frozen models) and is also handed to Ollama as the decoding grammar, so even malformed JSON cannot reach the engine.

<details>
<summary><b>中文速览（30 秒）</b></summary>

moray-surgeon 是一个「零幻觉」代码重构引擎：大模型只负责把自然语言需求编译成 **JSON 指令**，真正改代码的是 `libcst` 语法树手术刀。它不会删你的 import、不会改你的引号、不会重排你的格式；每一轮修改都在 Git 事务里执行，`ruff` / `mypy` / `pytest` 任一失败立即回滚，并把 diff 和报错喂回模型重试。

适合：大规模符号重命名、函数抽取、批量调用替换、补类型注解。不适合：从零生成模块、需要新增 import 的重构（引擎从不改 import）。

</details>

## Features

- 🛡️ **Zero-hallucination, schema-driven.** The model can only emit one of five instructions, validated against a frozen Pydantic model; the same JSON Schema is sent as Ollama's `format`, constraining decoding rather than just checking afterwards.
- 🔪 **AST precision with `libcst`.** Edits are CST node transformations, not string replacement or `ast.unparse`. Comments, blank lines, quote style, indent width and untouched expressions round-trip byte-for-byte.
- 🕵️‍♂️ **Jedi cross-file resolution.** `rename_symbol` resolves references at project scope, so renaming a function updates its callers across files without touching same-named attributes or keyword arguments — when Jedi can analyse the project.
- 🛑 **Git transaction and rollback.** Every attempt enters a transaction: uncommitted work is snapshotted with `git stash -u`, the refactor is committed only on a green verification, and any failure runs `git reset --hard HEAD` plus `git clean -fd`, then restores your snapshot.
- 🔄 **Auto-fix loop.** On a red gate the worktree diff *and* the tool output are captured before rollback and fed into the next plan request. The next attempt re-plans from a clean tree, up to `--max-attempts` rounds (default 3).
- 🔌 **MCP ready.** A stdio MCP server exposes the engine to Cursor, Claude Desktop and any MCP host: list actions, execute a structured plan, or run the full natural-language loop.
- 🧪 **Dry run.** `--dry-run` applies the plan inside the transaction, prints a syntax-highlighted unified diff, and rolls back without verifying or committing.

### Instruction set

| Action | Effect | Scope |
|---|---|---|
| `rename_symbol` | Rename a symbol and every Jedi-resolved reference | Project-wide, cross-file |
| `extract_function` | Move a line range into a new parameterless function and insert the call | Module top level only |
| `replace_call` | Swap the callee of matching calls | Arguments and layout preserved |
| `add_type_hints` | Add a missing `-> None` return annotation | Conservative by design |
| `noop` | Explicitly declare that no edit is needed | Retried with feedback if the whole plan is `noop` |

## Quick start

**Requirements:** Python 3.11+, Git, and a local [Ollama](https://ollama.com/) instance. Nothing leaves your machine.

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

> The engine never edits imports. The example above produces `logging.info(...)` calls, so `import logging` must already exist. If it is missing, expect a red gate and a rollback — that is the intended contract, not a workaround.

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
| `--model` | `qwen2.5-coder:7b` | Ollama model tag |
| `--ollama` | `http://localhost:11434` | Ollama base URL (http/https only) |
| `--max-attempts` | `3` | Plan → apply → verify rounds |
| `--context` | empty | Extra context appended to the prompt |
| `--files` | auto-scan, max 50 | Comma-separated, repo-relative files for the code map |
| `--mypy-target` | `src` | Path passed to `mypy` |
| `--pytest-args` | `-q` | Arguments passed to `pytest` |
| `--dry-run` | off | Show the diff and roll back without committing |

Exit codes: `0` success, `1` refactor or verification failure, `2` not a Git repository.

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

## Engineering standards

| Gate | Result |
|---|---|
| Test suite | **188 passed** |
| Coverage | **94.56%** (enforced gate: 90%) |
| Lint | **Ruff: 0 warnings** |
| Runtime | Python 3.11+ (verified on CPython 3.12.2) |

The suite is fully offline: Ollama calls are mocked at the `httpx.Client.post` boundary, while Git, `ruff`, `mypy` and `pytest` run for real in temporary repositories. Reproduce the numbers locally:

```bash
ruff check .
pytest -q
pytest --cov
```

Safety is part of the test surface, not an afterthought: path containment is checked after `resolve()`, every external command has a timeout, failed transactions restore the user's stash, and rollback errors are raised rather than swallowed. See [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) for the invariants behind each module.

## Known limitations

1. **No import editing.** The engine never adds or removes imports. A refactor that requires a new import fails verification and is rolled back.
2. **`rename_symbol` is best-effort outside Jedi's view.** If Jedi cannot analyse a file, the atom falls back to a single-file textual rename and logs a warning.
3. **`extract_function` is top-level only.** Nested scopes are not extracted, and a compound statement between the selected lines makes the atom refuse rather than silently reorder code.
4. **`add_type_hints` only writes `-> None`.** Parameter hints and richer return types are not guessed: a wrong hint is worse than a missing one.
5. **Auto-fix is bounded.** Each round gets a fresh transaction; after `--max-attempts` failures the tree stays at `HEAD` and the last verification feedback is reported.
6. **A Git repository with a `HEAD` commit is required.** Without one there is no rollback baseline, and the transaction refuses to start.

## License and contact

This project is source-available under the **PolyForm Noncommercial License 1.0.0**: free for personal, research, educational and other noncommercial use. It is not an OSI-approved open-source license, because noncommercial-only terms restrict a field of use.

**This project is for non-commercial use only. For commercial licensing, please contact: 848908615@qq.com**

Commercial use — including internal business tooling, paid services, or distribution as part of a commercial product — requires a separate written license. See [LICENSE](./LICENSE) for the binding text.

---

<div align="center">

**If a model never writes your code, it can never break your code.**

</div>
