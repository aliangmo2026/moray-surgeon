<div align="center">

# moray-surgeon

**AST-based zero-hallucination code refactoring engine.**

*The model writes no code. It writes a plan. `libcst` does the surgery.*

[![Tests](https://img.shields.io/badge/Tests-780%20passed-brightgreen)](#engineering-standards)
[![Coverage](https://img.shields.io/badge/Coverage-94%25-brightgreen)](#engineering-standards)
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

moray-surgeon removes the model from the editing path. The LLM compiles natural language into a JSON plan of thirteen typed operations; a deterministic AST engine applies it. The model never returns a file body or a statement suite: even imports are expressed as a module path plus identifiers, and the only expression-shaped field is `replace_call.new_call`, which must parse as a single Python expression and can only replace a callee. A hallucinated file has no slot to arrive in.

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

- 🛡️ **Zero-hallucination, schema-driven.** The model can only emit one of the thirteen instructions below, validated against a frozen Pydantic model; on the Ollama path the same JSON Schema is sent as the `format` decoding grammar, constraining generation rather than just checking afterwards.
- 🔪 **AST precision with `libcst`.** Edits are CST node transformations, not string replacement or `ast.unparse`. Comments, blank lines, quote style, indent width and untouched expressions round-trip byte-for-byte.
- 🔗 **Dependency-aware plans (DAG).** Every instruction carries a unique `id` and may list `depends_on`. Before the first file is read, `execute_plan` runs Kahn's algorithm over the plan: rename a class before rewriting its callers, add the import before the call that needs it. A plan with no edges keeps v0.2.0's authored-order semantics; duplicate ids, dangling references and cycles are rejected before any edit.
- 🕵️‍♂️ **Jedi cross-file resolution.** `rename_symbol` resolves references at project scope, so renaming a function updates its callers across files without touching same-named attributes or keyword arguments. A local [SQLite code index](#large-project-performance-the-code-index) narrows the search to the files that mention the symbol, so a rename on a 2,000-module project costs ~0.2 s instead of ~23 s — without weakening Jedi's verdict on a single site.
- 🛑 **Git transaction and rollback.** Every attempt enters a transaction: uncommitted work is snapshotted with `git stash -u`, the refactor is committed only on a green verification, and any failure runs `git reset --hard HEAD` plus `git clean -fd`, then restores your snapshot.
- 🧭 **AST scope guard (the second safety net).** Before the plan runs, every Python file is fingerprinted (per-class, per-function, per-module-level statement). After it runs, every changed node must be explained by the plan — a renamed symbol, a Jedi reference site, or a file-global atom. Anything else raises `ScopeViolationError`, rolls the attempt back and feeds the offending node names to the next prompt.
- 📦 **Controlled imports.** `add_import` and `remove_import` edit the import block through typed fields (module path + identifier), never free-form source. Imports are merged into the right statement, sorted with isort's `order-by-type` rule, wrapped when a line would exceed 88 characters, and never reorder unrelated statements.
- 🔄 **Auto-fix loop.** On a red gate (behavioural *or* structural) the worktree diff and the diagnostics are captured before rollback and fed into the next plan request. The next attempt re-plans from a clean tree, up to `--max-attempts` rounds (default 3).
- 🧠 **Bring your own model.** Run against a local Ollama daemon or any OpenAI-compatible or Anthropic-compatible cloud API (DeepSeek, Xiaomi Token Plan, Claude, vLLM, LM Studio, ...). Only the wire format changes: validation, repair, transactions and both gates are identical. The API key comes from `MORAY_API_KEY` (or `--api-key`), is never logged or echoed, and is never forwarded across a redirect.
- 🔌 **MCP ready.** A stdio MCP server exposes the engine to Cursor, Claude Desktop and any MCP host: list actions, execute a structured plan, or run the full natural-language loop.
- 🖱️ **LSP code actions.** `moray-surgeon lsp` is a Language Server: extract, inline, match-case conversion, type hints and dead-code removal show up as editor refactor actions, plus an AI-driven action for everything else. The result arrives as a `WorkspaceEdit` you review and save — the engine applies to disk and restores it, so nothing is committed behind your back.
- ⚖️ **Differential testing.** After pytest passes, every function whose body changed is executed before and after the refactor inside an in-process sandbox (restricted builtins, import allowlist, opcode + wall-clock watchdog) on identical inputs. A proven behavioural difference rolls the attempt back; functions the sandbox cannot verify are logged as skips, never silent passes.
- 🧪 **Dry run.** `--dry-run` applies the plan inside the transaction, prints a syntax-highlighted unified diff, and rolls back without verifying or committing.
- 🚦 **CI gateway.** `moray-surgeon audit` is the read-only counterpart of the refactor loop: it diffs the working tree against a base ref, re-fingerprints every changed Python file, applies the same scope gate when a plan is supplied, and runs `ruff`/`mypy`/`pytest` — then exits non-zero with a JSON report. No edit, no commit, no stash.
- 💥 **Proven under load.** An 8,046-line, 300-class synthetic project survived a 50-instruction single-transaction refactor; two poisoned-instruction runs rolled back in under 0.1 s. [Full report with diffs and timings](./docs/STRESS_TEST_REPORT.md).
- 🏛️ **Architecture before/after.** On success the orchestrator takes two moray-scribe symbol-graph snapshots (before the plan, after the gates) and stores a structural comparison in `RefactorReport.architecture_diff` — extracted classes, functions that moved across modules, removed symbols. The CLI prints a one-line 架构变更 summary with the detail. Requires the local moray-scribe checkout (`pip install -e ../moray-scribe` or the `scribe` extra); without it the feature degrades to an empty diff and never blocks a refactor.

### Instruction set

| Action | Effect | Scope |
|---|---|---|
| `rename_symbol` | Rename a symbol and every Jedi-resolved reference | Project-wide, cross-file |
| `extract_function` | Move a line range into a new parameterless function and insert the call | Module top level only |
| `extract_variable` | Hoist the expression at an exact source range into `name = <expr>` | Single file; refuses binding positions and scope/iteration boundaries |
| `inline_variable` | Replace every read of an SSA variable with its value and delete the assignment | Single file; refuses reassignment, closures, cross-file reads |
| `inline_function` | Inline a single-return function into every call site and delete it | Same file only; refuses recursion and duplicated side effects |
| `split_class` | Move receiver-free methods into a new sibling class | Same file; refuses `self`/`cls`/`super()` methods |
| `remove_dead_code` | Delete an unreferenced top-level function or class | Project-wide proof; refuses `__all__`, decorators, dynamic strings |
| `move_symbol` | Move a top-level function/class to another file and rewrite imports | Cross-file; refuses ambiguous or unsafe imports |
| `replace_call` | Swap the callee of matching calls | Arguments and layout preserved |
| `add_type_hints` | Add a missing `-> None` return annotation | Conservative by design |
| `add_import` | Add `import x` / `from x import y`, sorted into the import block | Module level, absolute modules |
| `remove_import` | Drop an import, or one name from a from-import | Module level, absolute modules |
| `convert_to_match_case` | Rewrite an if/elif/else chain into a `match` statement | Python 3.10+; refuses `and`/`not`, non-`==` tests and subjects a branch rebinds |
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

## Large-project performance: the code index

`rename_symbol` and `move_symbol` are the only project-aware atoms. Before the index existed they asked Jedi for *project-scope* references with the per-search file budget raised to the size of the repository (raising it is what keeps the search complete past Jedi's 30-file default). Correct, but the cost grew with the repository: on a 2,000-module project a single rename spent around half a minute locating references.

The engine now keeps a symbol table in SQLite and asks it the only question Jedi cannot answer cheaply — *which files mention this identifier at all?*

```text
<repo>/.moray-cache/index.sqlite3     name, kind, scope chain, file, line for every identifier
```

For each rename the atom now:

1. asks the index for the files that contain the symbol;
2. parses only those files with libcst to get exact positions;
3. asks Jedi to resolve each position (`goto(follow_imports=True)`) and keeps it only when it lands on one of the symbol's definitions;
4. rewrites with libcst, then re-indexes the files it wrote.

Jedi still decides every edit. The index proposes candidates only, so a stale row can drop a candidate — never rewrite a symbol it does not own. The rewrite set is identical to what the old project-wide search produced: everything that resolves to one of the target's definitions is rewritten, `obj.Symbol` attribute lookups and `kwarg=Symbol` keyword arguments are left alone, and a renamed import alias does not drag the original definition along.

The atom also chooses its strategy with the index: a symbol concentrated in a minority of the files is verified position by position (the common case, and the one the index makes cheap), while a symbol spread across most of the repository is resolved by one project-wide search, which shares its inference instead of paying it per occurrence. On a project whose modules carry rich import graphs the per-position path was measured 3.8× slower once a symbol appeared in 53 of 59 files; the cutoff keeps that case on the proven path.

### Measured

`python scripts/bench_index.py` builds a fresh repository per measurement (Jedi caches parsed modules per process, so timing twice on one tree measures the cache, not the algorithm) and times a **complete rename** on both sides, so the only difference is how the references are found. Synthetic modules on Windows 11 / CPython 3.12.2:

| Modules | Files | Before index | Index build (once) | Indexed rename | Speedup |
|---|---|---|---|---|---|
| 200 | 204 | 3.43 s | 0.87 s | **0.38 s** | 9.0× |
| 800 | 804 | 14.00 s | 7.95 s | **0.45 s** | 31.5× |
| 2,000 | 2,004 | 30.78 s | 24.78 s | **0.37 s** | 83.5× |

The indexed number is flat because it only touches the handful of files that mention the symbol; the build is paid once per repository and amortised over every later rename. The build figures include the cold filesystem and antivirus cost of writing a new database per temporary tree — the same tree refreshed again is a stat walk (~0.02 ms/file) plus one parse per changed file.

### Incremental refresh

The orchestrator refreshes the index **before** it opens a transaction, and again inside the transaction when the stash replaced the user's worktree (the stash reverts files, so the index built a moment earlier describes content that is no longer on disk). A refresh:

- stats every Python file (≈0.02 ms/file) and re-parses only files whose `mtime_ns` or size moved;
- hashes the file when the mtime moved, so a `touch` that leaves the content identical updates the row without a parse;
- drops the rows of deleted files;
- re-indexes exactly the files the engine itself wrote (`Indexer.reindex`), so the next instruction in the same plan sees the new symbols.

Symbol extraction is a read-only pass over the standard library `ast` (0.5 ms/file here, against 23.4 ms/file for a libcst `PositionProvider` pass). libcst remains the only thing that produces bytes: every position and every rewrite still comes from libcst, which is also why the index stores no columns — positions are recomputed on candidate files with libcst and therefore cannot drift from the ones being rewritten.

### When the cache is unavailable

The index is an accelerator, never a dependency:

- a read-only checkout, a locked database or a cache path that cannot be created makes the atoms fall back to the project-wide Jedi scan;
- the orchestrator then **suspends** the index for that run instead of rebuilding it inside the transaction — recreating `.moray-cache/` while a stash holds it would make `git stash apply` refuse to restore the user's files;
- `.moray-cache/` is added to `.git/info/exclude` (not `.gitignore`) the first time it is created, so it is never committed, never stashed and never removed by `git clean -fd`.

### Managing the cache

```bash
moray-surgeon refactor "..." --path . --clear-cache   # rebuild from scratch before running
moray-surgeon cache --path . --stats                  # files, symbols, database size
moray-surgeon cache --path . --clear-cache            # delete it; the next refactor rebuilds it
```

Nothing is lost by clearing: every row is derived from the sources.

### Parallel execution across cores

libcst parsing and rewriting are CPU-bound functions of one file's text, so a plan that edits many files does not have to run on one core. The dispatcher batches **consecutive, file-disjoint, dependency-free instructions whose atoms touch nothing but their own target** — `add_import`, `add_type_hints`, `convert_to_match_case`, `extract_function`, `extract_variable`, `remove_import`, `replace_call`, `split_class` — and applies them in a `ProcessPoolExecutor`. Project-aware atoms (`rename_symbol`, `move_symbol`, `inline_variable`, `inline_function`, `remove_dead_code`) read the rest of the project to prove safety, so they run in the parent as barriers; a worker would either see a stale tree or race the parent's writes.

Workers never write. The parent collects every result, rejects a batch if any instruction changed nothing, and only then writes the files — each through a temp-file-plus-rename — so a failing worker cancels the rest and leaves the whole batch untouched (`ConcurrencyError`, rolled back by the transaction like any other engine error).

The pool size is `MORAY_MAX_WORKERS`, defaulting to `min(cpu_count, 8)`; `1` disables the pool and every run stays in-process.

```bash
python scripts/bench_concurrency.py --files 10 --functions 1200 --repeat 2
```

Measured on Windows 11 / CPython 3.12.2 / 16 logical cores, 10 modules × 4,800 lines:

| Run | Wall time | Speedup |
|---|---|---|
| sequential (`max_workers=1`) | 5.79 s | 1.00× |
| process pool (8 workers) | **2.83 s** | **2.04×** |

Both runs produce byte-identical files (the script checks this before reporting). Process startup is not free: on ten 240-line modules the same pool is ~3× *slower* than one process (0.39 s → 1.34 s), because spawn costs more than the whole refactor. Parallelism pays off once per-file work dominates that fixed cost.

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
| `--clear-cache` | off | Delete the [code index](#large-project-performance-the-code-index) and rebuild it before running |

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

## LSP integration (editors)

`moray-surgeon lsp` runs a [Language Server](https://microsoft.github.io/language-server-protocol/) over stdio. Instead of a chat window, refactors appear as **code actions**: select code (or place the cursor), open the *Refactor...* menu, and pick a moray-surgeon action.

### What it offers

| Action | Trigger | Engine instruction |
|---|---|---|
| Extract selection to function | multi-line selection | `extract_function` |
| Extract expression to variable | any selection | `extract_variable` |
| Convert if/elif chain to match-case | cursor on an `if`/`elif`/`else` header | `convert_to_match_case` |
| Inline variable / Inline function | cursor on the name | `inline_variable` / `inline_function` |
| Remove dead code | cursor on the `def`/`class` line | `remove_dead_code` |
| Add type hints | cursor on a `def` line, or file-wide | `add_type_hints` |
| Refactor with AI (moray-surgeon) | always available | natural language → `IntentParser` |

**Preview-first architecture:** when you pick an action, the engine applies the plan to the files on disk, captures the result, and immediately restores the original bytes. What you receive is a `WorkspaceEdit` applied to your editor buffers — you review it, edit it, and save (or undo) yourself. No git state is touched, nothing is committed, and `execute_plan`'s refusal paths (unprovable match-case, non-SSA inlines, dead code with live references) become editor messages instead of rollbacks.

Two safety rules follow from the engine reading files from disk:

1. If an open Python file inside the project differs from its saved copy, the action is refused with a *save your files first* message.
2. The AI action needs a reachable model: a local Ollama daemon (`ollama serve`, `ollama pull qwen2.5-coder:7b`) or a cloud API configured at startup. When it cannot connect you get a `window/showMessage` with setup steps instead of a broken edit.

### Launching

```bash
pip install -e .          # installs pygls
moray-surgeon lsp         # stdio; logs go to ~/.moray-surgeon/lsp.log (override: MORAY_LSP_LOG)
```

`lsp` accepts the same provider flags as `refactor`: `--model`, `--base-url` / `--ollama`, `--api-type`, `--api-key` (with `MORAY_API_KEY` fallback). These configure the *AI action* only; the direct AST actions never touch a model.

### VSCode

VSCode has no built-in "attach arbitrary LSP server" switch. Two working paths:

**A. Generic LSP Proxy extension (no code).** Install [*Generic LSP Proxy*](https://marketplace.visualstudio.com/items?itemName=mjmorales.generic-lsp-proxy) and create `.vscode/lsp-proxy.json` in your project:

```json
[
  {
    "languageId": "python",
    "command": "moray-surgeon",
    "args": ["lsp"],
    "fileExtensions": [".py"]
  }
]
```

If `moray-surgeon` is not on VSCode's `PATH`, point `command` at the venv script (`.venv/Scripts/moray-surgeon.exe` on Windows, `.venv/bin/moray-surgeon` elsewhere). Pylance keeps running; VSCode merges code actions from every attached server, so moray-surgeon's entries appear under *Refactor...* next to Pylance's suggestions.

**B. A 20-line companion extension.** The official route is [`vscode-languageclient`](https://code.visualstudio.com/api/language-extensions/language-server-extension-guide): `ServerOptions` = `{ command: "moray-surgeon", args: ["lsp"] }`, `documentSelector` = `[{ language: "python" }]` — and the client handles initialize/didOpen/resolve for you.

### Other editors (built-in clients)

Neovim ≥ 0.8 (`~/.config/nvim/lua/lsp/moray.lua`, then add `require('lsp.moray')`):

```lua
local configs = require('lspconfig.configs')
configs.moray_surgeon = {
  default_config = {
    cmd = { 'moray-surgeon', 'lsp' },
    filetypes = { 'python' },
    root_dir = function(fname)
      return require('lspconfig.util').find_git_ancestor(fname)
    end,
  },
}
require('lspconfig').moray_surgeon.setup({})
```

Helix (`~/.config/helix/languages.toml`):

```toml
[[language]]
name = "python"
language-servers = ["pyright", "moray-surgeon"]

[language-server.moray-surgeon]
command = "moray-surgeon"
args = ["lsp"]
```

## Web dashboard

Prefer pointing and clicking to typing flags? The built-in console serves a single-page, dark-mode UI on `http://127.0.0.1:8080`:

```bash
moray-surgeon web                 # opens the browser automatically
moray-surgeon web --port 9000 --no-open
```

Enter a git repository path and the intent in plain language, press **Execute**, and watch the pipeline live over Server-Sent Events — plan request, AST surgery, ruff/mypy/pytest gates — with the final diff, architecture change and plan JSON rendered in tabs. The engine is the same orchestrator the CLI drives: one run at a time, full rollback on any red gate. `POST /api/refactor` is also a plain JSON API, so the console doubles as a local REST example.

<!-- TODO: drop a demo GIF here (e.g. docs/assets/web-dashboard.gif) -->

## GitHub Issue bot (autopilot pull requests)

`moray-surgeon bot` turns an issue into a verified branch without a local checkout. Triggered by `.github/workflows/moray-bot.yml` on `issues.opened`, it:

1. reads the issue title and body through the GitHub API and uses them as the refactor intent;
2. clones the repository into a temp directory and resets `moray-bot/issue-<N>` onto the default branch;
3. runs the same orchestrator as the CLI (plan, AST apply, ruff/mypy/pytest, architecture diff);
4. pushes the branch and opens a pull request whose body says `Closes #N`.

If any gate fails, nothing is pushed: the bot comments the failing `VerifyResult.feedback()` on the issue and applies the `needs-human` label (add `--close-on-failure` or `MORAY_BOT_CLOSE_ISSUE=1` to close it too).

Copy [docs/github_action_example.yml](docs/github_action_example.yml) to `.github/workflows/moray-bot.yml`. It needs `contents: write`, `issues: write` and `pull-requests: write`, the `MORAY_API_KEY` secret, and the repository setting that lets Actions create pull requests. The workflow runs the repository's own tests with a write token, so it only fires for issues opened by an `OWNER`, `MEMBER` or `COLLABORATOR`; adjust that `if:` deliberately, never remove it.

Configuration is environment-only: `GITHUB_TOKEN` (required), `GITHUB_API_URL` / `GITHUB_SERVER_URL` for GitHub Enterprise, and the usual `MORAY_API_TYPE` / `MORAY_MODEL` / `MORAY_BASE_URL` / `MORAY_API_KEY`. `MORAY_BOT_MAX_ATTEMPTS`, `MORAY_BOT_BASE_BRANCH`, `MORAY_BOT_BRANCH_PREFIX`, `MORAY_BOT_FAILURE_LABEL`, `MORAY_BOT_CLOSE_ISSUE`, `MORAY_BOT_GIT_NAME` and `MORAY_BOT_GIT_EMAIL` tune the bot itself. The token reaches git through `GIT_CONFIG_*` environment variables, never a remote URL or a log line, and GitHub endpoints must be https with a public host.

```bash
GITHUB_TOKEN=ghp_... moray-surgeon bot --issue 42 --repo acme/widgets
```

Two operational notes: commits pushed with the built-in `GITHUB_TOKEN` do not trigger other workflows (use a fine-grained PAT if the bot PR must run CI), and the bot branch is pushed with `--force-with-lease`, so a human commit pushed after the clone is never overwritten.

## Deploy as a Cloud Skill

Platforms like WorkBuddy accept a Serverless code package instead of a long-running service. `scripts/build_skill.py` builds one:

```bash
python scripts/build_skill.py
```

It stages a temporary directory (the `moray_surgeon/` package without `tests/` or `__pycache__`, a `requirements.txt` with the runtime dependencies, `pyproject.toml`/`LICENSE`/`README.md`, a `handler.py` shim, and the `SKILL.md` manifest the platform requires at the archive root), zips it into `moray-skill.zip` at the repository root, and fails with a non-zero exit code if the archive exceeds the 3 MB upload limit.

Upload `moray-skill.zip` to WorkBuddy's skill/Serverless page and configure the handler as `handler.handler` (the shim re-exports `moray_surgeon.cloud_entry.handler`, so any platform that expects `handler.py` at the package root works out of the box).

The handler accepts a JSON payload — either directly or inside an API-Gateway-style event `body` (base64 supported):

```json
{"action": "refactor", "repo_path": "/sandbox/repo", "intent": "把 parse_args 重命名为 parse_cli_options"}
```

| Field | Action | Meaning |
|---|---|---|
| `action` | both | `refactor` or `audit` |
| `repo_path` | both | Git repository root reachable **from the sandbox** |
| `intent` | `refactor` | Natural-language request |
| `dry_run` | `refactor` | Capture the diff and roll back instead of committing |
| `max_attempts` | `refactor` | Auto-fix rounds (default 3) |
| `base_ref` | `audit` | Git revision to diff against (default `HEAD`) |

Responses are always JSON: `{"statusCode", "headers", "body"}`, with `body.ok` / `body.error`. An unreachable `repo_path` (a local path on the caller's machine, a missing directory, or a non-Git folder) is caught and returned as a friendly `400` explaining that the path must be reachable inside the cloud sandbox — never as a crash.

The model endpoint is configured only through deployment environment variables, never through the request payload (so an untrusted request cannot aim the server at an arbitrary URL):

| Variable | Default | Meaning |
|---|---|---|
| `MORAY_API_TYPE` | `ollama` | `ollama`, `openai` or `anthropic` dialect |
| `MORAY_MODEL` | `qwen2.5-coder:7b` | Model name |
| `MORAY_BASE_URL` | `http://localhost:11434` | API root (http/https only; credentials and query strings are rejected) |
| `MORAY_API_KEY` | unset | Bearer/x-api-key credential for the cloud dialects |
| `MORAY_TIMEOUT` | `120` | Per-request timeout in seconds |

## Runtime-equivalence gate (differential testing)

pytest proves the suite still passes; it cannot prove uncovered edge cases still behave the same. After a green verification, the orchestrator runs one more gate: every function whose body changed is executed **before and after the refactor, on identical inputs**, and return values, printed output and raised exceptions are compared.

The check runs entirely in-process — no external process is started. Four walls make arbitrary code safe to execute here:

| Wall | Mechanism |
|---|---|
| Worker thread | Each sandbox run executes in its own daemon thread; the caller's deadline is honoured even if a worker hangs |
| Restricted builtins | `exec` receives an allowlist: no `open`, `eval`, `exec`, `getattr`/`setattr`, `globals`; `print` becomes a capturing implementation, so output is comparable data; `__build_class__` keeps normal class definitions working |
| Import allowlist | Only pure stdlib computation helpers (`math`, `itertools`, `functools`, `collections`, `re`, `json`, ...) import; everything else fails inside the sandbox and turns the run into a documented skip |
| Trace watchdog | A `sys.settrace` handler counts opcodes of every sandbox frame and enforces a wall-clock deadline; the kill switch derives from `BaseException` so sandbox-level `except Exception` cannot swallow it, and CPython runs trace handlers with tracing disabled so the sandboxed code can neither intercept nor disable it |

A skip is not a verdict. Functions that declare `global`/`nonlocal`, yield, await, import at call time, or reference I/O/randomness names (`os`, `open`, `random`, `time`, ...) are refused statically with a precise reason; a `NameError` raised by **both** versions is also a skip, because inside the sandbox it usually means a needed builtin was withheld — both sides failing identically proves nothing. What remains is a strong, bounded claim: *these two versions returned the same values, printed the same output and raised the same exceptions on the same inputs*.

The public surface is small:

```python
from moray_surgeon.verify import verify_equivalence, compare_functions, changed_function_names

verify_equivalence(original_source, new_source, "add")            # True / False
verify_equivalence(original_source, new_source, "add", 2, 3)      # explicit inputs
compare_functions(...)                                            # tri-state result with details
changed_function_names(before_source, after_source)               # qualnames whose code changed
```

Disable it per run with `refactor --no-differential` (or `Orchestrator(differential=False)`); skips and mismatches are visible in the trace (`Differential: skipped 'f' (...)` / `Differential mismatch in 'f'!`).

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
| Test suite | **780 passed** |
| Coverage | **94%** (enforced gate: 90%) |
| Lint | **Ruff: 0 warnings** |
| Stress test | **PASS** — 50 instructions on 8,046 lines / 300 classes, 54 files changed |
| Runtime | Python 3.11+ (verified on CPython 3.12.2) |

The suite is fully offline: model calls (Ollama and OpenAI-compatible) are mocked at the `httpx.Client.post` boundary, while Git, `ruff`, `mypy` and `pytest` run for real in temporary repositories. The stress harness is intentionally *not* part of `pytest -q` (it takes tens of seconds and generates a project); run it explicitly.

Safety is part of the test surface, not an afterthought: path containment is checked after `resolve()`, every external command has a timeout, failed transactions restore the user's stash, both gates capture their evidence before rollback, and rollback errors are raised rather than swallowed. See [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) for the invariants behind each module.

## Known limitations

1. **Controlled imports have a deliberate scope.** `add_import`/`remove_import` only touch top-level statements; imports inside `try:` blocks or functions are left alone. `module` must be an absolute dotted path (relative imports are never generated or merged). Existing statements are never reordered — only a merged alias list is sorted. A rename that unsorts a from-import is the project's isort gate's concern, not a silent engine reformat.
2. **`rename_symbol` is best-effort outside Jedi's view.** Candidates come from the local [code index](#large-project-performance-the-code-index) and every one of them is resolved by Jedi before it is rewritten; if the index cannot be used the atom falls back to Jedi's project-wide search, and if Jedi itself cannot analyse the project it falls back to a single-file textual rename and logs a warning.
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
14. **The code index is a candidate oracle, not a compiler.** It records identifiers per file, not types or bindings: a symbol that only appears inside a string or comment is (correctly) never a candidate, and keyword arguments are excluded on purpose because Jedi does not treat them as references to the parameter. Correctness never depends on the index being right — it only changes which files Jedi is asked about, and an unusable cache degrades to a full scan.
15. **Line ranges in a plan are absolute.** `convert_to_match_case`, `extract_function` and `extract_variable` address the file as it exists when their step runs, so two range-based steps in the *same* file interact: converting a chain near the top shifts the lines of everything below it. Order such steps bottom-up (or `depends_on` them across files) when a plan touches two ranges in one file. `convert_to_match_case` refuses a range that no longer covers a whole chain instead of guessing which one was meant.

## License and contact

This project is source-available under the **PolyForm Noncommercial License 1.0.0**: free for personal, research, educational and other noncommercial use. It is not an OSI-approved open-source license, because noncommercial-only terms restrict a field of use.

**This project is for non-commercial use only. For commercial licensing, please contact: 848908615@qq.com**

Commercial use — including internal business tooling, paid services, or distribution as part of a commercial product — requires a separate written license. See [LICENSE](./LICENSE) for the binding text.

---

<div align="center">

**If a model never writes your code, it can never break your code.**

</div>
