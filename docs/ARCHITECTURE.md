# Architecture

moray-surgeon is a one-way pipeline with one hard boundary: **the model plans, the engine mutates**. Nothing the model produces is ever written to disk directly, and nothing the engine writes survives a failed gate.

Two independent gates protect every attempt. The **verification gate** asks whether the project still works (`ruff` → `mypy` → `pytest`). The **AST scope gate** asks whether the engine changed anything the plan did not authorise. A plan must satisfy both; either failure rolls the whole attempt back.

The design has six layers:

1. **Control plane** — CLI and MCP entry points. `refactor` is transactional; `audit` is read-only and runs the same gates over a diff.
2. **Compilation** — intent → validated `RefactorPlan` (the only place a model is consulted, through a pluggable Ollama, OpenAI-compatible or Anthropic-compatible transport).
3. **Ordering** — `id`/`depends_on` → one deterministic execution sequence (Kahn's algorithm).
4. **Mutation** — Git transaction + deterministic AST edits.
5. **Structural evidence** — AST fingerprints: what changed, and was it in scope?
6. **Behavioural evidence** — verification gates: does the project still build and pass?

## Data flow

```text
        ┌─────────────────────────────────────────────────────────────┐
        │  ENTRY POINTS                                               │
        │  CLI   moray-surgeon refactor "..."    [Typer + Rich]       │
        │  MCP   moray-surgeon mcp              [stdio JSON-RPC]      │
        └─────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Orchestrator  ·  attempt 1 ... max_attempts                │
        │  injects the [Project Code Map] when given                  │
        └─────────────────────────────────────────────────────────────┘
                                       │ intent + context
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  IntentParser  ── POST chat endpoint ──▶ provider           │
        │  Ollama · OpenAI-compatible · Anthropic Messages            │
        │  schema in prompt · grammar/JSON mode · validate · repair   │
        └─────────────────────────────────────────────────────────────┘
                                       │ RefactorPlan (frozen JSON, extra=forbid)
                                       ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  TransactionManager            [git subprocess, 30 s]                       ║
║  enter: git stash -u   (snapshot uncommitted work)                          ║
║  exit : commit on green  |  reset --hard HEAD + clean -fd on any red gate   ║
║                                                                             ║
║   ┌───────────────────────────────────────────────────────────────────┐     ║
║   │ 1. ScopeGuard.snapshot_repository()                               │     ║
║   │    every .py file → {qualified node name → SHA-256 of its code}   │     ║
║   └───────────────────────────────────────────────────────────────────┘     ║
║                                  │                                          ║
║                                  ▼                                          ║
║   ┌───────────────────────────────────────────────────────────────────┐     ║
║   │ 2. Dag.topological_sort(plan) → execution order (Kahn)            │     ║
║   │    Dispatcher → AST engine (libcst, Jedi)                         │     ║
║   │    rename · extract · replace · type hints · add/remove import    │     ║
║   └───────────────────────────────────────────────────────────────────┘     ║
║                                  │ files written (UTF-8)                    ║
║                                  ▼                                          ║
║   ┌───────────────────────────────────────────────────────────────────┐     ║
║   │ 3. ScopeGuard.verify_repository_scope()      [STRUCTURAL GATE]    │     ║
║   │    re-fingerprint changed files, diff the maps                    │     ║
║   │    only nodes named by the plan, rename reference sites and       │     ║
║   │    file-global atoms may differ → otherwise ScopeViolationError   │     ║
║   └───────────────────────────────────────────────────────────────────┘     ║
║                                  │ violation                                ║
║                                  ├──────────────▶ capture diff, roll back   ║
║                                  ▼                                          ║
║   ┌───────────────────────────────────────────────────────────────────┐     ║
║   │ 4. Verifier                            [BEHAVIOURAL GATE]         │     ║
║   │    ruff → mypy → pytest   (stop at the first red gate)            │     ║
║   └───────────────────────────────────────────────────────────────────┘     ║
║                                  │ red gate                                 ║
║                                  ├──────────────▶ capture diff, roll back   ║
╚══════════════════════════════════╪══════════════════════════════════════════╝
                                   │ both gates green
                                   ▼
                    ┌───────────────────────────────┐
                    │  commit + RefactorReport      │
                    │  success = True               │
                    │  (dry run returns the diff)   │
                    └───────────────────────────────┘

     Failure path (either gate):
       diff + gate diagnostics ──▶ next plan prompt ──▶ next attempt from clean HEAD
```

The order is deliberate: the structural gate is cheap, deterministic and runs first, so a plan that silently rewrote an unrelated function is rejected before expensive tooling starts. Both gates capture the worktree diff **before** the exception propagates, so the rollback destroys nothing the next prompt needs.

## The attempt lifecycle

1. **Compose context.** The CLI scans the repository (or takes `--files`) and `generate_repo_map` emits one line per top-level class/function/method: signature, line number, first docstring line. Signatures are whitespace-collapsed and each file block is capped at 2,000 characters, so the prompt stays bounded. The map is prefixed with `【Project Code Map】` and reused on every retry.
2. **Parse.** `IntentParser` asks the configured provider for one chat turn. With `api_type="ollama"` the inlined JSON Schema travels twice: in the system prompt and as the `format` decoding grammar. With `api_type="openai"` the request goes to `{base_url}/chat/completions` with `response_format={"type": "json_object"}` and a bearer token. With `api_type="anthropic"` it goes to `{base_url}/v1/messages` with `x-api-key` + `anthropic-version`, the system prompt moved to the top-level `system` field, only `user`/`assistant` messages retained and `max_tokens=4096`; the answer is the first `type: "text"` block in `content`, because reasoning models prepend `thinking` blocks. Neither cloud dialect has a grammar field, so the schema is enforced by the prompt and validation instead. The response is sliced between its outermost braces, then validated as a `RefactorPlan`. Invalid model output gets exactly one repair round-trip; transport errors are never retried. No repository file is read at this stage.
3. **Snapshot.** The orchestrator opens a fresh `TransactionManager` (which stashes uncommitted work and refuses to continue without a `HEAD` commit), then fingerprints every Python file under the repository. The snapshot is taken **inside** the transaction, after the stash, so the baseline matches the tree the engine is about to edit.
4. **Apply.** `execute_plan` topologically sorts the instructions on `id`/`depends_on` first (Kahn's algorithm, ties broken by authored position), then runs them in that order; the first failure aborts the attempt. A plan made only of `noop` instructions is treated as a failed attempt with synthetic feedback, not as success. A duplicate id, a dangling dependency or a cycle raises before any file is read.
5. **Structural gate.** Every file whose bytes changed is re-parsed and its fingerprint map is diffed against the snapshot. A node is accepted only when an instruction explains it:
   - `rename_symbol`: the node is named by the rename, or its code contains the old/new name as an identifier (a Jedi-rewritten reference site), or it is a member of the renamed class.
   - `extract_function`: the new function, or the module-level block it moved.
   - `add_type_hints`: every function when no name is given, otherwise the named function.
   - `replace_call`: any node in the file it targets (the atom is file-global).
   - `add_import` / `remove_import`: import statement text is excluded from fingerprints entirely.
   - Anything else — including a created or deleted file — is a `ScopeViolationError`.
6. **Behavioural gate.** `ruff check .`, then `<python> -m mypy <target>`, then `<python> -m pytest <args>`, stopping at the first red gate. A missing tool, an invalid cwd or a timeout is returned as a failed `VerifyResult`; verification never raises.
7. **Settle.**
   - **Green:** the transaction commits (`stash apply` → `add -A` → `commit` → `stash drop`) and the report says `success=True`.
   - **Red (either gate):** the diff is captured before the exception reaches `__exit__`, which runs `reset --hard HEAD`, `clean -fd` and restores the stash. The diff and diagnostics become the next attempt's context.
   - **Dry run:** the diff is captured, `_DryRunComplete` forces a rollback, verification and commit are skipped, and the report returns `success=True` with the diff.

## Module responsibilities

| Module | Responsibility | Key invariant |
|---|---|---|
| `cli.py` | Typer entry points (`refactor`, `audit`, `mcp`, `version`); validates the repo root; renders plan, diff and report | Refuses to start outside a Git work tree (exit `2`); `audit` never writes |
| `audit.py` | Read-only CI gate: git change set → per-file fingerprints vs the base commit → optional scope gate → verification; JSON report | No edit, commit or stash; violations are collected, not raised |
| `mcp_server.py` | stdio MCP host with three tools; all failures become strings | stdout carries JSON-RPC only; logs go to a file |
| `orchestrator.py` | Attempt loop, repo-map injection, per-attempt transaction, scope gate, dry-run, diff capture | One transaction per attempt; both gates roll back through the same path |
| `intent/parser.py` | Model round-trip through the selected provider, fence stripping, schema + DAG validation, one repair round | Returns typed instructions, never a file body; a bad graph is retryable |
| `intent/providers.py` | Request/response dialects: Ollama native (`/api/chat`, schema grammar), OpenAI-compatible (`/chat/completions`, JSON mode, bearer auth) and Anthropic Messages (`/v1/messages`, top-level `system`, `x-api-key`) | The API key exists only in a request header — never in `repr`, logs or error messages; redirects are not followed |
| `schema/instructions.py` | Frozen, closed instruction models with `id`/`depends_on`; discriminated union; Ollama-safe schema inlining | `extra="forbid"`, `frozen=True`, no `$ref` survives inlining |
| `transaction/manager.py` | Git snapshot, commit and rollback | Failed transaction restores the exact starting state; rollback errors are raised |
| `engine/dag.py` | Kahn topological sort over `depends_on`; node-level graph validation | Stable order: ready ties follow authored position; invalid graphs never reach a file |
| `engine/dispatcher.py` | Dependency-ordered execution, path containment, error classification | Sort → resolve → containment → existence, in that order |
| `engine/rename.py` | Jedi project references → libcst position edits; textual fallback; temporary project-search budget | References outside `repo_root` are never rewritten |
| `engine/extract.py` | Module-level statements → new function + call site | Only CST body edits; never string slicing or re-parsing |
| `engine/replace_call.py` | Callee replacement | Arguments, comments and spacing on the `Call` node are preserved |
| `engine/type_hints.py` | Missing `-> None` annotations | Never guesses a richer type |
| `engine/import_manager.py` | Add/remove module-level imports, sorted and layout-preserving | Never reorders existing statements; no statement is deleted without keeping its comments |
| `engine/scope_guard.py` | Repository fingerprints, scope authorisation, created/deleted-file detection | Imports are excluded from fingerprints; every other node is compared |
| `engine/repo_map.py` | libcst-derived symbol map for prompt context | 2,000 chars per file, line-boundary truncation |
| `verify/runner.py` | ruff → mypy → pytest | Failures are data, never exceptions; explicit interpreter |
| `ui.py` | Rich plan table, diff view, live trace | Dynamic text is rendered with markup disabled |

## Instruction set

Seven atoms, each a frozen model in the discriminated union. Every atom inherits `id` (required, unique inside the plan) and `depends_on` (default empty) from the shared base; the dispatcher orders the batch on those fields before executing it.

| Action | Effect | Scope |
|---|---|---|
| `rename_symbol` | Rename a symbol and every Jedi-resolved reference | Project-wide, cross-file |
| `extract_function` | Move a line range into a new parameterless function and insert the call | Module top level only |
| `replace_call` | Swap the callee of matching calls | Arguments and layout preserved |
| `add_type_hints` | Add a missing `-> None` return annotation | Conservative by design |
| `add_import` | Add `import x` or `from x import y`, sorted into the import block | Module level only |
| `remove_import` | Remove an import (or one name from a from-import) | Module level only |
| `noop` | Explicitly declare that no edit is needed | Retried with feedback if the whole plan is `noop` |

## Data contracts

**`RefactorPlan`** — `{ "instructions": [...], "summary": str }`. Frozen and closed. Every instruction carries `id` (required), `depends_on` (optional list of ids), `action`, `target_file` and optional `reasoning`. The list order is the authoring order; execution order is the topological sort of `depends_on`, with authored position breaking ties.

`id` is validated against `^[A-Za-z0-9][A-Za-z0-9_.:-]*$` so it stays usable as a JSON key without invisible whitespace. `depends_on` entries must match the same alphabet and may not repeat. Graph-level rules — uniqueness, referential integrity, acyclicity — live in `engine/dag.py`, not in the model, because they are properties of the collection rather than of one instruction.

`target_file` is validated before the engine sees it: it must end in `.py`, be repository-relative, contain no `..`, and have no drive/anchor (`Path.anchor`, not just `is_absolute()`, so `/abs/x.py` is rejected on Windows too). The dispatcher repeats the check after `resolve()`, which also catches symlinks that escape the repository.

Import fields are validated as well: `module` must be an absolute dotted module path with no keywords, and `name`/`alias` must be Python identifiers. The engine re-validates before touching the CST, so a `model_construct`-built plan cannot bypass the schema.

**AST fingerprint** — `dict[str, str]`. One entry per qualified class/function (`Class.method`, `outer.inner`), plus `<module>` for module-level statements that are not imports or definitions. Values are SHA-256 digests of the node's rendered source. Duplicate qualified names (conditional definitions) get a stable `#2`/`#3` suffix. Imports are intentionally excluded: the import atoms exist to rewrite them, and their effect is covered by the behavioural gate.

**`RepositorySnapshot`** — the fingerprint map per file plus a SHA-256 of each file's raw bytes. The raw digest lets the post-attempt scan skip unchanged files without parsing them.

**`ScopeViolationError`** — carries `violations: tuple[str, ...]` with the offending qualified names (or file paths for created/deleted files) so callers can feed a machine-readable list back to the model.

**`VerifyResult`** — `{ ok, stage, stdout, stderr }`. `stage` is one of `ruff`, `mypy`, `pytest`, `all`, `scope` (AST gate) or `noop` (synthetic empty-plan signal). `feedback()` prefers stderr and truncates the prompt payload to 4,000 characters.

**`RefactorReport`** — `{ success, attempts, plan, log, last_verify, diff }`. `diff` is populated only by a dry run; `last_verify` is the last gate result whether the run succeeded or not.

## Failure model

| Failure | Raised / detected by | Outcome |
|---|---|---|
| Model endpoint unreachable, non-JSON response, malformed chat payload, or two invalid plans | `IntentParserError` | CLI exit `1`; no transaction was started |
| Model output fails schema validation once | Repair round-trip | Second answer must validate |
| Plan targets a missing file | `FileNotFoundError` | Transaction rolls back, run aborts (a new model round cannot fix this) |
| Path escapes the repository | `PathTraversalError` | Same as above |
| Atom rejects its input (bad line range, empty expression, invalid import) | `InstructionError` with `instruction_index` | Same as above |
| An instruction matches nothing (duplicate import, missing name) | `NoOpEditError` | Same as above |
| Verification red | `_VerificationFailed` (`stage` = ruff/mypy/pytest) | Roll back, capture diff, retry |
| AST scope violation | `ScopeViolationError` → `_VerificationFailed(stage="scope")` | Roll back, capture diff, retry with the offending node names |
| File created or deleted during an attempt | `ScopeViolationError` | Same as above (no atom may do this) |
| Plan contains only `noop` | Synthetic `VerifyResult(stage="noop")` | Counts as a failed attempt and retries |
| Plan repeats an instruction id | `DuplicateInstructionIdError` from `topological_sort` | Parser retries with the message in the repair prompt; a direct `execute_plan` aborts before any edit |
| `depends_on` names no step | `UnknownDependencyError` | Same as above |
| Dependency cycle | `CircularDependencyError`; the message names one concrete cycle | Same as above |
| Git operation fails | `GitError` (git stderr attached) | Fails loudly: a commit failure keeps the stash recoverable; a rollback failure propagates |
| Dry run completes | `_DryRunComplete` | Roll back, return the captured diff |

The distinction matters: gate failures (behavioural or structural) are *retryable* because another plan can fix them; schema, path and engine rejections are *deterministic* and abort immediately. A dependency error sits in between: `IntentParser` retries it because the graph came from the model and the message says exactly what to fix, while `execute_plan` (which receives an already-accepted plan) refuses to start.

## DAG ordering

`engine/dag.py` is deliberately small and total: it either returns a complete order or raises before the dispatcher reads a file.

- **Kahn's algorithm with a min-heap.** Nodes with no remaining dependency are pushed keyed on their authored index. Popping the smallest index first makes the order deterministic and, when there are no edges at all, reproduces the authored order exactly — the property that keeps v0.2.0 plans and the 50-instruction stress plan stable.
- **Validation is part of sorting.** Duplicate ids are rejected while building the index map (otherwise `depends_on` would be ambiguous). Each `depends_on` entry must resolve to a known id. If fewer nodes than the plan are emitted, the residual contains a cycle; a depth-first walk over the residual returns a concrete loop for the error message rather than a set of suspects.
- **Ordering is not parallelism.** Every atom reads and writes the same worktree, so independent steps still run sequentially. `depends_on` expresses "must happen after", not "may run concurrently".
- **Instruction indices are execution positions.** `InstructionError.instruction_index` and the dispatcher log's `[n]` refer to the sorted sequence, because that is the order in which files were touched.

## CI audit

`audit.py` mirrors the attempt contract without a transaction. It asks the same two questions over a diff instead of a plan application:

1. **Structural:** `git diff --name-status <base>` (plus untracked, non-ignored files) is the change set. Each changed Python file is fingerprinted from `git show <base>:<path>` and from disk; the symmetric difference of the fingerprint maps is the node-level evidence. With `--plan`, `verify_scope` runs per file, and created/deleted/renamed/non-Python/unparseable files are violations because no atom can produce them. Without a plan the same data is reported as inventory and only parse errors fail the run.
2. **Behavioural:** the unchanged `Verifier` runs ruff → mypy → pytest with the same explicit interpreter.

The result is one frozen `AuditReport` serialised to JSON. Gates are data here too: a red gate does not raise, it sets `ok=False`, so the CI job always gets a report. The only exceptions are setup failures — not a git work tree, an unresolvable base ref, or an unreadable or cyclic plan — which the CLI maps to exit `2`.

## Controlled imports

Imports are the most dangerous lines in a Python file: they execute code at import time, shadow names, and create cycles. They are therefore a typed atom, not a free-form edit:

- `AddImport(module, name=None, alias=None)` writes `import module [as alias]` or `from module import name [as alias]` and merges into an existing from-import when one exists. Aliases are sorted with isort's `order-by-type` grouping (constants, classes, lowercase names) and commas are rebuilt from the original layout, so a merge never produces `a,OrderedDict` or a stray trailing comma.
- `RemoveImport(module, name=None)` removes every import of the module when `name` is omitted, or one name from from-imports when it is given. A statement that loses its last name is removed entirely; comments above it are re-attached to the next statement (or the module footer) rather than deleted.
- Placement follows isort's default grouping: `__future__` first, then straight imports, then from-imports, each sorted case-insensitively by module. A merged line longer than 88 characters is wrapped in parentheses, so a long merge cannot trip `E501`.
- The atom never reorders existing statements and never touches imports inside `try:` blocks or functions. Relative imports are not merged or generated. Renaming a symbol inside a from-import is left to `rename_symbol`; the resulting member order is the project's isort gate's concern, not a silent reformat by the engine.

## Why these choices

- **The model is a compiler, not an editor.** It emits one of seven typed instructions. No field accepts a file body or a statement suite; even the import fields are module paths and identifiers, not source text. The only expression-shaped field (`replace_call.new_call`) must parse as a single expression and can only replace a callee. Prompt injection cannot smuggle a new module into a file.
- **Two gates, two questions.** Verification can only see behaviour: a silent rewrite of an unrelated function passes every test if that function happens to be untested. The AST gate compares the exact shape of every function, class and module-level statement, so "it works" does not excuse "it was not in the plan". Conversely, the AST gate cannot know whether the authorised edits are correct; verification owns that.
- **libcst, not the standard library `ast`.** `ast.unparse` regenerates source and loses comments, quote style and layout. libcst round-trips every untouched node byte-for-byte and re-renders only the changed nodes, using the module's own indent unit.
- **Git via subprocess, not GitPython.** The working tree, index and untracked files are the backup. Shelling out to the user's own git is the only faithful way to snapshot and restore all three, and every call has a hard timeout.
- **One transaction per attempt.** A failed attempt is rolled back before the next plan is requested, so the model never plans against a broken tree. Both gates capture their diff first because the rollback destroys it.
- **Explicit interpreter.** Every gate runs `<sys.executable> -m <tool>`, never a bare `python` from `PATH`. This repository is developed on a machine with both Python 3.8 and 3.12; only the chosen interpreter has the tooling.
- **Failures are data in the Verifier.** A red gate is the normal way ruff/mypy/pytest report findings. Turning it into an exception would crash the CLI instead of informing the next attempt.

## Extension points

- **New instruction:** add a frozen model to the discriminated union in `schema/instructions.py`, a branch in `engine/dispatcher.py`, an atom under `engine/`, and an authorisation rule in `engine/scope_guard.py`. Schema inlining and Ollama grammar generation happen automatically, and `id`/`depends_on` are inherited from `_BaseInstruction`, so DAG ordering and the repair prompt cover the new atom without extra work.
- **New verification gate:** add a `(stage, args)` entry to the `stages` tuple in `Verifier.verify`. Order is the priority order; the first failure short-circuits. Both `refactor` and `audit` inherit the new gate.
- **New model backend:** anything with `.parse(intent, repo_root, extra_context) -> RefactorPlan` can replace `IntentParser`; the orchestrator depends only on that method. The backend is responsible for producing a graph-valid plan, which `topological_sort` will check again at execution time.
- **New transport dialect:** add a `ChatProvider` implementation in `intent/providers.py` (build a `ChatRequest`, parse the assistant content) and register it in `build_provider`. Retries, JSON recovery, validation and repair are inherited unchanged; only the wire format is new.

## Operating envelope

Python 3.11+, Git with at least one commit, a reachable model endpoint (a local Ollama daemon or an OpenAI-compatible HTTPS API), UTF-8 source files. `extract_function` is limited to module-level statements; `add_type_hints` only writes `-> None`; imports are edited only through `add_import`/`remove_import`, which are module-level and absolute-module-only. The scope guard fingerprints every `.py` file outside `.git`, virtualenvs, caches and build output, so a repository larger than a few thousand files pays a measurable snapshot cost (and `rename_symbol` remains bounded by Jedi's project-search budget, which the engine raises to the repository's file count).

`refactor` requires a clean rollback baseline (a `HEAD` commit) and temporarily stashes uncommitted work. `audit` requires only that `--base-ref` resolves locally — CI must fetch the base branch (`fetch-depth: 0`), and it writes nothing except the optional `--json-output` report.

The cloud dialects authenticate with a header: `Authorization: Bearer` for `openai`, `x-api-key` plus the required `anthropic-version: 2023-06-01` for `anthropic`. API keys are read from `MORAY_API_KEY` (or `--api-key`), base URLs carrying embedded credentials or query strings are rejected, and redirects are not followed so a token is never forwarded to another host. Endpoints that require credentials in a query string (Azure-style `?api-version=...`) are outside the current transport.
