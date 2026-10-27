# Architecture

moray-surgeon is a one-way pipeline with one hard boundary: **the model plans, the engine mutates**. Nothing the model produces is ever written to disk directly, and nothing the engine writes survives a failed verification.

The design has four layers:

1. **Control plane** — CLI and MCP entry points.
2. **Compilation** — intent → validated `RefactorPlan` (the only place a model is consulted).
3. **Mutation** — Git transaction + deterministic AST edits.
4. **Evidence** — verification gates that either commit the transaction or feed the next attempt.

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
        │  IntentParser  ── HTTP POST /api/chat ──▶  Ollama           │
        │  system prompt + inlined JSON Schema + format=schema        │
        │  strip fences · validate · one repair round-trip            │
        └─────────────────────────────────────────────────────────────┘
                                       │ RefactorPlan (frozen JSON, extra=forbid)
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  TransactionManager            [git subprocess, 30 s]       │
        │  enter: git stash -u   (snapshot uncommitted work)          │
        │  exit : commit on green  |  reset --hard HEAD on red        │
        └─────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Dispatcher  →  AST Engine                                  │
        │  libcst CST transforms  +  Jedi project references          │
        │  rename · extract · replace · type hints · noop             │
        └─────────────────────────────────────────────────────────────┘
                                       │ files written (UTF-8)
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Verifier                                                   │
        │  ruff → mypy → pytest   (stop at the first red gate)        │
        └─────────────────────────────────────────────────────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
             green ▼                                       ▼ red
   ┌───────────────────────────────┐  ┌─────────────────────────────────────────┐
   │  commit + RefactorReport      │  │  reset --hard HEAD + git clean -fd      │
   │  success = True               │  │  restore the stashed worktree           │
   │  (dry run returns the diff)   │  │  capture diff + tool output             │
   └───────────────────────────────┘  └────────────────────┬────────────────────┘
                                                           │ diff + tool output
                                                           ▼
                                      ┌─────────────────────────────────────────┐
                                      │  Auto-Fix Loop  ·  up to 3 rounds       │
                                      │  next plan starts from a clean HEAD     │
                                      │  diff + stderr are the next prompt      │
                                      └─────────────────────────────────────────┘
                                                           │
                                                           └──── re-enters Orchestrator
```

Everything above `TransactionManager` is planning; everything inside it is mutation; everything after it is evidence. A red gate never leaves a half-edited worktree behind, because the rollback is owned by the same context manager that owns the edits.

## The attempt lifecycle

1. **Compose context.** The CLI scans the repository (or takes `--files`) and `generate_repo_map` emits one line per top-level class/function/method: signature, line range, first docstring line. Signatures are whitespace-collapsed and each file block is capped at 2,000 characters, so the prompt stays bounded. The map is prefixed with `【Project Code Map】` and reused on every retry.
2. **Parse.** `IntentParser` sends one chat request to Ollama with the inlined JSON Schema in both the system prompt and the `format` field. The response is sliced between its outermost braces, then validated as a `RefactorPlan`. Invalid model output gets exactly one repair round-trip; transport errors are never retried. No repository file is read at this stage.
3. **Apply.** The orchestrator opens a fresh `TransactionManager`, which snapshots uncommitted work (`git stash -u`) and refuses to continue without a `HEAD` commit. `execute_plan` runs the instructions strictly in order; the first failure aborts the attempt. A plan made only of `noop` instructions is treated as a failed attempt with synthetic feedback, not as success.
4. **Verify.** `ruff check .`, then `<python> -m mypy <target>`, then `<python> -m pytest <args>`, stopping at the first red gate. A missing tool, an invalid cwd or a timeout is returned as a failed `VerifyResult`; verification never raises.
5. **Settle.**
   - **Green:** the transaction commits (`stash apply` → `add -A` → `commit` → `stash drop`) and the report says `success=True`.
   - **Red:** the diff is captured *before* the exception propagates, so `__exit__` can safely run `reset --hard HEAD`, `clean -fd` and restore the stash. The diff and tool output become the next attempt's context.
   - **Dry run:** the diff is captured, `_DryRunComplete` forces a rollback, verification and commit are skipped, and the report returns `success=True` with the diff.

## Module responsibilities

| Module | Responsibility | Key invariant |
|---|---|---|
| `cli.py` | Typer entry points (`refactor`, `mcp`, `version`); validates the repo root; renders plan, diff and report | Refuses to start outside a Git work tree (exit `2`) |
| `mcp_server.py` | stdio MCP host with three tools; all failures become strings | stdout carries JSON-RPC only; logs go to a file |
| `orchestrator.py` | Attempt loop, repo-map injection, per-attempt transaction, dry-run, diff capture | One transaction per attempt; retry only on verification failure |
| `intent/parser.py` | Ollama chat call, fence stripping, schema validation, one repair round | Returns typed instructions, never a file body |
| `schema/instructions.py` | Frozen, closed instruction models; discriminated union; Ollama-safe schema inlining | `extra="forbid"`, `frozen=True`, no `$ref` survives inlining |
| `transaction/manager.py` | Git snapshot, commit and rollback | Failed transaction restores the exact starting state; rollback errors are raised |
| `engine/dispatcher.py` | Ordered execution, path containment, error classification | Resolve → containment → existence, in that order |
| `engine/rename.py` | Jedi project references → libcst position edits; textual fallback | References outside `repo_root` are never rewritten |
| `engine/extract.py` | Module-level statements → new function + call site | Only CST body edits; never string slicing or re-parsing |
| `engine/replace_call.py` | Callee replacement | Arguments, comments and spacing on the `Call` node are preserved |
| `engine/type_hints.py` | Missing `-> None` annotations | Never guesses a richer type |
| `engine/repo_map.py` | libcst-derived symbol map for prompt context | 2,000 chars per file, line-boundary truncation |
| `verify/runner.py` | ruff → mypy → pytest | Failures are data, never exceptions; explicit interpreter |
| `ui.py` | Rich plan table, diff view, live trace | Dynamic text is rendered with markup disabled |

## Data contracts

**`RefactorPlan`** — `{ "instructions": [...], "summary": str }`. Frozen and closed. Each instruction carries `action`, `target_file` and optional `reasoning`.

`target_file` is validated before the engine sees it: it must end in `.py`, be repository-relative, contain no `..`, and have no drive/anchor (`Path.anchor`, not just `is_absolute()`, so `/abs/x.py` is rejected on Windows too). The dispatcher repeats the check after `resolve()`, which also catches symlinks that escape the repository.

**`VerifyResult`** — `{ ok, stage, stdout, stderr }`. `stage` is one of `ruff`, `mypy`, `pytest`, `all`, or `noop` (the synthetic empty-plan signal). `feedback()` prefers stderr and truncates the prompt payload to 4,000 characters.

**`RefactorReport`** — `{ success, attempts, plan, log, last_verify, diff }`. `diff` is populated only by a dry run; `last_verify` is the last gate result whether the run succeeded or not.

## Failure model

| Failure | Raised / detected by | Outcome |
|---|---|---|
| Ollama unreachable, non-JSON response, or two invalid plans | `IntentParserError` | CLI exit `1`; no transaction was started |
| Model output fails schema validation once | Repair round-trip | Second answer must validate |
| Plan targets a missing file | `FileNotFoundError` | Transaction rolls back, run aborts (a new model round cannot fix this) |
| Path escapes the repository | `PathTraversalError` | Same as above |
| Atom rejects its input (bad line range, empty expression) | `InstructionError` with `instruction_index` | Same as above |
| An instruction matches nothing | `NoOpEditError` | Same as above |
| Verification red | `_VerificationFailed` | Roll back, capture diff, retry |
| Plan contains only `noop` | Synthetic `VerifyResult(stage="noop")` | Counts as a failed attempt and retries |
| Git operation fails | `GitError` (git stderr attached) | Fails loudly: a commit failure keeps the stash recoverable; a rollback failure propagates |
| Dry run completes | `_DryRunComplete` | Roll back, return the captured diff |

The distinction matters: verification failures are *retryable* because another plan can fix them; schema, path and engine rejections are *deterministic* and abort immediately.

## Why these choices

- **The model is a compiler, not an editor.** It emits one of five instructions. No field accepts a file body, a statement suite or an import; the only expression-shaped field (`replace_call.new_call`) must parse as a single expression and can only replace a callee. Prompt injection cannot smuggle a new module into a file.
- **libcst, not the standard library `ast`.** `ast.unparse` regenerates source and loses comments, quote style and layout. libcst round-trips every untouched node byte-for-byte and re-renders only the changed nodes, using the module's own indent unit.
- **Git via subprocess, not GitPython.** The working tree, index and untracked files are the backup. Shelling out to the user's own git is the only faithful way to snapshot and restore all three, and every call has a hard timeout.
- **One transaction per attempt.** A failed attempt is rolled back before the next plan is requested, so the model never plans against a broken tree. The diff is captured first because the rollback destroys it.
- **Explicit interpreter.** Every gate runs `<sys.executable> -m <tool>`, never a bare `python` from `PATH`. This repository is developed on a machine with both Python 3.8 and 3.12; only the chosen interpreter has the tooling.
- **Failures are data in the Verifier.** A red gate is the normal way ruff/mypy/pytest report findings. Turning it into an exception would crash the CLI instead of informing the next attempt.

## Extension points

- **New instruction:** add a frozen model to the discriminated union in `schema/instructions.py`, a branch in `engine/dispatcher.py`, and an atom under `engine/`. Schema inlining and Ollama grammar generation happen automatically.
- **New verification gate:** add a `(stage, args)` entry to the `stages` tuple in `Verifier.verify`. Order is the priority order; the first failure short-circuits.
- **New model backend:** anything with `.parse(intent, repo_root, extra_context) -> RefactorPlan` can replace `IntentParser`; the orchestrator depends only on that method.

## Operating envelope

Python 3.11+, Git with at least one commit, a reachable Ollama endpoint, UTF-8 source files. `extract_function` is limited to module-level statements by design; `add_type_hints` only writes `-> None`; the engine never adds or removes imports.
