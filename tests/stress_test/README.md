# Stress test harness

Reproducible, intentionally brutal end-to-end tests for moray-surgeon. Unlike
the unit suite (which uses fake parsers and fake verifiers), everything here is
real: a generated multi-module project, a real `RefactorPlan`, the real
dispatcher, real Jedi/libcst edits, real `ruff`/`mypy`/`pytest` gates, real git
transactions and the AST scope guard.

## Files

| File | Purpose |
|---|---|
| `generate_project.py` | Writes the 50-module / 7.9k-line / 300-class synthetic project and initialises git. |
| `run_bombing.py` | Builds both plans, runs them, captures evidence and writes `docs/STRESS_TEST_REPORT.md`. |

Neither file is named `test_*.py`, so the normal `pytest -q` run does not collect
them; the harness takes tens of seconds and is meant to be run explicitly.

## Quick start

```bash
# from the repository root, with the dev dependencies installed
python tests/stress_test/run_bombing.py

# keep the generated projects for inspection
python tests/stress_test/run_bombing.py --keep --workdir /tmp/moray-stress

# just generate a project (no engine involved)
python tests/stress_test/generate_project.py --target /tmp/moray-stress
```

Requires Python 3.11+, git, and `ruff`, `mypy`, `pytest` installed in the
interpreter that runs the harness (the same interpreter is passed to the
verifier explicitly, never a bare `python` from `PATH`).

## What the project looks like

- `stresspkg/domain/m00..m24` and `stresspkg/services/m25..m49`: 50 modules.
- Six classes per module (`Entity`, `Service`, `Repository`, `Handler`,
  `Policy`, `Snapshot`) = 300 classes, with two-level inheritance
  (`Handler(Service)`, `Snapshot(Entity)`).
- A runtime dependency ring: every module lazily imports the next one and the
  last wraps to the first; `if TYPE_CHECKING:` back-edges mirror the same ring
  for static analysis.
- 8,046 physical lines, isort-clean, mypy-clean, with a pytest smoke suite.

## Bombing A — 50 instructions

One `RefactorPlan` applied in a single transaction:

- 2 project-wide `rename_symbol` operations (158 and 153 references across 53
  and 52 files);
- 20 `add_import` + 20 `replace_call` pairs (the import must exist for mypy to
  accept the new callee); every `replace_call` declares `depends_on` on its
  matching `add_import`, so this run also exercises DAG ordering at scale;
- 4 `replace_call` rewrites in files that already import the helper;
- 4 `remove_import` operations dropping deprecated re-exports.

It passes only if the AST scope guard accepts every changed node **and**
`ruff`, `mypy` and `pytest` are green afterwards.

## Bombing B — one poisoned instruction

A six-instruction plan in which five instructions are valid and one rewrites a
`validate_record(...)` call. Two variants:

- **B-1** rewrites it to `Record(...)`: ruff passes, mypy rejects the argument
  type. The attempt is rolled back; the feedback handed to the next model call
  contains the full captured diff and the mypy diagnostics.
- **B-2** rewrites it to `reject_record(...)`: ruff and mypy pass, pytest fails
  at runtime. Same rollback and feedback path, with the pytest diagnostic.

The harness records the rollback duration and proves `git status` is clean and
`HEAD` is unchanged after each failed round.

## Evidence

`run_bombing.py` writes `docs/STRESS_TEST_REPORT.md` with the environment,
baseline gate output, plan composition, per-phase timings, the complete
dispatcher log, commit statistics, the captured diffs and the gate feedback. The
report is generated evidence and is intentionally committed.
