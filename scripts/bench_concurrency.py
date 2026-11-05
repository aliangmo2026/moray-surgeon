"""Benchmark sequential vs multi-process refactoring on generated modules.

The engine's cost is CPU-bound libcst parsing and rewriting, so a plan that
touches N independent files should scale with cores. This script builds N
synthetic modules that all need return annotations, runs the identical plan
sequentially (``max_workers=1``) and through the process pool, and checks that
both runs produced byte-identical files before reporting the speedup.

Run it from the repository root (the installed package is imported)::

    python scripts/bench_concurrency.py --files 10 --functions 1200

Only the engine is measured: no git transaction, no verifier, no model calls.
Each run starts from a fresh copy of the same generated tree, so the two modes
see identical input.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

from moray_surgeon.engine import execute_plan
from moray_surgeon.engine.concurrency import resolve_max_workers
from moray_surgeon.schema import AddTypeHints, RefactorPlan


def module_source(module_index: int, function_count: int) -> str:
    """Return a synthetic module with ``function_count`` un-annotated functions.

    The functions are distinct per module so the generated tree looks like a
    real package rather than N copies of one file.
    """
    blocks = [
        f"def fn_{module_index}_{index}(a, b):\n"
        f"    total = a + b\n"
        f"    return total * {index + 1}\n\n"
        for index in range(function_count)
    ]
    return "".join(blocks)


def build_tree(root: Path, file_count: int, function_count: int) -> None:
    """Write ``file_count`` generated modules under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    for module_index in range(file_count):
        (root / f"mod_{module_index:02d}.py").write_text(
            module_source(module_index, function_count), encoding="utf-8"
        )


def build_plan(file_count: int) -> RefactorPlan:
    """One whole-file annotation instruction per module, with no dependencies."""
    return RefactorPlan(
        instructions=[
            AddTypeHints(
                id=f"hints_{index}",
                target_file=f"mod_{index:02d}.py",
                function_name=None,
            )
            for index in range(file_count)
        ]
    )


def run_once(root: Path, plan: RefactorPlan, max_workers: int) -> float:
    """Run ``plan`` once and return the wall-clock seconds it took."""
    started = time.perf_counter()
    execute_plan(plan, root, max_workers=max_workers)
    return time.perf_counter() - started


def snapshot(root: Path) -> dict[str, bytes]:
    """Return every ``.py`` file's bytes, keyed by relative path."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*.py"))
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", type=int, default=10, help="number of modules (default 10)")
    parser.add_argument(
        "--functions",
        type=int,
        default=1200,
        help="functions per module; each needs one annotation (default 1200)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="pool size; 0 uses MORAY_MAX_WORKERS or min(cpu, 8)",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="measure each mode this many times (default 1)"
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="exit non-zero when parallel is slower than this ratio (default 1.0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate the tree, run both modes and print the comparison."""
    args = parse_args(argv)
    if args.files < 1 or args.functions < 1 or args.repeat < 1:
        print("--files, --functions and --repeat must all be >= 1", file=sys.stderr)
        return 2

    workers = resolve_max_workers(args.workers or None)
    print(f"pool size: {workers}")
    if workers <= 1:
        print("parallelism disabled (set MORAY_MAX_WORKERS > 1 to enable it)")

    plan = build_plan(args.files)
    sequential_times: list[float] = []
    parallel_times: list[float] = []

    with tempfile.TemporaryDirectory(prefix="moray-bench-") as tmp:
        template = Path(tmp) / "template"
        build_tree(template, args.files, args.functions)
        sample = next(template.glob("*.py"))
        print(
            f"{args.files} files x {args.functions} functions "
            f"({sum(1 for _ in sample.read_text(encoding='utf-8').splitlines())} lines/file)"
        )

        for run in range(args.repeat):
            sequential_root = Path(tmp) / f"sequential-{run}"
            parallel_root = Path(tmp) / f"parallel-{run}"
            shutil.copytree(template, sequential_root)
            shutil.copytree(template, parallel_root)

            sequential_times.append(run_once(sequential_root, plan, max_workers=1))
            parallel_times.append(run_once(parallel_root, plan, max_workers=workers))

            if snapshot(sequential_root) != snapshot(parallel_root):
                print("FAIL: parallel output differs from the sequential output", file=sys.stderr)
                return 1

    sequential = statistics.median(sequential_times)
    parallel = statistics.median(parallel_times)
    speedup = sequential / parallel if parallel else float("inf")
    print(f"sequential: {sequential:7.3f} s  (best {min(sequential_times):.3f} s)")
    print(f"parallel  : {parallel:7.3f} s  (best {min(parallel_times):.3f} s)")
    print(f"speedup   : {speedup:7.2f}x")
    print("identical output: yes")
    if speedup < args.min_speedup:
        print(
            f"FAIL: speedup {speedup:.2f}x is below --min-speedup {args.min_speedup:.2f}x",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
