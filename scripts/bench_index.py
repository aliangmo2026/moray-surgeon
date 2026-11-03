#!/usr/bin/env python3
"""Benchmark the code index against the pre-index project-wide Jedi scan.

Usage (the package must be importable, e.g. ``pip install -e .``)::

    python scripts/bench_index.py
    python scripts/bench_index.py --modules 200 800 2000 --users 5

Every measurement builds a *fresh* repository in its own temporary directory.
That matters: Jedi caches parsed modules per process, so timing a second search
against the same tree measures the cache, not the algorithm.

Reported numbers:

* ``legacy`` — one full ``rename_symbol`` with the index disabled, i.e. exactly
  what the tool did before the index existed: raise Jedi's per-search file budget
  to the size of the repository and ask for every project reference.
* ``index build`` — the one-time cost of scanning the tree into SQLite.
* ``indexed`` — the same rename against an already-built index, i.e. what a
  second and later refactor pays.

Both rename columns include the libcst rewrite, so the search cost is the only
difference between them.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from moray_surgeon.engine import rename as rename_module
from moray_surgeon.engine.indexer import Indexer, reset_indexers
from moray_surgeon.engine.rename import rename_symbol
from moray_surgeon.exceptions import CacheError

_SYMBOLS = 4


def build_repository(modules: int, users: int) -> Path:
    """Create a repository of ``modules`` modules.

    ``users`` of the modules import :data:`_SYMBOLS` classes from their own
    ``shared_<n>.py``; the rest define an unrelated class, so they must never be
    rewritten.

    Args:
        modules: Number of ordinary modules to generate.
        users: Number of modules that use the shared symbols.

    Returns:
        The repository root.
    """
    root = Path(tempfile.mkdtemp()) / "repo"
    root.mkdir()
    for symbol in range(_SYMBOLS):
        (root / f"shared_{symbol}.py").write_text(
            f"class Widget{symbol}:\n"
            "    pass\n"
            "\n"
            "\n"
            f"def make_{symbol}() -> Widget{symbol}:\n"
            f"    return Widget{symbol}()\n",
            encoding="utf-8",
        )
    for index in range(modules):
        if index % max(1, modules // max(1, users)) == 0:
            body = "".join(
                f"from shared_{symbol} import Widget{symbol}\n"
                "\n"
                "\n"
                f"def use_{symbol}_{index}():\n"
                f"    return Widget{symbol}()\n"
                "\n"
                "\n"
                for symbol in range(_SYMBOLS)
            )
        else:
            body = (
                f"class Local{index}:\n"
                "    pass\n"
                "\n"
                "\n"
                f"def touch{index}():\n"
                f"    return Local{index}()\n"
            )
        (root / f"m{index:05d}.py").write_text(body, encoding="utf-8")
    return root


def time_legacy(modules: int, users: int) -> float:
    """Seconds one rename spends when the index does not exist."""
    root = build_repository(modules, users)
    original = rename_module.get_indexer

    def unavailable(repo_root: Path, cache_dir: Path | None = None) -> None:
        raise CacheError("index disabled for the benchmark")

    rename_module.get_indexer = unavailable
    try:
        started = time.perf_counter()
        rename_symbol(root, "shared_0.py", "Widget0", "Gadget0")
        return time.perf_counter() - started
    finally:
        rename_module.get_indexer = original
        shutil.rmtree(root.parent, ignore_errors=True)


def time_indexed(modules: int, users: int) -> tuple[float, float]:
    """Seconds for the index build and the median indexed rename afterwards."""
    root = build_repository(modules, users)
    try:
        reset_indexers()
        indexer = Indexer(root)
        started = time.perf_counter()
        indexer.refresh()
        build_seconds = time.perf_counter() - started
        samples = []
        for symbol in range(_SYMBOLS):
            started = time.perf_counter()
            rename_symbol(
                root, f"shared_{symbol}.py", f"Widget{symbol}", f"Gadget{symbol}", indexer
            )
            samples.append(time.perf_counter() - started)
        return build_seconds, statistics.median(samples)
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep and print one table row per repository size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modules", type=int, nargs="+", default=[200, 800, 2000], help="module counts"
    )
    parser.add_argument("--users", type=int, default=5, help="modules importing the symbols")
    options = parser.parse_args(argv)

    print(
        f"{'modules':>8} {'files':>7} {'legacy':>11} {'index build':>12} "
        f"{'indexed':>10} {'speedup':>8}"
    )
    for modules in options.modules:
        legacy = time_legacy(modules, options.users)
        build_seconds, indexed = time_indexed(modules, options.users)
        print(
            f"{modules:>8} {modules + _SYMBOLS:>7} {legacy:>10.2f}s {build_seconds:>11.2f}s "
            f"{indexed:>9.2f}s {legacy / indexed:>7.1f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
