"""Before/after architecture snapshots for the orchestrator, via moray-scribe.

A *snapshot* is the repository's moray-scribe symbol graph serialized as
deterministic JSON (sorted nodes and edges), so two captures of the same
structure are byte-identical and a plain string comparison detects change.

Every entry point degrades gracefully: when moray-scribe is not installed, or
any file refuses to parse, the capture returns ``None`` and the diff is empty.
Architecture reporting is a lens on the refactor — it must never be the reason
a refactor fails.

The comparison tracks the structural kinds (classes, functions, methods) and
recognises cross-module moves: a symbol removed from one module while an
equal-named symbol of the same kind appears in another is reported as a move,
which is what ``move_symbol`` and ``split_class`` produce.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Only structural kinds participate in the diff; variables/constants/imports
# churn too easily to say anything meaningful about architecture.
_TRACKED_KINDS = ("class", "function", "method")

# Scan everything Python, including tests, but skip the noise directories.
_INCLUDE = ["**/*.py"]
_EXCLUDE = [
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/node_modules/**",
    "**/build/**",
    "**/dist/**",
]


@dataclass(frozen=True)
class ArchitectureDiff:
    """Outcome of comparing two architecture snapshots.

    Attributes:
        summary: One-line Chinese summary for the terminal (the
            architecture-change line); empty when nothing changed.
        text: Full human-readable comparison stored in
            ``RefactorReport.architecture_diff``; empty when nothing changed.
    """

    summary: str
    text: str


def capture_architecture(repo_root: Path) -> str | None:
    """Serialize the repository's symbol graph as deterministic JSON.

    Returns ``None`` when moray-scribe is unavailable or any file refuses to
    parse — the caller then skips the comparison instead of failing.
    """
    try:
        from moray_scribe.grapher import build_symbol_graph
        from moray_scribe.scanner import extract_file, iter_python_files
    except ImportError:
        return None
    try:
        bundles = [
            extract_file(path, root=repo_root)
            for path in iter_python_files(repo_root, include=_INCLUDE, exclude=_EXCLUDE)
        ]
        graph = build_symbol_graph(bundles)
    except Exception:  # a snapshot failure must never fail a refactor
        return None
    nodes = sorted(
        (
            {
                "id": node.id,
                "kind": node.kind.value,
                "name": node.name,
                "module": node.module,
                "qualname": node.qualname,
            }
            for node in graph.nodes.values()
            if node.kind.value in _TRACKED_KINDS
        ),
        key=lambda node: node["id"],
    )
    edges = sorted([edge.source, edge.target, edge.kind.value] for edge in graph.edges)
    return json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, sort_keys=True)


def diff_architecture(before: str | None, after: str | None) -> ArchitectureDiff:
    """Compare two snapshots; an empty diff when either side is missing."""
    if before is None or after is None or before == after:
        return ArchitectureDiff(summary="", text="")
    before_nodes = {
        entry["id"]: entry for entry in json.loads(before)["nodes"]
    }
    after_nodes = {entry["id"]: entry for entry in json.loads(after)["nodes"]}
    added_ids = sorted(set(after_nodes) - set(before_nodes))
    removed_ids = sorted(set(before_nodes) - set(after_nodes))

    moved: list[tuple[dict, dict]] = []
    still_added: list[str] = []
    for added_id in added_ids:
        added_entry = after_nodes[added_id]
        counterpart = next(
            (
                removed_id
                for removed_id in removed_ids
                if before_nodes[removed_id]["kind"] == added_entry["kind"]
                and before_nodes[removed_id]["name"] == added_entry["name"]
                and (
                    before_nodes[removed_id]["module"] != added_entry["module"]
                    or before_nodes[removed_id].get("qualname", "")
                    != added_entry.get("qualname", "")
                )
            ),
            None,
        )
        if counterpart is None:
            still_added.append(added_id)
        else:
            removed_ids.remove(counterpart)
            moved.append((before_nodes[counterpart], added_entry))
    still_removed = removed_ids

    added_entries = [after_nodes[added_id] for added_id in still_added]
    removed_entries = [before_nodes[removed_id] for removed_id in still_removed]
    summary = _summary(added_entries, removed_entries, moved)
    text = _render(summary, added_entries, removed_entries, moved)
    return ArchitectureDiff(summary=summary, text=text)


def _label(entry: dict) -> str:
    """A kind/name/module description of one symbol entry, in Chinese."""
    kind_name = {"class": "类", "function": "函数", "method": "方法"}[entry["kind"]]
    return f"{kind_name} {entry['name']}（{entry['module']}）"  # noqa: RUF001


def _kind_counts(entries: Sequence[dict]) -> list[tuple[str, int]]:
    """Count entries per kind word, sorted for a stable summary."""
    counts: dict[str, int] = {}
    for entry in entries:
        kind_word = {"class": "类", "function": "函数", "method": "方法"}[entry["kind"]]
        counts[kind_word] = counts.get(kind_word, 0) + 1
    return [(kind_word, count) for kind_word, count in sorted(counts.items())]


def _summary(
    added_entries: Sequence[dict],
    removed_entries: Sequence[dict],
    moved: Sequence[tuple[dict, dict]],
) -> str:
    """The single architecture-change summary line for the terminal."""
    parts: list[str] = []
    added_classes = sum(1 for entry in added_entries if entry["kind"] == "class")
    if added_classes:
        parts.append(f"提取了 {added_classes} 个新类")
    moved_classes = sum(1 for _, entry in moved if entry["kind"] == "class")
    moved_methods = sum(1 for _, entry in moved if entry["kind"] == "method")
    moved_functions = len(moved) - moved_classes - moved_methods
    if moved_functions:
        parts.append(f"移动了 {moved_functions} 个函数")
    if moved_methods:
        parts.append(f"移动了 {moved_methods} 个方法")
    if moved_classes:
        parts.append(f"移动了 {moved_classes} 个类")
    for kind_word, count in _kind_counts(
        [entry for entry in added_entries if entry["kind"] != "class"]
    ):
        parts.append(f"新增 {count} 个{kind_word}")
    for kind_word, count in _kind_counts(removed_entries):
        parts.append(f"移除 {count} 个{kind_word}")
    if not parts:
        return ""
    return "架构变更：" + "，".join(parts)  # noqa: RUF001


def _render(
    summary: str,
    added_entries: Sequence[dict],
    removed_entries: Sequence[dict],
    moved: Sequence[tuple[dict, dict]],
) -> str:
    """The multi-line report stored in ``RefactorReport.architecture_diff``."""
    lines = [summary, ""]
    if added_entries:
        lines.append("新增符号:")
        lines.extend(f"  + {_label(entry)}" for entry in added_entries)
    if moved:
        lines.append("移动符号:")
        lines.extend(
            f"  → {before_entry['name']}: {before_entry['id']} -> {after_entry['id']}"
            for before_entry, after_entry in moved
        )
    if removed_entries:
        lines.append("移除符号:")
        lines.extend(f"  - {_label(entry)}" for entry in removed_entries)
    return "\n".join(lines)
