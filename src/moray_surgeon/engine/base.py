"""Immutable result type shared by every surgical edit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EditResult:
    """Outcome of one atomic libcst operation.

    Attributes:
        new_source: The rewritten module source. When nothing matched, this is
            byte-for-byte the input source (libcst round-trips untouched nodes).
        changes: Number of concrete edits performed. Never negative; zero means
            the operation was a no-op.

    Raises:
        ValueError: If ``changes`` is negative.
    """

    new_source: str
    changes: int

    def __post_init__(self) -> None:
        """Reject nonsensical counters early so callers can trust the report."""
        if self.changes < 0:
            raise ValueError("changes must be non-negative")
