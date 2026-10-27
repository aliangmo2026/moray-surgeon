"""Transaction layer: all-or-nothing refactors backed by git."""

from __future__ import annotations

from moray_surgeon.transaction.manager import TransactionManager

__all__ = ["TransactionManager"]
