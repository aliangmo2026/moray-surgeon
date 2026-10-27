from __future__ import annotations

from moray_surgeon.exceptions import MorayError

__all__ = ["CircularSchemaError", "MorayError", "SchemaInlineError"]


class CircularSchemaError(MorayError):
    """Raised when $ref inlining walks into a cycle in the $defs graph."""


class SchemaInlineError(MorayError):
    """Raised when $ref inlining hits an unsupported, missing or leaky schema."""
