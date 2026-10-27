"""Intent layer: natural language in, validated RefactorPlan out."""

from __future__ import annotations

from moray_surgeon.intent.parser import IntentParser, IntentParserError

__all__ = ["IntentParser", "IntentParserError"]
