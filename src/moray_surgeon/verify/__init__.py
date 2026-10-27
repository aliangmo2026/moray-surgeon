"""Verification layer: run the project gates and package the outcome.

Nothing here raises on a failed gate: the CLI/orchestrator decides what a red
result means. This module only turns subprocess outcomes into data.
"""

from __future__ import annotations

from moray_surgeon.verify.runner import Verifier, VerifyResult

__all__ = ["Verifier", "VerifyResult"]
