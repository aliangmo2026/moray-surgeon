"""Verification layer: run the project gates and package the outcome.

Nothing here raises on a failed gate: the CLI/orchestrator decides what a red
result means. The runner turns subprocess outcomes into data; the differential
module executes refactored functions side by side in an in-process sandbox.
"""

from __future__ import annotations

from moray_surgeon.verify.differential import (
    DifferentialResult,
    changed_function_names,
    compare_functions,
    verify_equivalence,
)
from moray_surgeon.verify.runner import Verifier, VerifyResult

__all__ = [
    "DifferentialResult",
    "Verifier",
    "VerifyResult",
    "changed_function_names",
    "compare_functions",
    "verify_equivalence",
]
