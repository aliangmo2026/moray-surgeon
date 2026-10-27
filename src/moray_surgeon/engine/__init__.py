"""Public surface of the surgical edit engine.

Four libcst atoms plus ``execute_plan``, which orchestrates them over a
``RefactorPlan``. Most atoms are source-in/source-out; ``rename_symbol`` is
project-aware (Jedi resolves references across files) and rewrites affected
files in place, so it must run inside the caller's transaction.
"""

from __future__ import annotations

from moray_surgeon.engine.dispatcher import execute_plan
from moray_surgeon.engine.extract import extract_function
from moray_surgeon.engine.rename import rename_symbol
from moray_surgeon.engine.replace_call import replace_call
from moray_surgeon.engine.repo_map import generate_repo_map
from moray_surgeon.engine.type_hints import add_type_hints

__all__ = [
    "add_type_hints",
    "execute_plan",
    "extract_function",
    "generate_repo_map",
    "rename_symbol",
    "replace_call",
]
