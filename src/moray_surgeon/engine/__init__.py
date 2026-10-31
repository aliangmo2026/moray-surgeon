"""Public surface of the surgical edit engine.

Six libcst atoms, the scope guard, the plan-DAG sorter and ``execute_plan``,
which orders the atoms by dependency and runs them over a ``RefactorPlan``. Most
atoms are source-in/source-out; ``rename_symbol`` is project-aware (Jedi resolves
references across files) and rewrites affected files in place, so it must run
inside the caller's transaction. The scope guard fingerprints the whole
repository before an attempt and proves afterwards that only authorised nodes
changed.
"""

from __future__ import annotations

from moray_surgeon.engine.dag import topological_sort
from moray_surgeon.engine.dispatcher import execute_plan
from moray_surgeon.engine.extract import extract_function
from moray_surgeon.engine.extract_variable import extract_variable
from moray_surgeon.engine.import_manager import add_import, remove_import
from moray_surgeon.engine.inline_function import inline_function
from moray_surgeon.engine.inline_variable import inline_variable
from moray_surgeon.engine.move_symbol import move_symbol
from moray_surgeon.engine.remove_dead_code import remove_dead_code
from moray_surgeon.engine.rename import rename_symbol
from moray_surgeon.engine.replace_call import replace_call
from moray_surgeon.engine.repo_map import generate_repo_map
from moray_surgeon.engine.scope_guard import (
    get_ast_fingerprint,
    snapshot_repository,
    verify_repository_scope,
    verify_scope,
)
from moray_surgeon.engine.split_class import split_class
from moray_surgeon.engine.type_hints import add_type_hints

__all__ = [
    "add_import",
    "add_type_hints",
    "execute_plan",
    "extract_function",
    "extract_variable",
    "generate_repo_map",
    "get_ast_fingerprint",
    "inline_function",
    "inline_variable",
    "move_symbol",
    "remove_dead_code",
    "remove_import",
    "rename_symbol",
    "replace_call",
    "snapshot_repository",
    "split_class",
    "topological_sort",
    "verify_repository_scope",
    "verify_scope",
]
