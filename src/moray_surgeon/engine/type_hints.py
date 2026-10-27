"""Add the conservative return hint ``-> None`` to unannotated functions.

MVP scope: only *missing* return annotations are filled, and the only annotation
this atom ever writes is ``None``. Parameter hints and richer return types are
intentionally not guessed — a wrong hint is worse than a missing one for a
zero-hallucination refactoring engine.
"""

from __future__ import annotations

import libcst as cst

from moray_surgeon.engine.base import EditResult


class _ReturnAnnotationTransformer(cst.CSTTransformer):
    """Attach ``-> None`` to ``FunctionDef`` nodes that have no return hint."""

    def __init__(self, function_name: str | None) -> None:
        self._function_name = function_name
        self.changes = 0

    # libcst dispatches by this exact CamelCase hook name.
    def leave_FunctionDef(  # noqa: N802
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        """Annotate only the requested function(s), and only when hint-less."""
        if self._function_name is not None and updated_node.name.value != self._function_name:
            return updated_node
        if updated_node.returns is not None:
            return updated_node
        self.changes += 1
        return updated_node.with_changes(returns=cst.Annotation(cst.Name("None")))


def add_type_hints(source: str, function_name: str | None = None) -> EditResult:
    """Add ``-> None`` to every target function that lacks a return annotation.

    Args:
        source: Python module source. Re-parsed once, never re-generated.
        function_name: If given, only ``FunctionDef`` nodes with this name are
            considered (every same-named nested definition included). If
            ``None``, every function in the module — including methods and
            nested functions — is considered.

    Returns:
        An :class:`EditResult` whose ``changes`` counts functions that actually
        gained an annotation. Functions that already declare ``returns`` are
        skipped and not counted, so a fully annotated file yields ``changes=0``
        and an unchanged source. A named function that does not exist is also a
        no-op (``changes=0``), matching the silent-skip contract above.

    Raises:
        ValueError: If ``function_name`` is provided but empty or whitespace.
    """
    if function_name is not None and not function_name.strip():
        raise ValueError("function_name must be a non-empty name or None")

    transformer = _ReturnAnnotationTransformer(function_name)
    new_module = cst.parse_module(source).visit(transformer)
    return EditResult(new_source=new_module.code, changes=transformer.changes)
