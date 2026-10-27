"""Domain error hierarchy shared by every moray-surgeon layer.

This module owns the single ``MorayError`` base class; ``schema.exceptions``
re-exports it, so ``isinstance`` checks and ``except MorayError`` behave the same
no matter which layer raised.
"""

from __future__ import annotations


class MorayError(Exception):
    """Base class for every moray-surgeon domain error."""


class PathTraversalError(MorayError):
    """Raised when an instruction points outside the repository root.

    This is the second line of defense: the schema layer already rejects
    absolute and ``..``-bearing ``target_file`` values, but a plan built without
    validation (e.g. ``model_construct``) must not escape the repo either.
    """


class NoOpEditError(MorayError):
    """Raised when an instruction executed but matched nothing.

    A plan that claims work exists yet edits zero nodes is an authoring bug;
    skipping it silently would let the CLI report success for a no-change run.
    """


class InstructionError(MorayError):
    """Plan-level failure annotated with the offending instruction's position."""

    def __init__(self, message: str, instruction_index: int) -> None:
        super().__init__(message)
        self.instruction_index = instruction_index


class GitError(MorayError):
    """Raised when a git command fails or the repo cannot host a transaction.

    ``stderr`` keeps git's own diagnostic text verbatim; the formatted message
    already embeds it, but callers may want to surface the raw output.
    """

    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr
