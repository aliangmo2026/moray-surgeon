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


class ConcurrencyError(MorayError):
    """A worker in a parallel refactor batch failed, so the batch was abandoned.

    The parallel path computes every batch member in a child process and only
    writes once all of them succeeded, so a failure here means *no* file of the
    batch was touched. ``instruction_index`` names the step whose worker raised
    (``None`` when the pool itself broke before any task could be attributed);
    the original exception is chained as ``__cause__`` so callers can still tell
    a rejected instruction (``ValueError``) from infrastructure trouble.
    """

    def __init__(self, message: str, instruction_index: int | None = None) -> None:
        super().__init__(message)
        self.instruction_index = instruction_index


class DependencyError(MorayError):
    """The plan's step graph cannot be executed as written.

    Covers the three ways a ``depends_on`` graph fails before any file is
    touched: a repeated id, an id that names no step, or a cycle. The error is
    deterministic — replanning the same JSON cannot fix it — so callers report it
    rather than retrying blindly; ``IntentParser`` does retry, because there the
    graph came from the model and the model can be told exactly what was wrong.
    """


class DuplicateInstructionIdError(DependencyError):
    """Two instructions share one id, so ``depends_on`` would be ambiguous."""


class UnknownDependencyError(DependencyError):
    """``depends_on`` names an id that is not part of the plan."""


class CircularDependencyError(DependencyError):
    """The ``depends_on`` graph contains a cycle, so no execution order exists."""


class GitError(MorayError):
    """Raised when a git command fails or the repo cannot host a transaction.

    ``stderr`` keeps git's own diagnostic text verbatim; the formatted message
    already embeds it, but callers may want to surface the raw output.
    """

    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


class CacheError(MorayError):
    """The local code index could not be opened, read or written.

    The index only ever narrows a search, so callers degrade to a full scan
    instead of failing: a read-only checkout, a locked database or a directory
    that is really a file must not stop a refactor.
    """


class ScopeViolationError(MorayError):
    """Raised when an attempt changed AST nodes its plan did not authorise.

    ``violations`` holds the qualified names of the offending nodes (or the
    affected file paths for created/deleted files), so callers can feed a
    machine-readable list back to the model or the log.
    """

    def __init__(self, message: str, violations: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.violations = violations


class DifferentialError(MorayError):
    """The differential sandbox could not produce equivalence evidence.

    A skip is not a verdict: impure functions, sandbox-blocked imports, budget
    exhaustion, methods or missing functions all mean "no evidence". Callers
    must treat the refactor as untested rather than equivalent -- the
    orchestrator logs the skip and moves on instead of blocking the commit.
    """


class UnsupportedLanguageError(MorayError):
    """No :class:`~moray_surgeon.engine.backends.base.LanguageBackend` matches a file.

    The dispatcher resolves a backend from ``target_file``'s suffix *before*
    reading or writing any byte, so a plan naming a file whose language has no
    registered backend (today: anything except Python and TypeScript) fails up
    front with this error instead of halfway inside a language-specific parser.
    """


class UnsupportedOperationError(MorayError):
    """A registered backend does not implement the requested atom.

    A language backend may support only part of the protocol — the TypeScript
    backend implements parsing, verification, rename and five other atoms, but
    not the Python-specific ``add_type_hints``/``convert_to_match_case`` or the
    project-aware ``move_symbol`` family. Raising this instead of silently
    editing nothing keeps the failure visible to the transaction.
    """


class TypeScriptBridgeError(MorayError):
    """The Node.js TypeScript bridge could not be started or answered badly.

    Covers a missing ``node`` executable, missing ``ts-morph`` dependencies, a
    timeout, a malformed response and a JSON-RPC internal error. Application
    errors (an invalid range, an unknown symbol) are reported as ``ValueError``
    so the dispatcher classifies them exactly like a Python atom rejection.
    """


class TimeTravelError(MorayError):
    """A historical rewrite could not be completed safely.

    The rewrite happens in a temporary linked worktree on a new branch. Every
    failure path — a failed refactor, a rebase conflict, a merge commit in the
    window — aborts and removes that worktree and the new branch. The original
    branch, its commits and the user's working tree are never modified.
    """


class HookError(MorayError):
    """The pre-commit guard could not be installed.

    Raised for a path that is not a Git repository, an existing hook that was not
    installed by moray-surgeon (unless ``--force`` is used), or a filesystem
    failure while writing ``hooks/pre-commit``.
    """


class FlywheelError(MorayError):
    """A fine-tuning dataset could not be extracted from the repository.

    Covers an unusable repository/rev, an unsupported output format, and git or
    archive failures. Individual commits and files that cannot be analysed are
    skipped; only a failure that prevents the export itself is raised.
    """

