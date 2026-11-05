"""Differential testing: prove a refactored function still behaves identically.

The engine rewrites code structurally and pytest proves the suite still passes,
but uncovered edge cases can hide semantic drift. This module executes the
original and the refactored function side by side on identical inputs and
compares return values, printed output and raised exceptions.

Everything runs in-process on purpose — no external process is started. The
sandbox rests on four independent walls:

1. Worker thread. The caller's deadline is honoured even if a worker hangs; the
   thread is a daemon, so it can never block interpreter shutdown.
2. Restricted builtins. ``exec`` receives an allowlist: no ``open``, no
   ``eval``/``exec``, no dynamic ``getattr``/``setattr``/``globals``. Class
   definitions still work (``__build_class__`` is included) and ``print`` is a
   capturing implementation, so printed output becomes comparable data.
3. Import allowlist. Only stdlib modules that are pure computation helpers
   (math, itertools, functools, collections, ...) can be imported; everything
   else raises ``ImportError`` inside the sandbox, which turns the run into a
   documented skip instead of a hidden I/O attempt.
4. Trace budget. A ``sys.settrace`` handler counts opcodes of every sandbox
   frame and enforces a wall-clock deadline. Exceeding either raises an
   exception derived from ``BaseException`` — sandbox-level ``except Exception``
   handlers cannot swallow it — and CPython runs trace handlers with tracing
   disabled, so sandbox code can neither intercept nor disable the watchdog.

Purity gate: functions that declare ``global``/``nonlocal``, yield, await,
import at call time, or reference process/file/network names (``os``, ``open``,
``random``, ``time``...) are *skipped*, not verified. A ``NameError`` raised by
BOTH versions is also a skip: inside this sandbox it usually means a needed
builtin was withheld, and both sides failing identically proves nothing.

Threat model: this sandbox contains verified project code against accidents —
infinite loops, stray output, accidental filesystem or process access — not
determined adversaries. Code that must be executed untrusted belongs in a
separate process.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import random
import re
import sys
import threading
import time
import types
import typing
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from io import StringIO
from types import FrameType, ModuleType
from typing import cast

import libcst as cst

from moray_surgeon.exceptions import DifferentialError

# A trace handler per typeshed's shape; used for settrace() and its return type.
_TraceFunc = Callable[[FrameType, str, object], object]

__all__ = [
    "DifferentialResult",
    "changed_function_names",
    "compare_functions",
    "verify_equivalence",
]

VERDICT_EQUIVALENT = "equivalent"
VERDICT_DIFFERENT = "different"
VERDICT_SKIPPED = "skipped"

# Generous for the small, pure functions this gate targets; the trace handler
# makes opcode events the primary budget and the wall clock the backstop.
_DEFAULT_OPCODE_BUDGET = 2_000_000
_DEFAULT_WALL_SECONDS = 10.0
# Wall-clock is checked every N opcodes: cheap enough to be invisible, frequent
# enough to catch long C-level calls (regex backtracking) at the next bytecode.
_CLOCK_CHECK_INTERVAL = 4096

_SANDBOX_FILENAME = "<moray-differential>"
_DETAIL_LIMIT = 200

_ALLOWED_MODULES = frozenset(
    {
        "abc",
        "bisect",
        "cmath",
        "collections",
        "contextlib",
        "copy",
        "dataclasses",
        "decimal",
        "difflib",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "re",
        "statistics",
        "string",
        "textwrap",
        "typing",
    }
)

# Dynamic evaluation and attribute mutation give sandbox code reach it should
# not have; everything else on the list fails early with a clear reason.
_DENIED_CALLS = frozenset(
    {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "quit",
        "setattr",
        "vars",
    }
)

# Referencing these names as attribute bases or call targets (``os.path``,
# ``time.sleep``) means I/O, randomness or concurrency: untestable here.
_DENIED_REFERENCES = _DENIED_CALLS | frozenset(
    {
        "asyncio",
        "code",
        "codeop",
        "ctypes",
        "curses",
        "datetime",
        "gc",
        "getpass",
        "http",
        "httpx",
        "imaplib",
        "importlib",
        "marshal",
        "multiprocessing",
        "os",
        "pathlib",
        "pdb",
        "pickle",
        "poplib",
        "random",
        "requests",
        "smtplib",
        "socket",
        "ssl",
        "subprocess",
        "sys",
        "telnetlib",
        "tempfile",
        "threading",
        "time",
        "tkinter",
        "turtle",
        "urllib",
        "uuid",
        "webbrowser",
    }
)

# Process/network verbs worth refusing on any receiver; filesystem verbs are
# left out because every path to them needs an import or builtin we withhold.
_DENIED_ATTRS = frozenset(
    {
        "Popen",
        "connect",
        "kill",
        "popen",
        "recv",
        "recvfrom",
        "send",
        "sendall",
        "sendto",
        "system",
        "terminate",
    }
)

_SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hasattr",
    "hash",
    "hex",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "Exception",
    "FloatingPointError",
    "IndexError",
    "KeyError",
    "LookupError",
    "MemoryError",
    "NameError",
    "NotImplementedError",
    "OverflowError",
    "OSError",
    "RecursionError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "UnicodeError",
    "ValueError",
    "ZeroDivisionError",
)

_SAFE_BUILTINS: dict[str, object] = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
_SAFE_BUILTINS["__build_class__"] = builtins.__build_class__
# __import__ is wired in below, right after _guarded_import is defined.

_STRING_POOL = ("alpha", "beta", "gamma", "delta", "a", "", "hello world", "42")
_HEX_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")

_PREWARMED = False


@dataclass(frozen=True)
class DifferentialResult:
    """Outcome of one differential comparison.

    Attributes:
        verdict: ``"equivalent"`` when every trial matched, ``"different"`` on
            the first observed mismatch, ``"skipped"`` when no evidence could
            be produced.
        detail: Human-readable difference or skip reason; empty when equivalent.
        trials: Number of input sets executed per version.
    """

    verdict: str
    detail: str = ""
    trials: int = 0


@dataclass(frozen=True)
class _CallRecord:
    """What one sandboxed call did: returned, printed, or raised."""

    returned: object
    printed: str
    error: tuple[str, str] | None


@dataclass
class _WorkerOutcome:
    """Mutable mailbox for the sandbox worker thread."""

    records: list[_CallRecord] = field(default_factory=list)
    used_args: list[tuple[object, ...]] = field(default_factory=list)
    skip: str | None = None
    budget: str | None = None


class _BudgetExceeded(BaseException):
    """Trace-handler kill switch: the sandbox ran past its opcode or wall budget.

    Deriving from ``BaseException`` is load-bearing: sandbox code that wraps its
    body in ``except Exception`` must not be able to swallow the watchdog.
    """


def verify_equivalence(
    original_source: str, new_source: str, func_name: str, *args: object, **kwargs: object
) -> bool:
    """Return True only when both sources' ``func_name`` behaves identically.

    Args:
        original_source: The module source before the refactor.
        new_source: The module source after the refactor.
        func_name: Module-level function to compare.
        *args: Optional explicit positional inputs; every given argument is
            passed to both versions. When omitted, random inputs are generated
            from the original function's signature.
        **kwargs: Optional explicit keyword inputs.

    Returns:
        True when every trial's return value, printed output and exception
        behaviour matched; False when any trial differed.

    Raises:
        DifferentialError: The sandbox could not produce evidence (impure
            function, blocked import, budget exhaustion, missing function, ...).
            A skip means "untested", never "equivalent".
    """
    result = compare_functions(
        original_source,
        new_source,
        func_name,
        args=args or None,
        kwargs=kwargs or None,
    )
    if result.verdict == VERDICT_SKIPPED:
        raise DifferentialError(result.detail)
    return result.verdict == VERDICT_EQUIVALENT


def compare_functions(
    original_source: str,
    new_source: str,
    func_name: str,
    *,
    args: Sequence[object] | None = None,
    kwargs: Mapping[str, object] | None = None,
    trials: int = 3,
    seed: int | None = None,
    opcode_budget: int = _DEFAULT_OPCODE_BUDGET,
    wall_seconds: float = _DEFAULT_WALL_SECONDS,
) -> DifferentialResult:
    """Run both versions of ``func_name`` in sandboxes and compare behaviour.

    Never raises: every obstacle becomes a ``"skipped"`` verdict with the reason
    in ``detail``, so callers (the orchestrator gate) can log and move on.

    Args:
        original_source: Module source before the refactor.
        new_source: Module source after the refactor.
        func_name: Module-level function to compare.
        args: Explicit positional inputs for both versions. None generates
            ``trials`` random input sets from the original signature.
        kwargs: Explicit keyword inputs for both versions.
        trials: How many random input sets to try when ``args`` is None.
        seed: RNG seed; derived from ``func_name`` when None, so identical
            calls produce identical inputs across processes.
        opcode_budget: Interpreter-step budget per sandbox run.
        wall_seconds: Wall-clock deadline per sandbox run.

    Returns:
        A tri-state result; see :class:`DifferentialResult`.
    """
    try:
        return _compare(
            original_source,
            new_source,
            func_name,
            args=args,
            kwargs=kwargs,
            trials=trials,
            seed=seed,
            opcode_budget=opcode_budget,
            wall_seconds=wall_seconds,
        )
    except DifferentialError as exc:
        return DifferentialResult(VERDICT_SKIPPED, str(exc))
    except Exception as exc:  # the gate must never break the refactor loop
        return DifferentialResult(VERDICT_SKIPPED, f"unexpected differential failure: {exc!r}")


def changed_function_names(original_source: str, new_source: str) -> list[str]:
    """Qualnames of functions whose code differs between the two sources.

    Covers module-level functions and class methods. Names present in only one
    version (added, removed or renamed symbols) are not reported: there is no
    before/after pair to compare.

    Raises:
        ValueError: Either source is not parseable (the caller decides whether
            that means "skip" or a hard error).
    """
    before = _collect_functions(original_source)
    after = _collect_functions(new_source)
    return sorted(name for name in after.keys() & before.keys() if before[name] != after[name])


def _compare(
    original_source: str,
    new_source: str,
    func_name: str,
    *,
    args: Sequence[object] | None,
    kwargs: Mapping[str, object] | None,
    trials: int,
    seed: int | None,
    opcode_budget: int,
    wall_seconds: float,
) -> DifferentialResult:
    """Body of :func:`compare_functions`; raises on skips, never on mismatches."""
    _prewarm_allowed_modules()
    call_kwargs: dict[str, object] = dict(kwargs or {})

    for label, source in (("original", original_source), ("refactored", new_source)):
        problem = _purity_problem(_function_source(source, func_name))
        if problem is not None:
            return DifferentialResult(
                VERDICT_SKIPPED, f"{label} version of '{func_name}': {problem}"
            )

    effective_seed = seed if seed is not None else zlib.crc32(func_name.encode("utf-8"))

    original_records, used_args = _sandbox_run(
        original_source,
        func_name,
        None if args is None else [tuple(args)],
        call_kwargs,
        trials,
        effective_seed,
        opcode_budget,
        wall_seconds,
    )
    new_records, _ = _sandbox_run(
        new_source,
        func_name,
        used_args,
        call_kwargs,
        trials,
        effective_seed,
        opcode_budget,
        wall_seconds,
    )

    try:
        for trial, (before, after) in enumerate(
            zip(original_records, new_records, strict=True), start=1
        ):
            if (
                before.error is not None
                and after.error is not None
                and before.error[0] == "NameError"
                and after.error[0] == "NameError"
            ):
                raise DifferentialError(
                    "both versions raised NameError inside the sandbox; the withheld "
                    "builtins make this evidence unusable, so the check is skipped"
                )
            difference = _compare_records(before, after)
            if difference is not None:
                return DifferentialResult(
                    VERDICT_DIFFERENT, f"trial {trial}: {difference}", len(used_args)
                )
    except DifferentialError as exc:
        return DifferentialResult(VERDICT_SKIPPED, str(exc), len(used_args))

    return DifferentialResult(VERDICT_EQUIVALENT, "", len(used_args))


def _sandbox_run(
    source: str,
    func_name: str,
    arg_sets: list[tuple[object, ...]] | None,
    call_kwargs: dict[str, object],
    trials: int,
    seed: int,
    opcode_budget: int,
    wall_seconds: float,
) -> tuple[list[_CallRecord], list[tuple[object, ...]]]:
    """Execute one namespace in a traced worker thread.

    Args:
        source: Module source to exec.
        func_name: Function to extract and call.
        arg_sets: Explicit inputs, or None to generate them from the signature
            inside the worker (the generated sets are returned so the second
            namespace receives the *identical* inputs).
        call_kwargs: Keyword inputs for every call.
        trials: Number of generated input sets.
        seed: RNG seed for generation.
        opcode_budget: Interpreter-step budget.
        wall_seconds: Wall-clock deadline.

    Returns:
        The call records and the argument sets actually used.

    Raises:
        DifferentialError: The sandbox could not produce evidence.
    """
    outcome = _WorkerOutcome()
    deadline = time.monotonic() + wall_seconds

    def worker() -> None:
        try:
            previous = sys.gettrace()
            # Our tracer returns "object" (itself, the previous tracer, or None);
            # typeshed's TraceFunction protocol is stricter than reality.
            sys.settrace(_make_tracer(deadline, opcode_budget))  # type: ignore[arg-type]
            try:
                namespace = _exec_module(source)
                candidate = namespace.get(func_name)
                if not callable(candidate):
                    raise DifferentialError(f"function '{func_name}' not found or not callable")
                func = cast(Callable[..., object], candidate)
                sets = arg_sets if arg_sets is not None else _generate_arg_sets(func, trials, seed)
                outcome.records = [_invoke(namespace, func, item, call_kwargs) for item in sets]
                outcome.used_args = list(sets)
            finally:
                sys.settrace(previous)
        except _BudgetExceeded as exc:
            outcome.budget = str(exc)
        except DifferentialError as exc:
            outcome.skip = str(exc)
        except Exception as exc:  # sandbox internals must never crash the caller
            outcome.skip = f"sandbox internals failed: {type(exc).__name__}: {exc}"

    thread = threading.Thread(target=worker, name="moray-differential", daemon=True)
    thread.start()
    thread.join(timeout=wall_seconds + 5.0)
    if thread.is_alive():
        raise DifferentialError(
            "sandbox worker missed its deadline and was abandoned; no equivalence evidence"
        )
    if outcome.budget is not None:
        raise DifferentialError(
            f"sandbox budget exhausted ({outcome.budget}); cannot bound the runtime, skipping"
        )
    if outcome.skip is not None:
        raise DifferentialError(outcome.skip)
    return outcome.records, outcome.used_args


def _exec_module(source: str) -> dict[str, object]:
    """Exec ``source`` with restricted builtins; every failure is a skip."""
    namespace: dict[str, object] = {
        "__builtins__": _SAFE_BUILTINS,
        "__name__": "moray_differential_module",
        "__doc__": None,
        "__package__": None,
        "__spec__": None,
        # Module-level prints are not part of the verdict; give them a sink.
        "print": _make_print(StringIO()),
    }
    try:
        # dont_inherit=True: without it, compile() inherits THIS module's
        # `from __future__ import annotations`, and sandbox annotations would be
        # stored as strings, breaking signature-driven input generation.
        code = compile(source, _SANDBOX_FILENAME, "exec", dont_inherit=True)
    except SyntaxError as exc:
        raise DifferentialError(f"sandbox could not parse the source: {exc}") from None
    # S102 acknowledged (security review): the exec below IS the differential
    # gate the task mandates. The code it runs is this repository's own file
    # content -- never raw model output -- and only inside four walls:
    # restricted builtins (no open/eval/getattr/globals/import), an import
    # allowlist, a daemon worker thread with a join deadline, and an
    # opcode/wall-clock watchdog whose kill switch derives from BaseException.
    # Full threat model lives in the module docstring.
    try:
        exec(code, namespace)  # noqa: S102 - sandboxed by design, see module docstring
    except _BudgetExceeded:
        raise
    except Exception as exc:
        raise DifferentialError(
            f"module-level execution failed ({type(exc).__name__}: {_shorten(str(exc))}); "
            "the sandbox likely withholds an import or builtin the module needs"
        ) from None
    return namespace


def _invoke(
    namespace: dict[str, object],
    func: Callable[..., object],
    arguments: tuple[object, ...],
    call_kwargs: Mapping[str, object],
) -> _CallRecord:
    """Call one input set with print captured; sandbox exceptions become data."""
    buffer = StringIO()
    namespace["print"] = _make_print(buffer)
    try:
        returned = func(*arguments, **call_kwargs)
    except _BudgetExceeded:
        raise
    except BaseException as exc:  # sandbox code may raise anything it likes
        return _CallRecord(None, buffer.getvalue(), (type(exc).__name__, _shorten(str(exc), 300)))
    return _CallRecord(returned, buffer.getvalue(), None)


def _make_tracer(deadline: float, opcode_budget: int) -> _TraceFunc:
    """Build the trace handler enforcing the opcode and wall-clock budgets.

    The CPython detail this handler must respect: a frame's *local* trace
    function is whatever the caller's trace returned for the ``call`` event, so
    once we hand a frame to another tracer, every frame it calls -- including
    the sandbox frames -- is traced by THAT tracer. Delegating non-sandbox
    frames to coverage's tracer would therefore blind the watchdog after the
    first hop. Instead we always keep ourselves installed as the local trace and
    merely *forward* events to the previous tracer (e.g. coverage), which keeps
    line coverage counting while the watchdog stays in charge.
    """
    state = {"opcodes": 0}
    previous = sys.gettrace()

    def tracer(frame: FrameType, event: str, arg: object) -> object:
        if frame.f_code.co_filename != _SANDBOX_FILENAME:
            if event == "line" and time.monotonic() > deadline:
                raise _BudgetExceeded("wall-clock deadline exceeded")
            if previous is not None and event in ("line", "return", "exception"):
                # Forward these to the previous tracer (e.g. pytest-cov) so line
                # counting keeps working, but DISCARD its return value: the
                # return would make it the frame's local trace again, and then
                # sandbox callees are traced by coverage instead of the
                # watchdog -- the exact bug this handler exists to prevent.
                previous(frame, event, arg)  # type: ignore[arg-type]
            return tracer
        frame.f_trace_opcodes = True
        if event == "opcode":
            state["opcodes"] += 1
            if state["opcodes"] > opcode_budget:
                raise _BudgetExceeded(f"opcode budget of {opcode_budget} exhausted")
            if state["opcodes"] % _CLOCK_CHECK_INTERVAL == 0 and time.monotonic() > deadline:
                raise _BudgetExceeded("wall-clock deadline exceeded")
        elif event == "line" and time.monotonic() > deadline:
            raise _BudgetExceeded("wall-clock deadline exceeded")
        return tracer

    return tracer


def _make_print(buffer: StringIO) -> object:
    """A print() replacement that writes into the trial's capture buffer."""

    def sandbox_print(
        *values: object, sep: str = " ", end: str = "\n", file: object = None, flush: bool = False
    ) -> None:
        buffer.write(sep.join(str(value) for value in values) + end)

    return sandbox_print


def _guarded_import(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: Sequence[str] | None = None,
    level: int = 0,
) -> ModuleType:
    """Import hook for the sandbox: allowlist or nothing."""
    if name.split(".", 1)[0] not in _ALLOWED_MODULES:
        raise ImportError(f"import of '{name}' is blocked by the differential sandbox")
    return builtins.__import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS["__import__"] = _guarded_import


_PREWARM_SKIP: set[str] = set()


def _prewarm_allowed_modules() -> None:
    """Import the allowlist once, outside tracing, so sandboxed imports are cheap."""
    global _PREWARMED
    if _PREWARMED:
        return
    for module_name in sorted(_ALLOWED_MODULES):
        if module_name in _PREWARM_SKIP:
            continue
        try:
            importlib.import_module(module_name)
        except Exception:
            # A platform without an optional stdlib module must not break the
            # gate; the sandboxed import will fail at call time and skip.
            _PREWARM_SKIP.add(module_name)
    _PREWARMED = True


def _generate_arg_sets(
    func: object, trials: int, seed: int
) -> list[tuple[object, ...]]:
    """Synthesise ``trials`` input sets from the original function's signature."""
    try:
        signature = inspect.signature(func)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DifferentialError(f"cannot inspect the function signature: {exc}") from None
    rng = random.Random(seed)
    sets: list[tuple[object, ...]] = []
    for _ in range(trials):
        values: list[object] = []
        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise DifferentialError(
                    f"parameter '{parameter.name}' is variadic; pass explicit args instead"
                )
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                if parameter.default is inspect.Parameter.empty:
                    raise DifferentialError(
                        f"keyword-only parameter '{parameter.name}' has no default; "
                        "pass explicit kwargs instead"
                    )
                continue
            if parameter.default is not inspect.Parameter.empty:
                values.append(parameter.default)
                continue
            values.append(_random_value(parameter.annotation, parameter.name, rng))
        sets.append(tuple(values))
    return sets


def _random_value(annotation: object, name: str, rng: random.Random) -> object:
    """One random value matching an annotation; unbuildable types skip the run."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return rng.randint(-8, 8)
    if annotation is type(None):
        return None
    if annotation is int:
        return rng.randint(-8, 8)
    if annotation is float:
        return round(rng.uniform(-10.0, 10.0), 3)
    if annotation is str:
        return rng.choice(_STRING_POOL)
    if annotation is bool:
        return rng.choice((True, False))
    if annotation is bytes:
        return rng.choice((b"a", b"abc", b""))
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        return _random_value(rng.choice(typing.get_args(annotation)), name, rng)
    if origin in (list, set, frozenset, tuple):
        args = typing.get_args(annotation)
        item_type = args[0] if args else int
        items = [_random_value(item_type, name, rng) for _ in range(rng.randint(1, 4))]
        return tuple(items) if origin is tuple else origin(items)
    if origin is dict:
        args = typing.get_args(annotation)
        key_type = args[0] if len(args) == 2 else str
        value_type = args[1] if len(args) == 2 else int
        return {
            _random_value(key_type, name, rng): _random_value(value_type, name, rng)
            for _ in range(rng.randint(1, 3))
        }
    raise DifferentialError(
        f"cannot synthesise a value for parameter '{name}' annotated as {annotation!r}; "
        "pass explicit args instead"
    )


def _compare_records(before: _CallRecord, after: _CallRecord) -> str | None:
    """Compare one trial pair; None means equivalent, else a difference report."""
    if before.printed != after.printed:
        return f"print output differs: {_shorten(before.printed)!r} vs {_shorten(after.printed)!r}"
    if (before.error is None) != (after.error is None):
        if before.error is not None:
            return f"only the original version raised {before.error[0]}: {before.error[1]}"
        if after.error is not None:
            return f"only the refactored version raised {after.error[0]}: {after.error[1]}"
    if before.error is not None and after.error is not None:
        if before.error != after.error:
            return f"exceptions differ: {before.error} vs {after.error}"
        return None
    if not _values_equal(before.returned, after.returned):
        original_repr = _shorten(repr(before.returned))
        refactored_repr = _shorten(repr(after.returned))
        return f"return values differ: {original_repr} vs {refactored_repr}"
    return None


def _values_equal(original: object, refactored: object) -> bool:
    """Equality that survives cross-namespace objects and NaN.

    Two instances of same-named classes defined in separate namespaces cannot
    use ``==`` meaningfully, so the comparison falls through three layers:
    direct ``==``, structural ``__dict__`` state, then a scrubbed repr (hex
    addresses normalised). A *default* repr carries no state, so it never
    certifies equality on its own -- differing ``__dict__`` state wins.
    """
    if type(original).__name__ != type(refactored).__name__:
        return False
    try:
        if bool(original == refactored):
            return True
    except Exception:
        pass
    try:
        original_state = getattr(original, "__dict__", None)
        refactored_state = getattr(refactored, "__dict__", None)
        if isinstance(original_state, dict) and isinstance(refactored_state, dict):
            return original_state == refactored_state
    except Exception:
        pass
    try:
        original_repr = _HEX_ADDRESS.sub("0xADDR", repr(original))
        refactored_repr = _HEX_ADDRESS.sub("0xADDR", repr(refactored))
        if original_repr == refactored_repr and "object at 0xADDR" not in original_repr:
            return True  # a custom repr that actually renders state
    except Exception:
        pass
    return False


def _function_source(source: str, func_name: str) -> str:
    """Extract one module-level function's source for the purity scan."""
    if "." in func_name:
        raise DifferentialError(
            f"'{func_name}' is a method; differential testing runs on module-level functions only"
        )
    try:
        module = cst.parse_module(source)
    except Exception as exc:
        raise DifferentialError(f"source does not parse ({type(exc).__name__})") from None
    for statement in module.body:
        if isinstance(statement, cst.FunctionDef) and statement.name.value == func_name:
            return module.code_for_node(statement)
    raise DifferentialError(f"module-level function '{func_name}' not found")


def _collect_functions(source: str) -> dict[str, str]:
    """Rendered source of every module-level function and method, by qualname."""
    module = cst.parse_module(source)
    collected: dict[str, str] = {}
    for statement in module.body:
        if isinstance(statement, cst.FunctionDef):
            collected[statement.name.value] = module.code_for_node(statement)
        elif isinstance(statement, cst.ClassDef):
            for inner in statement.body.body:
                if isinstance(inner, cst.FunctionDef):
                    qualname = f"{statement.name.value}.{inner.name.value}"
                    collected[qualname] = module.code_for_node(inner)
    return collected


def _purity_problem(function_source: str) -> str | None:
    """Static skip reasons; None when the function looks sandbox-verifiable.

    Conservative by design: the sandbox's runtime walls catch most I/O anyway
    (missing builtins raise NameError), so this scan mainly refuses early and
    with a precise reason.
    """
    try:
        tree = ast.parse(function_source)
    except SyntaxError as exc:
        return f"source does not parse ({exc})"
    node = next(
        (item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if node is None:
        return "the extracted source is not a plain function definition"
    if isinstance(node, ast.AsyncFunctionDef):
        return "async functions are skipped"
    for item in ast.walk(node):
        if isinstance(item, (ast.Global, ast.Nonlocal)):
            return f"'{type(item).__name__.lower()}' statement detected"
        if isinstance(item, (ast.Yield, ast.YieldFrom, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            return "generator/await behaviour is skipped"
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            return "call-time imports are skipped"
        if isinstance(item, ast.Call):
            target = item.func
            if isinstance(target, ast.Name) and target.id in _DENIED_CALLS:
                return f"call to '{target.id}' is refused"
            if isinstance(target, ast.Attribute) and target.attr in _DENIED_ATTRS:
                return f"call to '.{target.attr}' is refused"
        if isinstance(item, ast.Attribute):
            if item.attr in _DENIED_ATTRS:
                return f"attribute access '.{item.attr}' is refused"
            if isinstance(item.value, ast.Name) and item.value.id in _DENIED_REFERENCES:
                return f"use of '{item.value.id}' is refused"
    return None


def _shorten(text: str, limit: int = _DETAIL_LIMIT) -> str:
    """Bound diagnostic text so feedback stays readable."""
    if len(text) <= limit:
        return text
    return text[:limit] + "... (truncated)"
