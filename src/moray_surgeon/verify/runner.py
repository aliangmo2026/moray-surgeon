"""Run ruff, mypy and pytest against a repository and report the first failure.

The Verifier is deliberately the only place that shells out besides the
transaction layer. Two rules shape it:

* Failures are data, not exceptions. A red gate, a missing tool, a bad ``cwd`` or
  a timeout all come back as a failed :class:`VerifyResult` so the orchestrator
  can feed it to the model instead of crashing the CLI.
* The interpreter is explicit. Every stage runs ``<python_executable> -m <tool>``
  with the interpreter the caller chose, never a bare ``python`` from PATH: this
  machine has both 3.8 and 3.12, and only the chosen one has the tooling.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_COMMAND_TIMEOUT_SECONDS = 300
_FEEDBACK_LIMIT = 4000


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of one verification run.

    Attributes:
        ok: True only when every stage passed.
        stage: The gate that produced this result: ``"ruff"``, ``"mypy"``,
            ``"pytest"``, ``"all"`` for a fully green run, or ``"noop"`` for the
            orchestrator's synthetic empty-plan feedback.
        stdout: Captured standard output. ruff, mypy and pytest all write their
            diagnostics here, so this is usually the useful field.
        stderr: Captured standard error, kept verbatim even when empty.
    """

    ok: bool
    stage: str
    stdout: str = ""
    stderr: str = ""

    def feedback(self) -> str:
        """Render failure text for the LLM's self-repair prompt.

        Prefers stderr (real crashes, interpreter errors) and falls back to
        stdout (the normal channel for ruff/mypy/pytest diagnostics). Truncates to
        ``_FEEDBACK_LIMIT`` characters so the prompt stays bounded.
        """
        detail = self.stderr or self.stdout
        return f"[{self.stage} failed]\n{detail[:_FEEDBACK_LIMIT]}"


class Verifier:
    """Run the three project gates in order, stopping at the first failure."""

    def __init__(
        self,
        repo_root: Path,
        python_executable: str = sys.executable,
        mypy_target: str = "src",
        pytest_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Bind the verifier to a repository, an interpreter and gate options.

        Args:
            repo_root: Working directory for every stage command.
            python_executable: Interpreter used for ``-m ruff`` / ``-m mypy`` /
                ``-m pytest``. Defaults to the running interpreter so the tools
                always resolve from the same environment, never from a bare
                ``python`` that may be a different version.
            mypy_target: Argument passed to ``mypy`` (defaults to ``src``).
            pytest_args: Arguments passed to ``pytest``; defaults to ``["-q"]``.
            env: Environment passed to every stage as-is. When given it replaces
                the inherited environment, so pass a full mapping (for example
                ``{**os.environ, "PYTHONPATH": "src"}``) if you need inheritance.
        """
        self._repo_root = repo_root.resolve()
        self._python_executable = python_executable
        self._mypy_target = mypy_target
        self._pytest_args = list(pytest_args) if pytest_args is not None else ["-q"]
        self._environment = env

    def verify(self) -> VerifyResult:
        """Run ruff, mypy and pytest in order; return the first failure.

        Returns:
            ``VerifyResult(ok=True, stage="all")`` when all three gates pass, or
            the failing stage's result unchanged. Later stages never run after a
            failure.
        """
        stages = (
            ("ruff", ("check", ".")),
            ("mypy", (self._mypy_target,)),
            ("pytest", tuple(self._pytest_args)),
        )
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        for stage, stage_args in stages:
            result = self._run_command(self._stage_command(stage, stage_args), stage)
            if not result.ok:
                return result
            stdout_parts.append(result.stdout)
            stderr_parts.append(result.stderr)

        return VerifyResult(
            ok=True,
            stage="all",
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
        )

    def _stage_command(self, module: str, args: Sequence[str]) -> list[str]:
        """Build ``<interpreter> -m <module> <args>`` for one stage."""
        return [self._python_executable, "-m", module, *args]

    def _run_command(self, command: Sequence[str], stage: str) -> VerifyResult:
        """Execute one command, converting every failure mode into a result.

        Never raises for operational problems: a missing executable, an invalid
        ``cwd``, or a timeout is reported through the returned ``VerifyResult``.

        Args:
            command: Full argv. ``command[0]`` may be an absolute interpreter path
                or a bare executable name resolved through ``shutil.which``.
            stage: Stage label attached to the result.

        Returns:
            A result whose ``ok`` mirrors the exit code. A non-zero exit is not an
            exception: it is the normal way ruff/mypy/pytest report findings.
        """
        argv = list(command)
        executable = _resolve_executable(argv[0])
        if executable is None:
            return VerifyResult(ok=False, stage=stage, stderr=f"command not found: {argv[0]}")
        argv[0] = executable

        try:
            completed = subprocess.run(
                argv,
                cwd=self._repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_COMMAND_TIMEOUT_SECONDS,
                check=False,
                env=self._environment,
            )
        except subprocess.TimeoutExpired:
            return VerifyResult(ok=False, stage=stage, stderr="timed out")
        except OSError as exc:
            return VerifyResult(ok=False, stage=stage, stderr=f"failed to execute {argv[0]}: {exc}")

        return VerifyResult(
            ok=completed.returncode == 0,
            stage=stage,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def _resolve_executable(executable: str) -> str | None:
    """Resolve a path string or bare tool name to an existing executable.

    Absolute interpreter paths (``sys.executable``) are used as-is when present;
    a bare name such as ``"ruff"`` is looked up on PATH. Returning ``None`` lets
    ``_run_command`` report a failed result instead of raising.
    """
    path = Path(executable)
    if path.is_absolute() and path.is_file():
        return str(path)
    return shutil.which(executable)
