"""Install a pre-commit guard that runs the read-only audit gate.

The hook is a POSIX shell script. Git for Windows runs hooks through its bundled
``sh``, so one file works on every platform. It prefers the ``moray-surgeon``
console script and falls back to the interpreter that installed the hook, because
GUI Git clients often start hooks with a minimal ``PATH``.

The hook runs ``moray-surgeon audit``, which compares the working tree with
``HEAD``, fingerprints the changed Python files and then runs ruff → mypy →
pytest. Any non-zero result blocks the commit. ``git commit --no-verify`` is the
documented one-commit escape hatch.

Note that the audit gate is deliberately conservative: the verification tools see
the whole working tree, not only the staged index. An untracked file with a
syntax error can therefore block a commit until it is fixed or stashed.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from moray_surgeon.exceptions import HookError

_GIT_TIMEOUT_SECONDS = 60
_HOOK_MARKER = "# moray-surgeon pre-commit hook (managed)"
# Git's well-known empty tree: used to audit the very first commit.
_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def install_pre_commit_hook(
    repo_root: Path,
    *,
    force: bool = False,
    python_executable: str | Path | None = None,
) -> Path:
    """Write ``hooks/pre-commit`` so every commit must pass the audit.

    An existing hook is replaced only when it carries the moray-surgeon marker
    (idempotent reinstall) or when ``force`` is set; a foreign hook is never
    silently overwritten. ``core.hooksPath`` is honoured, so repositories that
    already route hooks elsewhere (husky, a shared hooks directory) work too.

    Args:
        repo_root: Any path inside the repository.
        force: Replace a hook that was not installed by moray-surgeon.
        python_executable: Interpreter recorded as the fallback launcher.
            Defaults to the interpreter running this code.

    Returns:
        The absolute path of the installed hook.

    Raises:
        HookError: The path is not a Git work tree, an existing foreign hook is
            in the way, or the file could not be written.
    """
    root = _repo_root(repo_root)
    hooks_dir = _hooks_directory(root)
    hook = hooks_dir / "pre-commit"
    interpreter = Path(python_executable or sys.executable).as_posix()
    script = _hook_script(interpreter)

    if hook.is_symlink():
        if not force:
            raise HookError(
                f"{hook} is a symlink; pass --force to replace it with a regular file"
            )
        try:
            hook.unlink()
        except OSError as exc:
            raise HookError(f"cannot remove the symlinked hook {hook}: {exc}") from exc
    elif hook.exists():
        try:
            existing = hook.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise HookError(f"cannot read the existing hook {hook}: {exc}") from exc
        if _HOOK_MARKER not in existing and not force:
            raise HookError(
                f"{hook} already exists and was not installed by moray-surgeon; "
                "pass --force to replace it"
            )

    try:
        hooks_dir.mkdir(parents=True, exist_ok=True)
        # Bytes, not write_text: a CRLF shebang breaks Git Bash on Windows.
        hook.write_bytes(script.encode("utf-8"))
        os.chmod(hook, 0o755)
    except OSError as exc:
        raise HookError(f"cannot install the pre-commit hook at {hook}: {exc}") from exc
    return hook


def _hook_script(interpreter: str) -> str:
    """Render the hook script, quoting the interpreter for the shell."""
    quoted = shlex.quote(interpreter)
    return (
        "#!/bin/sh\n"
        f"{_HOOK_MARKER}. Remove this file to disable.\n"
        "# Bypass one commit with: git commit --no-verify\n"
        "set -u\n"
        "\n"
        'repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {\n'
        '    echo "moray-surgeon: cannot find the repository root" >&2\n'
        "    exit 1\n"
        "}\n"
        "\n"
        "# The first commit has no HEAD to diff against.\n"
        "if git rev-parse --verify --quiet HEAD >/dev/null; then\n"
        '    base_ref="HEAD"\n'
        "else\n"
        f'    base_ref="{_EMPTY_TREE}"\n'
        "fi\n"
        "\n"
        'if [ -d "$repo_root/src" ]; then\n'
        '    mypy_target="src"\n'
        "else\n"
        '    mypy_target="."\n'
        "fi\n"
        "\n"
        "if command -v moray-surgeon >/dev/null 2>&1; then\n"
        '    exec moray-surgeon audit --path "$repo_root" \\\n'
        '        --base-ref "$base_ref" --mypy-target "$mypy_target"\n'
        "fi\n"
        f"exec {quoted} -m moray_surgeon audit --path "
        '"$repo_root" --base-ref "$base_ref" --mypy-target "$mypy_target"\n'
    )


def _repo_root(path: Path) -> Path:
    """Resolve the repository root from any path inside it."""
    candidate = path.expanduser().resolve()
    if not candidate.exists():
        raise HookError(f"{path} does not exist")
    result = _git(candidate, "rev-parse", "--show-toplevel", check=False)
    if result.returncode != 0:
        raise HookError(f"{path} is not inside a git working tree")
    return Path(result.stdout.strip()).resolve()


def _hooks_directory(root: Path) -> Path:
    """Return the directory Git will execute hooks from."""
    configured = _git(root, "config", "--get", "core.hooksPath", check=False)
    raw = configured.stdout.strip() if configured.returncode == 0 else ""
    if not raw:
        raw = _git(root, "rev-parse", "--git-path", "hooks").stdout.strip()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    return candidate


def _git(
    cwd: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run one git command with a timeout; never use a shell."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HookError(
            f"git {' '.join(args)} timed out after {_GIT_TIMEOUT_SECONDS}s"
        ) from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
        raise HookError(f"git {' '.join(args)} failed: {detail}")
    return result
