#!/usr/bin/env python3
"""Build the Serverless skill package ``moray-skill.zip`` for cloud platforms.

Usage:

    python scripts/build_skill.py

The script stages a temporary directory containing

* ``moray_surgeon/`` — the core package (``tests/`` and ``__pycache__`` excluded),
* ``requirements.txt`` — the runtime dependencies the cloud entry needs,
* ``pyproject.toml`` / ``LICENSE`` / ``README.md`` — provenance for reviewers,
* ``handler.py`` — the platform-facing shim re-exporting
  ``moray_surgeon.cloud_entry.handler`` (configure the platform handler as
  ``handler.handler``),
* ``SKILL.md`` — the skill manifest the platform requires at the archive root.

then zips it all into ``moray-skill.zip`` at the repository root and verifies
the archive stays under the platform's 3 MB limit.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO_ROOT / "src" / "moray_surgeon"
OUTPUT_ZIP = REPO_ROOT / "moray-skill.zip"
MAX_ZIP_BYTES = 3 * 1024 * 1024  # WorkBuddy's upload cap

# Only the libraries the cloud entry actually imports at runtime. The CLI
# extras (typer, rich, mcp) are deliberately left out to keep the package slim;
# handler.py never imports cli.py or mcp_server.py.
REQUIREMENTS = """\
libcst>=1.4
pydantic>=2.7
httpx>=0.27
jedi>=0.19.0
"""

# Re-exported so the platform can configure its handler as `handler.handler`
# without knowing our package layout.
HANDLER_SHIM = """\
\"\"\"Platform-facing handler shim for Serverless deployment.\"\"\"

from moray_surgeon.cloud_entry import handler  # noqa: F401
"""

# WorkBuddy rejects an upload whose archive root has no SKILL.md, and requires
# it to open with a YAML frontmatter block (`---` alone on the first line) with
# a fixed field set: version, bilingual display names and bilingual
# descriptions. The version string is kept in sync with moray_surgeon.__version__.
SKILL_MD = """\
---
name: moray-surgeon
version: 0.3.0
display_name: 零幻觉代码重构与审计引擎
display_name_en: Moray Surgeon
description_zh: 基于 AST 的零幻觉代码重构与审计引擎。支持自然语言重构与安全审计。
description_en: AST-based zero-hallucination code refactoring and audit engine.
---

# moray-surgeon

基于 AST 的零幻觉代码重构与审计引擎。收到自然语言指令后，由模型生成结构化的
`RefactorPlan`，再由 libcst 按语法树精确执行；每次修改都在 Git 事务中进行，
并通过 `ruff` → `mypy` → `pytest` 三道闸门后才提交，失败自动回滚并重试。

## 支持的能力（action）

| action | 说明 | 必填参数 | 可选参数 |
|---|---|---|---|
| `refactor` | 执行重构：模型规划 → AST 执行 → 验证 → 提交/回滚 | `repo_path`（云端可达的 Git 仓库目录）、`intent`（自然语言指令） | `dry_run`（只看 diff 不提交）、`context`、`target_files`、`max_attempts` |
| `audit` | 执行只读代码审计：工作树与基线对比，列出改动的 AST 节点与验证结果 | `repo_path`（云端可达的 Git 仓库目录） | `base_ref`（对比基线，默认 `HEAD`） |

`repo_path` 必须是云端沙箱内可达的 Git 仓库根目录（含 `.git`）；调用方本机路径无法访问。模型端点通过部署环境变量 `MORAY_API_TYPE` / `MORAY_MODEL` / `MORAY_BASE_URL` / `MORAY_API_KEY` 配置，不接受来自请求的 URL。

## Payload Example

```json
{
  "action": "refactor",
  "repo_path": "/sandbox/repo",
  "intent": "把 parse_args 重命名为 parse_cli_options",
  "dry_run": true
}
```

```json
{
  "action": "audit",
  "repo_path": "/sandbox/repo",
  "base_ref": "HEAD"
}
```

响应为 `{"statusCode", "headers", "body"}`，`body` 中 `ok` 表示成败，失败时
`error` 给出可读原因（含 `repo_path` 不可达时的友好提示）。
"""


def stage(tmp: Path) -> None:
    """Copy the package and metadata into ``tmp`` and write the shim files."""
    shutil.copytree(
        PACKAGE_SRC,
        tmp / "moray_surgeon",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "tests"),
    )
    (tmp / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (tmp / "handler.py").write_text(HANDLER_SHIM, encoding="utf-8")
    (tmp / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    for name in ("pyproject.toml", "LICENSE", "README.md"):
        shutil.copy2(REPO_ROOT / name, tmp / name)


def zip_stage(tmp: Path) -> int:
    """Zip the staged tree into ``OUTPUT_ZIP``; return the byte size."""
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(tmp.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(tmp).as_posix())
    return OUTPUT_ZIP.stat().st_size


def main() -> int:
    """Stage, zip, verify size; return a process exit code."""
    with tempfile.TemporaryDirectory(prefix="moray-skill-") as tmpdir:
        tmp = Path(tmpdir)
        stage(tmp)
        size = zip_stage(tmp)

    print(f"Built {OUTPUT_ZIP} ({size / 1024:.1f} KB)")
    if size > MAX_ZIP_BYTES:
        print(
            f"ERROR: package exceeds the {MAX_ZIP_BYTES / 1024 / 1024:.0f} MB "
            "upload limit; trim the staged tree and rebuild.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
