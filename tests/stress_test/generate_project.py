"""Generate the synthetic project used by the moray-surgeon stress suite.

The generated package is deliberately shaped like a real codebase rather than a
toy: 50 modules in two subpackages, 300 classes across six inheritance roles,
a runtime dependency ring closed with lazy imports, and enough formatting and
comments that a naive text-level refactor would leave visible damage.

Usage::

    python tests/stress_test/generate_project.py --target /tmp/moray-stress

The project is self-contained (stdlib only, pytest for its smoke tests) and is
verified with the same ruff/mypy/pytest gates the engine runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MODULE_COUNT = 50
DOMAIN_COUNT = 25
CLASS_COUNT = 6

_PACKAGE = "stresspkg"
_DOMAIN = f"{_PACKAGE}.domain"
_SERVICES = f"{_PACKAGE}.services"


@dataclass(frozen=True)
class ProjectStats:
    """Shape of a generated project, as reported by the stress harness.

    Attributes:
        root: Directory that contains the project.
        modules: Number of generated ``mNN`` modules.
        classes: Number of generated classes.
        files: Number of files written.
        lines: Total physical lines across all written files.
    """

    root: Path
    modules: int
    classes: int
    files: int
    lines: int

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly representation for the stress report."""
        return {
            "root": str(self.root),
            "modules": self.modules,
            "classes": self.classes,
            "files": self.files,
            "lines": self.lines,
        }


def module_spec(index: int) -> tuple[str, str]:
    """Return ``(package, dotted_module)`` for one ring position."""
    package = _DOMAIN if index < DOMAIN_COUNT else _SERVICES
    return package, f"{package}.m{index:02d}"


def next_module_spec(index: int) -> tuple[str, str]:
    """Return the ring successor of ``index`` (the last one wraps to m00)."""
    return module_spec((index + 1) % MODULE_COUNT)


_MODULE_TEMPLATE = '''"""Synthetic module @@IDX@@ used by the moray-surgeon stress suite.

The module participates in a 50-node dependency ring: ``neighbour_key`` reaches
the next module with a lazy import and the last module wraps back to the first.
The typing-only import below mirrors the same edge for static analysis.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from stresspkg.core.base import BaseEntity, BaseRepository, BaseService
from stresspkg.core.errors import LookupFailure
from stresspkg.core.validation import Record, format_key, reject_record, validate_record

if TYPE_CHECKING:
    from @@NEXT_MODULE@@ import Entity@@NEXT_IDX@@

LOGGER = logging.getLogger(__name__)
DEFAULT_LIMIT = 10
KIND_LABEL = "entity_@@IDX@@"
# A non-call reference keeps the validator import used even when a refactor
# rewrites every call site (the Bombing B poison relies on this).
RECORD_CHECKS = (validate_record,)


def neighbour_key(key: str, depth: int = 1) -> str:
    """Follow the lazy edge to the next module in the ring."""
    if depth <= 0:
        return key
    from @@NEXT_MODULE@@ import neighbour_key as next_key

    return next_key(key, depth - 1)


def normalise_key(key: str) -> str:
    """Normalise a key without importing anything else."""
    return key.strip().lower().replace(" ", "-")


class Entity@@IDX@@(BaseEntity):
    """A concrete entity with a lazily linked neighbour."""

    kind = KIND_LABEL

    def __init__(self, key: str, payload: dict[str, int]) -> None:
        super().__init__(key)
        self.payload = payload
        self._cache: OrderedDict[str, int] = OrderedDict()

    def describe(self) -> str:
        return str(self.key)

    def peer(self) -> Entity@@NEXT_IDX@@ | None:
        """Return the typed peer from the next module (typing-only import)."""
        return None

    def score(self, limit: int = DEFAULT_LIMIT) -> int:
        total = 0
        for name, value in self.payload.items():
            record = Record(key=name, value=value)
            if validate_record(record):
                total += value
            else:
                reject_record(record)
        return min(total, limit)

    def to_json(self) -> str:
        return json.dumps(self.payload)

    def from_json(self, text: str) -> dict[str, int]:
        return json.loads(text)

    def label(self) -> str:
        return format_key(self.key)


class Service@@IDX@@(BaseService):
    """Service owning a registry of entities."""

    def __init__(self) -> None:
        super().__init__()
        self.registry: dict[str, Entity@@IDX@@] = {}

    def register(self, entity: Entity@@IDX@@) -> None:
        self.registry[entity.key] = entity
        self._cache[entity.key] = entity.score()

    def lookup(self, key: str) -> Entity@@IDX@@ | None:
        return self.registry.get(key)

    def require(self, key: str) -> Entity@@IDX@@:
        entity = self.lookup(key)
        if entity is None:
            raise LookupFailure(f"missing entity {key}")
        return entity

    def total_score(self) -> int:
        return sum(entity.score() for entity in self.registry.values())


class Repository@@IDX@@(BaseRepository):
    """Read/write facade over a service."""

    def __init__(self, service: Service@@IDX@@) -> None:
        super().__init__()
        self.service = service

    def store(self, entity: Entity@@IDX@@) -> None:
        self.service.register(entity)

    def fetch(self, key: str) -> Entity@@IDX@@:
        return self.service.require(key)

    def count(self) -> int:
        return len(self.service.registry)


class Handler@@IDX@@(Service@@IDX@@):
    """A handler that reads twice from the inherited registry."""

    def handle(self, entity: Entity@@IDX@@) -> int:
        self.register(entity)
        return entity.score() * 2

    def describe(self) -> str:
        return f"handler-@@IDX@@:{normalise_key(self.__class__.__name__)}"


class Policy@@IDX@@(BaseEntity):
    """A policy entity with a bounded value."""

    kind = "policy_@@IDX@@"

    def __init__(self, key: str, ceiling: int = 100) -> None:
        super().__init__(key)
        self.ceiling = ceiling

    def allows(self, value: int) -> bool:
        return 0 <= value <= self.ceiling


class Snapshot@@IDX@@(Entity@@IDX@@):
    """A point-in-time copy of an entity."""

    def clone(self) -> Snapshot@@IDX@@:
        return Snapshot@@IDX@@(self.key, dict(self.payload))

    def describe(self) -> str:
        return f"snapshot:{super().describe()}"


# End of module m@@IDX@@. The ring is closed by m49 -> m00.
'''


_PYPROJECT = """[project]
name = "moray-stress-project"
version = "0.1.0"
requires-python = ">=3.11"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]

[tool.mypy]
python_version = "3.11"
warn_unused_ignores = true
no_implicit_optional = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
"""

_CORE_BASE = '''"""Base types shared by every synthetic domain module."""

from __future__ import annotations

import json
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


class BaseEntity:
    """Identity plus attributes, the root of the entity hierarchy."""

    kind = "base"

    def __init__(self, key: str) -> None:
        self.key = key
        self.attributes: dict[str, object] = {}

    def describe(self) -> str:
        return f"{self.kind}:{self.key}"

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "key": self.key, **self.attributes}


class BaseService:
    """Shared registry plumbing for service-like objects."""

    def __init__(self) -> None:
        self._cache: dict[str, int] = {}

    def cached(self, key: str) -> int | None:
        return self._cache.get(key)


class BaseRepository:
    """Storage root with a small json round-trip helper."""

    def __init__(self) -> None:
        self.storage: dict[str, Any] = {}

    def dump(self) -> str:
        return json.dumps(self.storage, sort_keys=True)


class Registry:
    """A counted registry used by the smoke tests."""

    def __init__(self) -> None:
        self._items: list[BaseEntity] = []

    def add(self, entity: BaseEntity) -> None:
        self._items.append(entity)

    def __len__(self) -> int:
        return len(self._items)
'''

_CORE_ERRORS = '''"""Exception hierarchy for the synthetic package."""

from __future__ import annotations


class DomainFailure(Exception):
    """Base class for every domain error."""


class ValidationFailure(DomainFailure):
    """Raised when a record fails validation."""


class RecordRejected(ValidationFailure):
    """Raised when a record is explicitly rejected."""


class LookupFailure(DomainFailure):
    """Raised when a required entity is missing."""
'''

_CORE_VALIDATION = '''"""Record validation and payload encoding helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass

from stresspkg.core.errors import RecordRejected


@dataclass(frozen=True)
class Record:
    """One key/value pair flowing through the entity graph."""

    key: str
    value: int


def validate_record(record: Record) -> bool:
    """Accept a record whose key is non-empty and value non-negative."""
    return bool(record.key) and record.value >= 0


def reject_record(record: Record) -> bool:
    """Reject a record loudly; this function never returns."""
    raise RecordRejected(f"record rejected: {record.key}")


def format_key(key: str) -> str:
    """Format a key for display."""
    return key.strip().lower().replace(" ", "-")


def encode_payload(value: object) -> str:
    """Encode an arbitrary payload as sorted JSON text."""
    return json.dumps(value, default=str, sort_keys=True)
'''

_CORE_LEGACY = '''"""Deprecated symbols kept only so the stress plan can drop re-exports."""

from __future__ import annotations

from stresspkg.core.base import BaseEntity, BaseRepository, BaseService


class LegacyRecord(BaseEntity):
    """A record shape kept for old callers."""

    kind = "legacy"

    def describe(self) -> str:
        return f"legacy:{self.key}"


class ObsoleteService(BaseService):
    """A service kept for old callers."""

    def count(self) -> int:
        return len(self._cache)


class RetiredHandler(BaseService):
    """A handler kept for old callers."""

    def retired(self) -> bool:
        return True


class DeprecatedPolicy(BaseEntity):
    """A policy kept for old callers."""

    kind = "deprecated"

    def describe(self) -> str:
        return f"deprecated:{self.key}"


class LegacyRepository(BaseRepository):
    """A repository kept for old callers."""

    def size(self) -> int:
        return len(self.storage)
'''

_PACKAGE_INIT = '''"""Synthetic package used by the moray-surgeon stress suite."""

from stresspkg.core.base import BaseEntity as BaseEntity
from stresspkg.core.base import BaseRepository as BaseRepository
from stresspkg.core.base import BaseService as BaseService
from stresspkg.core.base import Registry as Registry
from stresspkg.core.errors import LookupFailure as LookupFailure
from stresspkg.core.errors import RecordRejected as RecordRejected
from stresspkg.core.errors import ValidationFailure as ValidationFailure
from stresspkg.core.legacy import DeprecatedPolicy as DeprecatedPolicy
from stresspkg.core.legacy import LegacyRecord as LegacyRecord
from stresspkg.core.legacy import ObsoleteService as ObsoleteService
from stresspkg.core.legacy import RetiredHandler as RetiredHandler
from stresspkg.core.validation import Record as Record
from stresspkg.core.validation import format_key as format_key
from stresspkg.core.validation import validate_record as validate_record
'''

_SUBPACKAGE_INIT = '''"""Modules for one half of the synthetic dependency ring."""
'''

_SMOKE_TESTS = '''"""Smoke tests for the synthetic stress project."""

from __future__ import annotations

from stresspkg.core.base import Registry
from stresspkg.domain.m00 import Entity00, Service00
from stresspkg.domain.m04 import Entity04
from stresspkg.domain.m20 import Entity20
from stresspkg.services.m49 import Entity49, neighbour_key


def test_entities_report_their_scores() -> None:
    entity = Entity00("alpha", {"a": 1, "b": 2})
    assert entity.score() == 3
    assert entity.describe() == "alpha"


def test_validation_scores_only_valid_records() -> None:
    entity = Entity04("beta", {"a": 1, "b": 2})
    assert entity.score() == 3


def test_service_round_trip() -> None:
    service = Service00()
    service.register(Entity00("alpha", {"a": 4}))
    assert service.require("alpha").score() == 4
    assert service.total_score() == 4


def test_registry_counts_entities() -> None:
    registry = Registry()
    registry.add(Entity00("alpha", {}))
    registry.add(Entity20("gamma", {}))
    assert len(registry) == 2


def test_ring_edge_resolves() -> None:
    assert neighbour_key("delta", depth=3) == "delta"


def test_last_module_scores_too() -> None:
    assert Entity49("omega", {"z": 9}).score() == 9
'''

_README = """# moray-surgeon stress project

Generated by `tests/stress_test/generate_project.py`; do not edit by hand.

- 50 modules (`stresspkg/domain/m00..m24`, `stresspkg/services/m25..m49`)
- 300 classes across six inheritance roles
- a runtime dependency ring closed with lazy imports
- `ruff`, `mypy` and `pytest` are the verification gates
"""

_GITIGNORE = """__pycache__/
*.py[cod]
.mypy_cache/
.pytest_cache/
.ruff_cache/
"""


def render_module(index: int) -> str:
    """Render one module source from the fixed template.

    The only substitutions are generated identifiers (a two-digit index and a
    module path), never user input; the template is checked into the repository.
    """
    _, next_dotted = next_module_spec(index)
    source = _MODULE_TEMPLATE
    source = source.replace("@@NEXT_MODULE@@", next_dotted)
    source = source.replace("@@NEXT_IDX@@", f"{(index + 1) % MODULE_COUNT:02d}")
    source = source.replace("@@IDX@@", f"{index:02d}")
    return source


def generate_project(root: Path) -> ProjectStats:
    """Write the synthetic project under ``root`` and return its shape.

    Args:
        root: Target directory. Created (with parents) if missing; existing
            files at generated paths are overwritten.

    Returns:
        Statistics used by the report: module/class/file/line counts.
    """
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def write(relative: str, text: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)

    write("pyproject.toml", _PYPROJECT)
    write("README.md", _README)
    write(".gitignore", _GITIGNORE)
    write(f"{_PACKAGE}/__init__.py", _PACKAGE_INIT)
    write(f"{_PACKAGE}/core/__init__.py", '"""Core types and helpers."""\n')
    write(f"{_PACKAGE}/core/base.py", _CORE_BASE)
    write(f"{_PACKAGE}/core/errors.py", _CORE_ERRORS)
    write(f"{_PACKAGE}/core/validation.py", _CORE_VALIDATION)
    write(f"{_PACKAGE}/core/legacy.py", _CORE_LEGACY)
    write(f"{_PACKAGE}/domain/__init__.py", _SUBPACKAGE_INIT)
    write(f"{_PACKAGE}/services/__init__.py", _SUBPACKAGE_INIT)

    for index in range(MODULE_COUNT):
        package, _dotted = module_spec(index)
        directory = package.replace(".", "/")
        write(f"{directory}/m{index:02d}.py", render_module(index))

    write("tests/test_smoke.py", _SMOKE_TESTS)

    lines = sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in written
        if path.suffix in {".py", ".toml", ".md"}
    )
    return ProjectStats(
        root=root,
        modules=MODULE_COUNT,
        classes=MODULE_COUNT * CLASS_COUNT,
        files=len(written),
        lines=lines,
    )


def init_git_repository(root: Path) -> None:
    """Initialise a git repository and commit the generated tree.

    The engine's transaction layer needs a HEAD commit to roll back to, and the
    stress harness needs a clean baseline to prove rollback restored it.
    """
    env = {
        "GIT_AUTHOR_NAME": "moray stress",
        "GIT_AUTHOR_EMAIL": "stress@example.com",
        "GIT_COMMITTER_NAME": "moray stress",
        "GIT_COMMITTER_EMAIL": "stress@example.com",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "commit.gpgsign",
        "GIT_CONFIG_VALUE_0": "false",
        "GIT_CONFIG_KEY_1": "core.autocrlf",
        "GIT_CONFIG_VALUE_1": "false",
    }
    for args in (
        ("init", "-q"),
        ("add", "-A"),
        ("commit", "-q", "-m", "stress baseline"),
    ):
        subprocess.run(
            ["git", *args],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: generate the project and optionally initialise git."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="skip git init (useful when only inspecting the generated tree)",
    )
    args = parser.parse_args(argv)

    stats = generate_project(args.target)
    if not args.no_git:
        init_git_repository(stats.root)
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
