from __future__ import annotations

import textwrap

import pytest

from moray_surgeon.engine.move_symbol import move_symbol


def _src(text: str) -> str:
    """Dedent a triple-quoted fixture that starts on its own line."""
    return textwrap.dedent(text).lstrip("\n")


_MODELS = _src(
    """
    \"\"\"Domain models.\"\"\"

    from dataclasses import dataclass

    # A user of the system.
    @dataclass
    class User:
        \"\"\"Represents a user.\"\"\"

        name: str
    """
)

_ENTITIES = _src(
    """
    \"\"\"Entity types.\"\"\"

    from dataclasses import dataclass
    """
)


def _repo(tmp_path, models=_MODELS, entities=_ENTITIES, extra_files=()):
    (tmp_path / "models.py").write_text(models, encoding="utf-8")
    (tmp_path / "entities.py").write_text(entities, encoding="utf-8")
    for name, content in extra_files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def test_move_symbol_golden(tmp_path) -> None:
    """The class moves with its comments; the importer is rewritten."""
    repo = _repo(
        tmp_path,
        extra_files=(
            ("service.py", "from models import User\n\n\ndef run(user):\n    return user\n"),
        ),
    )

    result = move_symbol(repo, repo / "models.py", repo / "entities.py", "User")

    assert "class User" not in (repo / "models.py").read_text()
    entities = (repo / "entities.py").read_text()
    assert "# A user of the system." in entities
    assert '"""Represents a user."""' in entities
    assert "class User:" in entities
    assert (repo / "service.py").read_text().startswith("from entities import User\n")
    assert result.changes == 3  # deletion + insertion + one rewritten importer
    assert result.new_source == (repo / "models.py").read_text()


def test_move_symbol_appends_at_the_end_with_separation(tmp_path) -> None:
    """Existing target code stays on top; the moved node lands below two blanks."""
    legacy = (
        '"""Entity types."""\n'
        "\nfrom dataclasses import dataclass\n\n\ndef legacy():\n    return 1\n"
    )
    repo = _repo(tmp_path, entities=legacy)

    move_symbol(repo, repo / "models.py", repo / "entities.py", "User")

    entities = (repo / "entities.py").read_text()
    assert entities.index("def legacy():") < entities.index("class User:")
    assert "\n\n\n# A user of the system." in entities


def test_move_symbol_rewrites_import_alias(tmp_path) -> None:
    """``from models import User as Account`` keeps its alias."""
    repo = _repo(
        tmp_path,
        extra_files=(("service.py", "from models import User as Account\n\n\nprint(Account)\n"),),
    )

    move_symbol(repo, repo / "models.py", repo / "entities.py", "User")

    assert (repo / "service.py").read_text().startswith(
        "from entities import User as Account\n"
    )


def test_move_symbol_drops_name_from_multi_import(tmp_path) -> None:
    """A from-import carrying other names keeps them and gains the new import."""
    repo = _repo(
        tmp_path,
        models=_src(
            """
            class User:
                pass


            class Group:
                pass
            """
        ),
        entities="",
        extra_files=(("service.py", "from models import User, Group\n\n\nprint(User, Group)\n"),),
    )

    move_symbol(repo, repo / "models.py", repo / "entities.py", "User")

    service = (repo / "service.py").read_text()
    assert "from models import Group" in service
    assert "from entities import User" in service


def test_move_symbol_rewrites_relative_import_within_package(tmp_path) -> None:
    """Dots survive a same-package move."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text("class User:\n    pass\n", encoding="utf-8")
    (pkg / "sibling.py").write_text("from .models import User\nprint(User)\n", encoding="utf-8")
    (pkg / "entities.py").write_text("", encoding="utf-8")

    move_symbol(tmp_path, pkg / "models.py", pkg / "entities.py", "User")

    assert (pkg / "sibling.py").read_text().startswith("from .entities import User\n")


def test_move_symbol_rewrites_dotted_absolute_import(tmp_path) -> None:
    """``from pkg.models import User`` becomes ``from pkg.entities import User``."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text("class User:\n    pass\n", encoding="utf-8")
    (pkg / "entities.py").write_text("", encoding="utf-8")
    (tmp_path / "app.py").write_text("from pkg.models import User\nprint(User)\n", encoding="utf-8")

    move_symbol(tmp_path, pkg / "models.py", pkg / "entities.py", "User")

    assert (tmp_path / "app.py").read_text().startswith("from pkg.entities import User\n")


def test_move_symbol_without_importers(tmp_path) -> None:
    """No references anywhere: a plain two-file move."""
    repo = _repo(tmp_path)

    result = move_symbol(repo, repo / "models.py", repo / "entities.py", "User")

    assert result.changes == 2
    assert "class User" in (repo / "entities.py").read_text()


@pytest.mark.parametrize(
    ("models", "entities", "extra_files", "match"),
    [
        pytest.param(
            _MODELS + "\n\n\nmake = User\n",
            _ENTITIES,
            (),
            "still referenced at models.py",
            id="leftover-reference-in-source",
        ),
        pytest.param(
            _MODELS,
            "User = 1\n",
            (),
            "already binds 'User'",
            id="target-binds-name",
        ),
        pytest.param(
            _MODELS,
            "from models import User\n",
            (),
            "already binds 'User'",
            id="target-imports-name",
        ),
        pytest.param(
            _MODELS,
            _ENTITIES,
            (("plugin.py", "import models\nprint(models.User)\n"),),
            "without a rewritable",
            id="attribute-style-usage",
        ),
        pytest.param(
            '__all__ = ["User"]\n\n\n@dataclass\nclass User:\n    name: str\n'.replace(
                "@dataclass\n", ""
            ),
            _ENTITIES,
            (),
            "exported by __all__",
            id="exported-by-all",
        ),
        pytest.param(
            _MODELS,
            '"""Entity types."""\n',
            (),
            "not available in entities.py",
            id="missing-dependency",
        ),
        pytest.param(
            "class User:\n    pass\n\n\nclass User:\n    pass\n",
            "",
            (),
            "defined 2 times",
            id="duplicate-definition",
        ),
        pytest.param(
            "class Widget:\n    pass\n",
            "",
            (),
            "no top-level function or class",
            id="symbol-missing",
        ),
    ],
)
def test_move_symbol_refuses_unsafe_moves(tmp_path, models, entities, extra_files, match) -> None:
    """Every ambiguity refuses the whole move; no file is touched."""
    repo = _repo(tmp_path, models=models, entities=entities, extra_files=extra_files)
    before = {path: path.read_text(encoding="utf-8") for path in repo.rglob("*.py")}

    with pytest.raises(ValueError, match=match):
        move_symbol(repo, repo / "models.py", repo / "entities.py", "User")

    after = {path: path.read_text(encoding="utf-8") for path in repo.rglob("*.py")}
    assert after == before


def test_move_symbol_allows_builtin_dependencies(tmp_path) -> None:
    """Names that resolve to builtins need no import in the target."""
    repo = _repo(
        tmp_path,
        models=_src(
            """
            def describe(value):
                return str(len(value))
            """
        ),
        entities='"""Helpers."""\n',
    )

    result = move_symbol(repo, repo / "models.py", repo / "entities.py", "describe")

    assert result.changes == 2
    assert "def describe" in (repo / "entities.py").read_text()


def test_move_symbol_rejects_same_file(tmp_path) -> None:
    """Moving a symbol onto its own file is refused."""
    repo = _repo(tmp_path)

    with pytest.raises(ValueError, match="different files"):
        move_symbol(repo, repo / "models.py", repo / "models.py", "User")


def test_move_symbol_rejects_relative_import_crossing_packages(tmp_path) -> None:
    """A relative import that cannot reach the target refuses the move."""
    pkg = tmp_path / "pkg"
    other = tmp_path / "other"
    pkg.mkdir()
    other.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (other / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text("class User:\n    pass\n", encoding="utf-8")
    (pkg / "sibling.py").write_text("from .models import User\n", encoding="utf-8")
    (other / "entities.py").write_text("", encoding="utf-8")
    before = {
        path: path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.py")
    }

    with pytest.raises(ValueError, match=r"would not reach entities\.py"):
        move_symbol(tmp_path, pkg / "models.py", other / "entities.py", "User")

    after = {path: path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.py")}
    assert after == before
