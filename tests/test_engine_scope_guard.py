from __future__ import annotations

from pathlib import Path

import libcst as cst
import pytest

from moray_surgeon.engine import extract_function
from moray_surgeon.engine.scope_guard import (
    MODULE_KEY,
    get_ast_fingerprint,
    snapshot_repository,
    verify_repository_scope,
    verify_scope,
)
from moray_surgeon.exceptions import ScopeViolationError
from moray_surgeon.schema import (
    AddImport,
    AddTypeHints,
    ExtractFunction,
    RefactorPlan,
    RemoveImport,
    RenameSymbol,
    ReplaceCall,
)

_SOURCE = """import os


def old_name(value):
    return old_name(value)


def untouched():
    return 1


class Widget:
    def method(self):
        return 2
"""


def _plan(*instructions: object) -> RefactorPlan:
    return RefactorPlan(instructions=list(instructions))


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# get_ast_fingerprint
# --------------------------------------------------------------------------- #


def test_fingerprint_indexes_functions_classes_and_module_body():
    fingerprint = get_ast_fingerprint(_SOURCE)

    assert set(fingerprint) == {
        MODULE_KEY,
        "old_name",
        "untouched",
        "Widget",
        "Widget.method",
    }
    assert all(len(digest) == 64 for digest in fingerprint.values())


def test_fingerprint_is_stable_for_identical_source():
    assert get_ast_fingerprint(_SOURCE) == get_ast_fingerprint(_SOURCE)


def test_fingerprint_ignores_layout_but_not_code():
    reformatted = _SOURCE.replace("\n\n\ndef untouched", "\n\n\ndef untouched")

    assert get_ast_fingerprint(_SOURCE) == get_ast_fingerprint(reformatted)
    assert (
        get_ast_fingerprint(_SOURCE)["untouched"]
        != get_ast_fingerprint(_SOURCE.replace("return 1", "return 2"))["untouched"]
    )


def test_fingerprint_of_module_level_change_is_visible():
    changed = _SOURCE.replace("import os", "import os\n\nLIMIT = 10")

    before = get_ast_fingerprint(_SOURCE)
    after = get_ast_fingerprint(changed)

    assert before[MODULE_KEY] != after[MODULE_KEY]
    assert before["untouched"] == after["untouched"]


def test_import_changes_do_not_touch_any_fingerprint():
    before = get_ast_fingerprint(_SOURCE)
    after = get_ast_fingerprint(_SOURCE.replace("import os", "import json\nimport os"))

    assert before == after


def test_duplicate_qualified_names_get_a_stable_suffix():
    source = (
        "import sys\n\n"
        "if sys.version_info >= (3, 12):\n"
        "    def f():\n"
        "        return 1\n"
        "else:\n"
        "    def f():\n"
        "        return 2\n"
    )

    fingerprint = get_ast_fingerprint(source)

    assert set(fingerprint) == {MODULE_KEY, "f", "f#2"}
    assert fingerprint["f"] != fingerprint["f#2"]


def test_fingerprint_of_nested_functions_uses_qualified_names():
    source = "def outer():\n    def inner():\n        return 1\n    return inner()\n"

    assert set(get_ast_fingerprint(source)) == {MODULE_KEY, "outer", "outer.inner"}


def test_get_ast_fingerprint_rejects_invalid_python():
    with pytest.raises(cst.ParserSyntaxError):
        get_ast_fingerprint("def broken(:\n")


# --------------------------------------------------------------------------- #
# verify_scope
# --------------------------------------------------------------------------- #


def test_verify_scope_accepts_an_unchanged_file():
    verify_scope(
        get_ast_fingerprint(_SOURCE),
        _SOURCE,
        _plan(
            RenameSymbol(id="s1", target_file="app.py", old_name="old_name", new_name="new_name")
        ),
        target_file="app.py",
    )


def test_verify_scope_accepts_a_rename_and_its_reference_sites():
    renamed = _SOURCE.replace("old_name", "new_name")

    verify_scope(
        get_ast_fingerprint(_SOURCE),
        renamed,
        _plan(
            RenameSymbol(id="s2", target_file="app.py", old_name="old_name", new_name="new_name")
        ),
        target_file="app.py",
    )


def test_verify_scope_accepts_a_class_rename_with_its_members():
    renamed = _SOURCE.replace("class Widget:", "class Gizmo:")

    verify_scope(
        get_ast_fingerprint(_SOURCE),
        renamed,
        _plan(RenameSymbol(id="s3", target_file="app.py", old_name="Widget", new_name="Gizmo")),
        target_file="app.py",
    )


def test_verify_scope_rejects_a_member_of_an_unrelated_class():
    tampered = _SOURCE.replace(
        "class Widget:\n    def method(self):\n        return 2",
        "class Widget:\n    def method(self):\n        return 3",
    )

    with pytest.raises(ScopeViolationError) as excinfo:
        verify_scope(
            get_ast_fingerprint(_SOURCE),
            tampered,
            _plan(
                RenameSymbol(
                    id="s4", target_file="app.py", old_name="old_name", new_name="new_name"
                )
            ),
            target_file="app.py",
        )

    assert excinfo.value.violations == ("Widget", "Widget.method")


def test_verify_scope_accepts_a_rename_in_a_non_target_file():
    before = "from app import old_name\n\n\ndef caller():\n    return old_name(1)\n"
    after = "from app import new_name\n\n\ndef caller():\n    return new_name(1)\n"

    verify_scope(
        get_ast_fingerprint(before),
        after,
        _plan(
            RenameSymbol(id="s5", target_file="app.py", old_name="old_name", new_name="new_name")
        ),
        target_file="other.py",
    )


def test_verify_scope_rejects_collateral_damage_to_an_unrelated_function():
    tampered = _SOURCE.replace("return 1", "return 999")

    with pytest.raises(ScopeViolationError) as excinfo:
        verify_scope(
            get_ast_fingerprint(_SOURCE),
            tampered,
            _plan(
                RenameSymbol(
                    id="s6", target_file="app.py", old_name="old_name", new_name="new_name"
                )
            ),
            target_file="app.py",
        )

    assert excinfo.value.violations == ("untouched",)
    assert "app.py" in str(excinfo.value)


def test_verify_scope_rejects_a_deleted_function():
    deleted = _SOURCE.replace("\n\ndef untouched():\n    return 1\n", "\n")

    with pytest.raises(ScopeViolationError) as excinfo:
        verify_scope(
            get_ast_fingerprint(_SOURCE),
            deleted,
            _plan(
                RenameSymbol(
                    id="s7", target_file="app.py", old_name="old_name", new_name="new_name"
                )
            ),
            target_file="app.py",
        )

    assert "untouched" in excinfo.value.violations


def test_verify_scope_rejects_a_change_in_a_non_target_file():
    tampered = _SOURCE.replace("return 1", "return 2")

    with pytest.raises(ScopeViolationError):
        verify_scope(
            get_ast_fingerprint(_SOURCE),
            tampered,
            _plan(
                RenameSymbol(
                    id="s8", target_file="app.py", old_name="old_name", new_name="new_name"
                )
            ),
            target_file="other.py",
        )


def test_verify_scope_allows_any_change_in_a_replace_call_target_file():
    changed = _SOURCE.replace("return old_name(value)", "return other(value)")

    verify_scope(
        get_ast_fingerprint(_SOURCE),
        changed,
        _plan(
            ReplaceCall(
                id="s9", target_file="app.py", old_call="os.getcwd", new_call="pathlib.Path.cwd"
            )
        ),
        target_file="app.py",
    )


def test_verify_scope_treats_unnamed_type_hints_as_file_global():
    changed = _SOURCE.replace("def untouched():", "def untouched() -> None:")

    verify_scope(
        get_ast_fingerprint(_SOURCE),
        changed,
        _plan(AddTypeHints(id="s10", target_file="app.py", function_name=None)),
        target_file="app.py",
    )


def test_verify_scope_scopes_named_type_hints_to_their_function():
    changed = _SOURCE.replace("def untouched():", "def untouched() -> None:")

    with pytest.raises(ScopeViolationError):
        verify_scope(
            get_ast_fingerprint(_SOURCE),
            changed,
            _plan(AddTypeHints(id="s11", target_file="app.py", function_name="old_name")),
            target_file="app.py",
        )


def test_verify_scope_allows_named_type_hints_on_the_named_function():
    changed = _SOURCE.replace("def old_name(value):", "def old_name(value) -> None:")

    verify_scope(
        get_ast_fingerprint(_SOURCE),
        changed,
        _plan(AddTypeHints(id="s12", target_file="app.py", function_name="old_name")),
        target_file="app.py",
    )


def test_verify_scope_ignores_instructions_aimed_at_another_file():
    changed = _SOURCE.replace("def untouched():", "def untouched() -> None:")

    with pytest.raises(ScopeViolationError):
        verify_scope(
            get_ast_fingerprint(_SOURCE),
            changed,
            _plan(
                ReplaceCall(id="s13", target_file="app.py", old_call="print", new_call="log.info")
            ),
            target_file="other.py",
        )


def test_verify_scope_allows_module_level_edits_in_a_target_file():
    changed = _SOURCE + "\nLIMIT = 10\n"

    verify_scope(
        get_ast_fingerprint(_SOURCE),
        changed,
        _plan(
            RenameSymbol(id="s14", target_file="app.py", old_name="old_name", new_name="new_name")
        ),
        target_file="app.py",
    )


def test_verify_scope_allows_extract_function_to_move_module_statements():
    source = "X = 1\nY = 2\n\n\ndef other():\n    return X + Y\n"
    result = extract_function(source, "helper", 1, 2)

    verify_scope(
        get_ast_fingerprint(source),
        result.new_source,
        _plan(
            ExtractFunction(
                id="s15",
                target_file="app.py",
                function_name="helper",
                start_line=1,
                end_line=2,
            )
        ),
        target_file="app.py",
    )


def test_verify_scope_rejects_import_atoms_changing_a_function():
    changed = _SOURCE.replace("return 1", "return 2")

    with pytest.raises(ScopeViolationError):
        verify_scope(
            get_ast_fingerprint(_SOURCE),
            changed,
            _plan(AddImport(id="s16", target_file="app.py", module="json")),
            target_file="app.py",
        )
    with pytest.raises(ScopeViolationError):
        verify_scope(
            get_ast_fingerprint(_SOURCE),
            changed,
            _plan(RemoveImport(id="s17", target_file="app.py", module="os")),
            target_file="app.py",
        )


def test_verify_scope_without_a_file_identity_is_permissive():
    changed = _SOURCE.replace("return 1", "return 2")

    verify_scope(
        get_ast_fingerprint(_SOURCE),
        changed,
        _plan(ReplaceCall(id="s18", target_file="app.py", old_call="print", new_call="log.info")),
    )


def test_verify_scope_rejects_a_syntax_invalid_result():
    with pytest.raises(cst.ParserSyntaxError):
        verify_scope(get_ast_fingerprint(_SOURCE), "def broken(:\n", _plan())


# --------------------------------------------------------------------------- #
# Repository snapshot
# --------------------------------------------------------------------------- #


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _write(repo / "app.py", _SOURCE)
    _write(repo / "pkg" / "helper.py", "def helper():\n    return 1\n")
    _write(repo / ".venv" / "lib" / "vendored.py", "def vendored():\n    return 1\n")
    _write(repo / "__pycache__" / "cached.py", "def cached():\n    return 1\n")
    _write(repo / "build" / "copied.py", "def copied():\n    return 1\n")
    return repo


def test_snapshot_skips_venvs_caches_and_build_output(tmp_path):
    snapshot = snapshot_repository(_repo(tmp_path))

    assert set(snapshot.fingerprints) == {"app.py", "pkg/helper.py"}


def test_verify_repository_scope_accepts_a_clean_repository(tmp_path):
    repo = _repo(tmp_path)
    snapshot = snapshot_repository(repo)

    verify_repository_scope(snapshot, repo, _plan())


def test_verify_repository_scope_accepts_the_plans_own_changes(tmp_path):
    repo = _repo(tmp_path)
    snapshot = snapshot_repository(repo)
    _write(repo / "app.py", _SOURCE.replace("old_name", "new_name"))

    verify_repository_scope(
        snapshot,
        repo,
        _plan(
            RenameSymbol(id="s19", target_file="app.py", old_name="old_name", new_name="new_name")
        ),
    )


def test_verify_repository_scope_rejects_collateral_damage(tmp_path):
    repo = _repo(tmp_path)
    snapshot = snapshot_repository(repo)
    _write(repo / "pkg" / "helper.py", "def helper():\n    return 999\n")

    with pytest.raises(ScopeViolationError) as excinfo:
        verify_repository_scope(
            snapshot,
            repo,
            _plan(
                RenameSymbol(
                    id="s20", target_file="app.py", old_name="old_name", new_name="new_name"
                )
            ),
        )

    assert "pkg/helper.py" in str(excinfo.value)
    assert excinfo.value.violations == ("helper",)


def test_verify_repository_scope_rejects_a_created_file(tmp_path):
    repo = _repo(tmp_path)
    snapshot = snapshot_repository(repo)
    _write(repo / "pkg" / "extra.py", "def extra():\n    return 1\n")

    with pytest.raises(ScopeViolationError, match="created"):
        verify_repository_scope(snapshot, repo, _plan())


def test_verify_repository_scope_rejects_a_deleted_file(tmp_path):
    repo = _repo(tmp_path)
    snapshot = snapshot_repository(repo)
    (repo / "pkg" / "helper.py").unlink()

    with pytest.raises(ScopeViolationError, match="deleted"):
        verify_repository_scope(snapshot, repo, _plan())


def test_snapshot_tracks_unparseable_files_without_crashing(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "broken.py", "def broken(:\n")
    snapshot = snapshot_repository(repo)

    assert snapshot.fingerprints["broken.py"] == {"<raw>": snapshot.digests["broken.py"]}
    with pytest.raises(ScopeViolationError):
        _write(repo / "broken.py", "def broken():\n    return 1\n")
        verify_repository_scope(snapshot, repo, _plan())


def test_snapshot_survives_a_non_utf8_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "latin.py").write_bytes(b"# caf\xe9\ndef f():\n    return 1\n")

    snapshot = snapshot_repository(repo)

    assert "latin.py" in snapshot.fingerprints
    verify_repository_scope(snapshot, repo, _plan())
