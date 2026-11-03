"""Tests for the SQLite code index and its incremental refresh."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from moray_surgeon.engine.indexer import (
    CACHE_DIRECTORY,
    DATABASE_FILENAME,
    Indexer,
    IndexSummary,
    clear_cache,
    get_indexer,
    reset_indexers,
)
from moray_surgeon.exceptions import CacheError


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_indexers()
    yield
    reset_indexers()


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


_MODEL = (
    "import os\n"
    "from collections import OrderedDict\n"
    "\n"
    "\n"
    "class Widget:\n"
    "    def __init__(self, name, size=3):\n"
    "        self.name = name\n"
    "        self.size = size\n"
    "\n"
    "    def ping(self, target=os.getcwd()):\n"
    "        local = OrderedDict()\n"
    "        return target, local\n"
)


def test_refresh_records_kinds_and_scope_chains(tmp_path):
    root = _repo(tmp_path)
    _write(root / "model.py", _MODEL)

    indexer = Indexer(root)
    stats = indexer.refresh()

    assert stats.total_files == 1
    assert stats.parsed_files == 1
    assert stats.symbols > 0

    by_name = {}
    for name in ("Widget", "ping", "name", "local", "os", "OrderedDict"):
        by_name[name] = {(hit.kind, hit.scope) for hit in indexer.symbols_named(name)}

    assert ("class", "") in by_name["Widget"]
    assert ("method", "Widget") in by_name["ping"]
    assert ("param", "Widget.__init__") in by_name["name"]
    assert ("variable", "Widget.ping") in by_name["local"]
    assert ("import", "") in by_name["os"]
    assert ("import", "") in by_name["OrderedDict"]

    hit = indexer.symbols_named("Widget")[0]
    assert hit.file == "model.py"
    assert hit.line == 5


def test_refresh_skips_dependency_and_cache_directories(tmp_path):
    root = _repo(tmp_path)
    _write(root / "app.py", "value = 1\n")
    _write(root / ".venv" / "lib.py", "vendored = 1\n")
    _write(root / "node_modules" / "pkg.py", "packaged = 1\n")

    indexer = Indexer(root)
    stats = indexer.refresh()

    assert stats.total_files == 1
    assert indexer.files_with("vendored") == ()
    assert indexer.files_with("packaged") == ()


def test_second_refresh_parses_nothing(tmp_path):
    root = _repo(tmp_path)
    _write(root / "a.py", "class A:\n    pass\n")
    _write(root / "b.py", "from a import A\n")

    indexer = Indexer(root)
    first = indexer.refresh()
    second = indexer.refresh()

    assert second.parsed_files == 0
    assert second.unchanged_files == first.total_files
    assert second.symbols == first.symbols


def test_only_the_changed_file_is_reparsed(tmp_path):
    root = _repo(tmp_path)
    _write(root / "a.py", "class A:\n    pass\n")
    _write(root / "b.py", "value = 1\n")

    indexer = Indexer(root)
    indexer.refresh()
    _write(root / "b.py", "value = 1\nfreshly_added = 2\n")

    stats = indexer.refresh()

    assert stats.parsed_files == 1
    assert stats.unchanged_files == 1
    assert indexer.files_with("freshly_added") == ((root / "b.py").resolve(),)
    # Symbols from the untouched file are still there.
    assert indexer.files_with("A") == ((root / "a.py").resolve(),)


def test_touching_a_file_without_changing_it_skips_the_parse(tmp_path):
    root = _repo(tmp_path)
    target = _write(root / "a.py", "class A:\n    pass\n")
    indexer = Indexer(root)
    indexer.refresh()

    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    stats = indexer.refresh()

    assert stats.rehashed_files == 1
    assert stats.parsed_files == 0
    assert indexer.files_with("A") == (target.resolve(),)


def test_deleted_file_is_dropped_from_the_index(tmp_path):
    root = _repo(tmp_path)
    gone = _write(root / "gone.py", "class Vanished:\n    pass\n")
    _write(root / "stays.py", "class Stayed:\n    pass\n")
    indexer = Indexer(root)
    indexer.refresh()

    gone.unlink()
    stats = indexer.refresh()

    assert stats.removed_files == 1
    assert indexer.files_with("Vanished") == ()
    assert indexer.files_with("Stayed")


def test_unparseable_file_is_indexed_once_without_symbols(tmp_path):
    root = _repo(tmp_path)
    _write(root / "broken.py", "def oops(:\n")
    indexer = Indexer(root)

    first = indexer.refresh()
    second = indexer.refresh()

    assert first.failed_files == 1
    assert first.total_files == 1
    assert second.failed_files == 0
    assert second.unchanged_files == 1
    assert indexer.files_with("oops") == ()


def test_force_reparses_every_file(tmp_path):
    root = _repo(tmp_path)
    _write(root / "a.py", "value = 1\n")
    _write(root / "b.py", "other = 2\n")
    indexer = Indexer(root)
    indexer.refresh()

    stats = indexer.refresh(force=True)

    assert stats.parsed_files == 2
    assert stats.unchanged_files == 0


def test_strings_and_comments_are_not_candidates(tmp_path):
    root = _repo(tmp_path)
    _write(root / "text_only.py", 'TEXT = "Symbol"\n# Symbol in a comment\n')
    _write(root / "real.py", "class Symbol:\n    pass\n")
    indexer = Indexer(root)

    found = indexer.files_with("Symbol")

    assert found == ((root / "real.py").resolve(),)


def test_keyword_arguments_are_not_candidates(tmp_path):
    """``call(name=1)`` must not make a file a candidate for renaming ``name``."""
    root = _repo(tmp_path)
    _write(root / "caller.py", "def run(fn):\n    return fn(name=1)\n")
    indexer = Indexer(root)

    assert indexer.files_with("name") == ()


def test_lookups_are_parameter_bound(tmp_path):
    root = _repo(tmp_path)
    _write(root / "a.py", "class Widget:\n    pass\n")
    indexer = Indexer(root)
    indexer.refresh()

    assert indexer.files_with("' OR 1=1 --") == ()
    assert indexer.files_with('Widget" OR "1') == ()
    assert indexer.files_with("Widget; DROP TABLE symbols") == ()
    # The table survived the injection attempts.
    assert indexer.files_with("Widget")


def test_files_with_respects_a_limit(tmp_path):
    root = _repo(tmp_path)
    for index in range(3):
        _write(root / f"m{index}.py", "from shared import Thing\n")
    indexer = Indexer(root)

    assert len(indexer.files_with("Thing", limit=2)) == 2


def test_empty_lookup_returns_nothing(tmp_path):
    root = _repo(tmp_path)
    _write(root / "a.py", "value = 1\n")
    indexer = Indexer(root)

    assert indexer.files_with("") == ()
    assert indexer.symbols_named("") == ()


def test_clear_removes_the_database_and_is_idempotent(tmp_path):
    root = _repo(tmp_path)
    _write(root / "a.py", "value = 1\n")
    indexer = Indexer(root)
    indexer.refresh()

    assert indexer.db_path.exists()
    assert indexer.clear() is True
    assert not indexer.db_path.exists()
    assert indexer.clear() is False


def test_clear_cache_module_function(tmp_path):
    root = _repo(tmp_path)
    _write(root / "a.py", "value = 1\n")
    Indexer(root).refresh()

    assert clear_cache(root) is True
    assert clear_cache(root) is False


def test_summary_reports_counts_and_handles_a_missing_database(tmp_path):
    root = _repo(tmp_path)
    indexer = Indexer(root)

    empty = indexer.summary()
    assert empty == IndexSummary(path=indexer.db_path, files=0, symbols=0, size_bytes=0)

    _write(root / "a.py", "class A:\n    pass\n")
    indexer.refresh()
    summary = indexer.summary()
    assert summary.files == 1
    assert summary.symbols > 0
    assert summary.size_bytes > 0


def test_refresh_raises_cache_error_when_the_cache_dir_is_a_file(tmp_path):
    root = _repo(tmp_path)
    _write(root / CACHE_DIRECTORY, "not a directory")
    _write(root / "a.py", "value = 1\n")

    with pytest.raises(CacheError):
        Indexer(root).refresh()


def test_clear_raises_cache_error_when_the_database_cannot_be_deleted(tmp_path):
    root = _repo(tmp_path)
    (root / CACHE_DIRECTORY / DATABASE_FILENAME).mkdir(parents=True)

    with pytest.raises(CacheError):
        Indexer(root).clear()


def test_relative_cache_dir_is_resolved_against_the_repository(tmp_path):
    root = _repo(tmp_path)
    indexer = Indexer(root, cache_dir="custom-cache")

    assert indexer.cache_dir == (root / "custom-cache").resolve()
    assert indexer.db_path == indexer.cache_dir / DATABASE_FILENAME


def test_cache_directory_is_added_to_git_exclude_once(tmp_path):
    root = _repo(tmp_path)
    (root / ".git" / "info").mkdir(parents=True)
    _write(root / "a.py", "value = 1\n")

    indexer = Indexer(root)
    indexer.refresh()
    indexer.refresh()

    exclude = (root / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert exclude.splitlines().count(f"/{CACHE_DIRECTORY}/") == 1


def test_cache_directory_is_not_added_outside_a_git_repository(tmp_path):
    root = _repo(tmp_path)
    _write(root / "a.py", "value = 1\n")

    Indexer(root).refresh()

    assert not (root / ".git").exists()


def test_get_indexer_memoises_per_repository(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()

    assert get_indexer(first) is get_indexer(first)
    assert get_indexer(first) is not get_indexer(second)

    shared = get_indexer(first)
    reset_indexers()
    assert get_indexer(first) is not shared


def test_invalidate_rebuilds_only_the_reported_files(tmp_path):
    root = _repo(tmp_path)
    _write(root / "a.py", "value = 1\n")
    _write(root / "b.py", "other = 2\n")
    indexer = Indexer(root)
    indexer.refresh()

    _write(root / "a.py", "value = 1\nextra = 3\n")
    indexer.invalidate([root / "a.py"])
    found = indexer.files_with("extra")

    assert found == ((root / "a.py").resolve(),)
    # The untouched file was not re-read: its symbols still answer queries.
    assert indexer.files_with("other") == ((root / "b.py").resolve(),)


def test_reindex_ignores_paths_outside_the_repository(tmp_path):
    root = _repo(tmp_path)
    outside = _write(tmp_path / "outside.py", "secret = 1\n")
    _write(root / "a.py", "value = 1\n")
    indexer = Indexer(root)
    indexer.refresh()

    indexer.reindex([outside])

    assert indexer.files_with("secret") == ()
    assert indexer.files_with("value")


def test_unknown_files_are_skipped_by_reindex(tmp_path):
    root = _repo(tmp_path)
    _write(root / "a.py", "value = 1\n")
    indexer = Indexer(root)
    indexer.refresh()

    missing = root / "never_created.py"
    indexer.reindex([missing])

    assert indexer.files_with("value")


def test_console_noise_is_kept_out_of_the_root_logger(tmp_path, caplog):
    """The module attaches a NullHandler so MCP stdio is never polluted."""
    root = _repo(tmp_path)
    _write(root / "broken.py", "def oops(:\n")

    with caplog.at_level(logging.WARNING):
        Indexer(root).refresh()

    assert "could not parse" in caplog.text


def test_suspended_index_refuses_queries_until_the_next_refresh(tmp_path):
    """A failed refresh must not be retried inside a transaction it could break."""
    root = _repo(tmp_path)
    _write(root / "a.py", "value = 1\n")
    indexer = Indexer(root)
    indexer.refresh()

    indexer.suspend()

    assert indexer.suspended is True
    with pytest.raises(CacheError):
        indexer.files_with("value")
    with pytest.raises(CacheError):
        indexer.symbols_named("value")
    with pytest.raises(CacheError):
        indexer.reindex([root / "a.py"])

    indexer.refresh()

    assert indexer.suspended is False
    assert indexer.files_with("value") == ((root / "a.py").resolve(),)


def test_file_count_reports_the_indexed_files(tmp_path):
    root = _repo(tmp_path)
    _write(root / "a.py", "value = 1\n")
    _write(root / "pkg" / "b.py", "other = 2\n")

    assert Indexer(root).file_count() == 2


def test_sparse_symbol_uses_per_position_verification(tmp_path, monkeypatch):
    """A symbol in one of many files is verified file by file."""
    from moray_surgeon.engine import rename as rename_module

    root = _repo(tmp_path)
    _write(root / "shared.py", "class Widget:\n    pass\n")
    for index in range(12):
        _write(root / f"m{index:02d}.py", f"class Local{index}:\n    pass\n")
    _write(root / "user.py", "from shared import Widget\n\nw = Widget()\n")
    used: list[str] = []
    original = rename_module._indexed_plan

    def recording(*args: object, **kwargs: object):
        used.append("indexed")
        return original(*args, **kwargs)

    monkeypatch.setattr(rename_module, "_indexed_plan", recording)

    result = rename_module.rename_symbol(root, "shared.py", "Widget", "Gadget")

    assert used == ["indexed"]
    assert result.changes == 3  # the class, the import binding and the call
    assert "from shared import Gadget" in (root / "user.py").read_text(encoding="utf-8")


def test_dense_symbol_falls_back_to_one_project_wide_search(tmp_path, monkeypatch):
    """A symbol in most files is cheaper to resolve with a single Jedi search."""
    from moray_surgeon.engine import rename as rename_module

    root = _repo(tmp_path)
    _write(root / "shared.py", "class Widget:\n    pass\n")
    for index in range(9):
        _write(
            root / f"m{index:02d}.py",
            f"from shared import Widget\n\n\ndef use_{index}():\n    return Widget()\n",
        )
    used: list[str] = []
    original = rename_module._indexed_plan

    def recording(*args: object, **kwargs: object):
        used.append("indexed")
        return original(*args, **kwargs)

    monkeypatch.setattr(rename_module, "_indexed_plan", recording)

    result = rename_module.rename_symbol(root, "shared.py", "Widget", "Gadget")

    assert used == []
    assert result.changes == 19  # the class plus 9 imports and 9 calls
    for index in range(9):
        text = (root / f"m{index:02d}.py").read_text(encoding="utf-8")
        assert "from shared import Gadget" in text
        assert "return Gadget()" in text
