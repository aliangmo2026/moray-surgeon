from __future__ import annotations

import pytest

from moray_surgeon.engine import add_import, remove_import


def _add(source: str, module: str, name: str | None = None, alias: str | None = None):
    return add_import(source, module, name, alias)


def _remove(source: str, module: str, name: str | None = None):
    return remove_import(source, module, name)


# --------------------------------------------------------------------------- #
# add_import
# --------------------------------------------------------------------------- #


def test_add_straight_import_to_empty_module():
    result = _add("", "os")

    assert result.new_source == "import os\n"
    assert result.changes == 1


def test_add_import_keeps_the_block_sorted():
    source = "import sys\nimport time\n"

    result = _add(source, "os")

    assert result.new_source == "import os\nimport sys\nimport time\n"
    assert result.changes == 1


def test_add_import_places_straight_before_from_imports():
    source = "import sys\nfrom collections import OrderedDict\n"

    result = _add(source, "os")

    assert result.new_source == "import os\nimport sys\nfrom collections import OrderedDict\n"


def test_add_import_never_reorders_existing_statements():
    source = "from collections import OrderedDict\nimport sys\n"

    result = _add(source, "os")

    assert result.new_source == "from collections import OrderedDict\nimport os\nimport sys\n"


def test_add_import_keeps_blank_line_between_block_and_code():
    source = "import os\n\n\ndef f():\n    return 1\n"

    result = _add(source, "sys")

    assert result.new_source == "import os\nimport sys\n\n\ndef f():\n    return 1\n"


def test_add_import_keeps_a_gap_above_the_block():
    source = '"""Doc."""\n\nimport sys\n\nX = 1\n'

    result = _add(source, "os")

    assert result.new_source == '"""Doc."""\n\nimport os\nimport sys\n\nX = 1\n'


def test_add_import_keeps_comments_attached_to_their_import():
    source = "import os\n\n# third-party\nimport requests\n"

    result = _add(source, "json")

    assert result.new_source == (
        "import json\nimport os\n\n# third-party\nimport requests\n"
    )


def test_add_first_import_adds_blank_line_after_docstring():
    source = '"""Doc."""\nX = 1\n'

    result = _add(source, "os")

    assert result.new_source == '"""Doc."""\n\nimport os\nX = 1\n'


def test_add_first_import_preserves_shebang():
    source = "#!/usr/bin/env python\nx = 1\n"

    result = _add(source, "os")

    assert result.new_source == "#!/usr/bin/env python\nimport os\nx = 1\n"


def test_add_import_stays_after_future_imports():
    source = "from __future__ import annotations\n\nX = 1\n"

    result = _add(source, "os")

    assert result.new_source == "from __future__ import annotations\nimport os\n\nX = 1\n"


def test_add_import_merges_into_existing_from_statement_sorted():
    source = "from collections import OrderedDict, defaultdict\n"

    result = _add(source, "collections", "Counter")

    assert result.new_source == "from collections import Counter, OrderedDict, defaultdict\n"
    assert result.changes == 1


def test_add_import_merges_alias_into_existing_from_statement():
    source = "from collections import OrderedDict\n"

    result = _add(source, "collections", "defaultdict", alias="dd")

    assert result.new_source == "from collections import OrderedDict, defaultdict as dd\n"


def test_add_import_merges_names_in_isort_order_by_type():
    source = "from x import A, b\n"

    result = _add(source, "x", "C_D")

    assert result.new_source == "from x import C_D, A, b\n"


def test_add_import_wraps_a_merge_that_would_exceed_the_line_limit():
    module = "a_very_long_module_name_for_imports"
    names = ", ".join(f"name_{index:02d}" for index in range(10))
    source = f"from {module} import {names}\n"

    result = _add(source, module, "name_99")

    expected = "\n".join(
        [
            f"from {module} import (",
            *(f"    name_{index:02d}," for index in range(10)),
            "    name_99,",
            ")",
            "",
        ]
    )
    assert result.new_source == expected


def test_add_import_does_not_wrap_a_short_merge():
    result = _add("from x import a\n", "x", "b")

    assert result.new_source == "from x import a, b\n"


def test_add_import_keeps_trailing_comma_of_parenthesized_import():
    source = "from collections import (\n    OrderedDict,\n)\n"

    result = _add(source, "collections", "Counter")

    assert result.new_source == "from collections import (\n    Counter,\n    OrderedDict,\n)\n"


def test_add_import_is_a_noop_when_straight_import_exists():
    source = "import os\n\nX = 1\n"

    result = _add(source, "os")

    assert result.changes == 0
    assert result.new_source == source


def test_add_import_with_alias_is_not_a_duplicate_of_plain_import():
    source = "import os\n"

    result = _add(source, "os", alias="operating_system")

    assert result.new_source == "import os\nimport os as operating_system\n"


def test_add_from_import_is_a_noop_when_name_and_alias_match():
    source = "from collections import defaultdict as dd\n"

    result = _add(source, "collections", "defaultdict", alias="dd")

    assert result.changes == 0
    assert result.new_source == source


def test_add_from_import_adds_a_differently_aliased_same_name():
    source = "from collections import defaultdict\n"

    result = _add(source, "collections", "defaultdict", alias="dd")

    assert result.new_source == "from collections import defaultdict, defaultdict as dd\n"


def test_add_import_never_touches_imports_inside_try_blocks():
    source = "try:\n    import simplejson as json\nexcept ImportError:\n    json = None\n"

    result = _add(source, "json")

    assert result.new_source.startswith("import json\n")
    assert "\n    import simplejson as json" in result.new_source


def test_add_import_does_not_merge_relative_imports():
    source = "from . import sibling\n"

    result = _add(source, "pkg", "sibling")

    assert result.new_source == "from . import sibling\nfrom pkg import sibling\n"


def test_add_import_does_not_merge_into_a_different_module():
    source = "from other import thing\n"

    result = _add(source, "pkg", "thing")

    assert result.new_source == "from other import thing\nfrom pkg import thing\n"


def test_add_import_does_not_merge_into_a_star_import():
    source = "from pkg import *\n"

    result = _add(source, "pkg", "thing")

    assert result.new_source == "from pkg import *\nfrom pkg import thing\n"


@pytest.mark.parametrize(
    ("module", "name", "alias"),
    [
        ("", None, None),
        ("os.path, sys", None, None),
        ("os.", None, None),
        (".relative", None, None),
        ("import", None, None),
        ("os", "not a name", None),
        ("os", "class", None),
        ("os", "", None),
        ("os", None, "not a name"),
        ("os", None, "def"),
        ("os", None, ""),
    ],
)
def test_add_import_rejects_invalid_arguments(module, name, alias):
    with pytest.raises(ValueError):
        _add("import sys\n", module, name, alias)


# --------------------------------------------------------------------------- #
# remove_import
# --------------------------------------------------------------------------- #


def test_remove_straight_import_removes_the_line():
    result = _remove("import os\nimport sys\n", "os")

    assert result.new_source == "import sys\n"
    assert result.changes == 1


def test_remove_alias_from_multi_name_import_cleans_the_comma():
    result = _remove("import os, sys, time\n", "sys")

    assert result.new_source == "import os, time\n"


def test_remove_last_alias_from_multi_name_import_cleans_the_comma():
    result = _remove("import os, sys\n", "sys")

    assert result.new_source == "import os\n"


def test_remove_name_from_from_import_keeps_the_rest():
    result = _remove(
        "from collections import Counter, OrderedDict, defaultdict\n",
        "collections",
        "Counter",
    )

    assert result.new_source == "from collections import OrderedDict, defaultdict\n"


def test_remove_only_name_removes_the_whole_line():
    result = _remove("from collections import OrderedDict\nX = 1\n", "collections", "OrderedDict")

    assert result.new_source == "X = 1\n"
    assert result.changes == 1


def test_remove_parenthesized_name_keeps_trailing_comma():
    source = "from collections import (\n    Counter,\n    OrderedDict,\n)\n"

    result = _remove(source, "collections", "Counter")

    assert result.new_source == "from collections import (\n    OrderedDict,\n)\n"


def test_remove_module_removes_from_import_statement():
    result = _remove("from os import path\nimport sys\n", "os")

    assert result.new_source == "import sys\n"


def test_remove_module_removes_star_import():
    result = _remove("from os.path import *\nimport sys\n", "os.path")

    assert result.new_source == "import sys\n"


def test_remove_name_does_not_touch_star_import():
    source = "from os.path import *\n"

    result = _remove(source, "os.path", "join")

    assert result.changes == 0
    assert result.new_source == source


def test_remove_does_not_touch_straight_import_when_name_is_given():
    source = "import os\n"

    result = _remove(source, "os", "path")

    assert result.changes == 0
    assert result.new_source == source


def test_remove_does_not_touch_a_different_module():
    source = "import os\nimport sys\n"

    result = _remove(source, "json")

    assert result.changes == 0
    assert result.new_source == source


def test_remove_does_not_touch_a_different_from_import():
    source = "from other import thing\n"

    result = _remove(source, "pkg")

    assert result.changes == 0
    assert result.new_source == source


def test_remove_does_not_touch_relative_imports():
    source = "from . import os\n"

    result = _remove(source, "os")

    assert result.changes == 0
    assert result.new_source == source


def test_remove_keeps_comment_above_a_deleted_import():
    source = "# runtime dependency\nimport os\nX = 1\n"

    result = _remove(source, "os")

    assert result.new_source == "# runtime dependency\nX = 1\n"


def test_remove_moves_comment_onto_the_next_statement():
    source = "X = 1\n# runtime dependency\nimport os\nY = 2\n"

    result = _remove(source, "os")

    assert result.new_source == "X = 1\n# runtime dependency\nY = 2\n"


def test_remove_moves_comment_onto_a_partially_rewritten_statement():
    source = "X = 1\n# note\nimport os\nimport os, json\n"

    result = _remove(source, "os")

    assert result.new_source == "X = 1\n# note\nimport json\n"
    assert result.changes == 2


def test_remove_comment_survives_when_import_is_last_statement():
    source = "X = 1\n\n# runtime dependency\nimport os\n"

    result = _remove(source, "os")

    assert result.new_source == "X = 1\n# runtime dependency\n"


def test_remove_first_import_does_not_leave_blank_lines_at_top():
    source = "\nimport os\nX = 1\n"

    result = _remove(source, "os")

    assert result.new_source == "X = 1\n"


def test_remove_first_import_strips_the_blank_line_before_code():
    source = "import os\n\nX = 1\n"

    result = _remove(source, "os")

    assert result.new_source == "X = 1\n"


def test_remove_last_of_three_parenthesized_names_keeps_close_indent():
    source = "from x import (\n    a,\n    b,\n    c,\n)\n"

    result = _remove(source, "x", "c")

    assert result.new_source == "from x import (\n    a,\n    b,\n)\n"


def test_remove_shebang_is_never_dropped():
    source = "#!/usr/bin/env python\nimport os\nX = 1\n"

    result = _remove(source, "os")

    assert result.new_source == "#!/usr/bin/env python\nX = 1\n"


def test_remove_skips_semicolon_statements():
    source = "import os; import sys\n"

    result = _remove(source, "os")

    assert result.changes == 0
    assert result.new_source == source


def test_remove_from_import_is_noop_when_name_is_absent():
    source = "from collections import OrderedDict\n"

    result = _remove(source, "collections", "Counter")

    assert result.changes == 0
    assert result.new_source == source


@pytest.mark.parametrize(
    ("module", "name"),
    [("", None), ("os..path", None), ("os", "not a name"), ("os", "class"), ("os", "")],
)
def test_remove_import_rejects_invalid_arguments(module, name):
    with pytest.raises(ValueError):
        _remove("import os\n", module, name)


def test_remove_reports_one_change_per_rewritten_statement():
    source = "import os\nfrom os import path\n"

    result = _remove(source, "os")

    assert result.changes == 2
    assert result.new_source == ""
