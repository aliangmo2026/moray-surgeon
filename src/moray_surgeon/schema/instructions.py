from __future__ import annotations

import copy
import keyword
import re
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from moray_surgeon.schema.exceptions import CircularSchemaError, SchemaInlineError

_REF_PREFIX = "#/$defs/"

# An instruction id is a plan-local key referenced by ``depends_on``; keeping it
# to an identifier-like alphabet keeps the JSON that reaches the model simple and
# stops a stray space or a newline from becoming an invisible part of the key.
_INSTRUCTION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
_INSTRUCTION_ID_REGEX = re.compile(_INSTRUCTION_ID_PATTERN)


class _BaseInstruction(BaseModel):
    """Shared contract for every instruction: frozen, closed, auditable.

    ``id`` names the step; ``depends_on`` lists the ids that must run first. The
    dispatcher topologically sorts the plan on these two fields, so a model can
    describe "rename the class, then rewrite its callers, then add the import"
    without relying on list position.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        pattern=_INSTRUCTION_ID_PATTERN,
        description=(
            "Unique id of this step inside the plan, for example 'step1'. "
            "Other steps reference it from depends_on."
        ),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of the steps that must execute before this one.",
    )
    reasoning: str = ""

    @field_validator("depends_on")
    @classmethod
    def _validate_depends_on(cls, value: list[str]) -> list[str]:
        """Reject malformed and repeated ids; graph validity is the DAG's job."""
        seen: set[str] = set()
        for dependency in value:
            if not _INSTRUCTION_ID_REGEX.fullmatch(dependency):
                raise ValueError(
                    f"depends_on entries must be instruction ids, got {dependency!r}"
                )
            if dependency in seen:
                raise ValueError(f"depends_on repeats id {dependency!r}")
            seen.add(dependency)
        return value


class _FileInstruction(_BaseInstruction):
    """An instruction bound to one target Python file inside the repo."""

    target_file: str

    @field_validator("target_file")
    @classmethod
    def _validate_target_file(cls, value: str) -> str:
        """Reject non-.py, absolute and traversal paths; return POSIX form.

        `Path.anchor` is checked instead of only `is_absolute()` because on
        Windows a rooted path like "/abs/x.py" is not "absolute" but must
        still be rejected.
        """
        path = Path(value)
        if path.suffix != ".py":
            raise ValueError("target_file must end with .py")
        if ".." in path.parts:
            raise ValueError("target_file must not contain '..'")
        if path.anchor:
            raise ValueError("target_file must be relative and drive-less")
        return path.as_posix()


class RenameSymbol(_FileInstruction):
    """Rename a symbol (function/class/variable) within one file."""

    action: Literal["rename_symbol"] = "rename_symbol"
    old_name: str
    new_name: str


class ExtractFunction(_FileInstruction):
    """Extract lines [start_line, end_line] into a new named function.

    NOTE: start_line <= end_line is enforced by the engine layer, not here.
    """

    action: Literal["extract_function"] = "extract_function"
    function_name: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class ExtractVariable(_FileInstruction):
    """Extract the expression at an exact source range into a new variable.

    The engine locates the unique expression node occupying the range and
    rewrites it to ``variable_name`` while inserting
    ``variable_name = <expression>`` on its own line just before the statement
    that contains the range.

    Coordinates follow libcst's ``PositionProvider``: lines are 1-based,
    columns are 0-based, and the end coordinates point *one past* the
    expression's last character (exclusive, like editor selection ranges).
    NOTE: ordering of the four coordinates is enforced by the engine layer,
    not here.
    """

    action: Literal["extract_variable"] = "extract_variable"
    start_line: int = Field(ge=1)
    start_column: int = Field(ge=0)
    end_line: int = Field(ge=1)
    end_column: int = Field(ge=0)
    variable_name: str

    @field_validator("variable_name")
    @classmethod
    def _validate_variable_name(cls, value: str) -> str:
        """The generated assignment must be syntactically valid."""
        return _validate_required_identifier(value)


class InlineVariable(_FileInstruction):
    """Inline the single assignment of ``variable_name`` into every reference.

    The engine requires the variable to be bound exactly once in its scope
    (SSA form), read only after that assignment, never captured by a nested
    function/class, and never referenced from another module. Anything else is
    refused with ``ValueError`` instead of producing a semantic change.
    """

    action: Literal["inline_variable"] = "inline_variable"
    variable_name: str

    @field_validator("variable_name")
    @classmethod
    def _validate_variable_name(cls, value: str) -> str:
        """The lookup target must be a real Python identifier."""
        return _validate_required_identifier(value)


class RemoveDeadCode(_FileInstruction):
    """Delete a top-level function or class that nothing references.

    Identify the dead symbol with exactly one of ``function_name`` or
    ``class_name``. The engine confirms project-wide deadness with Jedi and
    refuses on any doubt: any reference, any ``__all__`` entry, any string
    literal that could be a dynamic ``getattr`` key, any decorator on the
    symbol, any star import of its module.
    """

    action: Literal["remove_dead_code"] = "remove_dead_code"
    function_name: str | None = None
    class_name: str | None = None

    @model_validator(mode="after")
    def _exactly_one_name(self) -> RemoveDeadCode:
        """Both names (or neither) is an ambiguous instruction: refuse."""
        if (self.function_name is None) == (self.class_name is None):
            raise ValueError("exactly one of function_name or class_name is required")
        name = self.function_name or self.class_name
        assert name is not None
        _validate_required_identifier(name)
        return self

    @property
    def symbol_name(self) -> str:
        """The name of the symbol slated for deletion."""
        return self.function_name or self.class_name  # type: ignore[return-value]


class MoveSymbol(_BaseInstruction):
    """Move a top-level function or class from one file to another.

    The node (with its leading comments and docstring) is appended to
    ``target_file`` and deleted from ``source_file``. ``from <source> import
    <symbol>`` statements in other in-repo modules are rewritten to import from
    the target module; anything the engine cannot prove safe — relative imports
    crossing directories, attribute-style usage, leftover references in the
    source file itself — refuses the whole move.
    """

    action: Literal["move_symbol"] = "move_symbol"
    source_file: str
    target_file: str
    symbol_name: str

    @field_validator("source_file", "target_file")
    @classmethod
    def _validate_files(cls, value: str) -> str:
        """Both endpoints obey the same rules as every other target_file."""
        path = Path(value)
        if path.suffix != ".py":
            raise ValueError("source_file and target_file must end with .py")
        if ".." in path.parts:
            raise ValueError("source_file and target_file must not contain '..'")
        if path.anchor:
            raise ValueError("source_file and target_file must be relative and drive-less")
        return path.as_posix()

    @field_validator("symbol_name")
    @classmethod
    def _validate_symbol_name(cls, value: str) -> str:
        """Only a real identifier can be a top-level definition."""
        return _validate_required_identifier(value)

    @model_validator(mode="after")
    def _files_must_differ(self) -> MoveSymbol:
        """Moving a symbol onto its own file is a no-op at best."""
        if self.source_file == self.target_file:
            raise ValueError("source_file and target_file must be different files")
        return self


class InlineFunction(_FileInstruction):
    """Inline a single-return function into every same-module call site.

    The engine requires the function to be a unique top-level ``def`` whose
    body is exactly one ``return <expr>``; the expression's parameters are
    substituted with the caller's arguments at every call site and the
    definition is deleted. Anything ambiguous — recursion, side-effecting
    bodies used more than once, parameters used twice with impure arguments,
    callers outside the file — refuses the inline.
    """

    action: Literal["inline_function"] = "inline_function"
    function_name: str

    @field_validator("function_name")
    @classmethod
    def _validate_function_name(cls, value: str) -> str:
        """The lookup target must be a real Python identifier."""
        return _validate_required_identifier(value)


class SplitClass(_FileInstruction):
    """Move methods of one class into a new sibling class.

    For this iteration the moved methods must not touch their receiver at all:
    any ``self`` / ``cls`` usage (or ``super()``) refuses the split, so only
    state-free methods can move. Instance-state migration is future work.
    """

    action: Literal["split_class"] = "split_class"
    class_name: str
    methods_to_move: list[str] = Field(min_length=1)
    new_class_name: str

    @field_validator("class_name", "new_class_name")
    @classmethod
    def _validate_class_names(cls, value: str) -> str:
        """Both class names must be real Python identifiers."""
        return _validate_required_identifier(value)

    @field_validator("methods_to_move")
    @classmethod
    def _validate_method_names(cls, value: list[str]) -> list[str]:
        """Every method name must be an identifier and the list must be unique."""
        for name in value:
            _validate_required_identifier(name)
        if len(set(value)) != len(value):
            raise ValueError("methods_to_move must not repeat a method name")
        return value

    @model_validator(mode="after")
    def _classes_must_differ(self) -> SplitClass:
        """Splitting a class onto itself is a no-op at best."""
        if self.new_class_name == self.class_name:
            raise ValueError("new_class_name must differ from class_name")
        return self


class AddTypeHints(_FileInstruction):
    """Add type hints to one function, or to the whole file if function_name is None."""

    action: Literal["add_type_hints"] = "add_type_hints"
    function_name: str | None = None


class ReplaceCall(_FileInstruction):
    """Replace one call expression with another within one file."""

    action: Literal["replace_call"] = "replace_call"
    old_call: str
    new_call: str


class AddImport(_FileInstruction):
    """Add one module-level import, keeping the file's import block sorted.

    ``name=None`` writes ``import <module> [as alias]``; otherwise
    ``from <module> import <name> [as alias]`` is written. When a matching
    ``from`` statement already exists the name is merged into it instead of
    creating a second statement.
    """

    action: Literal["add_import"] = "add_import"
    module: str
    name: str | None = None
    alias: str | None = None

    @field_validator("module")
    @classmethod
    def _validate_module(cls, value: str) -> str:
        """Reject anything that is not an absolute dotted module path."""
        return _validate_module_path(value)

    @field_validator("name", "alias")
    @classmethod
    def _validate_optional_identifier(cls, value: str | None) -> str | None:
        """Reject names that could not be written after ``import``/``as``."""
        return _validate_optional_identifier(value)


class RemoveImport(_FileInstruction):
    """Remove a module-level import from one file.

    With ``name=None`` every import of ``module`` is removed, including
    ``from <module> import ...`` statements. With ``name`` set, only that name
    is removed from ``from <module> import ...`` statements.
    """

    action: Literal["remove_import"] = "remove_import"
    module: str
    name: str | None = None

    @field_validator("module")
    @classmethod
    def _validate_module(cls, value: str) -> str:
        """Reject anything that is not an absolute dotted module path."""
        return _validate_module_path(value)

    @field_validator("name")
    @classmethod
    def _validate_optional_name(cls, value: str | None) -> str | None:
        """Reject names that could never appear in an import statement."""
        return _validate_optional_identifier(value)


def _validate_module_path(value: str) -> str:
    """Validate an absolute dotted module path such as ``pkg.sub.mod``."""
    parts = value.split(".")
    if not value or not all(part.isidentifier() for part in parts):
        raise ValueError(f"module must be a dotted Python module path, got {value!r}")
    if any(keyword.iskeyword(part) for part in parts):
        raise ValueError(f"module must not contain a Python keyword, got {value!r}")
    return value


def _validate_optional_identifier(value: str | None) -> str | None:
    """Validate an optional identifier; None passes through untouched."""
    if value is None:
        return None
    return _validate_required_identifier(value)


def _validate_required_identifier(value: str) -> str:
    """Validate a mandatory identifier; None never passes through."""
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(f"expected a Python identifier, got {value!r}")
    return value


class Noop(_BaseInstruction):
    """Explicit no-operation carrying a human-readable message."""

    action: Literal["noop"] = "noop"
    message: str = "Nothing to do."


Instruction = Annotated[
    RenameSymbol
    | ExtractFunction
    | ExtractVariable
    | InlineVariable
    | InlineFunction
    | SplitClass
    | RemoveDeadCode
    | MoveSymbol
    | AddTypeHints
    | ReplaceCall
    | AddImport
    | RemoveImport
    | Noop,
    Field(discriminator="action"),
]


class RefactorPlan(BaseModel):
    """A batch of refactor instructions plus a human summary.

    The list order is the authoring order, not necessarily the execution order:
    ``engine.dag.topological_sort`` reorders the batch so every ``depends_on``
    edge is honoured, and the dispatcher executes that order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    instructions: list[Instruction] = Field(default_factory=list)
    summary: str = ""

    @classmethod
    def export_schema(cls) -> dict[str, Any]:
        """Return an Ollama-safe JSON Schema: every $ref inlined, discriminator kept.

        Ollama's `format` converts JSON Schema to GBNF grammar and mishandles
        $ref/$defs, so the default pydantic output is normalized here into a
        self-contained oneOf tree before it ever reaches the model.
        """
        document = _inline_document(cls.model_json_schema())
        _assert_no_ref_keys(document)
        return document


def _expand_ref(
    name: str,
    defs: dict[str, Any],
    visiting: set[str],
    memo: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Expand one $defs entry into a fully inlined copy, memoized per name.

    `visiting` holds the names on the current expansion stack so a cycle is
    detected (and reported) instead of recursing forever.
    """
    cached = memo.get(name)
    if cached is not None:
        return copy.deepcopy(cached)
    if name in visiting:
        raise CircularSchemaError(f"circular $ref while expanding {name!r}")
    definition = defs.get(name)
    if definition is None:
        raise SchemaInlineError(f"dangling $ref target {name!r}")
    visiting.add(name)
    try:
        expanded = _inline_node(definition, defs, visiting, memo)
    finally:
        visiting.discard(name)
    if not isinstance(expanded, dict):
        raise SchemaInlineError(f"$ref target {name!r} must inline to an object schema")
    memo[name] = expanded
    # Deep copies keep later sibling merges from mutating the shared memo entry.
    return copy.deepcopy(expanded)


def _inline_discriminator(
    discriminator: dict[str, object],
    defs: dict[str, Any],
    visiting: set[str],
    memo: dict[str, dict[str, Any]],
) -> dict[str, object]:
    """Replace discriminator mapping pointers with the inlined branch schemas.

    Mapping values are "#/$defs/X" strings in pydantic output; after inlining
    those pointers would dangle, so each value becomes the expanded branch.
    """
    mapping = discriminator.get("mapping")
    if not isinstance(mapping, dict):
        return discriminator
    new_mapping: dict[str, object] = {}
    for tag, target in mapping.items():
        if isinstance(target, str) and target.startswith(_REF_PREFIX):
            new_mapping[tag] = _expand_ref(target[len(_REF_PREFIX) :], defs, visiting, memo)
        else:
            new_mapping[tag] = target
    return {**discriminator, "mapping": new_mapping}


def _inline_node(
    node: object,
    defs: dict[str, Any],
    visiting: set[str],
    memo: dict[str, dict[str, Any]],
) -> object:
    """Recursively inline every $ref inside an arbitrary JSON Schema fragment."""
    if isinstance(node, dict):
        new_node: dict[str, object] = {}
        for key, value in node.items():
            if key == "$ref":
                continue
            new_node[key] = _inline_node(value, defs, visiting, memo)
        ref = node.get("$ref")
        if ref is not None:
            if not isinstance(ref, str) or not ref.startswith(_REF_PREFIX):
                raise SchemaInlineError(f"unsupported $ref form: {ref!r}")
            # Sibling keywords next to $ref override the target (JSON Schema 2020-12).
            new_node = {**_expand_ref(ref[len(_REF_PREFIX) :], defs, visiting, memo), **new_node}
        discriminator = new_node.get("discriminator")
        if isinstance(discriminator, dict):
            new_node["discriminator"] = _inline_discriminator(discriminator, defs, visiting, memo)
        return new_node
    if isinstance(node, list):
        return [_inline_node(item, defs, visiting, memo) for item in node]
    return node


def _inline_document(document: dict[str, Any]) -> dict[str, Any]:
    """Inline every $ref of a JSON Schema document and drop the $defs section."""
    defs = document.get("$defs", {})
    if not isinstance(defs, dict):
        raise SchemaInlineError("$defs must be an object when present")
    root = {key: value for key, value in document.items() if key != "$defs"}
    return cast("dict[str, Any]", _inline_node(root, defs, set(), {}))


def _assert_no_ref_keys(node: object) -> int:
    """Raise SchemaInlineError if any "$ref"/"$defs" key survives; return node count.

    This is the production-side invariant check: an inlined schema handed to
    Ollama must be fully self-contained at every depth, not just at the top.
    """
    count = 1
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("$ref", "$defs"):
                raise SchemaInlineError(f"uninlined schema key survived: {key!r}")
            count += _assert_no_ref_keys(value)
    elif isinstance(node, list):
        for item in node:
            count += _assert_no_ref_keys(item)
    return count
