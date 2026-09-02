"""Config-parameter schema for this model, driven by ``fdo.json``.

``fdo.json``'s ``additionalProperty`` array is the single declaration of
which ``config.json`` keys the model accepts. This module loads it and
validates a parsed config dict against it:

* required keys must be present
* values must match the declared type (integer unless ``valueType`` says
  otherwise) and lie within any declared ``minValue`` / ``maxValue``
* omitted optional keys are filled from ``value`` (the default)

Cross-field rules that a per-key schema cannot express (for example,
``history_length + prediction_offset`` must not exceed the number of input
rows) stay in ``run.py``; this module only covers per-key declarations.

The same file is used at runtime (``run.py`` calls :func:`validate_config`,
including via ``--check-config``) and as a lint (``tests/test_config_schema.py``
fails the build if ``fdo.json`` is malformed or drifts from the model
constants).
"""

from __future__ import annotations

import json
from pathlib import Path

# fdo.json sits at the repo root; this module is src/config_schema.py.
_FDO_PATH = Path(__file__).resolve().parent.parent / "fdo.json"

_ALLOWED_VALUE_TYPES = {"integer", "number", "string"}


class SchemaError(Exception):
    """Raised when ``fdo.json`` itself is missing or malformed.

    This is a packaging/build problem, not a user config problem -- callers
    should surface it distinctly from :func:`validate_config` errors.
    """


def _is_int(value: object) -> bool:
    """Return True for a real integer (bools are not integers here)."""
    return isinstance(value, int) and not isinstance(value, bool)


class ParamSpec:
    """One declared config parameter, parsed from an ``additionalProperty`` entry."""

    __slots__ = (
        "name",
        "required",
        "value_type",
        "default",
        "min_value",
        "max_value",
        "description",
    )

    def __init__(self, entry: dict) -> None:
        if not isinstance(entry, dict):
            raise SchemaError(f"additionalProperty entry is not an object: {entry!r}")

        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise SchemaError(f"additionalProperty entry has no string 'name': {entry!r}")
        self.name = name

        required = entry.get("valueRequired", False)
        if not isinstance(required, bool):
            raise SchemaError(f"{name}: 'valueRequired' must be a boolean")
        self.required = required

        value_type = entry.get("valueType", "integer")
        if value_type not in _ALLOWED_VALUE_TYPES:
            raise SchemaError(
                f"{name}: 'valueType' must be one of {sorted(_ALLOWED_VALUE_TYPES)}"
            )
        self.value_type = value_type

        self.default = entry.get("value")
        self.min_value = entry.get("minValue")
        self.max_value = entry.get("maxValue")
        self.description = entry.get("description", "")

        if self.required and self.default is not None:
            raise SchemaError(f"{name}: a required parameter must not also declare a default 'value'")

        if value_type == "integer":
            for label, bound in (
                ("minValue", self.min_value),
                ("maxValue", self.max_value),
                ("value", self.default),
            ):
                if bound is not None and not _is_int(bound):
                    raise SchemaError(f"{name}: '{label}' must be an integer")

        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise SchemaError(f"{name}: 'minValue' is greater than 'maxValue'")


def load_specs(path: Path | None = None) -> list[ParamSpec]:
    """Parse ``fdo.json``'s ``additionalProperty`` list into :class:`ParamSpec`.

    Args:
        path: Override for the ``fdo.json`` location (tests only).

    Returns:
        One :class:`ParamSpec` per declared parameter, in declaration order.

    Raises:
        SchemaError: If the file is missing, is not valid JSON, has no
            non-empty ``additionalProperty`` list, contains a malformed
            entry, or declares a duplicate parameter name.
    """
    path = path or _FDO_PATH
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SchemaError(f"{path} not found") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path} is not valid JSON: {exc}") from exc

    props = raw.get("additionalProperty")
    if not isinstance(props, list) or not props:
        raise SchemaError(f"{path}: 'additionalProperty' must be a non-empty list")

    specs = [ParamSpec(entry) for entry in props]

    names = [spec.name for spec in specs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SchemaError(f"{path}: duplicate parameter name(s): {', '.join(duplicates)}")

    return specs


def validate_config(
    config: dict,
    specs: list[ParamSpec] | None = None,
) -> tuple[dict, list[str], list[str]]:
    """Validate ``config`` against the declared parameter specs.

    Args:
        config: Parsed ``config.json`` contents.
        specs: Parameter specs; loaded from ``fdo.json`` when omitted.

    Returns:
        ``(resolved, errors, warnings)``. ``resolved`` maps every declared
        key to its value, with omitted optional keys filled from their
        default; it is only meaningful when ``errors`` is empty. ``errors``
        are fatal, human-readable messages (``run.py`` prints each as an
        ``ERROR:`` line and exits non-zero). ``warnings`` are non-fatal
        (printed as ``WARNING:`` lines); currently only "unrecognised key".

    Raises:
        SchemaError: If ``specs`` is omitted and ``fdo.json`` is malformed.
    """
    if specs is None:
        specs = load_specs()

    errors: list[str] = []
    warnings: list[str] = []
    resolved: dict = {}
    known = {spec.name for spec in specs}

    for spec in specs:
        if spec.name not in config:
            if spec.required:
                errors.append(f"config.json missing required key: '{spec.name}'")
            else:
                resolved[spec.name] = spec.default
            continue

        value = config[spec.name]

        if spec.value_type == "integer" and not _is_int(value):
            errors.append(f"config.json key '{spec.name}' must be an integer")
            continue
        if spec.value_type == "number" and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            errors.append(f"config.json key '{spec.name}' must be a number")
            continue
        if spec.value_type == "string" and not isinstance(value, str):
            errors.append(f"config.json key '{spec.name}' must be a string")
            continue

        if spec.min_value is not None and value < spec.min_value:
            errors.append(f"config.json key '{spec.name}' must be >= {spec.min_value}")
            continue
        if spec.max_value is not None and value > spec.max_value:
            errors.append(f"config.json key '{spec.name}' must be <= {spec.max_value}")
            continue

        resolved[spec.name] = value

    for name in sorted(set(config) - known):
        warnings.append(f"config.json has an unrecognised key: '{name}' (ignored)")

    return resolved, errors, warnings
