"""Config check for this model, driven by the model card ``fdo.json``.

The card's ``episerve:parameters`` (a JSON Schema of ``config.json``) is the
single declaration of which keys the model accepts. This module loads it and
checks a parsed config against it with ``jsonschema``:

* required keys must be present
* values must have the declared type and lie within the declared bounds
* keys the card does not declare are errors (``additionalProperties: false``)
* omitted optional keys are filled from their ``default``

Cross-field rules that a per-key schema cannot express (for example,
``history_length + prediction_offset`` must not exceed the number of input
rows) are listed as ``rules`` in the card and checked in ``run.py``.

The same file is used at runtime (``run.py`` calls :func:`validate_config`,
including via ``--check-config``) and as a lint (``tests/test_config_schema.py``
fails the build if the parameters are malformed or drift from the model
constants).
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as _JsonSchemaError

# fdo.json sits at the repo root; this module is src/config_schema.py.
_FDO_PATH = Path(__file__).resolve().parent.parent / "fdo.json"


class SchemaError(Exception):
    """Raised when ``fdo.json`` itself is missing or malformed.

    This is a packaging/build problem, not a user config problem -- callers
    should surface it distinctly from :func:`validate_config` errors.
    """


def load_parameters(path: Path | None = None) -> dict:
    """Return the card's ``episerve:parameters``.

    Args:
        path: Override for the ``fdo.json`` location (tests only).

    Raises:
        SchemaError: If the file is missing or not valid JSON, or its
            ``episerve:parameters`` is not a JSON Schema object with a
            non-empty ``properties``.
    """
    path = path or _FDO_PATH
    try:
        card = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SchemaError(f"{path} not found") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path} is not valid JSON: {exc}") from exc

    params = card.get("episerve:parameters") if isinstance(card, dict) else None
    if not isinstance(params, dict) or not isinstance(params.get("properties"), dict) or not params["properties"]:
        raise SchemaError(f"{path}: 'episerve:parameters' must be an object with a non-empty 'properties'")
    try:
        Draft202012Validator.check_schema(params)
    except _JsonSchemaError as exc:
        raise SchemaError(f"{path}: 'episerve:parameters' is not a valid JSON Schema: {exc.message}") from exc
    return params


def _message(error) -> str:
    """One readable line per violation, naming the key."""
    key = error.absolute_path[0] if error.absolute_path else None
    v, arg = error.validator, error.validator_value
    if v == "required":
        missing = error.message.split("'")[1]
        return f"config.json missing required key: '{missing}'"
    if v == "additionalProperties" and isinstance(error.instance, dict):
        unknown = sorted(set(error.instance) - set(error.schema.get("properties", {})))
        return "config.json has unrecognised key(s): " + ", ".join(f"'{k}'" for k in unknown)
    if key is None:
        return f"config.json: {error.message}"
    if v == "type":
        article = "an" if str(arg)[0] in "aeiou" else "a"
        return f"config.json key '{key}' must be {article} {arg}"
    if v == "minimum":
        return f"config.json key '{key}' must be >= {arg}"
    if v == "maximum":
        return f"config.json key '{key}' must be <= {arg}"
    return f"config.json key '{key}': {error.message}"


def validate_config(config: dict, parameters: dict | None = None) -> tuple[dict, list[str]]:
    """Check ``config`` against the declared parameters.

    Args:
        config: Parsed ``config.json`` contents.
        parameters: The card's ``episerve:parameters``; loaded from ``fdo.json`` when omitted.

    Returns:
        ``(resolved, errors)``. ``resolved`` maps every declared key to its
        value, with omitted optional keys filled from their default; it is only
        meaningful when ``errors`` is empty. ``errors`` are human-readable
        messages (``run.py`` prints each as an ``ERROR:`` line and exits non-zero).

    Raises:
        SchemaError: If ``parameters`` is omitted and ``fdo.json`` is malformed.
    """
    if parameters is None:
        parameters = load_parameters()

    validator = Draft202012Validator(parameters)
    errors = [_message(e) for e in sorted(validator.iter_errors(config), key=lambda e: list(map(str, e.absolute_path)))]

    resolved = {}
    if isinstance(config, dict):
        for name, spec in parameters["properties"].items():
            if name in config:
                resolved[name] = config[name]
            elif "default" in spec:
                resolved[name] = spec["default"]
    return resolved, errors
