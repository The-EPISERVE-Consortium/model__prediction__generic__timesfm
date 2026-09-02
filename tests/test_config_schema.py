"""Tests for src/config_schema.py and the fdo.json parameter declaration.

Doubles as the CI lint for fdo.json: a malformed or drifted fdo.json fails
these tests, which run inside the built image in CI.
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Stub timesfm so importing `model` (for its constants) needs no torch stack.
if "timesfm" not in sys.modules:
    _stub = types.ModuleType("timesfm")
    _stub.ForecastConfig = MagicMock()
    _stub.TimesFM_2p5_200M_torch = MagicMock()
    sys.modules["timesfm"] = _stub

sys.path.insert(0, str(REPO_ROOT / "src"))

import model  # noqa: E402
from config_schema import ParamSpec, SchemaError, load_specs, validate_config  # noqa: E402


# --- fdo.json parses and declares the expected parameters --------------------

def test_fdo_json_declares_the_three_params():
    names = [spec.name for spec in load_specs()]
    assert names == ["history_length", "prediction_length", "prediction_offset"]


def test_fdo_bounds_match_model_constants():
    """fdo.json is the declared source of truth; assert it agrees with the
    model constants so the two cannot silently drift."""
    specs = {spec.name: spec for spec in load_specs()}

    assert specs["history_length"].required is True
    assert specs["history_length"].min_value == model._MIN_CONTEXT
    assert specs["history_length"].max_value is None  # over-long is truncated, not rejected

    assert specs["prediction_length"].required is True
    assert specs["prediction_length"].min_value == 1
    assert specs["prediction_length"].max_value == model._MAX_PREDICTION_STEPS

    assert specs["prediction_offset"].required is False
    assert specs["prediction_offset"].min_value == 0
    assert specs["prediction_offset"].default == 0


def test_every_declared_param_is_an_integer_type():
    for spec in load_specs():
        assert spec.value_type == "integer"


# --- validate_config: happy path -------------------------------------------------

def test_valid_config_fills_default_offset():
    resolved, errors, warnings = validate_config({"history_length": 10, "prediction_length": 5})
    assert errors == []
    assert warnings == []
    assert resolved == {"history_length": 10, "prediction_length": 5, "prediction_offset": 0}


def test_explicit_offset_is_kept():
    resolved, errors, _ = validate_config(
        {"history_length": 10, "prediction_length": 5, "prediction_offset": 4}
    )
    assert errors == []
    assert resolved["prediction_offset"] == 4


def test_boundary_values_are_accepted():
    resolved, errors, _ = validate_config(
        {"history_length": model._MIN_CONTEXT, "prediction_length": model._MAX_PREDICTION_STEPS,
         "prediction_offset": 0}
    )
    assert errors == []


# --- validate_config: errors ---------------------------------------------------

@pytest.mark.parametrize(
    ("config", "needle"),
    [
        ({"prediction_length": 5}, "missing required key: 'history_length'"),
        ({"history_length": 10}, "missing required key: 'prediction_length'"),
        ({"history_length": 2, "prediction_length": 5}, "'history_length' must be >= 3"),
        ({"history_length": 10, "prediction_length": 0}, "'prediction_length' must be >= 1"),
        ({"history_length": 10, "prediction_length": 513}, "'prediction_length' must be <= 512"),
        ({"history_length": 10, "prediction_length": 5, "prediction_offset": -1},
         "'prediction_offset' must be >= 0"),
        ({"history_length": 1.5, "prediction_length": 5}, "'history_length' must be an integer"),
        ({"history_length": True, "prediction_length": 5}, "'history_length' must be an integer"),
        ({"history_length": "10", "prediction_length": 5}, "'history_length' must be an integer"),
    ],
)
def test_invalid_config_reports_clear_error(config, needle):
    _, errors, _ = validate_config(config)
    assert any(needle in e for e in errors), errors


def test_unknown_key_is_a_warning_not_an_error():
    resolved, errors, warnings = validate_config(
        {"history_length": 10, "prediction_length": 5, "prediciton_offset": 3}
    )
    assert errors == []
    assert any("prediciton_offset" in w for w in warnings)
    assert "prediciton_offset" not in resolved


def test_multiple_errors_are_all_reported():
    _, errors, _ = validate_config({"history_length": 1, "prediction_length": 99999})
    assert len(errors) == 2


# --- load_specs: fdo.json lint ------------------------------------------------

def test_load_specs_rejects_non_list_additional_property(tmp_path):
    bad = tmp_path / "fdo.json"
    bad.write_text(json.dumps({"additionalProperty": "nope"}))
    with pytest.raises(SchemaError):
        load_specs(bad)


def test_load_specs_rejects_missing_file(tmp_path):
    with pytest.raises(SchemaError):
        load_specs(tmp_path / "does-not-exist.json")


def test_load_specs_rejects_invalid_json(tmp_path):
    bad = tmp_path / "fdo.json"
    bad.write_text("{ not json")
    with pytest.raises(SchemaError):
        load_specs(bad)


def test_load_specs_rejects_duplicate_names(tmp_path):
    bad = tmp_path / "fdo.json"
    bad.write_text(json.dumps({"additionalProperty": [
        {"name": "x", "valueRequired": True},
        {"name": "x", "valueRequired": False},
    ]}))
    with pytest.raises(SchemaError):
        load_specs(bad)


@pytest.mark.parametrize("entry", [
    {"valueRequired": True},                                # no name
    {"name": "x", "valueRequired": "yes"},                  # non-bool required
    {"name": "x", "valueType": "date"},                     # unknown value type
    {"name": "x", "minValue": 1.5},                         # non-int bound for integer
    {"name": "x", "minValue": 10, "maxValue": 1},           # min > max
    {"name": "x", "valueRequired": True, "value": 0},       # required + default
])
def test_param_spec_rejects_malformed_entry(entry):
    with pytest.raises(SchemaError):
        ParamSpec(entry)
