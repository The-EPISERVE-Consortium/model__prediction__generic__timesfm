"""Tests for src/config_schema.py and the model card's parameter declaration.

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
from config_schema import SchemaError, load_parameters, validate_config  # noqa: E402


# --- fdo.json parses and declares the expected parameters --------------------

def test_fdo_json_declares_the_three_params():
    assert list(load_parameters()["properties"]) == ["history_length", "prediction_length", "prediction_offset"]


def test_fdo_bounds_match_model_constants():
    """The card is the declared source of truth; assert it agrees with the
    model constants so the two cannot silently drift."""
    params = load_parameters()
    props = params["properties"]

    assert set(params["required"]) == {"history_length", "prediction_length"}
    assert params["additionalProperties"] is False

    assert props["history_length"]["minimum"] == model._MIN_CONTEXT
    assert "maximum" not in props["history_length"]  # over-long is truncated, not rejected

    assert props["prediction_length"]["minimum"] == 1
    assert props["prediction_length"]["maximum"] == model._MAX_PREDICTION_STEPS

    assert props["prediction_offset"]["minimum"] == 0
    assert props["prediction_offset"]["default"] == 0


def test_every_declared_param_is_an_integer_type():
    for spec in load_parameters()["properties"].values():
        assert spec["type"] == "integer"


# --- validate_config: happy path -------------------------------------------------

def test_valid_config_fills_default_offset():
    resolved, errors = validate_config({"history_length": 10, "prediction_length": 5})
    assert errors == []
    assert resolved == {"history_length": 10, "prediction_length": 5, "prediction_offset": 0}


def test_explicit_offset_is_kept():
    resolved, errors = validate_config(
        {"history_length": 10, "prediction_length": 5, "prediction_offset": 4}
    )
    assert errors == []
    assert resolved["prediction_offset"] == 4


def test_boundary_values_are_accepted():
    resolved, errors = validate_config(
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
        ({"history_length": None, "prediction_length": 5}, "'history_length' must be an integer"),
    ],
)
def test_invalid_config_reports_clear_error(config, needle):
    _, errors = validate_config(config)
    assert any(needle in e for e in errors), errors


def test_unknown_key_is_an_error():
    _, errors = validate_config(
        {"history_length": 10, "prediction_length": 5, "prediciton_offset": 3, "zz": 1}
    )
    assert errors == ["config.json has unrecognised key(s): 'prediciton_offset', 'zz'"]


def test_config_must_be_an_object():
    resolved, errors = validate_config([1, 2])
    assert resolved == {} and errors == ["config.json: [1, 2] is not of type 'object'"]


def test_multiple_errors_are_all_reported():
    _, errors = validate_config({"history_length": 1, "prediction_length": 99999})
    assert len(errors) == 2


# --- load_parameters: fdo.json lint ---------------------------------------------

@pytest.mark.parametrize("card", [
    {},                                                           # no parameters
    {"episerve:parameters": "nope"},                              # not an object
    {"episerve:parameters": {"type": "object", "properties": {}}},  # nothing declared
    {"episerve:parameters": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": "x"}},  # bad schema
])
def test_load_parameters_rejects_malformed_card(tmp_path, card):
    bad = tmp_path / "fdo.json"
    bad.write_text(json.dumps(card))
    with pytest.raises(SchemaError):
        load_parameters(bad)


def test_load_parameters_rejects_missing_file(tmp_path):
    with pytest.raises(SchemaError):
        load_parameters(tmp_path / "does-not-exist.json")


def test_load_parameters_rejects_invalid_json(tmp_path):
    bad = tmp_path / "fdo.json"
    bad.write_text("{ not json")
    with pytest.raises(SchemaError):
        load_parameters(bad)
