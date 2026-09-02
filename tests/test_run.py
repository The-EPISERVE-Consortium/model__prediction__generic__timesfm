import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# Stub the timesfm package in this test process (as test_model.py does) so we
# can import model constants without pulling in the real TimesFM/torch stack.
# The subprocess-based CLI tests below stub timesfm via PYTHONPATH instead.
if "timesfm" not in sys.modules:
    timesfm_stub = types.ModuleType("timesfm")
    timesfm_stub.ForecastConfig = MagicMock()
    timesfm_stub.TimesFM_2p5_200M_torch = MagicMock()
    sys.modules["timesfm"] = timesfm_stub

sys.path.insert(0, str(REPO_ROOT / "src"))
from model import _MIN_CONTEXT, _MAX_CONTEXT


TIMESFM_STUB = """
import numpy as np


class ForecastConfig:
    \"\"\"Stub TimesFM forecast config.\"\"\"

    def __init__(self, **kwargs):
        \"\"\"Store config keyword arguments.

        Args:
            **kwargs: Forecast config options.
        \"\"\"
        self.kwargs = kwargs


class _TimesFM:
    \"\"\"Stub TimesFM model.\"\"\"

    def compile(self, config):
        \"\"\"Store the compile config.

        Args:
            config: Forecast config object.
        \"\"\"
        self.config = config

    def forecast(self, inputs, horizon):
        \"\"\"Return deterministic point and quantile forecasts.

        Args:
            inputs: Forecast input arrays.
            horizon: Forecast horizon.

        Returns:
            Tuple of point and quantile forecast arrays.
        \"\"\"
        batch = len(inputs)
        point = np.full((batch, horizon), 45.0)
        quantiles = np.zeros((batch, horizon, 10))
        quantiles[..., 1] = 30.0
        quantiles[..., 9] = 60.0
        return point, quantiles


class TimesFM_2p5_200M_torch:
    \"\"\"Stub TimesFM torch loader.\"\"\"

    @classmethod
    def from_pretrained(cls, name):
        \"\"\"Return a stub TimesFM model.

        Args:
            name: Model name.

        Returns:
            Stub TimesFM model.
        \"\"\"
        return _TimesFM()
"""


def _run_with_config_text(tmp_path, config_text, work_dir=None, args=()):
    """Run the CLI with a temporary config file.

    Args:
        tmp_path: Temporary directory provided by pytest.
        config_text: Raw config.json text to write.
        work_dir: Optional work directory to expose through WORK_DIR.
        args: Extra command-line arguments for run.py (e.g. ``--check-config``).

    Returns:
        Completed subprocess result for the CLI invocation.
    """
    work_dir = work_dir or tmp_path / "work"
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "config.json").write_text(config_text)
    (tmp_path / "timesfm.py").write_text(TIMESFM_STUB)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    env["WORK_DIR"] = str(work_dir)

    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "run.py"), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_with_config(tmp_path, config, work_dir=None, args=()):
    """Run the CLI with a temporary JSON config.

    Args:
        tmp_path: Temporary directory provided by pytest.
        config: Config dictionary to write to work/input/config.json.
        work_dir: Optional work directory to expose through WORK_DIR.
        args: Extra command-line arguments for run.py (e.g. ``--check-config``).

    Returns:
        Completed subprocess result for the CLI invocation.
    """
    return _run_with_config_text(tmp_path, json.dumps(config), work_dir=work_dir, args=args)


def _run_without_config(tmp_path, work_dir):
    """Run the CLI without writing config.json.

    Args:
        tmp_path: Temporary directory provided by pytest.
        work_dir: Work directory to expose through WORK_DIR.

    Returns:
        Completed subprocess result for the CLI invocation.
    """
    (tmp_path / "timesfm.py").write_text(TIMESFM_STUB)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    env["WORK_DIR"] = str(work_dir)

    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "run.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"history_length": 0, "prediction_length": 1}, f"'history_length' must be >= {_MIN_CONTEXT}"),
        ({"history_length": -1, "prediction_length": 1}, f"'history_length' must be >= {_MIN_CONTEXT}"),
        ({"history_length": 1, "prediction_length": 1}, f"'history_length' must be >= {_MIN_CONTEXT}"),
        # history_length=2 is below the model's minimum context; the CLI must
        # reject it up front (with the model's bound), not after data loading.
        ({"history_length": 2, "prediction_length": 1}, f"'history_length' must be >= {_MIN_CONTEXT}"),
        ({"history_length": _MIN_CONTEXT, "prediction_length": 0}, "'prediction_length' must be >= 1"),
        ({"history_length": _MIN_CONTEXT, "prediction_length": -1}, "'prediction_length' must be >= 1"),
    ],
)
def test_invalid_lengths_exit_before_loading_data(tmp_path, config, message):
    """Verify invalid config lengths fail with clear CLI errors."""
    result = _run_with_config(tmp_path, config)

    assert result.returncode == 1
    assert message in result.stderr
    assert "input.parquet not found" not in result.stderr


def test_malformed_config_exits_with_clear_error(tmp_path):
    """Verify malformed JSON fails without a Python traceback."""
    result = _run_with_config_text(tmp_path, "{")

    assert result.returncode == 1
    assert "ERROR: invalid config.json" in result.stderr
    assert "Traceback" not in result.stderr


def test_noninteger_config_exits_with_clear_error(tmp_path):
    """Verify non-integer config values fail without a Python traceback."""
    result = _run_with_config(tmp_path, {"history_length": "abc", "prediction_length": 1})

    assert result.returncode == 1
    assert "ERROR: config.json key 'history_length' must be an integer" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("history_length", 40.5),
        ("prediction_length", "2"),
        ("prediction_offset", True),
    ],
)
def test_config_integer_fields_reject_non_integer_types(tmp_path, key, value):
    """Verify integer config fields reject floats, strings, and booleans."""
    config = {"history_length": _MIN_CONTEXT, "prediction_length": 1, "prediction_offset": 0}
    config[key] = value

    result = _run_with_config(tmp_path, config)

    assert result.returncode == 1
    assert f"ERROR: config.json key '{key}' must be an integer" in result.stderr
    assert "input.parquet not found" not in result.stderr


def test_missing_config_reports_resolved_path(tmp_path):
    """Verify missing config errors include the actual resolved path."""
    work_dir = tmp_path / "custom-work"
    result = _run_without_config(tmp_path, work_dir)

    assert result.returncode == 1
    assert f"{work_dir.resolve()}/input/config.json not found" in result.stderr


def test_missing_input_reports_resolved_path(tmp_path):
    """Verify missing parquet errors include the actual resolved path."""
    work_dir = tmp_path / "custom-work"
    result = _run_with_config(
        tmp_path,
        {"history_length": _MIN_CONTEXT, "prediction_length": 1},
        work_dir=work_dir,
    )

    assert result.returncode == 1
    assert f"{work_dir.resolve()}/input/input.parquet not found" in result.stderr


def test_check_config_accepts_a_valid_config_without_data(tmp_path):
    """--check-config validates config.json and exits 0 without input.parquet."""
    result = _run_with_config(
        tmp_path,
        {"history_length": 10, "prediction_length": 5},
        args=("--check-config",),
    )

    assert result.returncode == 0, result.stderr
    assert "config.json OK" in result.stdout
    # No input.parquet exists; the mode must not have looked for one or run.
    assert "input.parquet" not in result.stderr
    assert "Written" not in result.stdout


def test_check_config_reports_missing_required_key(tmp_path):
    """--check-config surfaces a missing required key and exits non-zero."""
    result = _run_with_config(
        tmp_path, {"history_length": 10}, args=("--check-config",)
    )

    assert result.returncode == 1
    assert "missing required key: 'prediction_length'" in result.stderr
    assert "config.json OK" not in result.stdout


def test_check_config_still_warns_about_over_long_history(tmp_path):
    """--check-config passes but still emits the >max-context WARNING."""
    result = _run_with_config(
        tmp_path,
        {"history_length": _MAX_CONTEXT + 1, "prediction_length": 5},
        args=("--check-config",),
    )

    assert result.returncode == 0, result.stderr
    assert "exceeds model max context" in result.stderr


@pytest.mark.parametrize("columns", [["x"], ["x", "y", "z"], ["x", "y", "z1", "z2"]])
def test_rejects_parquet_without_exactly_two_columns(tmp_path, columns):
    """input.parquet must be exactly x + y; anything else is a clear error."""
    work_dir = tmp_path / "work"
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True)
    pd.DataFrame({c: np.arange(10, dtype=float) for c in columns}).to_parquet(
        input_dir / "input.parquet"
    )

    result = _run_with_config(
        tmp_path,
        {"history_length": 3, "prediction_length": 2},
        work_dir=work_dir,
    )

    assert result.returncode == 1
    assert "must have exactly 2 columns" in result.stderr
    assert "Traceback" not in result.stderr


def test_string_x_prediction_offset_writes_absolute_positions(tmp_path):
    """Verify string x output positions are shifted to absolute row positions."""
    work_dir = tmp_path / "work"
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "week": [f"2022-W{i:02d}" for i in range(1, 7)],
            "cases": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    ).to_parquet(input_dir / "input.parquet")

    result = _run_with_config(
        tmp_path,
        {"history_length": 3, "prediction_length": 2, "prediction_offset": 2},
        work_dir=work_dir,
    )

    assert result.returncode == 0, result.stderr
    output = pd.read_csv(work_dir / "output" / "predictions.tsv", sep="\t")
    assert list(output["week"]) == [5, 6]
    assert list(output["x_auto_converted"]) == [5, 6]


def test_history_length_exceeding_max_context_truncates_window(tmp_path):
    """Verify history_length > _MAX_CONTEXT truncates to the most recent
    _MAX_CONTEXT rows before validation.

    A long interior NaN gap placed entirely within the dropped prefix must not
    cause a false rejection, and the window actually used must be the most
    recent _MAX_CONTEXT rows (matching the CLI's WARNING).
    """
    work_dir = tmp_path / "work"
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True)
    n = _MAX_CONTEXT + 60
    x = np.arange(n, dtype=float)
    y = np.random.rand(n) * 10 + 5
    # 20-row interior NaN gap (well over the interpolation limit of 8) placed
    # entirely within the prefix that gets dropped (rows 0 .. n-_MAX_CONTEXT-1).
    gap_start = 40
    y[gap_start:gap_start + 20] = np.nan
    pd.DataFrame({"x": x, "y": y}).to_parquet(input_dir / "input.parquet")

    result = _run_with_config(
        tmp_path,
        {"history_length": n, "prediction_length": 5},
        work_dir=work_dir,
    )

    assert result.returncode == 0, result.stderr
    # The window was truncated: used rows start at n - _MAX_CONTEXT.
    assert f"Using rows {n - _MAX_CONTEXT}" in result.stdout
    output = pd.read_csv(work_dir / "output" / "predictions.tsv", sep="\t")
    assert len(output) == 5


def test_model_runtime_error_exits_cleanly_without_traceback(tmp_path):
    """Verify a non-ValueError from the model exits cleanly with an ERROR line."""
    work_dir = tmp_path / "work"
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "week": [f"2022-W{i:02d}" for i in range(1, 7)],
            "cases": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    ).to_parquet(input_dir / "input.parquet")
    (input_dir / "config.json").write_text(
        json.dumps({"history_length": 3, "prediction_length": 2})
    )

    error_stub = TIMESFM_STUB.replace(
        "return point, quantiles",
        "raise RuntimeError('simulated model failure')",
    )
    (tmp_path / "timesfm.py").write_text(error_stub)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    env["WORK_DIR"] = str(work_dir)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "run.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "ERROR: model failed" in result.stderr
    assert "Traceback" not in result.stderr
