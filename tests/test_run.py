import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_with_config_text(tmp_path, config_text, work_dir=None):
    """Run the CLI with a temporary config file.

    Args:
        tmp_path: Temporary directory provided by pytest.
        config_text: Raw config.json text to write.
        work_dir: Optional work directory to expose through WORK_DIR.

    Returns:
        Completed subprocess result for the CLI invocation.
    """
    work_dir = work_dir or tmp_path / "work"
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "config.json").write_text(config_text)
    (tmp_path / "timesfm.py").write_text("")

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


def _run_with_config(tmp_path, config, work_dir=None):
    """Run the CLI with a temporary JSON config.

    Args:
        tmp_path: Temporary directory provided by pytest.
        config: Config dictionary to write to work/input/config.json.
        work_dir: Optional work directory to expose through WORK_DIR.

    Returns:
        Completed subprocess result for the CLI invocation.
    """
    return _run_with_config_text(tmp_path, json.dumps(config), work_dir=work_dir)


def _run_without_config(tmp_path, work_dir):
    """Run the CLI without writing config.json.

    Args:
        tmp_path: Temporary directory provided by pytest.
        work_dir: Work directory to expose through WORK_DIR.

    Returns:
        Completed subprocess result for the CLI invocation.
    """
    (tmp_path / "timesfm.py").write_text("")
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
        ({"history_length": 0, "prediction_length": 1}, "history_length must be >= 2"),
        ({"history_length": -1, "prediction_length": 1}, "history_length must be >= 2"),
        ({"history_length": 1, "prediction_length": 1}, "history_length must be >= 2"),
        ({"history_length": 2, "prediction_length": 0}, "prediction_length must be > 0"),
        ({"history_length": 2, "prediction_length": -1}, "prediction_length must be > 0"),
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
    assert "ERROR: invalid config.json numeric value" in result.stderr
    assert "Traceback" not in result.stderr


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
        {"history_length": 2, "prediction_length": 1},
        work_dir=work_dir,
    )

    assert result.returncode == 1
    assert f"{work_dir.resolve()}/input/input.parquet not found" in result.stderr
