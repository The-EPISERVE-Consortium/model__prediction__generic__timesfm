import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_with_config(tmp_path, config):
    """Run the CLI with a temporary work directory and config.

    Args:
        tmp_path: Temporary directory provided by pytest.
        config: Config dictionary to write to work/input/config.json.

    Returns:
        Completed subprocess result for the CLI invocation.
    """
    input_dir = tmp_path / "work" / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "config.json").write_text(json.dumps(config))
    (tmp_path / "timesfm.py").write_text("")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)

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
        ({"history_length": 0, "prediction_length": 1}, "history_length must be > 0"),
        ({"history_length": -1, "prediction_length": 1}, "history_length must be > 0"),
        ({"history_length": 1, "prediction_length": 0}, "prediction_length must be > 0"),
        ({"history_length": 1, "prediction_length": -1}, "prediction_length must be > 0"),
    ],
)
def test_invalid_lengths_exit_before_loading_data(tmp_path, config, message):
    """Verify invalid config lengths fail with clear CLI errors."""
    result = _run_with_config(tmp_path, config)

    assert result.returncode == 1
    assert message in result.stderr
    assert "input.parquet not found" not in result.stderr
