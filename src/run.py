import json
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import predict, _MAX_PREDICTION_STEPS, _MAX_CONTEXT


def _require_config_int(config: dict, key: str, default=None) -> int:
    """Return a config value only if it is a real JSON integer.

    Args:
        config: Parsed config dictionary.
        key: Config key to read.
        default: Optional default value for missing keys.

    Returns:
        Integer config value.

    Raises:
        ValueError: If the value is missing without a default, is a boolean, or
            is not an integer.
    """
    value = config.get(key, default)
    if value is None:
        raise ValueError(f"config.json missing required key: '{key}'")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"config.json key '{key}' must be an integer")
    return value


_work = Path(os.environ["WORK_DIR"]) if "WORK_DIR" in os.environ else Path("./work")
if "WORK_DIR" not in os.environ and not _work.exists():
    _work = Path("/work")
_work = _work.resolve()
INPUT  = _work / "input"
OUTPUT = _work / "output"

# ── Config ────────────────────────────────────────────────────────────────────
config_path = INPUT / "config.json"
if not config_path.exists():
    print(f"ERROR: {config_path} not found", file=sys.stderr)
    sys.exit(1)

try:
    config = json.loads(config_path.read_text())
except json.JSONDecodeError as exc:
    print(f"ERROR: invalid config.json: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    history_length = _require_config_int(config, "history_length")
    prediction_length = _require_config_int(config, "prediction_length")
    prediction_offset = _require_config_int(config, "prediction_offset", default=0)
except ValueError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
print(f"Config: history_length={history_length}, prediction_length={prediction_length}, prediction_offset={prediction_offset}")

if history_length < 2:
    print("ERROR: history_length must be >= 2", file=sys.stderr)
    sys.exit(1)

if prediction_length <= 0:
    print("ERROR: prediction_length must be > 0", file=sys.stderr)
    sys.exit(1)

if prediction_length > _MAX_PREDICTION_STEPS:
    print(
        f"ERROR: prediction_length={prediction_length} exceeds max {_MAX_PREDICTION_STEPS} steps",
        file=sys.stderr,
    )
    sys.exit(1)

if prediction_offset < 0:
    print("ERROR: prediction_offset must be >= 0", file=sys.stderr)
    sys.exit(1)

if history_length > _MAX_CONTEXT:
    print(
        f"WARNING: history_length={history_length} exceeds model max context "
        f"{_MAX_CONTEXT}; only the most recent {_MAX_CONTEXT} points will be used.",
        file=sys.stderr,
    )

# ── Data ──────────────────────────────────────────────────────────────────────
data_path = INPUT / "input.parquet"
if not data_path.exists():
    print(f"ERROR: {data_path} not found", file=sys.stderr)
    sys.exit(1)

df_full = pd.read_parquet(data_path)
print(f"Loaded {len(df_full)} rows, columns: {list(df_full.columns)}")

if len(df_full.columns) < 2:
    print("ERROR: input.parquet must have at least 2 columns (x + at least one y)", file=sys.stderr)
    sys.exit(1)

if history_length + prediction_offset > len(df_full):
    print(
        f"ERROR: not enough input data — "
        f"history_length ({history_length}) + prediction_offset ({prediction_offset}) "
        f"= {history_length + prediction_offset} rows required, "
        f"but input.parquet only has {len(df_full)} rows. "
        f"Reduce history_length/prediction_offset or provide more input data.",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Prepare ───────────────────────────────────────────────────────────────────
x_col      = df_full.columns[0]
y_cols     = list(df_full.columns[1:])
total_rows = len(df_full)
is_string_x = pd.api.types.is_string_dtype(df_full[x_col]) or pd.api.types.is_object_dtype(df_full[x_col])

end_idx   = total_rows - prediction_offset
start_idx = end_idx - history_length
df = df_full.iloc[start_idx:end_idx].reset_index(drop=True)
x_series = df[x_col]
y_df     = df[y_cols]

print(f"x column: {x_col!r}, y columns: {y_cols}")
print(f"Using rows {start_idx}–{end_idx - 1} ({x_series.iloc[0]} – {x_series.iloc[-1]})")

# ── Predict ───────────────────────────────────────────────────────────────────
try:
    predictions = predict(x_series, y_df, prediction_length)
except ValueError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)

# ── Fix x_auto_converted to absolute positions ────────────────────────────────
if is_string_x and "x_auto_converted" in predictions.columns:
    predictions["x_auto_converted"] += start_idx
    predictions[x_col] = predictions["x_auto_converted"]

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT.mkdir(parents=True, exist_ok=True)

out_path = OUTPUT / "predictions.tsv"
predictions.to_csv(out_path, sep="\t", index=False)
print(f"Written {len(predictions)} rows to {out_path}")

print("Done.")
