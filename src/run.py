import json
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config_schema import validate_config, SchemaError
from model import predict, _MAX_CONTEXT

# `--check-config`: validate config.json against fdo.json and exit, without
# reading input data or loading the TimesFM weights.
CHECK_ONLY = "--check-config" in sys.argv[1:]


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

# Validate config.json against the parameter declaration in fdo.json:
# required keys, integer typing, and any minValue/maxValue bounds. Defaults
# for omitted optional keys come from fdo.json too.
try:
    resolved, config_errors, config_warnings = validate_config(config)
except SchemaError as exc:
    print(f"ERROR: model config schema (fdo.json) is invalid: {exc}", file=sys.stderr)
    sys.exit(1)

for warning in config_warnings:
    print(f"WARNING: {warning}", file=sys.stderr)

if config_errors:
    for error in config_errors:
        print(f"ERROR: {error}", file=sys.stderr)
    sys.exit(1)

history_length = resolved["history_length"]
prediction_length = resolved["prediction_length"]
prediction_offset = resolved["prediction_offset"]
print(f"Config: history_length={history_length}, prediction_length={prediction_length}, prediction_offset={prediction_offset}")

if history_length > _MAX_CONTEXT:
    print(
        f"WARNING: history_length={history_length} exceeds model max context "
        f"{_MAX_CONTEXT}; only the most recent {_MAX_CONTEXT} points will be used.",
        file=sys.stderr,
    )

if CHECK_ONLY:
    print("config.json OK")
    sys.exit(0)

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
if history_length > _MAX_CONTEXT:
    # The model consumes at most the most recent _MAX_CONTEXT points (see the
    # WARNING above). Drop the older prefix here so NaN-gap validation and
    # interpolation inside predict() operate on exactly the window the model
    # receives, instead of silently relying on TimesFM to truncate internally.
    start_idx = end_idx - _MAX_CONTEXT
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
except Exception as exc:
    # The model layer (TimesFM/torch) can raise non-ValueError exceptions such
    # as OSError on a failed first-run weight download or RuntimeError on an
    # OOM/context error. Keep the same clean-error convention as every other
    # failure path instead of leaking a raw traceback.
    print(f"ERROR: model failed: {exc}", file=sys.stderr)
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
