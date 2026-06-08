import json
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import predict, _MAX_PREDICTION_STEPS

_work  = Path("./work") if Path("./work").exists() else Path("/work")
INPUT  = _work / "input"
OUTPUT = _work / "output"

# ── Config ────────────────────────────────────────────────────────────────────
config_path = INPUT / "config.json"
if not config_path.exists():
    print("ERROR: /work/input/config.json not found", file=sys.stderr)
    sys.exit(1)

config = json.loads(config_path.read_text())

for key in ("history_length", "prediction_length"):
    if key not in config:
        print(f"ERROR: config.json missing required key: '{key}'", file=sys.stderr)
        sys.exit(1)

history_length    = int(config["history_length"])
prediction_length = int(config["prediction_length"])
prediction_offset = int(config.get("prediction_offset", 0))
print(f"Config: history_length={history_length}, prediction_length={prediction_length}, prediction_offset={prediction_offset}")

if prediction_length > _MAX_PREDICTION_STEPS:
    print(
        f"ERROR: prediction_length={prediction_length} exceeds max {_MAX_PREDICTION_STEPS} steps",
        file=sys.stderr,
    )
    sys.exit(1)

if prediction_offset < 0:
    print("ERROR: prediction_offset must be >= 0", file=sys.stderr)
    sys.exit(1)

# ── Data ──────────────────────────────────────────────────────────────────────
data_path = INPUT / "input.parquet"
if not data_path.exists():
    print("ERROR: /work/input/input.parquet not found", file=sys.stderr)
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
predictions = predict(x_series, y_df, prediction_length)

# ── Fix x_auto_converted to absolute positions ────────────────────────────────
if is_string_x and "x_auto_converted" in predictions.columns:
    predictions["x_auto_converted"] += start_idx
    predictions[x_col] = predictions["x_auto_converted"]

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT.mkdir(parents=True, exist_ok=True)

out_path = OUTPUT / "predictions.tsv"
predictions.to_csv(out_path, sep="\t", index=False)
print(f"Written {len(predictions)} rows to {out_path}")

if is_string_x:
    df_annotated = df_full.copy()
    df_annotated.insert(1, "x_auto_converted", range(1, total_rows + 1))
    ann_path = OUTPUT / "input_annotated.parquet"
    df_annotated.to_parquet(ann_path, index=False)
    print(f"Written {len(df_annotated)} rows to {ann_path}")

print("Done.")
