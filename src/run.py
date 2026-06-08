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
print(f"Config: history_length={history_length}, prediction_length={prediction_length}")

if prediction_length > _MAX_PREDICTION_STEPS:
    print(
        f"ERROR: prediction_length={prediction_length} exceeds max {_MAX_PREDICTION_STEPS} steps",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Data ──────────────────────────────────────────────────────────────────────
data_path = INPUT / "input.parquet"
if not data_path.exists():
    print("ERROR: /work/input/input.parquet not found", file=sys.stderr)
    sys.exit(1)

df = pd.read_parquet(data_path)
print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

if len(df.columns) < 2:
    print("ERROR: input.parquet must have at least 2 columns (x + at least one y)", file=sys.stderr)
    sys.exit(1)

if history_length > len(df):
    print(
        f"ERROR: history_length={history_length} exceeds available rows ({len(df)})",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Prepare ───────────────────────────────────────────────────────────────────
x_col  = df.columns[0]
y_cols = list(df.columns[1:])

df = df.tail(history_length).reset_index(drop=True)
x_series = df[x_col]
y_df     = df[y_cols]

print(f"x column: {x_col!r}, y columns: {y_cols}")
print(f"Using {len(df)} rows ({x_series.iloc[0]} – {x_series.iloc[-1]})")

# ── Predict ───────────────────────────────────────────────────────────────────
predictions = predict(x_series, y_df, prediction_length)

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT.mkdir(parents=True, exist_ok=True)
out_path = OUTPUT / "predictions.tsv"
predictions.to_csv(out_path, sep="\t", index=False)
print(f"Written {len(predictions)} rows to {out_path}")
print("Done.")
