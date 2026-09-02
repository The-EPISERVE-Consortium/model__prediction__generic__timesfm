# model__prediction__generic__timesfm

Zero-shot time-series forecasting with [Google TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
(200M-param foundation model) — no training or fine-tuning. Takes a parquet with
exactly two columns — x (first) and y (second) — and forecasts y with
10th/90th-percentile bounds.

**In:** `input.parquet` + `config.json` under `/work/input/`.
**Out:** `predictions.tsv` under `/work/output/`.

## Input

| Path | Description |
|---|---|
| `/work/input/input.parquet` | Exactly two columns: x (first) and y (second, the series to forecast). |
| `/work/input/config.json` | Run parameters — see [Config](#config). |

**x column**, by dtype:

- **datetime / numeric / timedelta** — unique, monotonic, evenly spaced; future x is extrapolated from that step.
- **string** — unique and monotonic, taken as-is (no sorting or date parsing). The output x column becomes integer positions and an `x_auto_converted` column is added: **1-indexed absolute row positions** in the input, so forecast step *i* is input row `end_of_history + i` and lines up with the source rows even with `prediction_offset` or a truncated history.

**y column** — the series to forecast. Interior NaN gaps of ≤ 8 rows are
interpolated; longer ones are rejected. Leading/trailing NaN runs are filled from
the nearest value. Needs ≥ `history_length + prediction_offset` rows.

## Output

`/work/output/predictions.tsv` — one row per forecast step (`<y>` is the y column's name):

| Column | Description |
|---|---|
| `<x_col>` | Extrapolated x value (synthetic integer position for string x). |
| `x_auto_converted` | 1-indexed absolute input-row position (string x only). |
| `<y>` / `<y>_q10` / `<y>_q90` | Point forecast / 10th / 90th percentile. |

## Config

Parameters are declared in [`fdo.json`](fdo.json) (`additionalProperty`) — the
source of truth for names, whether each is required, defaults, and
`minValue`/`maxValue`. `src/run.py` validates `config.json` against it at startup
(unknown keys warn and are ignored); cross-field rules it can't express (e.g.
`history_length + prediction_offset` ≤ input rows) are checked once the data loads.

`prediction_offset` skips rows at the end before the history window — set it to
`prediction_length` to forecast over already-known data for back-testing.

**Validate a config without running the model:**

```bash
docker run --rm -v $(pwd)/work/input:/work/input <image> --check-config
```

Checks `config.json` against `fdo.json` and exits `0` (`config.json OK`) or `1`
(`ERROR:` lines); reads no data, loads no weights. Locally:
`WORK_DIR=./work python src/run.py --check-config`.

## Develop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest tests/ -v          # TimesFM is mocked — no model download

docker build -t episerve/generic-timesfm:dev .
docker run --rm -v $(pwd)/work/input:/work/input -v $(pwd)/work/output:/work/output \
  episerve/generic-timesfm:dev
```

Model weights (~800 MB) download from HuggingFace on first real run.

## Release

Push to `main` → CI builds, tests inside the image, and pushes `:latest` to GHCR.
`git tag v0.1.0 && git push origin v0.1.0` also publishes `:v0.1.0`.
