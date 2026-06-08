# model__prediction__generic__timesfm

Zero-shot time-series forecasting using [Google TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (200M parameter foundation model). No training or fine-tuning required.

Accepts any parquet file with a leading x column and one or more y columns. Each y column is forecast independently with 10th/90th percentile uncertainty intervals.

## Input

| Path | Description |
|---|---|
| `/work/input/input.parquet` | Time-series data. First column = x axis; all remaining columns = y series to forecast. |
| `/work/input/config.json` | Run parameters (see below). |

### Input parquet rules

- First column: x values. Supported types:
  - **datetime** — step size is inferred from the median diff; future x values are extrapolated accordingly
  - **numeric** — same as datetime
  - **string** — rows are taken as-is (no sorting, no date parsing); a synthetic integer position column `x_auto_converted` is added to the output for alignment
- Remaining columns: one y series each, treated independently
- Must have at least 2 columns and enough rows to satisfy `history_length + prediction_offset`

## Output

| Path | Description |
|---|---|
| `/work/output/predictions.tsv` | Forecast results (tab-separated). |
| `/work/output/input_annotated.parquet` | All input rows plus an `x_auto_converted` column (1…N). Only written when the x column is a string type. |

### Prediction columns

For each y column `col` in the input:

| Column | Description |
|---|---|
| `<x_col>` | Extrapolated x value for this forecast step |
| `x_auto_converted` | Integer position (only present when x column is a string type) |
| `col` | Point forecast |
| `col_q10` | 10th percentile |
| `col_q90` | 90th percentile |

## Config parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `history_length` | yes | — | Number of rows to use as model context. Taken as the window ending at `total_rows - prediction_offset`. |
| `prediction_length` | yes | — | Number of steps to forecast ahead. Maximum: 512. |
| `prediction_offset` | no | `0` | Rows to skip at the end of the input before the history window. Use this to predict over already-known data for back-testing. |

### prediction_offset example

With 100 rows in the input and the config below, the model uses rows 31–70 as context and predicts steps 71–100 — which overlap with the real data, enabling direct comparison:

```json
{
  "history_length": 40,
  "prediction_length": 30,
  "prediction_offset": 30
}
```

Without `prediction_offset` (or with `prediction_offset: 0`) the model uses the last `history_length` rows and predicts beyond the end of the available data.

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Tests (TimesFM is mocked — no model download needed)
pytest tests/ -v
```

## Running with Docker

```bash
docker build -t episerve/generic-timesfm:dev .

docker run --rm \
  -v $(pwd)/work/input:/work/input \
  -v $(pwd)/work/output:/work/output \
  episerve/generic-timesfm:dev
```

The model weights (~800 MB) are downloaded from HuggingFace on first run.

## Release

```bash
git tag v0.1.0
git push origin v0.1.0
```

The GitHub Actions workflow builds the Docker image, runs the test suite, and pushes to `ghcr.io/the-episerve-consortium/model__prediction__generic__timesfm`.
