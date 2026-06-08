# CLAUDE.md

## What this is

A Docker-based forecast container for the **EPISERVE** platform. Uses **Google TimesFM 2.5** (200M parameter time-series foundation model) for zero-shot forecasting — no training step required.

Accepts any parquet with a leading x column (datetime or numeric) and one or more y columns. Each y column is forecast independently for `prediction_length` steps ahead, with 10th/90th percentile uncertainty intervals.

Repository follows the naming convention `model__<type>__<dataset>__<variant>`. Published image: `ghcr.io/the-episerve-consortium/model__prediction__generic__timesfm`.

## I/O Contract

| Path | Direction | Description |
|---|---|---|
| `/work/input/config.json` | → into container | `history_length` (int, required) and `prediction_length` (int, required, max 512) |
| `/work/input/input.parquet` | → into container | First column = x axis; all other columns = independent y series (arbitrary names) |
| `/work/output/predictions.tsv` | ← out of container | Forecast with uncertainty intervals |

**Input parquet rules:**
- First column: x values (datetime or numeric), used to infer step size and extrapolate future x values
- Remaining columns: y series to forecast (each treated independently)
- Must have at least 2 columns and `history_length` rows

**Output columns:** x column name (preserved) + for each y column `col`: `col`, `col_q10`, `col_q90`

**Config:**
- `history_length`: how many of the most recent rows to use as model context (required; error if > available rows)
- `prediction_length`: number of steps ahead to forecast (required; error if > 512)

## Commands

```bash
# Install (PyTorch CPU-only to keep env lightweight)
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Run tests (no model download — TimesFM is mocked)
pytest tests/ -v

# Build Docker image
docker build -t episerve/generic-timesfm:dev .

# Test with Docker (requires work/input/ to be populated)
docker run --rm \
  -v $(pwd)/work/input:/work/input \
  -v $(pwd)/work/output:/work/output \
  episerve/generic-timesfm:dev
```

## Architecture

| File | Role |
|---|---|
| `src/model.py` | `predict(x_series, y_df, prediction_length)` — loads TimesFM lazily, runs batched inference over all y columns |
| `src/run.py` | Entrypoint: validates config, reads parquet, slices history, calls predict, writes TSV |
| `tests/test_model.py` | Unit tests — TimesFM mocked, no network calls |

## TimesFM notes

- Model: `google/timesfm-2.5-200m-pytorch` (~800 MB, downloaded from HuggingFace on first run)
- Zero-shot: no fine-tuning, no training data required
- CPU-only inference; all y columns are batched into a single `forecast()` call
- `max_horizon` compiled at 512 steps; `prediction_length` can be anything ≤ 512
- Quantile output: index 0 = lowest level (~q10), index -1 = highest level (~q90)
- x step size is inferred as the median diff of the input x series; future x values are extrapolated linearly

## Release

```bash
git tag v0.1.0
git push origin v0.1.0
```
