import re
import numpy as np
import pandas as pd
import timesfm

_tfm = None

# Max forecast steps (must be a multiple of 128).
_MAX_PREDICTION_STEPS = 512

_ISO_WEEK_RE = re.compile(r"^\d{4}-W\d{2}$")


def _model():
    global _tfm
    if _tfm is None:
        tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        tfm.compile(timesfm.ForecastConfig(
            max_context=512,
            max_horizon=_MAX_PREDICTION_STEPS,
            normalize_inputs=True,
            fix_quantile_crossing=True,
        ))
        _tfm = tfm
    return _tfm


def _extrapolate_x(x_series: pd.Series, n: int) -> list:
    """Return n future x values extrapolated from x_series."""
    if len(x_series) < 2:
        raise ValueError("x_series must contain at least 2 values to infer step size")

    if pd.api.types.is_datetime64_any_dtype(x_series):
        step = x_series.diff().dropna().median()
        last = x_series.iloc[-1]
        return [last + step * (i + 1) for i in range(n)]

    if pd.api.types.is_string_dtype(x_series) or pd.api.types.is_object_dtype(x_series):
        sample = str(x_series.iloc[0])
        if _ISO_WEEK_RE.match(sample):
            # e.g. "2025-W45" — parse as ISO week (Monday of that week) then step weekly
            parsed = pd.to_datetime(x_series + "-1", format="%G-W%V-%u")
            step = parsed.diff().dropna().median()
            last = parsed.iloc[-1]
            future = [last + step * (i + 1) for i in range(n)]
            return [f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}" for dt in future]
        try:
            parsed = pd.to_datetime(x_series)
            step = parsed.diff().dropna().median()
            last = parsed.iloc[-1]
            return [last + step * (i + 1) for i in range(n)]
        except Exception:
            pass
        # non-parseable strings: use integer positions as fallback
        return list(range(len(x_series), len(x_series) + n))

    # numeric
    step = float(x_series.diff().dropna().median())
    last = float(x_series.iloc[-1])
    return [last + step * (i + 1) for i in range(n)]


def predict(x_series: pd.Series, y_df: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    """
    x_series:          pd.Series of x values (datetime, numeric, or ISO-week string).
    y_df:              pd.DataFrame where each column is an independent time series to forecast.
    prediction_length: number of steps ahead to predict.

    Returns: DataFrame with one row per predicted step.
             Columns: x column name + for each y column col: col, col_q10, col_q90.
    """
    if prediction_length > _MAX_PREDICTION_STEPS:
        raise ValueError(
            f"prediction_length={prediction_length} exceeds max {_MAX_PREDICTION_STEPS} steps"
        )

    inputs = [
        y_df[col].fillna(0.0).to_numpy(dtype=np.float64)
        for col in y_df.columns
    ]

    point, quantiles = _model().forecast(inputs=inputs, horizon=prediction_length)
    # point:     (n_cols, prediction_length)
    # quantiles: (n_cols, prediction_length, 10)

    future_x = _extrapolate_x(x_series, prediction_length)

    rows = []
    for i in range(prediction_length):
        row = {x_series.name: future_x[i]}
        for j, col in enumerate(y_df.columns):
            row[col]          = float(point[j, i])
            row[f"{col}_q10"] = float(quantiles[j, i, 0])
            row[f"{col}_q90"] = float(quantiles[j, i, -1])
        rows.append(row)

    return pd.DataFrame(rows)
