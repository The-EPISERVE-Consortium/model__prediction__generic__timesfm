import numpy as np
import pandas as pd
import timesfm

_tfm = None

# Max forecast steps (must be a multiple of 128).
_MAX_PREDICTION_STEPS = 512


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


def _infer_step(x_series: pd.Series):
    """Infer median step size from an ordered x series (datetime or numeric)."""
    diffs = x_series.diff().dropna()
    if diffs.empty:
        raise ValueError("x_series must contain at least 2 values to infer step size")
    if pd.api.types.is_datetime64_any_dtype(x_series):
        return diffs.median()
    return float(diffs.median())


def predict(x_series: pd.Series, y_df: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    """
    x_series:          pd.Series of x values (datetime or numeric); determines future x axis.
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

    step = _infer_step(x_series)
    last_x = x_series.iloc[-1]
    future_x = [last_x + step * (i + 1) for i in range(prediction_length)]

    rows = []
    for i in range(prediction_length):
        row = {x_series.name: future_x[i]}
        for j, col in enumerate(y_df.columns):
            row[col]          = float(point[j, i])
            row[f"{col}_q10"] = float(quantiles[j, i, 0])
            row[f"{col}_q90"] = float(quantiles[j, i, -1])
        rows.append(row)

    return pd.DataFrame(rows)
