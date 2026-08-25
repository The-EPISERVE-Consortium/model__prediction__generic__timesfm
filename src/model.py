import numpy as np
import pandas as pd
import timesfm

_tfm = None

# Max forecast steps (must be a multiple of 128).
_MAX_PREDICTION_STEPS = 512

# Max history points fed to the model. TimesFM 2.5 supports up to 16384; more
# context lets the model capture multi-cycle seasonality (e.g. yearly patterns
# in epidemiological data). History longer than this is truncated to the most
# recent _MAX_CONTEXT points by the model.
_MAX_CONTEXT = 2048


def _validate_ordered_x(x_series: pd.Series, kind: str) -> None:
    """Validate x values that must represent an ordered series.

    Args:
        x_series: Observed x-axis values.
        kind: Human-readable x-axis kind for error messages.

    Raises:
        ValueError: If x values contain missing, duplicate, or non-monotonic
            entries.
    """
    if x_series.isna().any():
        raise ValueError(f"{kind} x values must not contain NaN")
    if not x_series.is_unique:
        raise ValueError(f"{kind} x values must be unique")
    if not x_series.is_monotonic_increasing:
        raise ValueError(f"{kind} x values must be monotonic increasing")


def _validate_uniform_step(diffs: pd.Series, kind: str) -> None:
    """Validate that x-axis diffs have a consistent positive step.

    Args:
        diffs: Consecutive differences between x-axis values.
        kind: Human-readable x-axis kind for error messages.

    Raises:
        ValueError: If diffs are not positive and uniform.
    """
    first_diff = diffs.iloc[0]
    if pd.api.types.is_timedelta64_dtype(diffs):
        if not (diffs == first_diff).all():
            raise ValueError(f"{kind} x values must have a uniform step")
        return

    if not np.allclose(diffs.to_numpy(dtype=np.float64), float(first_diff)):
        raise ValueError(f"{kind} x values must have a uniform step")


def _model():
    """Return the cached TimesFM model instance.

    Returns:
        timesfm.TimesFM: Compiled TimesFM model instance.
    """
    global _tfm
    if _tfm is None:
        tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        tfm.compile(timesfm.ForecastConfig(
            max_context=_MAX_CONTEXT,
            max_horizon=_MAX_PREDICTION_STEPS,
            normalize_inputs=True,
            fix_quantile_crossing=True,
        ))
        _tfm = tfm
    return _tfm


def _extrapolate_x(x_series: pd.Series, n: int) -> list:
    """Return future x values extrapolated from the observed x series.

    Args:
        x_series: Observed x-axis values.
        n: Number of future values to generate.

    Returns:
        Future x-axis values.

    Raises:
        ValueError: If fewer than two x values are provided, x values are
            missing, duplicated, non-monotonic, or datetime/numeric values do not
            have a uniform step.
    """
    if len(x_series) < 2:
        raise ValueError("x_series must contain at least 2 values to infer step size")

    if pd.api.types.is_datetime64_any_dtype(x_series):
        _validate_ordered_x(x_series, "datetime")
        diffs = x_series.diff().dropna()
        _validate_uniform_step(diffs, "datetime")
        step = diffs.iloc[0]
        last = x_series.iloc[-1]
        return [last + step * (i + 1) for i in range(n)]

    if pd.api.types.is_string_dtype(x_series) or pd.api.types.is_object_dtype(x_series):
        _validate_ordered_x(x_series, "string")
        # String x: rows are already ordered by the string column; future positions
        # continue the integer sequence (len+1, len+2, ...).
        base = len(x_series)
        return list(range(base + 1, base + 1 + n))

    # numeric
    _validate_ordered_x(x_series, "numeric")
    diffs = x_series.diff().dropna()
    _validate_uniform_step(diffs, "numeric")
    step = float(diffs.iloc[0])
    last = float(x_series.iloc[-1])
    return [last + step * (i + 1) for i in range(n)]


def predict(x_series: pd.Series, y_df: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    """Forecast future values for each y column.

    Args:
        x_series: Series of x values, such as datetime, numeric, or ISO-week
            strings.
        y_df: DataFrame where each column is an independent time series to
            forecast.
        prediction_length: Number of steps ahead to predict.

    Returns:
        DataFrame with one row per predicted step. Columns include the x column
        name and, for each y column, the point forecast plus q10/q90 intervals.

    Raises:
        ValueError: If prediction_length is outside the supported range, a y
            column contains only missing values, or x values cannot be
            extrapolated.
    """
    if prediction_length <= 0:
        raise ValueError("prediction_length must be > 0")
    if prediction_length > _MAX_PREDICTION_STEPS:
        raise ValueError(
            f"prediction_length={prediction_length} exceeds max {_MAX_PREDICTION_STEPS} steps"
        )

    numeric_y = y_df.astype(np.float64)
    all_nan_cols = numeric_y.columns[numeric_y.isna().all()].tolist()
    if all_nan_cols:
        raise ValueError(
            "y columns must contain at least one non-NaN value: "
            + ", ".join(map(str, all_nan_cols))
        )

    # Missing values are gaps, not zeros — filling with 0.0 injects artificial
    # cliffs that corrupt the model's input normalization. Interpolate interior
    # gaps linearly and carry the nearest value over leading/trailing NaNs.
    inputs = [
        numeric_y[col]
        .interpolate(method="linear", limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy(dtype=np.float64)
        for col in y_df.columns
    ]

    is_string_x = (
        pd.api.types.is_string_dtype(x_series)
        or pd.api.types.is_object_dtype(x_series)
    )
    future_x = _extrapolate_x(x_series, prediction_length)

    point, quantiles = _model().forecast(inputs=inputs, horizon=prediction_length)
    # point:     (n_cols, prediction_length)
    # quantiles: (n_cols, prediction_length, 10)
    # Quantile channels are [mean, q0.1, q0.2, ..., q0.9] — index 0 is the mean
    # (NOT q0.1), index 5 is the median, index 9 is q0.9.
    _Q10, _Q90 = 1, 9

    rows = []
    x_name = x_series.name if x_series.name is not None else "x"
    for i in range(prediction_length):
        row = {x_name: future_x[i]}
        if is_string_x:
            row["x_auto_converted"] = future_x[i]
        for j, col in enumerate(y_df.columns):
            row[col]          = float(point[j, i])
            row[f"{col}_q10"] = float(quantiles[j, i, _Q10])
            row[f"{col}_q90"] = float(quantiles[j, i, _Q90])
        rows.append(row)

    return pd.DataFrame(rows)
