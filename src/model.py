from datetime import date

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

# Min history points fed to the model. TimesFM requires a minimum context to
# produce a forecast; shorter windows (which run.py's `history_length >= 2`
# check permits) would otherwise reach forecast() and fail with an uncaught
# non-ValueError. Reject them up front with a descriptive error.
_MIN_CONTEXT = 3

# Max consecutive missing y values to bridge before forecasting. Longer gaps are
# treated as too sparse to fabricate safely.
_MAX_INTERPOLATION_GAP = 8


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
        diff_ns = diffs.map(lambda value: pd.Timedelta(value).value).to_numpy(dtype=np.float64)
        first_diff_ns = float(pd.Timedelta(first_diff).value)
        if not np.allclose(diff_ns, first_diff_ns):
            raise ValueError(f"{kind} x values must have a uniform step")
        return

    if not np.allclose(diffs.to_numpy(dtype=np.float64), float(first_diff)):
        raise ValueError(f"{kind} x values must have a uniform step")


def _coerce_datetime_like_x(x_series: pd.Series) -> pd.Series:
    """Convert object x values to datetime when they are datetime-like objects.

    Args:
        x_series: Observed x-axis values.

    Returns:
        Original x series, or a datetime-converted copy when object values are
        already datetime-like.
    """
    if pd.api.types.is_datetime64_any_dtype(x_series):
        return x_series
    if not pd.api.types.is_object_dtype(x_series):
        return x_series

    values = x_series.dropna()
    if values.empty:
        return x_series

    datetime_types = (date, np.datetime64, pd.Timestamp)
    if values.map(lambda value: isinstance(value, datetime_types)).all():
        # pd.to_datetime on a Series already returns a Series preserving the
        # name and index; wrapping it again in pd.Series would discard the
        # caller-provided index for a fresh RangeIndex.
        return pd.to_datetime(x_series)

    return x_series


def _datetime_step(x_series: pd.Series):
    """Infer the datetime step from a validated datetime x series.

    Args:
        x_series: Datetime x-axis values.

    Returns:
        Date offset or timedelta used to extrapolate future timestamps.

    Raises:
        ValueError: If the datetime values do not have an inferable calendar
            frequency or uniform timedelta step.
    """
    freq = pd.infer_freq(pd.DatetimeIndex(x_series))
    if freq is not None:
        return pd.tseries.frequencies.to_offset(freq)

    diffs = x_series.diff().dropna()
    _validate_uniform_step(diffs, "datetime")
    return diffs.iloc[0]


def _validate_numeric_y(y_df: pd.DataFrame) -> None:
    """Validate that all y columns have numeric dtypes.

    Args:
        y_df: DataFrame of y columns to forecast.

    Raises:
        ValueError: If any y column name is duplicated, or if any y column is
            not numeric.
    """
    duplicate_cols = y_df.columns[y_df.columns.duplicated()].unique().tolist()
    if duplicate_cols:
        raise ValueError(
            "y columns must have unique names, found duplicates: "
            + ", ".join(map(str, duplicate_cols))
        )
    non_numeric_cols = [
        col for col in y_df.columns
        if not pd.api.types.is_numeric_dtype(y_df[col])
    ]
    if non_numeric_cols:
        raise ValueError(
            "y columns must be numeric: " + ", ".join(map(str, non_numeric_cols))
        )


def _longest_interior_nan_run(series: pd.Series) -> int:
    """Return the longest consecutive NaN run between the first and last valid
    values of a series (an interior gap).

    Leading and trailing NaN runs are excluded because `predict()` carries the
    nearest value over them with `.bfill()/.ffill()` -- any leading/trailing
    run is recoverable regardless of length. Only gaps bounded by valid values
    on both sides are limited by the interpolation step.

    Args:
        series: Series to inspect for missing values.

    Returns:
        Length of the longest interior run of missing values.
    """
    values = series.to_numpy()
    valid_positions = np.flatnonzero(~pd.isna(values))
    if len(valid_positions) < 2:
        return 0
    first_valid, last_valid = int(valid_positions[0]), int(valid_positions[-1])
    longest = 0
    current = 0
    for i in range(first_valid + 1, last_valid):
        if pd.isna(values[i]):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _validate_interpolation_gaps(y_df: pd.DataFrame) -> None:
    """Validate that y columns do not contain long missing runs.

    Args:
        y_df: Numeric y DataFrame to inspect.

    Raises:
        ValueError: If any y column has a consecutive NaN run longer than the
            supported interpolation limit.
    """
    long_gap_cols = []
    for col in y_df.columns:
        longest_run = _longest_interior_nan_run(y_df[col])
        if longest_run > _MAX_INTERPOLATION_GAP:
            long_gap_cols.append(f"{col} ({longest_run})")
    if long_gap_cols:
        raise ValueError(
            "y columns contain NaN gaps longer than "
            f"{_MAX_INTERPOLATION_GAP} consecutive values: "
            + ", ".join(long_gap_cols)
        )


def _validate_forecast_shapes(point, quantiles, prediction_length: int, n_cols: int) -> None:
    """Validate model forecast array shapes before reading fixed channels.

    Args:
        point: Point forecast array returned by TimesFM.
        quantiles: Quantile forecast array returned by TimesFM.
        prediction_length: Expected forecast horizon.
        n_cols: Expected number of independent y series.

    Raises:
        ValueError: If forecast outputs do not have the expected dimensions.
    """
    if point.shape != (n_cols, prediction_length):
        raise ValueError(
            "TimesFM point forecast has unexpected shape: "
            f"{point.shape}; expected {(n_cols, prediction_length)}"
        )
    if quantiles.ndim != 3 or quantiles.shape[:2] != (n_cols, prediction_length):
        raise ValueError(
            "TimesFM quantile forecast has unexpected shape: "
            f"{quantiles.shape}; expected ({n_cols}, {prediction_length}, channels)"
        )
    if quantiles.shape[2] != 10:
        raise ValueError(
            "TimesFM quantile forecast must have 10 channels ordered "
            "[mean, q0.1, ..., q0.9]"
        )


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
            missing, duplicated, non-monotonic, or datetime/numeric/timedelta
            values do not have a uniform step.
    """
    if len(x_series) < 2:
        raise ValueError("x_series must contain at least 2 values to infer step size")

    x_series = _coerce_datetime_like_x(x_series)

    if pd.api.types.is_datetime64_any_dtype(x_series):
        _validate_ordered_x(x_series, "datetime")
        step = _datetime_step(x_series)
        last = x_series.iloc[-1]
        return [last + step * (i + 1) for i in range(n)]

    if pd.api.types.is_timedelta64_dtype(x_series):
        _validate_ordered_x(x_series, "timedelta")
        diffs = x_series.diff().dropna()
        _validate_uniform_step(diffs, "timedelta")
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
        x_series: Series of x values, such as datetime, numeric, timedelta, or
            ISO-week strings.
        y_df: DataFrame where each column is an independent time series to
            forecast.
        prediction_length: Number of steps ahead to predict.

    Returns:
        DataFrame with one row per predicted step. Columns include the x column
        name and, for each y column, the point forecast plus q10/q90 intervals.

    Raises:
        ValueError: If prediction_length is outside the supported range, if x
            and y have different lengths, if y column names are duplicated, if
            a y column contains invalid missing values, or if x values cannot
            be extrapolated.
    """
    if prediction_length <= 0:
        raise ValueError("prediction_length must be > 0")
    if prediction_length > _MAX_PREDICTION_STEPS:
        raise ValueError(
            f"prediction_length={prediction_length} exceeds max {_MAX_PREDICTION_STEPS} steps"
        )
    if len(x_series) < _MIN_CONTEXT:
        raise ValueError(
            f"x_series must contain at least {_MIN_CONTEXT} points "
            f"(got {len(x_series)}) to forecast"
        )
    if len(x_series) != len(y_df):
        raise ValueError(
            "x_series and y_df must have the same number of rows: "
            f"x_series has {len(x_series)} rows but y_df has {len(y_df)} rows"
        )

    x_series = _coerce_datetime_like_x(x_series)
    _validate_numeric_y(y_df)
    numeric_y = y_df.astype(np.float64)
    all_nan_cols = numeric_y.columns[numeric_y.isna().all()].tolist()
    if all_nan_cols:
        raise ValueError(
            "y columns must contain at least one non-NaN value: "
            + ", ".join(map(str, all_nan_cols))
        )
    _validate_interpolation_gaps(numeric_y)

    # Missing values are gaps, not zeros — filling with 0.0 injects artificial
    # cliffs that corrupt the model's input normalization. Interpolate interior
    # gaps linearly and carry the nearest value over leading/trailing NaNs.
    inputs = [
        numeric_y[col]
        .interpolate(
            method="linear",
            limit=_MAX_INTERPOLATION_GAP,
            limit_direction="both",
        )
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
    _validate_forecast_shapes(point, quantiles, prediction_length, len(y_df.columns))
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
