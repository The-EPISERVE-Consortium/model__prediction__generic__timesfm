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
# produce a forecast; shorter windows would otherwise reach forecast() and
# fail with an uncaught non-ValueError. fdo.json declares this as
# history_length's minValue (kept in sync by tests/test_config_schema.py), so
# run.py rejects anything lower up front; predict() also guards it directly
# for library callers.
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
        # Diffs are in nanoseconds (a day is ~8.64e13 ns), so np.allclose's
        # default rtol=1e-5 would silently tolerate ~0.86 s of irregularity on
        # a daily series (minutes on an annual one). Use an absolute 1 ns
        # tolerance so the check is magnitude-independent: steps must be
        # uniform to the nanosecond.
        diff_ns = diffs.map(lambda value: pd.Timedelta(value).value).to_numpy(dtype=np.float64)
        first_diff_ns = float(pd.Timedelta(first_diff).value)
        if not np.allclose(diff_ns, first_diff_ns, rtol=0, atol=1):
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


def _validate_numeric_y(y_series: pd.Series) -> None:
    """Validate that the y series has a numeric dtype.

    Args:
        y_series: The single time series to forecast.

    Raises:
        ValueError: If the y series is not numeric.
    """
    if not pd.api.types.is_numeric_dtype(y_series):
        name = y_series.name if y_series.name is not None else "y"
        raise ValueError(f"y column '{name}' must be numeric")


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


def _validate_interpolation_gaps(y_series: pd.Series) -> None:
    """Validate that the y series does not contain a long missing run.

    Args:
        y_series: Numeric y series to inspect.

    Raises:
        ValueError: If the y series has a consecutive interior NaN run longer
            than the supported interpolation limit.
    """
    longest_run = _longest_interior_nan_run(y_series)
    if longest_run > _MAX_INTERPOLATION_GAP:
        name = y_series.name if y_series.name is not None else "y"
        raise ValueError(
            f"y column '{name}' has a NaN gap longer than "
            f"{_MAX_INTERPOLATION_GAP} consecutive values ({longest_run})"
        )


def _validate_forecast_shapes(point, quantiles, prediction_length: int) -> None:
    """Validate model forecast array shapes before reading fixed channels.

    The model is fed a single series, so TimesFM returns a batch of one.

    Args:
        point: Point forecast array returned by TimesFM.
        quantiles: Quantile forecast array returned by TimesFM.
        prediction_length: Expected forecast horizon.

    Raises:
        ValueError: If forecast outputs do not have the expected dimensions.
    """
    if point.shape != (1, prediction_length):
        raise ValueError(
            "TimesFM point forecast has unexpected shape: "
            f"{point.shape}; expected {(1, prediction_length)}"
        )
    if quantiles.ndim != 3 or quantiles.shape[:2] != (1, prediction_length):
        raise ValueError(
            "TimesFM quantile forecast has unexpected shape: "
            f"{quantiles.shape}; expected (1, {prediction_length}, channels)"
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


def predict(x_series: pd.Series, y_series: pd.Series, prediction_length: int) -> pd.DataFrame:
    """Forecast future values of a single y series.

    Args:
        x_series: Series of x values, such as datetime, numeric, timedelta, or
            ISO-week strings.
        y_series: The single time series to forecast.
        prediction_length: Number of steps ahead to predict.

    Returns:
        DataFrame with one row per predicted step: the x column, then the y
        point forecast plus its q10/q90 interval.

    Raises:
        ValueError: If prediction_length is outside the supported range, the y
            series contains invalid missing values, or x values cannot be
            extrapolated.
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

    x_series = _coerce_datetime_like_x(x_series)
    _validate_numeric_y(y_series)
    numeric_y = y_series.astype(np.float64)
    y_name = y_series.name if y_series.name is not None else "y"
    if numeric_y.isna().all():
        raise ValueError(f"y column '{y_name}' must contain at least one non-NaN value")
    _validate_interpolation_gaps(numeric_y)

    # Missing values are gaps, not zeros — filling with 0.0 injects artificial
    # cliffs that corrupt the model's input normalization. Interpolate interior
    # gaps linearly and carry the nearest value over leading/trailing NaNs.
    series = (
        numeric_y
        .interpolate(method="linear", limit=_MAX_INTERPOLATION_GAP, limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy(dtype=np.float64)
    )

    is_string_x = (
        pd.api.types.is_string_dtype(x_series)
        or pd.api.types.is_object_dtype(x_series)
    )
    future_x = _extrapolate_x(x_series, prediction_length)

    point, quantiles = _model().forecast(inputs=[series], horizon=prediction_length)
    _validate_forecast_shapes(point, quantiles, prediction_length)
    # point:     (1, prediction_length)
    # quantiles: (1, prediction_length, 10)
    # Quantile channels are [mean, q0.1, q0.2, ..., q0.9] — index 0 is the mean
    # (NOT q0.1), index 5 is the median, index 9 is q0.9.
    _Q10, _Q90 = 1, 9

    rows = []
    x_name = x_series.name if x_series.name is not None else "x"
    for i in range(prediction_length):
        row = {x_name: future_x[i]}
        if is_string_x:
            row["x_auto_converted"] = future_x[i]
        row[y_name]          = float(point[0, i])
        row[f"{y_name}_q10"] = float(quantiles[0, i, _Q10])
        row[f"{y_name}_q90"] = float(quantiles[0, i, _Q90])
        rows.append(row)

    return pd.DataFrame(rows)
