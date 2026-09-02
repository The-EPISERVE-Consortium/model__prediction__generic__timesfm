import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

if "timesfm" not in sys.modules:
    timesfm_stub = types.ModuleType("timesfm")
    timesfm_stub.ForecastConfig = MagicMock()
    timesfm_stub.TimesFM_2p5_200M_torch = MagicMock()
    sys.modules["timesfm"] = timesfm_stub

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import model as model_module
from model import predict, _MAX_PREDICTION_STEPS


def _make_xy(n=100, y_name="cases", freq="D"):
    """Create a sample (x_series, y_series) pair.

    Args:
        n: Number of rows.
        y_name: Name of the y series.
        freq: Pandas frequency string for the datetime x column.

    Returns:
        Tuple of the x series (named "date") and the y series.
    """
    x_series = pd.Series(pd.date_range("2022-01-01", periods=n, freq=freq), name="date")
    y_series = pd.Series(np.random.rand(n) * 50 + 10, name=y_name)
    return x_series, y_series


def _mock_forecast(inputs, horizon):
    """Return deterministic (point, quantile) forecast arrays for a batch of one.

    Args:
        inputs: Model input arrays (length 1 -- the model forecasts a single series).
        horizon: Number of forecast steps.

    Returns:
        Tuple of point forecasts and quantile forecasts.
    """
    batch = len(inputs)
    point = np.full((batch, horizon), 45.0)
    # Channels are [mean, q0.1, q0.2, ..., q0.9]: index 0 is the mean, index 1
    # is q0.1, index 9 is q0.9.
    quantiles = np.zeros((batch, horizon, 10))
    quantiles[..., 0] = 45.0   # mean
    quantiles[..., 1] = 30.0   # q0.1
    quantiles[..., -1] = 60.0  # q0.9
    return point, quantiles


@pytest.fixture(autouse=True)
def mock_timesfm():
    """Mock the TimesFM torch backend used by the model.

    Yields:
        MagicMock: Mocked TimesFM torch backend class.
    """
    mock_instance = MagicMock()
    mock_instance.forecast.side_effect = _mock_forecast
    model_module._tfm = None
    with patch("timesfm.TimesFM_2p5_200M_torch", create=True) as mock_cls:
        mock_cls.from_pretrained.return_value = mock_instance
        yield mock_cls
    model_module._tfm = None


def test_output_row_count():
    x, y = _make_xy()
    assert len(predict(x, y, prediction_length=7)) == 7


def test_output_columns():
    x, y = _make_xy(y_name="value")
    result = predict(x, y, prediction_length=3)
    assert set(result.columns) == {"date", "value", "value_q10", "value_q90"}


def test_output_column_order():
    x, y = _make_xy(y_name="value")
    result = predict(x, y, prediction_length=3)
    assert list(result.columns) == ["date", "value", "value_q10", "value_q90"]


def test_x_extrapolation_daily():
    x, y = _make_xy(n=50, freq="D")
    result = predict(x, y, prediction_length=3)
    last = x.iloc[-1]
    for i, row in enumerate(result.itertuples(), 1):
        assert row.date == last + pd.Timedelta(days=i)


def test_x_extrapolation_weekly():
    x, y = _make_xy(n=50, freq="W")
    result = predict(x, y, prediction_length=3)
    last = x.iloc[-1]
    for i, row in enumerate(result.itertuples(), 1):
        assert row.date == last + pd.Timedelta(weeks=i)


def test_x_extrapolation_daily_across_dst():
    """Use calendar daily frequency across DST transitions."""
    x = pd.Series(
        pd.date_range("2022-03-26", periods=4, freq="D", tz="Europe/Berlin"),
        name="date",
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0], name="cases")
    result = predict(x, y, prediction_length=2)
    expected = pd.date_range("2022-03-30", periods=2, freq="D", tz="Europe/Berlin")
    assert list(result["date"]) == list(expected)


def test_x_extrapolation_numeric():
    n = 50
    x = pd.Series(np.arange(n, dtype=float), name="x")
    y = pd.Series(np.random.rand(n), name="y")
    result = predict(x, y, prediction_length=3)
    for i, row in enumerate(result.itertuples(), 1):
        assert row.x == pytest.approx(n - 1 + i)


def test_x_extrapolation_timedelta_microseconds():
    """Extrapolate uniform non-ns timedelta x values."""
    x = pd.Series(np.array([0, 1000, 2000], dtype="timedelta64[us]"), name="delta")
    y = pd.Series([1.0, 2.0, 3.0], name="cases")
    result = predict(x, y, prediction_length=2)
    expected = [pd.Timedelta(microseconds=3000), pd.Timedelta(microseconds=4000)]
    assert list(result["delta"]) == expected


def test_q10_reads_q01_channel_not_mean():
    # q10 must come from channel 1 (q0.1=30.0), not channel 0 (mean=45.0).
    x, y = _make_xy(y_name="cases")
    result = predict(x, y, prediction_length=5)
    assert (result["cases_q10"] == 30.0).all()
    assert (result["cases_q90"] == 60.0).all()


def test_quantile_ordering():
    x, y = _make_xy(y_name="cases")
    result = predict(x, y, prediction_length=5)
    assert (result["cases_q10"] <= result["cases_q90"]).all()


def test_x_string_column_outputs_integer_positions():
    weeks = [f"2022-W{i + 1:02d}" for i in range(50)]
    x = pd.Series(weeks, name="Meldewoche")
    y = pd.Series(np.random.rand(50) * 100, name="cases")
    result = predict(x, y, prediction_length=3)
    assert result.columns[0] == "Meldewoche"
    assert list(result["Meldewoche"]) == [51, 52, 53]
    assert "x_auto_converted" in result.columns
    assert list(result["x_auto_converted"]) == [51, 52, 53]


def test_x_non_string_column_has_no_auto_converted():
    x, y = _make_xy(y_name="val")
    result = predict(x, y, prediction_length=3)
    assert "x_auto_converted" not in result.columns


def test_object_datetime_x_extrapolates_as_datetime():
    """Treat object columns containing datetime values as datetime x values."""
    x = pd.Series(
        [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)],
        dtype=object,
        name="date",
    )
    y = pd.Series([1.0, 2.0, 3.0], name="cases")
    result = predict(x, y, prediction_length=2)

    assert "x_auto_converted" not in result.columns
    assert list(result["date"]) == [pd.Timestamp("2022-01-04"), pd.Timestamp("2022-01-05")]


def test_exceeds_max_raises():
    x, y = _make_xy()
    with pytest.raises(ValueError, match="exceeds max"):
        predict(x, y, prediction_length=_MAX_PREDICTION_STEPS + 1)


def test_too_short_history_raises_clean_error():
    """Reject too-short history with a clear ValueError before reaching the model."""
    from model import _MIN_CONTEXT

    x = pd.Series(np.arange(_MIN_CONTEXT - 1, dtype=float), name="x")
    y = pd.Series(np.arange(_MIN_CONTEXT - 1, dtype=float), name="cases")
    with pytest.raises(ValueError, match=f"at least {_MIN_CONTEXT} points"):
        predict(x, y, prediction_length=2)


@pytest.mark.parametrize("prediction_length", [0, -1])
def test_prediction_length_must_be_positive(prediction_length):
    """Reject zero and negative forecast horizons."""
    x, y = _make_xy()
    with pytest.raises(ValueError, match="prediction_length must be > 0"):
        predict(x, y, prediction_length=prediction_length)


def test_all_nan_y_raises_clear_error():
    """Reject a y series that contains only NaN values."""
    x, _ = _make_xy(n=20)
    y = pd.Series([np.nan] * 20, name="cases")
    with pytest.raises(ValueError, match="'cases' must contain at least one non-NaN"):
        predict(x, y, prediction_length=3)


def test_long_nan_gap_raises_clear_error():
    """Reject a y series with a long consecutive interior NaN gap."""
    x = pd.Series(np.arange(12, dtype=float), name="x")
    y = pd.Series([1.0] + [np.nan] * 9 + [11.0, 12.0], name="cases")
    with pytest.raises(ValueError, match=r"'cases' has a NaN gap longer than 8 .*\(9\)"):
        predict(x, y, prediction_length=2)


def test_long_leading_nan_run_is_accepted():
    """Accept a long leading NaN run that .bfill() carries over safely."""
    x = pd.Series(np.arange(20, dtype=float), name="x")
    y = pd.Series([np.nan] * 9 + list(range(11, 22)), name="cases")
    result = predict(x, y, prediction_length=2)
    assert len(result) == 2


def test_long_trailing_nan_run_is_accepted():
    """Accept a long trailing NaN run that .ffill() carries over safely."""
    x = pd.Series(np.arange(20, dtype=float), name="x")
    y = pd.Series(list(range(1, 12)) + [np.nan] * 9, name="cases")
    result = predict(x, y, prediction_length=2)
    assert len(result) == 2


def test_non_numeric_y_raises_clear_error():
    """Reject a non-numeric y series, naming the column."""
    x = pd.Series([1.0, 2.0, 3.0], name="x")
    y = pd.Series(["low", "medium", "high"], name="cases")
    with pytest.raises(ValueError, match="'cases' must be numeric"):
        predict(x, y, prediction_length=2)


def test_duplicate_string_x_raises_clear_error():
    x = pd.Series(["2022-W01", "2022-W01", "2022-W02"], name="week")
    y = pd.Series([1.0, 2.0, 3.0], name="cases")
    with pytest.raises(ValueError, match="unique"):
        predict(x, y, prediction_length=2)


def test_nonmonotonic_string_x_raises_clear_error():
    x = pd.Series(["2022-W01", "2022-W03", "2022-W02"], name="week")
    y = pd.Series([1.0, 2.0, 3.0], name="cases")
    with pytest.raises(ValueError, match="monotonic"):
        predict(x, y, prediction_length=2)


def test_duplicate_numeric_x_raises_clear_error():
    x = pd.Series([1.0, 1.0, 2.0], name="x")
    y = pd.Series([1.0, 2.0, 3.0], name="cases")
    with pytest.raises(ValueError, match="unique"):
        predict(x, y, prediction_length=2)


def test_nonmonotonic_numeric_x_raises_clear_error():
    x = pd.Series([5.0, 1.0, 2.0, 3.0, 4.0], name="x")
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="cases")
    with pytest.raises(ValueError, match="monotonic"):
        predict(x, y, prediction_length=2)


def test_nan_numeric_x_raises_clear_error():
    x = pd.Series([1.0, np.nan, 3.0], name="x")
    y = pd.Series([1.0, 2.0, 3.0], name="cases")
    with pytest.raises(ValueError, match="NaN"):
        predict(x, y, prediction_length=2)


def test_irregular_numeric_x_raises_clear_error():
    x = pd.Series([1.0, 2.0, 4.0], name="x")
    y = pd.Series([1.0, 2.0, 3.0], name="cases")
    with pytest.raises(ValueError, match="uniform step"):
        predict(x, y, prediction_length=2)


def test_duplicate_datetime_x_raises_clear_error():
    x = pd.Series(pd.to_datetime(["2022-01-01", "2022-01-01", "2022-01-02"]), name="date")
    y = pd.Series([1.0, 2.0, 3.0], name="cases")
    with pytest.raises(ValueError, match="unique"):
        predict(x, y, prediction_length=2)


def test_irregular_datetime_x_raises_clear_error():
    x = pd.Series(pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-04"]), name="date")
    y = pd.Series([1.0, 2.0, 3.0], name="cases")
    with pytest.raises(ValueError, match="uniform step"):
        predict(x, y, prediction_length=2)


def test_subsecond_datetime_irregularity_rejected():
    """Reject datetime x values with sub-second step irregularity."""
    x = pd.Series(
        [
            pd.Timestamp("2022-01-01"),
            pd.Timestamp("2022-01-02"),
            pd.Timestamp("2022-01-03 00:00:00.100"),
            pd.Timestamp("2022-01-04"),
        ],
        name="date",
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0], name="cases")
    with pytest.raises(ValueError, match="uniform step"):
        predict(x, y, prediction_length=2)


def test_missing_x_name_defaults_to_x_column():
    """Use a stable x column name when x_series has no name."""
    x = pd.Series([1.0, 2.0, 3.0], name=None)
    y = pd.Series([1.0, 2.0, 3.0], name="cases")
    result = predict(x, y, prediction_length=2)
    assert list(result.columns) == ["x", "cases", "cases_q10", "cases_q90"]
    assert list(result["x"]) == [4.0, 5.0]


def test_missing_y_name_defaults_to_y_column():
    """Use a stable y column name when y_series has no name."""
    x = pd.Series([1.0, 2.0, 3.0], name="x")
    y = pd.Series([1.0, 2.0, 3.0], name=None)
    result = predict(x, y, prediction_length=2)
    assert list(result.columns) == ["x", "y", "y_q10", "y_q90"]


def test_unexpected_quantile_channel_count_raises(mock_timesfm):
    """Reject TimesFM quantile output with an unexpected channel count."""
    mock_instance = mock_timesfm.from_pretrained.return_value
    mock_instance.forecast.side_effect = lambda inputs, horizon: (
        np.full((len(inputs), horizon), 45.0),
        np.zeros((len(inputs), horizon, 9)),
    )
    x, y = _make_xy(y_name="cases")

    with pytest.raises(ValueError, match="10 channels"):
        predict(x, y, prediction_length=2)


def test_model_loaded_once(mock_timesfm):
    x, y = _make_xy()
    predict(x, y, prediction_length=4)
    predict(x, y, prediction_length=4)
    assert mock_timesfm.from_pretrained.call_count == 1
