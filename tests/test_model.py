import sys
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import model as model_module
from model import predict, _MAX_PREDICTION_STEPS


def _make_xy(n=100, y_cols=("a", "b"), freq="D"):
    dates = pd.date_range("2022-01-01", periods=n, freq=freq)
    x_series = pd.Series(dates, name="date")
    y_df = pd.DataFrame({col: np.random.rand(n) * 50 + 10 for col in y_cols})
    return x_series, y_df


def _mock_forecast(inputs, horizon):
    batch = len(inputs)
    point = np.full((batch, horizon), 45.0)
    quantiles = np.zeros((batch, horizon, 10))
    quantiles[..., 0]  = 30.0
    quantiles[..., -1] = 60.0
    return point, quantiles


@pytest.fixture(autouse=True)
def mock_timesfm():
    mock_instance = MagicMock()
    mock_instance.forecast.side_effect = _mock_forecast
    model_module._tfm = None
    with patch("timesfm.TimesFM_2p5_200M_torch") as mock_cls:
        mock_cls.from_pretrained.return_value = mock_instance
        yield mock_cls
    model_module._tfm = None


def test_output_row_count():
    x, y = _make_xy()
    assert len(predict(x, y, prediction_length=7)) == 7


def test_output_columns_named_correctly():
    x, y = _make_xy(y_cols=("temp", "pressure"))
    result = predict(x, y, prediction_length=3)
    expected = {"date", "temp", "temp_q10", "temp_q90", "pressure", "pressure_q10", "pressure_q90"}
    assert set(result.columns) == expected


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


def test_x_extrapolation_numeric():
    n = 50
    x = pd.Series(np.arange(n, dtype=float), name="x")
    y = pd.DataFrame({"y": np.random.rand(n)})
    result = predict(x, y, prediction_length=3)
    for i, row in enumerate(result.itertuples(), 1):
        assert row.x == pytest.approx(n - 1 + i)


def test_single_y_column():
    x, y = _make_xy(y_cols=("value",))
    result = predict(x, y, prediction_length=5)
    assert set(result.columns) == {"date", "value", "value_q10", "value_q90"}


def test_quantile_ordering():
    x, y = _make_xy()
    result = predict(x, y, prediction_length=5)
    for col in y.columns:
        assert (result[f"{col}_q10"] <= result[f"{col}_q90"]).all()


def test_x_string_column_outputs_integer_positions():
    weeks = [f"2022-W{i+1:02d}" for i in range(50)]
    x = pd.Series(weeks, name="Meldewoche")
    y = pd.DataFrame({"cases": np.random.rand(50) * 100})
    result = predict(x, y, prediction_length=3)
    assert result.columns[0] == "Meldewoche"
    assert list(result["Meldewoche"]) == [51, 52, 53]


def test_exceeds_max_raises():
    x, y = _make_xy()
    with pytest.raises(ValueError, match="exceeds max"):
        predict(x, y, prediction_length=_MAX_PREDICTION_STEPS + 1)


def test_model_loaded_once(mock_timesfm):
    x, y = _make_xy()
    predict(x, y, prediction_length=4)
    predict(x, y, prediction_length=4)
    assert mock_timesfm.from_pretrained.call_count == 1
