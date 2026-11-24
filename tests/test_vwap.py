import pandas as pd
import pytest

from indicators.vwap import calculate_vwap, rolling_vwap


def sample_data():
    return pd.DataFrame(
        {
            "high": [100, 110, 120],
            "low": [100, 110, 120],
            "close": [100, 110, 120],
            "volume": [100, 200, 300],
        },
        index=pd.to_datetime([
            "2026-01-05 09:00",
            "2026-01-05 09:01",
            "2026-01-06 09:00",
        ]),
    )


def test_session_vwap():
    result = calculate_vwap(sample_data())

    assert result.iloc[0] == pytest.approx(100)
    assert result.iloc[1] == pytest.approx(106.666667)
    assert result.iloc[2] == pytest.approx(120)


def test_rolling_vwap():
    result = rolling_vwap(sample_data(), window=2)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(106.666667)
    assert result.iloc[2] == pytest.approx(116)


def test_invalid_window():
    with pytest.raises(ValueError):
        rolling_vwap(sample_data(), window=0)
