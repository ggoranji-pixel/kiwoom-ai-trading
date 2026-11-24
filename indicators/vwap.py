"""
VWAP indicators for quantitative trading research.

Part of the Kiwoom AI Trading project.
Licensed under the MIT License.
"""

import pandas as pd


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """Calculate cumulative VWAP, resetting each trading day.

    Input: chronological OHLCV data with a DatetimeIndex
    in the exchange's local timezone.
    """
    typical_price = (
        data["high"] + data["low"] + data["close"]
    ) / 3

    volume = data["volume"]
    session = data.index.normalize()

    cumulative_pv = (
        typical_price * volume
    ).groupby(session).cumsum()

    cumulative_volume = volume.groupby(session).cumsum()

    return cumulative_pv / cumulative_volume.where(
        cumulative_volume > 0
    )


def rolling_vwap(
    data: pd.DataFrame,
    window: int = 5
) -> pd.Series:
    """Calculate rolling VWAP over a fixed number of bars."""
    if window < 1:
        raise ValueError("Window must be positive.")

    typical_price = (
        data["high"] + data["low"] + data["close"]
    ) / 3

    pv = typical_price * data["volume"]
    volume_sum = data["volume"].rolling(window).sum()

    return pv.rolling(window).sum() / volume_sum.where(
        volume_sum > 0
    )


def vwap_deviation(
    data: pd.DataFrame
) -> pd.Series:
    """Calculate percentage deviation from session VWAP."""
    vwap = calculate_vwap(data)

    return (
        data["close"] / vwap.where(vwap != 0) - 1
    ) * 100
