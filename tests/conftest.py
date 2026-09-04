from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def synthetic_bars() -> pl.DataFrame:
    symbols = [f"600{index:03d}.SH" for index in range(35)]
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(300)]
    symbol_column = np.repeat(symbols, len(dates))
    date_column = np.tile(np.array(dates, dtype="datetime64[D]"), len(symbols))
    day = np.tile(np.arange(len(dates), dtype=np.float64), len(symbols))
    stock = np.repeat(np.arange(len(symbols), dtype=np.float64), len(dates))
    close = 10.0 + stock * 0.1 + day * 0.01 + np.sin(day / 7 + stock) * 0.03
    previous = np.r_[
        *[np.r_[values[0], values[:-1]] for values in close.reshape(len(symbols), len(dates))]
    ]
    return pl.DataFrame(
        {
            "symbol": symbol_column,
            "trade_date": date_column,
            "open": (previous * 1.001).astype(np.float32),
            "high": (np.maximum(close, previous) * 1.01).astype(np.float32),
            "low": (np.minimum(close, previous) * 0.99).astype(np.float32),
            "close": close.astype(np.float32),
            "prev_close": previous.astype(np.float32),
            "volume": np.full(close.size, 5_000_000.0),
            "amount": np.full(close.size, 60_000_000.0),
            "name": np.repeat([f"测试{index}" for index in range(35)], len(dates)),
            "instrument_type": np.full(close.size, "stock"),
        }
    )
