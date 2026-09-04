from __future__ import annotations

from datetime import date

import polars as pl

from neural_alpha_ashare.config import load_config
from neural_alpha_ashare.data.store import ParquetStore


def test_default_config_contract() -> None:
    config = load_config("config/default.yaml")
    assert config.labels.horizons == (5, 20, 60)
    assert config.labels.weights == (0.2, 0.5, 0.3)
    assert config.portfolio.stock_commission == 0.00008499999
    assert config.portfolio.fund_commission == 0.00005000001
    assert config.portfolio.minimum_commission == 0
    assert config.portfolio.initial_capital == 200_000
    assert config.portfolio.research_capital == 1_000_000


def test_partition_upsert_keeps_latest(tmp_path) -> None:
    store = ParquetStore(tmp_path / "raw", tmp_path / "derived")
    first = pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": [date(2024, 1, 2)],
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "close": [10.0],
            "prev_close": [9.9],
            "volume": [100.0],
            "amount": [1000.0],
            "ingested_at": ["2024-01-02T16:00:00"],
        }
    )
    second = first.with_columns(
        pl.lit(10.1).alias("close"), pl.lit("2024-01-03T16:00:00").alias("ingested_at")
    )
    store.upsert_bars(first)
    store.upsert_bars(second)
    result = store.read_bars()
    assert result.height == 1
    assert result["close"].item() == 10.1
