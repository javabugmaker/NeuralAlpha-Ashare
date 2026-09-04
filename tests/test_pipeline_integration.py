from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl

from neural_alpha_ashare.config import (
    AppConfig,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    PathsConfig,
    WalkForwardConfig,
)
from neural_alpha_ashare.pipeline import ResearchPipeline


def test_local_pipeline_end_to_end(tmp_path) -> None:
    symbols = [f"600{index:03d}.SH" for index in range(35)]
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(500)]
    symbol_column = np.repeat(symbols, len(dates))
    date_column = np.tile(np.array(dates, dtype="datetime64[D]"), len(symbols))
    day = np.tile(np.arange(len(dates), dtype=np.float64), len(symbols))
    stock = np.repeat(np.arange(len(symbols), dtype=np.float64), len(dates))
    close = 10 + stock * 0.05 + day * 0.003 + np.sin(day / 11 + stock) * 0.1
    matrix = close.reshape(len(symbols), len(dates))
    previous = np.concatenate([np.r_[row[0], row[:-1]] for row in matrix])
    bars = pl.DataFrame(
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
        }
    )
    paths = PathsConfig(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data/raw",
        derived_dir=tmp_path / "data/derived",
        cache_dir=tmp_path / "data/cache",
        models_dir=tmp_path / "models",
        predictions_dir=tmp_path / "predictions",
        backtests_dir=tmp_path / "backtests",
        logs_dir=tmp_path / "logs",
        docs_dir=tmp_path / "docs",
    )
    base = AppConfig()
    config = replace(
        base,
        paths=paths,
        data=DataConfig(
            min_history_sessions=20,
            min_median_amount_20=1.0,
            min_feature_coverage=0.50,
        ),
        features=FeatureConfig(min_cross_section=30),
        model=ModelConfig(primary="ridge", rolling_train_years=2),
        walk_forward=WalkForwardConfig(
            initial_train_sessions=250,
            validation_sessions=70,
            test_sessions=50,
            step_sessions=50,
            purge_sessions=60,
            embargo_sessions=5,
        ),
    )
    pipeline = ResearchPipeline(config)
    assert pipeline.store.upsert_bars(bars) == bars.height
    built = pipeline.build()
    assert len(built["partitions"]) == 2
    trained = pipeline.train("ridge")
    assert trained["role"] == "champion"
    walked = pipeline.walk_forward("ridge")
    assert walked["folds"]
    summaries = pipeline.backtest(walked["predictions"], capital=200_000)
    assert "200000" in summaries
    assert (paths.backtests_dir / "200000/nav.parquet").exists()
