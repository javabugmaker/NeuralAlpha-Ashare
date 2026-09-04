from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from neural_alpha_ashare.config import LightGBMConfig, WalkForwardConfig
from neural_alpha_ashare.metrics import newey_west_t, prediction_metrics
from neural_alpha_ashare.models.lightgbm import LightGBMAlphaModel
from neural_alpha_ashare.models.ridge import RidgeAlphaModel
from neural_alpha_ashare.walk_forward import assert_fold_isolation, make_folds


def _model_frame() -> pl.DataFrame:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(30)]
    symbols = [f"600{index:03d}.SH" for index in range(40)]
    x = np.tile(np.linspace(-1, 1, len(symbols), dtype=np.float32), len(dates))
    day = np.repeat(np.arange(len(dates), dtype=np.float32), len(symbols))
    return pl.DataFrame(
        {
            "trade_date": np.repeat(np.array(dates, dtype="datetime64[D]"), len(symbols)),
            "symbol": np.tile(symbols, len(dates)),
            "eligible": [True] * (len(dates) * len(symbols)),
            "median_amount_20": [50_000_000.0] * (len(dates) * len(symbols)),
            "instrument_type": ["stock"] * (len(dates) * len(symbols)),
            "f_x": x,
            "f_day": day / 30,
            "target_5": x * 0.8 + day * 0.0001,
            "target_20": x * 0.6 + day * 0.0001,
            "target_60": x * 0.4 + day * 0.0001,
        }
    )


def test_ridge_multihead_ranks_per_date() -> None:
    frame = _model_frame()
    train = frame.filter(pl.col("trade_date") < date(2024, 1, 23))
    validation = frame.filter(pl.col("trade_date") >= date(2024, 1, 23))
    model = RidgeAlphaModel(alpha=1.0).fit(
        train, validation, ["f_x", "f_day"], (5, 20, 60), (0.2, 0.5, 0.3), "test"
    )
    prediction = model.predict(validation)
    bounds = prediction.group_by("trade_date").agg(
        pl.col("model_rank").min().alias("low"), pl.col("model_rank").max().alias("high")
    )
    assert bounds["low"].to_list() == [1] * bounds.height
    assert bounds["high"].to_list() == [40] * bounds.height
    assert prediction["model_alpha"].is_between(0, 1).all()


def test_lightgbm_cpu_smoke() -> None:
    frame = _model_frame()
    train = frame.head(800)
    validation = frame.tail(400)
    config = LightGBMConfig(
        n_estimators=30,
        learning_rate=0.1,
        num_leaves=7,
        max_depth=4,
        min_child_samples=5,
        early_stopping_rounds=5,
    )
    model = LightGBMAlphaModel(config, cpu_threads=2, max_train_rows=1000).fit(
        train, validation, ["f_x", "f_day"], (5, 20, 60), (0.2, 0.5, 0.3), "test"
    )
    assert model.predict(validation).height == validation.height


def test_metrics_and_purged_walk_forward() -> None:
    frame = _model_frame().with_columns(pl.col("target_20").alias("model_alpha"))
    metrics = prediction_metrics(frame)
    assert metrics["rank_ic_mean"] > 0.99
    assert np.isfinite(newey_west_t(np.array([0.01, 0.02, 0.03, 0.01])))
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(80)]
    config = WalkForwardConfig(
        initial_train_sessions=30,
        validation_sessions=10,
        test_sessions=10,
        step_sessions=10,
        purge_sessions=5,
        embargo_sessions=2,
    )
    folds = make_folds(dates, config, rolling_train_sessions=20)
    assert folds
    for fold in folds:
        assert_fold_isolation(fold)
        assert (fold.validation_start - fold.train_end).days >= 5
        assert (fold.test_start - fold.validation_end).days >= 2
