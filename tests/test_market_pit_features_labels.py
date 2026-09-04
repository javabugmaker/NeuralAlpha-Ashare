from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from neural_alpha_ashare.config import DataConfig, FeatureConfig, LabelConfig
from neural_alpha_ashare.data.pit import reconstruct_pit_prices
from neural_alpha_ashare.features import build_features, feature_columns
from neural_alpha_ashare.labels import build_labels
from neural_alpha_ashare.market import build_eligibility, decorate_market_rules


def test_board_and_historical_limit_rules() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["600000.SH", "300001.SZ", "300001.SZ", "688001.SH", "830001.BJ"],
            "trade_date": [
                date(2024, 1, 2),
                date(2020, 8, 21),
                date(2020, 8, 24),
                date(2024, 1, 2),
                date(2024, 1, 2),
            ],
            "name": ["浦发银行", "创业", "创业", "科创", "北交"],
            "prev_close": [10.0] * 5,
            "open": [11.0, 11.0, 12.0, 12.0, 13.0],
            "high": [11.0, 11.0, 12.0, 12.0, 13.0],
            "low": [11.0, 11.0, 12.0, 12.0, 13.0],
            "close": [11.0, 11.0, 12.0, 12.0, 13.0],
            "volume": [100.0] * 5,
            "amount": [1000.0] * 5,
        }
    )
    result = decorate_market_rules(frame)
    assert result["board"].to_list() == ["MAIN", "CHINEXT", "CHINEXT", "STAR", "BSE"]
    assert np.allclose(result["limit_ratio"].to_numpy(), [0.1, 0.1, 0.2, 0.2, 0.3])
    assert result["one_price_limit_up"].to_list() == [True] * 5


def test_pit_reconstruction_does_not_back_adjust_history() -> None:
    bars = pl.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "open": [10.0, 5.0],
            "high": [10.0, 5.0],
            "low": [10.0, 5.0],
            "close": [10.0, 5.0],
            "prev_close": [10.0, 5.0],
            "volume": [100.0, 200.0],
            "amount": [1000.0, 1000.0],
        }
    )
    result = reconstruct_pit_prices(bars)
    assert result["total_return"].to_list() == [0.0, 0.0]
    assert result["pit_close"].to_list() == [1.0, 1.0]


def test_features_are_vectorized_and_labels_mature(synthetic_bars: pl.DataFrame) -> None:
    data_config = DataConfig(min_history_sessions=20, min_median_amount_20=1.0)
    pit = reconstruct_pit_prices(synthetic_bars)
    eligible = build_eligibility(pit, data_config)
    featured = build_features(eligible, FeatureConfig(min_cross_section=30))
    labelled = build_labels(featured, LabelConfig())
    assert len(feature_columns(featured)) >= 70
    last = labelled.sort("trade_date").tail(35)
    assert last["target_60"].null_count() == 35
    assert labelled.filter(pl.col("target_20").is_not_null()).height > 0
    assert labelled["feature_coverage"].dtype == pl.Float32


def test_past_features_do_not_change_when_future_changes(synthetic_bars: pl.DataFrame) -> None:
    config = DataConfig(min_history_sessions=20, min_median_amount_20=1.0)

    def calculate(source: pl.DataFrame) -> pl.DataFrame:
        return build_features(
            build_eligibility(reconstruct_pit_prices(source), config), FeatureConfig()
        ).select("symbol", "trade_date", "f_return_20", "cs_return_20")

    baseline = calculate(synthetic_bars)
    last_date = synthetic_bars["trade_date"].max()
    changed = synthetic_bars.with_columns(
        pl.when(pl.col("trade_date") == last_date)
        .then(pl.col("close") * 2)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.col("trade_date") == last_date)
        .then(pl.col("high") * 2)
        .otherwise(pl.col("high"))
        .alias("high"),
    )
    earlier = baseline.filter(pl.col("trade_date") < last_date)
    changed_earlier = calculate(changed).filter(pl.col("trade_date") < last_date)
    assert earlier.equals(changed_earlier, null_equal=True)
