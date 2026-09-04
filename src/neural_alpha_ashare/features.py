from __future__ import annotations

import polars as pl

from .config import FeatureConfig

RETURN_WINDOWS = (1, 2, 3, 5, 10, 20, 40, 60, 120, 240)
VOL_WINDOWS = (5, 10, 20, 40, 60)
TREND_WINDOWS = (5, 10, 20, 40, 60, 120, 240)
RANGE_WINDOWS = (10, 20, 60, 120, 240)
FLOW_WINDOWS = (5, 10, 20, 60)


def feature_columns(frame: pl.DataFrame | pl.LazyFrame) -> list[str]:
    return sorted(name for name in frame.collect_schema().names() if name.startswith(("f_", "cs_")))


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator.abs() > 1e-12).then(numerator / denominator).otherwise(None)


def build_features(frame: pl.DataFrame, config: FeatureConfig) -> pl.DataFrame:
    """Build the complete feature matrix with Polars window expressions.

    Every expression at row ``t`` uses values no later than ``t``. There are no
    Python stock/date loops in the hot path.
    """

    if frame.is_empty():
        return frame
    ordered = frame.sort(["symbol", "trade_date"])
    close = pl.col("pit_close").cast(pl.Float64)
    open_ = pl.col("pit_open").cast(pl.Float64)
    high = pl.col("pit_high").cast(pl.Float64)
    low = pl.col("pit_low").cast(pl.Float64)
    ret1 = pl.col("total_return").cast(pl.Float64)

    return_exprs = [
        _safe_ratio(close, close.shift(window).over("symbol")).sub(1.0).alias(f"f_return_{window}")
        for window in RETURN_WINDOWS
    ]
    volatility_exprs: list[pl.Expr] = []
    for window in VOL_WINDOWS:
        volatility_exprs.extend(
            [
                ret1.rolling_std(window, min_samples=max(3, window // 2))
                .over("symbol")
                .alias(f"f_volatility_{window}"),
                ret1.clip(upper_bound=0.0)
                .rolling_std(window, min_samples=max(3, window // 2))
                .over("symbol")
                .alias(f"f_downside_vol_{window}"),
            ]
        )
    trend_exprs = [
        _safe_ratio(
            close,
            close.rolling_mean(window, min_samples=max(3, window // 2)).over("symbol"),
        )
        .sub(1.0)
        .alias(f"f_ma_ratio_{window}")
        for window in TREND_WINDOWS
    ]
    range_exprs: list[pl.Expr] = []
    for window in RANGE_WINDOWS:
        rolling_min = low.rolling_min(window, min_samples=max(3, window // 2)).over("symbol")
        rolling_max = high.rolling_max(window, min_samples=max(3, window // 2)).over("symbol")
        rolling_mean = close.rolling_mean(window, min_samples=max(3, window // 2)).over("symbol")
        rolling_std = close.rolling_std(window, min_samples=max(3, window // 2)).over("symbol")
        range_exprs.extend(
            [
                _safe_ratio(close - rolling_min, rolling_max - rolling_min).alias(
                    f"f_range_position_{window}"
                ),
                _safe_ratio(close - rolling_mean, rolling_std).alias(f"f_price_z_{window}"),
            ]
        )

    previous = close.shift(1).over("symbol")
    true_range = pl.max_horizontal(high - low, (high - previous).abs(), (low - previous).abs())
    atr_exprs = [
        _safe_ratio(
            true_range.rolling_mean(window, min_samples=max(3, window // 2)).over("symbol"),
            close,
        ).alias(f"f_atr_{window}")
        for window in (5, 14, 20, 60)
    ]
    flow_exprs: list[pl.Expr] = []
    for window in FLOW_WINDOWS:
        amount_mean = (
            pl.col("amount").rolling_mean(window, min_samples=max(3, window // 2)).over("symbol")
        )
        volume_mean = (
            pl.col("volume").rolling_mean(window, min_samples=max(3, window // 2)).over("symbol")
        )
        illiquidity = (
            (ret1.abs() / pl.col("amount").clip(lower_bound=1.0))
            .rolling_mean(window, min_samples=max(3, window // 2))
            .over("symbol")
        )
        flow_exprs.extend(
            [
                _safe_ratio(pl.col("amount"), amount_mean).alias(f"f_amount_ratio_{window}"),
                _safe_ratio(pl.col("volume"), volume_mean).alias(f"f_volume_ratio_{window}"),
                (illiquidity * 1e9).alias(f"f_illiquidity_{window}"),
            ]
        )

    shape_exprs = [
        _safe_ratio(open_, previous).sub(1.0).alias("f_gap"),
        _safe_ratio(close, open_).sub(1.0).alias("f_intraday_return"),
        _safe_ratio(high - low, previous).alias("f_daily_range"),
        _safe_ratio(high - pl.max_horizontal(open_, close), previous).alias("f_upper_shadow"),
        _safe_ratio(pl.min_horizontal(open_, close) - low, previous).alias("f_lower_shadow"),
        _safe_ratio(close - low, high - low).alias("f_close_location"),
        pl.col("amount").clip(lower_bound=1.0).log().alias("f_log_amount"),
        pl.col("listing_age_sessions").cast(pl.Float64).log1p().alias("f_log_listing_age"),
        pl.col("limit_ratio").cast(pl.Float64).alias("f_limit_ratio"),
        pl.col("close_at_limit_up").cast(pl.Float64).alias("f_close_at_limit_up"),
        pl.col("close_at_limit_down").cast(pl.Float64).alias("f_close_at_limit_down"),
    ]
    featured = ordered.with_columns(
        return_exprs
        + volatility_exprs
        + trend_exprs
        + range_exprs
        + atr_exprs
        + flow_exprs
        + shape_exprs
    )

    market_context = [
        pl.col("total_return").median().over("trade_date").alias("f_market_return_median_1"),
        pl.col("f_return_5").median().over("trade_date").alias("f_market_return_median_5"),
        pl.col("f_return_20").median().over("trade_date").alias("f_market_return_median_20"),
        pl.col("total_return").std().over("trade_date").alias("f_market_dispersion"),
        (pl.col("total_return") > 0).mean().over("trade_date").alias("f_market_breadth"),
        (pl.col("f_ma_ratio_20") > 0).mean().over("trade_date").alias("f_above_ma20"),
        (pl.col("f_ma_ratio_60") > 0).mean().over("trade_date").alias("f_above_ma60"),
        pl.col("close_at_limit_up").mean().over("trade_date").alias("f_limit_up_share"),
        pl.col("close_at_limit_down").mean().over("trade_date").alias("f_limit_down_share"),
    ]
    featured = featured.with_columns(market_context)

    cross_section_sources = [
        *[f"f_return_{window}" for window in RETURN_WINDOWS],
        *[f"f_volatility_{window}" for window in VOL_WINDOWS],
        *[f"f_ma_ratio_{window}" for window in TREND_WINDOWS],
        *[f"f_range_position_{window}" for window in RANGE_WINDOWS],
        "f_atr_14",
        "f_atr_60",
        "f_amount_ratio_20",
        "f_illiquidity_20",
        "f_gap",
        "f_intraday_return",
        "f_log_amount",
    ]
    cross_section_exprs = [
        (
            2.0
            * (pl.col(name).rank(method="average").over("trade_date") - 1.0)
            / (pl.col(name).count().over("trade_date") - 1.0).clip(lower_bound=1.0)
            - 1.0
        ).alias(f"cs_{name[2:]}")
        for name in cross_section_sources
    ]
    featured = featured.with_columns(cross_section_exprs)
    names = feature_columns(featured)
    coverage = pl.sum_horizontal(
        [pl.col(name).is_finite().fill_null(False).cast(pl.UInt8) for name in names]
    ) / len(names)
    return featured.with_columns(
        coverage.cast(pl.Float32).alias("feature_coverage"),
        (pl.len().over("trade_date") >= config.min_cross_section).alias("feature_row_valid"),
        *[pl.col(name).cast(pl.Float32) for name in names],
    )
