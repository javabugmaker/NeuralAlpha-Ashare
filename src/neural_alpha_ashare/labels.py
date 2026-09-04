from __future__ import annotations

import polars as pl

from .config import LabelConfig


def build_labels(frame: pl.DataFrame, config: LabelConfig) -> pl.DataFrame:
    """Create execution-aware, peer-neutral targets without feature leakage."""

    ordered = frame.sort(["symbol", "trade_date"])
    count = pl.len().over("trade_date").clip(lower_bound=1)
    liquidity_bucket = (
        (pl.col("median_amount_20").rank(method="average").over("trade_date") / count)
        .mul(config.liquidity_buckets)
        .ceil()
        .clip(1, config.liquidity_buckets)
        .cast(pl.Int8)
    )
    labelled = ordered.with_columns(liquidity_bucket.alias("liquidity_bucket"))
    for horizon in config.horizons:
        entry_open = pl.col("open").shift(-1).over("symbol")
        entry_volume = pl.col("volume").shift(-1).over("symbol")
        entry_blocked = pl.col("open_at_limit_up").shift(-1).over("symbol").fill_null(True)
        exit_close = pl.col("close").shift(-horizon).over("symbol")
        exit_volume = pl.col("volume").shift(-horizon).over("symbol")
        exit_blocked = pl.col("one_price_limit_down").shift(-horizon).over("symbol").fill_null(True)
        available_date = pl.col("trade_date").shift(-horizon).over("symbol")
        raw = (
            pl.when(
                pl.col("eligible")
                & (entry_open > 0)
                & (entry_volume > 0)
                & ~entry_blocked
                & (exit_close > 0)
                & (exit_volume > 0)
                & ~exit_blocked
            )
            .then(exit_close / entry_open - 1.0)
            .otherwise(None)
        )
        labelled = labelled.with_columns(
            raw.cast(pl.Float32).alias(f"raw_return_{horizon}"),
            (~entry_blocked & (entry_volume > 0)).alias(f"entry_feasible_{horizon}"),
            available_date.alias(f"label_available_date_{horizon}"),
        )
        peer_median = (
            pl.col(f"raw_return_{horizon}")
            .median()
            .over(["trade_date", "board", "liquidity_bucket"])
        )
        labelled = labelled.with_columns(
            (pl.col(f"raw_return_{horizon}") - peer_median).alias(f"peer_excess_{horizon}")
        )
        date_count = pl.col(f"peer_excess_{horizon}").count().over("trade_date")
        target = (
            2.0
            * (pl.col(f"peer_excess_{horizon}").rank(method="average").over("trade_date") - 1.0)
            / (date_count - 1.0).clip(lower_bound=1.0)
            - 1.0
        )
        labelled = labelled.with_columns(target.cast(pl.Float32).alias(f"target_{horizon}"))
    return labelled


def mature_labels(frame: pl.DataFrame, asof_date: str, horizon: int) -> pl.DataFrame:
    """Filter labels by the date on which their full outcome became observable."""

    available = f"label_available_date_{horizon}"
    target = f"target_{horizon}"
    return frame.filter(
        pl.col(target).is_not_null() & (pl.col(available) <= pl.lit(asof_date).str.to_date())
    )
