from __future__ import annotations

import polars as pl

from .config import DataConfig


def board_expression(symbol: str = "symbol") -> pl.Expr:
    code = pl.col(symbol).str.split(".").list.first()
    return (
        pl.when(code.str.contains(r"^(688|689)"))
        .then(pl.lit("STAR"))
        .when(code.str.contains(r"^(300|301)"))
        .then(pl.lit("CHINEXT"))
        .when(code.str.contains(r"^(4|8|920)"))
        .then(pl.lit("BSE"))
        .otherwise(pl.lit("MAIN"))
    )


def price_limit_ratio_expression() -> pl.Expr:
    chinext_reform = pl.date(2020, 8, 24)
    return (
        pl.when(pl.col("is_st"))
        .then(0.05)
        .when(pl.col("board") == "STAR")
        .then(0.20)
        .when(pl.col("board") == "BSE")
        .then(0.30)
        .when((pl.col("board") == "CHINEXT") & (pl.col("trade_date") >= chinext_reform))
        .then(0.20)
        .otherwise(0.10)
    )


def decorate_market_rules(frame: pl.DataFrame) -> pl.DataFrame:
    name = pl.col("name").fill_null("") if "name" in frame.columns else pl.lit("")
    decorated = frame.with_columns(
        board_expression().alias("board"),
        name.str.to_uppercase().str.contains(r"(^|[^A-Z])\*?ST").alias("is_st"),
    ).with_columns(price_limit_ratio_expression().cast(pl.Float32).alias("limit_ratio"))
    tolerance = pl.lit(0.0005)
    limit_up = (pl.col("prev_close") * (1.0 + pl.col("limit_ratio"))).round(2)
    limit_down = (pl.col("prev_close") * (1.0 - pl.col("limit_ratio"))).round(2)
    return decorated.with_columns(
        limit_up.cast(pl.Float32).alias("limit_up"),
        limit_down.cast(pl.Float32).alias("limit_down"),
        (pl.col("close") >= limit_up - tolerance).alias("close_at_limit_up"),
        (pl.col("close") <= limit_down + tolerance).alias("close_at_limit_down"),
        ((pl.col("high") - pl.col("low")).abs() <= tolerance).alias("one_price"),
        ((pl.col("open") >= limit_up - tolerance) & (pl.col("volume") > 0)).alias(
            "open_at_limit_up"
        ),
        ((pl.col("open") <= limit_down + tolerance) & (pl.col("volume") > 0)).alias(
            "open_at_limit_down"
        ),
    ).with_columns(
        (pl.col("one_price") & pl.col("close_at_limit_up")).alias("one_price_limit_up"),
        (pl.col("one_price") & pl.col("close_at_limit_down")).alias("one_price_limit_down"),
    )


def build_eligibility(frame: pl.DataFrame, config: DataConfig) -> pl.DataFrame:
    """Add only contemporaneously observable eligibility fields."""

    decorated = decorate_market_rules(frame).sort(["symbol", "trade_date"])
    instrument_ok = (
        pl.col("instrument_type").fill_null("stock") == "stock"
        if "instrument_type" in decorated.columns
        else pl.lit(True)
    )
    listed = pl.int_range(pl.len()).over("symbol") + 1
    median_amount = pl.col("amount").rolling_median(20, min_samples=10).over("symbol")
    board_ok = pl.col("board").is_in(list(config.allowed_boards))
    if config.include_bse:
        board_ok = board_ok | (pl.col("board") == "BSE")
    eligible = (
        instrument_ok
        & board_ok
        & (listed >= config.min_history_sessions)
        & (median_amount >= config.min_median_amount_20)
        & (pl.col("volume") > 0)
        & (pl.col("amount") > 0)
    )
    if config.exclude_st:
        eligible = eligible & ~pl.col("is_st")
    if config.exclude_one_price_limit:
        eligible = eligible & ~pl.col("one_price_limit_up")
    return decorated.with_columns(
        listed.cast(pl.Int32).alias("listing_age_sessions"),
        median_amount.cast(pl.Float32).alias("median_amount_20"),
        eligible.alias("eligible"),
    )
