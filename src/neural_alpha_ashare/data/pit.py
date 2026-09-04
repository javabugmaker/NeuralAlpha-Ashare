from __future__ import annotations

import polars as pl

REQUIRED_BAR_COLUMNS = {
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "amount",
}


def validate_bars(frame: pl.DataFrame) -> None:
    missing = REQUIRED_BAR_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"bars missing PIT columns: {sorted(missing)}")
    duplicates = frame.select(pl.struct("symbol", "trade_date").is_duplicated().any()).item()
    if duplicates:
        raise ValueError("duplicate symbol/trade_date rows")
    invalid = frame.select(
        (
            (pl.col("high") < pl.max_horizontal("open", "close", "low"))
            | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
            | (pl.col("close") <= 0)
            | (pl.col("prev_close") <= 0)
            | (pl.col("volume") < 0)
            | (pl.col("amount") < 0)
        ).any()
    ).item()
    if invalid:
        raise ValueError("bars contain invalid prices, volume, or amount")


def reconstruct_pit_prices(frame: pl.DataFrame) -> pl.DataFrame:
    """Create split/dividend-consistent indices using only same-day prev_close.

    TickFlow is requested with ``adjust='none'``. The daily total return
    ``close / prev_close - 1`` is point-in-time observable after that close and
    avoids today's backward-adjustment factor leaking into history.
    """

    validate_bars(frame)
    ordered = frame.sort(["symbol", "trade_date"])
    ret = (pl.col("close") / pl.col("prev_close") - 1.0).cast(pl.Float32)
    return (
        ordered.with_columns(ret.alias("total_return"))
        .with_columns(
            (pl.col("total_return") + 1.0)
            .cum_prod()
            .over("symbol")
            .cast(pl.Float32)
            .alias("pit_close")
        )
        .with_columns(
            (pl.col("pit_close") * pl.col("open") / pl.col("close"))
            .cast(pl.Float32)
            .alias("pit_open"),
            (pl.col("pit_close") * pl.col("high") / pl.col("close"))
            .cast(pl.Float32)
            .alias("pit_high"),
            (pl.col("pit_close") * pl.col("low") / pl.col("close"))
            .cast(pl.Float32)
            .alias("pit_low"),
        )
    )
