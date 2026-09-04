from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from .config import PortfolioConfig
from .metrics import performance_metrics


@dataclass(frozen=True)
class BacktestResult:
    nav: pl.DataFrame
    trades: pl.DataFrame
    holdings: pl.DataFrame
    metrics: dict[str, float]


class TransactionCostModel:
    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config

    def commission_rate(self, instrument_types: np.ndarray) -> np.ndarray:
        kinds = np.char.lower(instrument_types.astype(str))
        fund = np.isin(kinds, ["etf", "lof", "fund"])
        return np.where(fund, self.config.fund_commission, self.config.stock_commission)

    def commission(self, notional: np.ndarray, instrument_types: np.ndarray) -> np.ndarray:
        return np.maximum(
            np.abs(notional) * self.commission_rate(instrument_types),
            self.config.minimum_commission,
        )

    def impact_fraction(self, notional: np.ndarray, known_adv_amount: np.ndarray) -> np.ndarray:
        participation = np.divide(
            np.abs(notional),
            known_adv_amount,
            out=np.zeros_like(notional, dtype=np.float64),
            where=known_adv_amount > 0,
        )
        raw = self.config.impact_coefficient * np.power(participation, self.config.impact_exponent)
        return np.minimum(raw, self.config.max_impact_bps / 10_000.0)


def _empty_trades() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "trade_date": pl.Date,
            "symbol": pl.Utf8,
            "side": pl.Utf8,
            "quantity": pl.Int64,
            "price": pl.Float64,
            "notional": pl.Float64,
            "commission": pl.Float64,
            "stamp_duty": pl.Float64,
            "slippage_impact": pl.Float64,
        }
    )


def run_backtest(
    market: pl.DataFrame,
    predictions: pl.DataFrame,
    config: PortfolioConfig,
    initial_capital: float | None = None,
) -> BacktestResult:
    """Long-only A-share simulation with a vectorized per-session state kernel.

    The only Python loop is over trading sessions. Selection, sizing, capacity,
    fees and state updates within a session are NumPy array operations.
    """

    if market.is_empty() or predictions.is_empty():
        empty = pl.DataFrame()
        return BacktestResult(empty, _empty_trades(), empty, {})
    required_market = {
        "trade_date",
        "symbol",
        "open",
        "close",
        "volume",
        "open_at_limit_up",
        "open_at_limit_down",
    }
    missing = required_market - set(market.columns)
    if missing:
        raise ValueError(f"market missing execution columns: {sorted(missing)}")
    required_predictions = {"trade_date", "symbol", "model_alpha"}
    if missing_predictions := required_predictions - set(predictions.columns):
        raise ValueError(f"predictions missing columns: {sorted(missing_predictions)}")

    market_dates = market["trade_date"].unique().sort()
    calendar = pl.DataFrame({"trade_date": market_dates}).with_columns(
        pl.col("trade_date").shift(1).alias("signal_date")
    )
    prediction_columns = [
        pl.col("trade_date").alias("signal_date"),
        "symbol",
        pl.col("model_alpha").alias("signal_alpha"),
        (
            pl.col("model_rank")
            if "model_rank" in predictions.columns
            else pl.col("model_alpha").rank(method="ordinal", descending=True).over("trade_date")
        ).alias("signal_rank"),
        (pl.col("eligible") if "eligible" in predictions.columns else pl.lit(True)).alias(
            "signal_eligible"
        ),
        (
            pl.col("median_amount_20")
            if "median_amount_20" in predictions.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("signal_capacity"),
        (
            pl.col("instrument_type")
            if "instrument_type" in predictions.columns
            else pl.lit("stock")
        ).alias("signal_type"),
    ]
    signals = predictions.select(prediction_columns)
    joined = (
        market.join(calendar, on="trade_date", how="inner")
        .join(signals, on=["signal_date", "symbol"], how="left")
        .sort(["trade_date", "symbol"])
    )
    dates = joined["trade_date"].to_numpy()
    session_dates = joined["trade_date"].unique().sort().to_list()
    symbols_raw = joined["symbol"].to_numpy()
    unique_symbols, symbol_codes = np.unique(symbols_raw, return_inverse=True)
    _, date_codes = np.unique(dates, return_inverse=True)
    boundaries = np.r_[0, np.flatnonzero(np.diff(date_codes)) + 1, joined.height]
    symbol_count = unique_symbols.size

    open_prices = joined["open"].fill_null(np.nan).to_numpy().astype(np.float64)
    close_prices = joined["close"].fill_null(np.nan).to_numpy().astype(np.float64)
    volume = joined["volume"].fill_null(0).to_numpy().astype(np.float64)
    buy_blocked = joined["open_at_limit_up"].fill_null(True).to_numpy()
    sell_blocked = joined["open_at_limit_down"].fill_null(True).to_numpy()
    signal_rank = joined["signal_rank"].fill_null(2**30).to_numpy().astype(np.int64)
    signal_alpha = joined["signal_alpha"].fill_null(np.nan).to_numpy().astype(np.float64)
    signal_eligible = joined["signal_eligible"].fill_null(False).to_numpy()
    signal_capacity = joined["signal_capacity"].fill_null(np.nan).to_numpy().astype(np.float64)
    signal_type = joined["signal_type"].fill_null("stock").to_numpy().astype(str)

    capital = float(initial_capital or config.initial_capital)
    cash = capital
    positions = np.zeros(symbol_count, dtype=np.int64)
    acquired_day = np.full(symbol_count, -1, dtype=np.int64)
    holding_sessions = np.zeros(symbol_count, dtype=np.int32)
    exit_wait = np.zeros(symbol_count, dtype=np.int16)
    last_close = np.full(symbol_count, np.nan, dtype=np.float64)
    last_capacity = np.zeros(symbol_count, dtype=np.float64)
    instrument_types = np.full(symbol_count, "stock", dtype="U8")
    costs = TransactionCostModel(config)
    nav_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []

    for day_index in range(len(session_dates)):
        start, stop = boundaries[day_index], boundaries[day_index + 1]
        rows = np.arange(start, stop)
        codes = symbol_codes[start:stop]
        valid_close = np.isfinite(close_prices[rows]) & (close_prices[rows] > 0)
        last_close[codes[valid_close]] = close_prices[rows[valid_close]]
        known_capacity = np.nan_to_num(signal_capacity[rows], nan=0.0)
        has_signal = np.isfinite(signal_alpha[rows])
        last_capacity[codes[has_signal]] = known_capacity[has_signal]
        instrument_types[codes[has_signal]] = signal_type[rows][has_signal]

        day_turnover = 0.0
        should_rebalance = day_index > 0 and (day_index - 1) % config.rebalance_every_sessions == 0
        retry_pending_exits = bool(np.any((exit_wait > 0) & (positions > 0)))
        if should_rebalance or retry_pending_exits:
            day_rank = np.full(symbol_count, 2**30, dtype=np.int64)
            day_eligible = np.zeros(symbol_count, dtype=bool)
            day_open = np.full(symbol_count, np.nan, dtype=np.float64)
            day_volume = np.zeros(symbol_count, dtype=np.float64)
            day_buy_blocked = np.ones(symbol_count, dtype=bool)
            day_sell_blocked = np.ones(symbol_count, dtype=bool)
            day_rank[codes] = signal_rank[rows]
            day_eligible[codes] = signal_eligible[rows]
            day_open[codes] = open_prices[rows]
            day_volume[codes] = volume[rows]
            day_buy_blocked[codes] = buy_blocked[rows]
            day_sell_blocked[codes] = sell_blocked[rows]

            currently_held = positions > 0
            if should_rebalance:
                retained = currently_held & (day_rank <= config.exit_rank)
                forced_exit = currently_held & (holding_sessions >= config.max_holding_sessions)
                entrants = day_eligible & (day_rank <= config.entry_rank)
                desired = (retained | entrants) & ~forced_exit
                desired_count = int(desired.sum())
                equity_at_open = cash + float(
                    np.nansum(positions * np.where(np.isfinite(day_open), day_open, last_close))
                )
                weight = min(config.max_weight, 1.0 / desired_count) if desired_count else 0.0
                target_value = np.where(desired, equity_at_open * weight, 0.0)
                target_shares = (
                    np.floor(
                        np.divide(
                            target_value,
                            day_open * config.lot_size,
                            out=np.zeros(symbol_count),
                            where=np.isfinite(day_open) & (day_open > 0),
                        )
                    ).astype(np.int64)
                    * config.lot_size
                )
            else:
                pending = (exit_wait > 0) & currently_held
                desired = currently_held & ~pending
                target_shares = positions.copy()
                target_shares[pending] = 0

            sell_quantity = np.maximum(positions - target_shares, 0)
            sellable = (
                (sell_quantity > 0)
                & (acquired_day < day_index)
                & np.isfinite(day_open)
                & (day_open > 0)
                & (day_volume > 0)
                & ~day_sell_blocked
            )
            max_capacity_shares = (
                np.floor(
                    np.divide(
                        config.max_participation * last_capacity,
                        day_open * config.lot_size,
                        out=np.zeros(symbol_count),
                        where=np.isfinite(day_open) & (day_open > 0),
                    )
                ).astype(np.int64)
                * config.lot_size
            )
            sell_quantity = np.where(
                sellable, np.minimum(sell_quantity, max_capacity_shares), 0
            ).astype(np.int64)
            sell_ids = np.flatnonzero(sell_quantity > 0)
            if sell_ids.size:
                quantity = sell_quantity[sell_ids]
                reference = day_open[sell_ids]
                reference_notional = quantity * reference
                impact = costs.impact_fraction(reference_notional, last_capacity[sell_ids])
                execution = reference * (1.0 - config.slippage_bps / 10_000.0 - impact)
                notional = quantity * execution
                commission = costs.commission(notional, instrument_types[sell_ids])
                stamp = np.where(
                    np.isin(np.char.lower(instrument_types[sell_ids]), ["etf", "lof", "fund"]),
                    0.0,
                    notional * config.stock_sell_stamp_duty,
                )
                cash += float(np.sum(notional - commission - stamp))
                positions[sell_ids] -= quantity
                day_turnover += float(np.sum(notional))
                trade_rows.extend(
                    {
                        "trade_date": session_dates[day_index],
                        "symbol": unique_symbols[index],
                        "side": "SELL",
                        "quantity": int(qty),
                        "price": float(price),
                        "notional": float(value),
                        "commission": float(fee),
                        "stamp_duty": float(tax),
                        "slippage_impact": float(reference_price - price),
                    }
                    for index, qty, price, value, fee, tax, reference_price in zip(
                        sell_ids,
                        quantity,
                        execution,
                        notional,
                        commission,
                        stamp,
                        reference,
                        strict=True,
                    )
                )

            buy_quantity = np.maximum(target_shares - positions, 0)
            buyable = (
                (buy_quantity > 0)
                & should_rebalance
                & day_eligible
                & np.isfinite(day_open)
                & (day_open > 0)
                & (day_volume > 0)
                & ~day_buy_blocked
            )
            buy_quantity = np.where(
                buyable, np.minimum(buy_quantity, max_capacity_shares), 0
            ).astype(np.int64)
            buy_ids = np.flatnonzero(buy_quantity > 0)
            if buy_ids.size:
                quantity = buy_quantity[buy_ids]
                reference = day_open[buy_ids]
                reference_notional = quantity * reference
                impact = costs.impact_fraction(reference_notional, last_capacity[buy_ids])
                execution = reference * (1.0 + config.slippage_bps / 10_000.0 + impact)
                notional = quantity * execution
                commission = costs.commission(notional, instrument_types[buy_ids])
                total_cost = notional + commission
                if total_cost.sum() > cash:
                    scale = max(cash / float(total_cost.sum()), 0.0)
                    quantity = (
                        np.floor(quantity * scale / config.lot_size).astype(np.int64)
                        * config.lot_size
                    )
                    keep = quantity > 0
                    buy_ids, quantity, reference, impact = (
                        buy_ids[keep],
                        quantity[keep],
                        reference[keep],
                        impact[keep],
                    )
                    execution = reference * (1.0 + config.slippage_bps / 10_000.0 + impact)
                    notional = quantity * execution
                    commission = costs.commission(notional, instrument_types[buy_ids])
                    total_cost = notional + commission
                cash -= float(total_cost.sum())
                was_flat = positions[buy_ids] == 0
                positions[buy_ids] += quantity
                acquired_day[buy_ids[was_flat]] = day_index
                holding_sessions[buy_ids[was_flat]] = 0
                day_turnover += float(np.sum(notional))
                trade_rows.extend(
                    {
                        "trade_date": session_dates[day_index],
                        "symbol": unique_symbols[index],
                        "side": "BUY",
                        "quantity": int(qty),
                        "price": float(price),
                        "notional": float(value),
                        "commission": float(fee),
                        "stamp_duty": 0.0,
                        "slippage_impact": float(price - reference_price),
                    }
                    for index, qty, price, value, fee, reference_price in zip(
                        buy_ids,
                        quantity,
                        execution,
                        notional,
                        commission,
                        reference,
                        strict=True,
                    )
                )

            wants_exit = currently_held & ~desired
            exited = positions == 0
            exit_wait[wants_exit & ~exited] += 1
            exit_wait[~wants_exit | exited] = 0

        active = positions > 0
        holding_sessions[active] += 1
        holding_sessions[~active] = 0
        acquired_day[~active] = -1
        marked = np.nan_to_num(last_close, nan=0.0)
        nav_value = cash + float(np.dot(positions, marked))
        nav_rows.append(
            {
                "trade_date": session_dates[day_index],
                "nav": nav_value,
                "cash": cash,
                "market_value": nav_value - cash,
                "positions": int(active.sum()),
                "turnover": day_turnover / max(nav_value, 1.0),
                "exit_wait_breaches": int(
                    ((exit_wait > config.max_exit_wait_sessions) & active).sum()
                ),
            }
        )
        active_ids = np.flatnonzero(active)
        holding_rows.extend(
            {
                "trade_date": session_dates[day_index],
                "symbol": unique_symbols[index],
                "quantity": int(positions[index]),
                "close": float(marked[index]),
                "market_value": float(positions[index] * marked[index]),
                "holding_sessions": int(holding_sessions[index]),
                "exit_wait_sessions": int(exit_wait[index]),
            }
            for index in active_ids
        )

    nav = pl.from_dicts(nav_rows, infer_schema_length=None)
    trades = pl.from_dicts(trade_rows, infer_schema_length=None) if trade_rows else _empty_trades()
    holdings = (
        pl.from_dicts(holding_rows, infer_schema_length=None) if holding_rows else pl.DataFrame()
    )
    metrics = performance_metrics(nav)
    if not trades.is_empty():
        metrics.update(
            {
                "commission": float(trades["commission"].sum()),
                "stamp_duty": float(trades["stamp_duty"].sum()),
                "trade_count": float(trades.height),
            }
        )
    return BacktestResult(nav, trades, holdings, metrics)
