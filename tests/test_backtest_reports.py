from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from neural_alpha_ashare.backtest import TransactionCostModel, run_backtest
from neural_alpha_ashare.config import PortfolioConfig
from neural_alpha_ashare.reports import write_reports


def test_exact_commission_contract() -> None:
    config = PortfolioConfig()
    costs = TransactionCostModel(config)
    notional = np.array([100_000.0, 100_000.0])
    result = costs.commission(notional, np.array(["stock", "etf"]))
    assert result[0] == 100_000 * 0.00008499999
    assert result[1] == 100_000 * 0.00005000001


def test_t_plus_one_execution_and_sell_costs() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    market = pl.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["600000.SH"] * 3,
            "open": [10.0, 10.0, 10.2],
            "close": [10.0, 10.1, 10.2],
            "volume": [1_000_000.0] * 3,
            "open_at_limit_up": [False] * 3,
            "open_at_limit_down": [False] * 3,
        }
    )
    predictions = pl.DataFrame(
        {
            "trade_date": dates[:2],
            "symbol": ["600000.SH"] * 2,
            "model_alpha": [1.0, 0.0],
            "model_rank": [1, 100],
            "eligible": [True, False],
            "median_amount_20": [100_000_000.0] * 2,
            "instrument_type": ["stock"] * 2,
        }
    )
    config = PortfolioConfig(
        initial_capital=20_000,
        rebalance_every_sessions=1,
        max_weight=1.0,
        slippage_bps=0.0,
        impact_coefficient=0.0,
    )
    result = run_backtest(market, predictions, config)
    assert result.trades["side"].to_list() == ["BUY", "SELL"]
    assert result.trades["trade_date"].to_list() == dates[1:]
    sell = result.trades.filter(pl.col("side") == "SELL").row(0, named=True)
    assert sell["stamp_duty"] == sell["notional"] * 0.0005


def test_blocked_exit_is_retried_before_next_rebalance() -> None:
    dates = [date(2024, 1, day) for day in range(2, 7)]
    market = pl.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["600000.SH"] * 5,
            "open": [10.0] * 5,
            "close": [10.0] * 5,
            "volume": [1_000_000.0] * 5,
            "open_at_limit_up": [False] * 5,
            "open_at_limit_down": [False, False, False, True, False],
        }
    )
    predictions = pl.DataFrame(
        {
            "trade_date": dates[:3],
            "symbol": ["600000.SH"] * 3,
            "model_alpha": [1.0, 1.0, 0.0],
            "model_rank": [1, 1, 100],
            "eligible": [True, True, False],
            "median_amount_20": [100_000_000.0] * 3,
            "instrument_type": ["stock"] * 3,
        }
    )
    config = PortfolioConfig(
        initial_capital=20_000,
        rebalance_every_sessions=2,
        max_weight=1.0,
        slippage_bps=0.0,
        impact_coefficient=0.0,
    )
    result = run_backtest(market, predictions, config)
    assert result.trades["side"].to_list() == ["BUY", "SELL"]
    assert result.trades["trade_date"].to_list() == [dates[1], dates[4]]


def test_reports_are_static_and_show_quality(tmp_path) -> None:
    predictions = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "symbol": ["600000.SH"],
            "model_rank": [1],
            "model_alpha": [0.9],
            "eligible": [True],
        }
    )
    paths = write_reports(
        tmp_path, "研究台", "2024-01-02", predictions, {"sharpe": 1.2}, "DEGRADED"
    )
    assert len(paths) == 3
    content = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "DEGRADED" in content
    assert "cdn" not in content.lower()
