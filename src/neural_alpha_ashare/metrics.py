from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl


def rank_ic_by_date(
    frame: pl.DataFrame, prediction: str = "model_alpha", target: str = "target_20"
) -> pl.DataFrame:
    valid = frame.filter(pl.col(prediction).is_not_null() & pl.col(target).is_not_null())
    if valid.is_empty():
        return pl.DataFrame(schema={"trade_date": pl.Date, "rank_ic": pl.Float64, "n": pl.UInt32})
    return (
        valid.group_by("trade_date")
        .agg(
            pl.corr(prediction, target, method="spearman").alias("rank_ic"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 10)
        .sort("trade_date")
    )


def newey_west_t(values: np.ndarray, max_lag: int | None = None) -> float:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    size = clean.size
    if size < 3:
        return float("nan")
    lag = min(max_lag or int(4 * (size / 100) ** (2 / 9)), size - 1)
    demeaned = clean - clean.mean()
    long_run = float(np.dot(demeaned, demeaned) / size)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(demeaned[offset:], demeaned[:-offset]) / size)
        long_run += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run, 0.0) / size)
    return float(clean.mean() / standard_error) if standard_error > 0 else float("nan")


def prediction_metrics(
    frame: pl.DataFrame, prediction: str = "model_alpha", target: str = "target_20"
) -> dict[str, Any]:
    daily = rank_ic_by_date(frame, prediction, target)
    values = daily["rank_ic"].to_numpy() if not daily.is_empty() else np.array([])
    mean = float(np.nanmean(values)) if values.size else float("nan")
    std = float(np.nanstd(values, ddof=1)) if values.size > 1 else float("nan")
    metrics: dict[str, Any] = {
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "rank_ic_ir": mean / std if std and np.isfinite(std) else float("nan"),
        "rank_ic_newey_west_t": newey_west_t(values),
        "dates": int(values.size),
    }
    metrics.update(quantile_metrics(frame, prediction, target))
    return metrics


def quantile_metrics(
    frame: pl.DataFrame,
    prediction: str = "model_alpha",
    target: str = "target_20",
    quantiles: int = 5,
) -> dict[str, Any]:
    valid = frame.filter(pl.col(prediction).is_not_null() & pl.col(target).is_not_null())
    if valid.is_empty():
        return {
            "quantile_returns": [],
            "quantile_spread": float("nan"),
            "quantile_monotonicity": float("nan"),
        }
    count = pl.len().over("trade_date")
    bucket = (
        ((pl.col(prediction).rank(method="ordinal").over("trade_date") - 1) * quantiles / count)
        .floor()
        .clip(0, quantiles - 1)
        .cast(pl.Int8)
        .alias("_quantile")
    )
    averages = (
        valid.with_columns(bucket)
        .group_by("trade_date", "_quantile")
        .agg(pl.col(target).mean().alias("_return"))
        .group_by("_quantile")
        .agg(pl.col("_return").mean())
        .sort("_quantile")
    )
    values = averages["_return"].to_numpy().astype(np.float64)
    monotonicity = (
        float(np.corrcoef(np.arange(values.size), values)[0, 1])
        if values.size > 1 and np.std(values) > 0
        else float("nan")
    )
    return {
        "quantile_returns": values.tolist(),
        "quantile_spread": float(values[-1] - values[0]) if values.size > 1 else float("nan"),
        "quantile_monotonicity": monotonicity,
    }


def multi_horizon_metrics(
    frame: pl.DataFrame, horizons: tuple[int, ...], primary_horizon: int = 20
) -> dict[str, Any]:
    primary = primary_horizon if primary_horizon in horizons else horizons[len(horizons) // 2]
    result = prediction_metrics(frame, "model_alpha", f"target_{primary}")
    result["heads"] = {
        str(horizon): prediction_metrics(frame, f"prediction_{horizon}", f"target_{horizon}")
        for horizon in horizons
    }
    result["combined_by_horizon"] = {
        str(horizon): prediction_metrics(frame, "model_alpha", f"target_{horizon}")
        for horizon in horizons
    }
    return result


def performance_metrics(nav: pl.DataFrame) -> dict[str, float]:
    if nav.height < 2:
        return {}
    values = nav["nav"].to_numpy().astype(np.float64)
    returns = values[1:] / values[:-1] - 1.0
    annual_return = (values[-1] / values[0]) ** (252.0 / max(len(returns), 1)) - 1.0
    annual_volatility = float(np.std(returns, ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    peaks = np.maximum.accumulate(values)
    max_drawdown = float(np.min(values / peaks - 1.0))
    downside = returns[returns < 0]
    return {
        "total_return": float(values[-1] / values[0] - 1.0),
        "annual_return": float(annual_return),
        "annual_volatility": annual_volatility,
        "sharpe": float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252))
        if len(returns) > 1 and np.std(returns, ddof=1) > 0
        else 0.0,
        "sortino": float(np.mean(returns) / np.std(downside, ddof=1) * np.sqrt(252))
        if len(downside) > 1 and np.std(downside, ddof=1) > 0
        else 0.0,
        "max_drawdown": max_drawdown,
        "annual_turnover": float(nav["turnover"].mean() * 252)
        if "turnover" in nav.columns
        else 0.0,
    }
