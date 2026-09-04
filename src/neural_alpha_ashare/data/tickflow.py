from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from ..config import TickFlowConfig

LOGGER = logging.getLogger("neural_alpha_ashare")
BAR_COLUMNS = [
    "symbol",
    "trade_date",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "amount",
]

_TICKFLOW_NOTICE_CHARACTERS = "🆓✅❌⚠️💡"


def _make_tickflow_notice_safe(stream: Any) -> None:
    """Prevent TickFlow's emoji notice from crashing legacy Windows consoles.

    TickFlow.free() prints several emoji before constructing its client.  A
    Windows stream using GBK/CP936 cannot encode those characters with the
    default ``strict`` error handler.  Keep the user's current encoding (so
    Chinese output remains readable) and only replace unsupported glyphs.
    """
    encoding = getattr(stream, "encoding", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if not encoding or not callable(reconfigure):
        return
    try:
        _TICKFLOW_NOTICE_CHARACTERS.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        try:
            reconfigure(errors="replace")
        except (OSError, ValueError):  # pragma: no cover - closed/custom streams
            LOGGER.debug("Unable to relax stdout encoding errors", exc_info=True)


def _millis(value: str | pd.Timestamp, end_of_day: bool = False) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Shanghai")
    if end_of_day:
        timestamp = timestamp.normalize() + pd.Timedelta(days=1, milliseconds=-1)
    return int(timestamp.tz_convert("UTC").timestamp() * 1000)


def _columnar_to_frame(symbol: str, payload: Mapping[str, Sequence[Any]]) -> pd.DataFrame:
    timestamps = list(payload.get("timestamp", []))
    size = len(timestamps)
    if size == 0:
        return pd.DataFrame(columns=BAR_COLUMNS)

    def values(name: str, default: float = float("nan")) -> list[Any]:
        raw = payload.get(name)
        return list(raw) if raw is not None else [default] * size

    utc = pd.to_datetime(timestamps, unit="ms", utc=True)
    result = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": utc.tz_convert("Asia/Shanghai").date,
            "timestamp": timestamps,
            "open": values("open"),
            "high": values("high"),
            "low": values("low"),
            "close": values("close"),
            "prev_close": values("prev_close"),
            "volume": values("volume", 0),
            "amount": values("amount", 0),
        }
    )
    numeric = ["open", "high", "low", "close", "prev_close", "volume", "amount"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result["prev_close"] = result["prev_close"].fillna(result["close"].shift(1))
    return result[BAR_COLUMNS]


@dataclass(frozen=True)
class TickFlowUpdateResult:
    symbols_requested: int
    symbols_received: int
    rows_received: int
    latest_date: str | None
    catalog: pl.DataFrame
    bars: pl.DataFrame


class TickFlowFreeClient:
    """Small, injectable boundary around the official keyless TickFlow SDK."""

    def __init__(
        self, config: TickFlowConfig, cache_dir: Path | None = None, sdk: Any | None = None
    ) -> None:
        self.config = config
        self._owns_sdk = sdk is None
        if sdk is None:
            try:
                from tickflow import TickFlow
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("tickflow is required; run pip install -e .") from exc
            _make_tickflow_notice_safe(sys.stdout)
            sdk = TickFlow.free(
                timeout=float(config.timeout_seconds),
                max_retries=int(config.max_retries),
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        self.sdk = sdk

    def close(self) -> None:
        if self._owns_sdk and hasattr(self.sdk, "close"):
            self.sdk.close()

    def __enter__(self) -> TickFlowFreeClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def fetch_catalog(self) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for exchange in self.config.exchanges:
            for instrument_type in self.config.instrument_types:
                instruments = self.sdk.exchanges.get_instruments(
                    exchange, instrument_type=instrument_type
                )
                for instrument in instruments:
                    ext = instrument.get("ext") or {}
                    rows.append(
                        {
                            "symbol": instrument.get("symbol"),
                            "code": instrument.get("code"),
                            "name": instrument.get("name"),
                            "exchange": instrument.get("exchange", exchange),
                            "region": instrument.get("region", "CN"),
                            "instrument_type": instrument.get("type", instrument_type),
                            "listing_date": ext.get("listing_date"),
                            "float_shares": ext.get("float_shares"),
                            "total_shares": ext.get("total_shares"),
                        }
                    )
        if not rows:
            raise RuntimeError("TickFlow.free() returned an empty A-share catalog")
        catalog = pl.from_dicts(rows, infer_schema_length=None).filter(
            pl.col("symbol").is_not_null()
        )
        return (
            catalog.unique("symbol", keep="last")
            .with_columns(
                pl.col("listing_date").cast(pl.Utf8).str.to_date(strict=False),
                pl.col("instrument_type").cast(pl.Utf8).str.to_lowercase(),
                pl.lit(
                    pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).to_pydatetime()
                ).alias("observed_at"),
            )
            .sort("symbol")
        )

    def fetch_bars(
        self,
        symbols: Sequence[str],
        start_date: str | None = None,
        end_date: str | None = None,
        count: int = 10_000,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame(schema={column: pl.Null for column in BAR_COLUMNS})
        kwargs: dict[str, Any] = {
            "period": self.config.period,
            "count": min(int(count), 10_000),
            "adjust": "none",
            "as_dataframe": False,
            "show_progress": False,
            "max_workers": int(self.config.max_workers),
            "batch_size": min(int(self.config.batch_size), 100),
        }
        if start_date:
            kwargs["start_time"] = _millis(start_date)
        if end_date:
            kwargs["end_time"] = _millis(end_date, end_of_day=True)
        payload = self.sdk.klines.batch(list(symbols), **kwargs)
        parts = [_columnar_to_frame(symbol, data) for symbol, data in payload.items()]
        parts = [part for part in parts if not part.empty]
        if not parts:
            return pl.DataFrame(schema={column: pl.Null for column in BAR_COLUMNS})
        pandas_frame = pd.concat(parts, ignore_index=True)
        pandas_frame["ingested_at"] = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
        return (
            pl.from_pandas(pandas_frame)
            .with_columns(
                pl.col("trade_date").cast(pl.Date),
                pl.col("open", "high", "low", "close", "prev_close", "amount").cast(pl.Float32),
                pl.col("volume").cast(pl.Float64),
            )
            .sort(["trade_date", "symbol"])
        )

    def update(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: Iterable[str] | None = None,
    ) -> TickFlowUpdateResult:
        catalog = self.fetch_catalog()
        requested = list(symbols) if symbols is not None else catalog["symbol"].to_list()
        if self.config.benchmark not in requested:
            requested.append(self.config.benchmark)
        bars = self.fetch_bars(requested, start_date=start_date, end_date=end_date)
        received = bars["symbol"].n_unique() if not bars.is_empty() else 0
        latest = bars["trade_date"].max().isoformat() if not bars.is_empty() else None
        LOGGER.info(
            "TickFlow update requested=%s received=%s rows=%s latest=%s",
            len(requested),
            received,
            bars.height,
            latest,
        )
        return TickFlowUpdateResult(len(requested), received, bars.height, latest, catalog, bars)
