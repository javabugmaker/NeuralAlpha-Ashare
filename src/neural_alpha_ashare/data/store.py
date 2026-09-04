from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import polars as pl


class ParquetStore:
    """Atomic, year-partitioned Parquet store with predicate-pushdown reads."""

    def __init__(self, raw_dir: str | Path, derived_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.derived_dir = Path(derived_dir)
        self.universe_dir = self.raw_dir / "universe"
        self.manifest_dir = self.raw_dir / "manifests"
        for path in (self.raw_dir, self.derived_dir, self.universe_dir, self.manifest_dir):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            frame.write_parquet(temporary, compression="zstd", statistics=True)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def atomic_json(payload: Mapping[str, Any], path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _normalise_dates(frame: pl.DataFrame) -> pl.DataFrame:
        dtype = frame.schema.get("trade_date")
        if dtype == pl.Date:
            return frame
        if dtype == pl.Utf8:
            return frame.with_columns(pl.col("trade_date").str.to_date(strict=False))
        return frame.with_columns(pl.col("trade_date").cast(pl.Date))

    def upsert_bars(self, bars: pl.DataFrame) -> int:
        if bars.is_empty():
            return 0
        required = {"symbol", "trade_date", "open", "high", "low", "close", "volume"}
        missing = required - set(bars.columns)
        if missing:
            raise ValueError(f"bars missing columns: {sorted(missing)}")
        incoming = self._normalise_dates(bars).with_columns(
            pl.col("trade_date").dt.year().alias("_year")
        )
        written = 0
        for year in incoming.get_column("_year").unique().sort().to_list():
            target = self.raw_dir / f"year={year}" / "bars.parquet"
            chunk = incoming.filter(pl.col("_year") == year).drop("_year")
            if target.exists():
                chunk = pl.concat([pl.read_parquet(target), chunk], how="diagonal_relaxed")
            sort_columns = ["symbol", "trade_date"]
            if "ingested_at" in chunk.columns:
                sort_columns.append("ingested_at")
            chunk = chunk.sort(sort_columns).unique(
                subset=["symbol", "trade_date"], keep="last", maintain_order=True
            )
            chunk = chunk.sort(["trade_date", "symbol"])
            self._atomic_parquet(chunk, target)
            written += chunk.height
        return written

    def scan_bars(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> pl.LazyFrame:
        files = sorted(self.raw_dir.glob("year=*/bars.parquet"))
        if start_date:
            files = [p for p in files if int(p.parent.name.split("=")[1]) >= int(start_date[:4])]
        if end_date:
            files = [p for p in files if int(p.parent.name.split("=")[1]) <= int(end_date[:4])]
        if not files:
            return pl.DataFrame().lazy()
        lazy = pl.scan_parquet([str(path) for path in files])
        if start_date:
            lazy = lazy.filter(pl.col("trade_date") >= pl.lit(start_date).str.to_date())
        if end_date:
            lazy = lazy.filter(pl.col("trade_date") <= pl.lit(end_date).str.to_date())
        if symbols is not None:
            lazy = lazy.filter(pl.col("symbol").is_in(list(symbols)))
        if columns is not None:
            lazy = lazy.select(list(columns))
        return lazy

    def read_bars(self, **kwargs: Any) -> pl.DataFrame:
        return self.scan_bars(**kwargs).collect(engine="streaming")

    def latest_bar_date(self) -> str | None:
        files = sorted(self.raw_dir.glob("year=*/bars.parquet"))
        if not files:
            return None
        value = pl.scan_parquet(str(files[-1])).select(pl.col("trade_date").max()).collect().item()
        return value.isoformat() if value is not None else None

    def bar_years(self) -> list[int]:
        return sorted(
            int(path.parent.name.split("=")[1]) for path in self.raw_dir.glob("year=*/bars.parquet")
        )

    def write_universe_snapshot(self, catalog: pl.DataFrame, asof_date: str) -> Path:
        target = self.universe_dir / f"asof={asof_date}.parquet"
        snapshot = catalog.with_columns(
            pl.lit(asof_date).str.to_date().alias("snapshot_date"),
            pl.lit(asof_date).str.to_date().alias("information_date"),
        )
        self._atomic_parquet(snapshot.sort("symbol"), target)
        return target

    def universe_snapshot_dates(self) -> list[str]:
        return sorted(
            path.stem.split("=", 1)[1] for path in self.universe_dir.glob("asof=*.parquet")
        )

    def read_universe_asof(self, asof_date: str, strict: bool = True) -> pl.DataFrame:
        eligible = [date for date in self.universe_snapshot_dates() if date <= asof_date]
        if not eligible:
            if strict:
                raise ValueError(
                    f"no universe snapshot existed by {asof_date}; current membership would leak"
                )
            return pl.DataFrame()
        return pl.read_parquet(self.universe_dir / f"asof={eligible[-1]}.parquet")

    def write_derived_year(self, name: str, year: int, frame: pl.DataFrame) -> Path:
        target = self.derived_dir / name / f"year={year}" / f"{name}.parquet"
        self._atomic_parquet(frame, target)
        return target

    def scan_derived(self, name: str, columns: Iterable[str] | None = None) -> pl.LazyFrame:
        files = sorted((self.derived_dir / name).glob("year=*/*.parquet"))
        if not files:
            return pl.DataFrame().lazy()
        lazy = pl.scan_parquet([str(path) for path in files])
        return lazy.select(list(columns)) if columns is not None else lazy

    def write_manifest(self, name: str, payload: Mapping[str, Any]) -> Path:
        target = self.manifest_dir / f"{name}.json"
        self.atomic_json(payload, target)
        return target
