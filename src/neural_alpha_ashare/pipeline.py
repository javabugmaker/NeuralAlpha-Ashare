from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import polars as pl
import psutil

from .backtest import BacktestResult, run_backtest
from .config import AppConfig
from .data.pit import reconstruct_pit_prices
from .data.store import ParquetStore
from .data.tickflow import TickFlowFreeClient
from .features import build_features, feature_columns
from .labels import build_labels
from .market import build_eligibility
from .metrics import multi_horizon_metrics
from .models.catboost import CatBoostGPUAlphaModel
from .models.lightgbm import LightGBMAlphaModel
from .models.registry import ModelRegistry
from .models.ridge import RidgeAlphaModel
from .reports import write_reports
from .walk_forward import assert_fold_isolation, make_folds

LOGGER = logging.getLogger("neural_alpha_ashare")


class ResearchPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        config.ensure_directories()
        self.store = ParquetStore(config.paths.raw_dir, config.paths.derived_dir)
        self.registry = ModelRegistry(config.paths.models_dir)

    def doctor(self) -> dict[str, object]:
        memory_gb = psutil.virtual_memory().total / 1024**3
        status: dict[str, object] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "logical_cpus": psutil.cpu_count(),
            "memory_gb": round(memory_gb, 2),
            "memory_soft_limit_gb": self.config.runtime.memory_soft_limit_gb,
            "latest_bar_date": self.store.latest_bar_date(),
            "raw_years": self.store.bar_years(),
            "universe_snapshots": len(self.store.universe_snapshot_dates()),
        }
        for package in ("polars", "pyarrow", "lightgbm", "tickflow", "numba", "catboost"):
            try:
                module = __import__(package)
                status[package] = getattr(module, "__version__", "installed")
            except ImportError:
                status[package] = "MISSING"
        status["memory_ok"] = memory_gb >= self.config.runtime.memory_soft_limit_gb
        return status

    def update(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, object]:
        latest = self.store.latest_bar_date()
        if start_date is None:
            start_date = (
                (
                    date.fromisoformat(latest)
                    - timedelta(days=self.config.tickflow.overlap_calendar_days)
                ).isoformat()
                if latest
                else self.config.tickflow.history_start
            )
        with TickFlowFreeClient(
            self.config.tickflow, cache_dir=self.config.paths.cache_dir
        ) as client:
            result = client.update(start_date=start_date, end_date=end_date)
        written = self.store.upsert_bars(result.bars)
        observed_date = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        self.store.write_universe_snapshot(result.catalog, observed_date)
        manifest = {
            "service": "TickFlow.free()",
            "adjust": "none",
            "start_date": start_date,
            "end_date": end_date,
            "catalog_observed_date": observed_date,
            "symbols_requested": result.symbols_requested,
            "symbols_received": result.symbols_received,
            "rows_received": result.rows_received,
            "rows_in_partitions": written,
            "latest_date": result.latest_date,
        }
        self.store.write_manifest("latest_update", manifest)
        return manifest

    @staticmethod
    def _infer_type() -> pl.Expr:
        code = pl.col("symbol").str.split(".").list.first()
        stock = code.str.contains(r"^(600|601|603|605|000|001|002|003|300|301|688|689|4|8|920)")
        fund = code.str.contains(r"^(5|15|16)")
        return (
            pl.when(stock)
            .then(pl.lit("stock"))
            .when(fund)
            .then(pl.lit("fund"))
            .otherwise(pl.lit("unknown"))
        )

    def _attach_universe(self, bars: pl.DataFrame) -> tuple[pl.DataFrame, str]:
        snapshot_files = sorted(self.store.universe_dir.glob("asof=*.parquet"))
        if not snapshot_files:
            enriched = bars.with_columns(
                pl.lit("").alias("name"), self._infer_type().alias("instrument_type")
            )
            return enriched, "DEGRADED"
        snapshots = pl.concat(
            [
                pl.read_parquet(path).select(
                    "symbol", "snapshot_date", "name", "instrument_type", "listing_date"
                )
                for path in snapshot_files
            ],
            how="diagonal_relaxed",
        ).unique(["symbol", "snapshot_date"], keep="last")
        enriched = bars.sort(["symbol", "trade_date"]).join_asof(
            snapshots.sort(["symbol", "snapshot_date"]),
            left_on="trade_date",
            right_on="snapshot_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        first_snapshot = min(self.store.universe_snapshot_dates())
        quality = "PIT" if str(bars["trade_date"].min()) >= first_snapshot else "DEGRADED"
        enriched = enriched.with_columns(
            pl.col("name").fill_null(""),
            pl.col("instrument_type").fill_null(self._infer_type()),
        )
        return enriched, quality

    def build(self, year: int | None = None) -> dict[str, object]:
        years = [year] if year is not None else self.store.bar_years()
        if not years:
            raise FileNotFoundError("no raw bars; run alpha-ashare update first")
        built: list[dict[str, object]] = []
        max_horizon = max(self.config.labels.horizons)
        for target_year in years:
            start = date(target_year, 1, 1) - timedelta(days=400)
            end = date(target_year, 12, 31) + timedelta(days=max_horizon * 2 + 20)
            bars = self.store.read_bars(start_date=start.isoformat(), end_date=end.isoformat())
            if bars.is_empty():
                continue
            bars = bars.filter(
                pl.col("prev_close").is_not_null()
                & (pl.col("prev_close") > 0)
                & (pl.col("close") > 0)
            )
            enriched, universe_quality = self._attach_universe(bars)
            enriched = enriched.filter(
                (pl.col("symbol") != self.config.tickflow.benchmark)
                & (pl.col("instrument_type") == "stock")
            )
            pit = reconstruct_pit_prices(enriched)
            eligible = build_eligibility(pit, self.config.data)
            featured = build_features(eligible, self.config.features)
            labelled = build_labels(featured, self.config.labels).with_columns(
                (
                    pl.col("eligible")
                    & pl.col("feature_row_valid")
                    & (pl.col("feature_coverage") >= self.config.data.min_feature_coverage)
                ).alias("eligible"),
                pl.lit(universe_quality).alias("universe_quality"),
            )
            partition = labelled.filter(pl.col("trade_date").dt.year() == target_year).sort(
                ["trade_date", "symbol"]
            )
            if partition.is_empty():
                continue
            self.store.write_derived_year("research", target_year, partition)
            info = {
                "year": target_year,
                "rows": partition.height,
                "symbols": partition["symbol"].n_unique(),
                "features": len(feature_columns(partition)),
                "universe_quality": universe_quality,
                "latest_date": str(partition["trade_date"].max()),
            }
            self.store.write_manifest(f"build_{target_year}", info)
            built.append(info)
        return {"partitions": built}

    def _research_scan(self) -> pl.LazyFrame:
        scan = self.store.scan_derived("research")
        if not scan.collect_schema().names():
            raise FileNotFoundError("no research data; run alpha-ashare build")
        return scan

    def _research(self, start: date | None = None, end: date | None = None) -> pl.DataFrame:
        scan = self._research_scan()
        if start is not None:
            scan = scan.filter(pl.col("trade_date") >= start)
        if end is not None:
            scan = scan.filter(pl.col("trade_date") <= end)
        frame = scan.collect(engine="streaming")
        if frame.is_empty():
            raise FileNotFoundError("no research data; run alpha-ashare build")
        return frame.sort(["trade_date", "symbol"])

    def _research_dates(self, scan: pl.LazyFrame) -> list[date]:
        return (
            scan.filter(pl.col("eligible"))
            .select("trade_date")
            .unique()
            .sort("trade_date")
            .collect(engine="streaming")["trade_date"]
            .to_list()
        )

    def _model_columns(self, features: list[str]) -> list[str]:
        horizons = self.config.labels.horizons
        return [
            "trade_date",
            "symbol",
            "eligible",
            "median_amount_20",
            "instrument_type",
            "universe_quality",
            *features,
            *[f"target_{horizon}" for horizon in horizons],
            *[f"label_available_date_{horizon}" for horizon in horizons],
        ]

    def _signature(self, frame: pl.DataFrame, features: list[str]) -> str:
        payload = {
            "rows": frame.height,
            "start": str(frame["trade_date"].min()),
            "end": str(frame["trade_date"].max()),
            "features": features,
            "horizons": self.config.labels.horizons,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def _new_model(self, model_name: str):
        if model_name == "ridge":
            return RidgeAlphaModel(self.config.model.ridge_alpha)
        if model_name == "catboost-gpu":
            return CatBoostGPUAlphaModel(
                self.config.model.catboost,
                seed=self.config.runtime.seed,
                max_train_rows=min(self.config.model.max_train_rows, 1_200_000),
                max_validation_rows=min(self.config.model.max_validation_rows, 300_000),
            )
        if model_name != "lightgbm":
            raise ValueError("model must be lightgbm, ridge, or catboost-gpu")
        return LightGBMAlphaModel(
            self.config.model.lightgbm,
            cpu_threads=self.config.runtime.cpu_threads,
            seed=self.config.runtime.seed,
            max_train_rows=self.config.model.max_train_rows,
            max_validation_rows=self.config.model.max_validation_rows,
        )

    def train(self, model_name: str | None = None) -> dict[str, object]:
        name = model_name or self.config.model.primary
        scan = self._research_scan()
        features = feature_columns(scan)
        dates = self._research_dates(scan)
        validation_size = self.config.walk_forward.validation_sessions
        purge = self.config.walk_forward.purge_sessions
        if len(dates) <= validation_size + purge + 252:
            raise ValueError("not enough trading sessions for chronological train/validation")
        validation_start_index = len(dates) - validation_size
        train_end_index = validation_start_index - purge
        rolling = self.config.model.rolling_train_years * 252
        train_start_index = max(0, train_end_index - rolling)
        columns = self._model_columns(features)
        train_scan = scan.filter(pl.col("eligible")).filter(
            pl.col("trade_date").is_between(
                dates[train_start_index], dates[train_end_index - 1], closed="both"
            )
        )
        validation_scan = scan.filter(pl.col("eligible")).filter(
            pl.col("trade_date") >= dates[validation_start_index]
        )
        asof = dates[-1]
        for horizon in self.config.labels.horizons:
            train_scan = train_scan.filter(
                pl.col(f"label_available_date_{horizon}") <= dates[train_end_index - 1]
            )
            validation_scan = validation_scan.filter(
                pl.col(f"label_available_date_{horizon}") <= asof
            )
        train = (
            train_scan.select(columns)
            .tail(self.config.model.max_train_rows)
            .collect(engine="streaming")
        )
        validation = (
            validation_scan.select(columns)
            .tail(self.config.model.max_validation_rows)
            .collect(engine="streaming")
        )
        signature = self._signature(train, features)
        model = self._new_model(name).fit(
            train,
            validation,
            features,
            self.config.labels.horizons,
            self.config.labels.weights,
            signature,
        )
        predictions = model.predict(validation).join(
            validation.select(
                "trade_date",
                "symbol",
                *[f"target_{horizon}" for horizon in self.config.labels.horizons],
            ),
            on=["trade_date", "symbol"],
            how="left",
        )
        metrics = multi_horizon_metrics(predictions, self.config.labels.horizons)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact = self.config.paths.models_dir / f"{name}_{stamp}_{signature}.joblib"
        model.save(artifact)
        audit_status = sorted(train["universe_quality"].unique().to_list())
        metadata = model.metadata.to_dict()
        metadata["universe_quality"] = audit_status
        role = self.registry.register(artifact, metadata, metrics)
        return {
            "artifact": str(artifact),
            "role": role,
            "universe_quality": audit_status,
            "metrics": metrics,
        }

    def walk_forward(self, model_name: str | None = None) -> dict[str, object]:
        name = model_name or self.config.model.primary
        scan = self._research_scan()
        features = feature_columns(scan)
        columns = self._model_columns(features)
        dates = self._research_dates(scan)
        folds = make_folds(
            dates,
            self.config.walk_forward,
            rolling_train_sessions=self.config.model.rolling_train_years * 252,
        )
        if not folds:
            raise ValueError("not enough dates for a walk-forward fold")
        cache_paths: list[Path] = []
        fold_metrics: list[dict[str, object]] = []
        cache_dir = self.config.paths.predictions_dir / "walk_forward" / name
        cache_dir.mkdir(parents=True, exist_ok=True)
        for fold in folds:
            assert_fold_isolation(fold)
            base_signature = hashlib.sha256(
                json.dumps(
                    {
                        "start": str(dates[0]),
                        "end": str(dates[-1]),
                        "features": features,
                        "model": name,
                        "model_config": str(self.config.model),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:16]
            signature = hashlib.sha256(f"{base_signature}:{fold}".encode()).hexdigest()[:16]
            cache = cache_dir / f"fold_{fold.fold}_{signature}.parquet"
            if cache.exists():
                predicted = pl.read_parquet(cache)
            else:
                train_scan = scan.filter(pl.col("eligible")).filter(
                    pl.col("trade_date").is_between(fold.train_start, fold.train_end, closed="both")
                )
                validation_scan = scan.filter(pl.col("eligible")).filter(
                    pl.col("trade_date").is_between(
                        fold.validation_start, fold.validation_end, closed="both"
                    )
                )
                test_scan = scan.filter(pl.col("eligible")).filter(
                    pl.col("trade_date").is_between(fold.test_start, fold.test_end, closed="both")
                )
                for horizon in self.config.labels.horizons:
                    train_scan = train_scan.filter(
                        pl.col(f"label_available_date_{horizon}") <= fold.train_end
                    )
                    validation_scan = validation_scan.filter(
                        pl.col(f"label_available_date_{horizon}") <= fold.validation_end
                    )
                train = (
                    train_scan.select(columns)
                    .tail(self.config.model.max_train_rows)
                    .collect(engine="streaming")
                )
                validation = (
                    validation_scan.select(columns)
                    .tail(self.config.model.max_validation_rows)
                    .collect(engine="streaming")
                )
                test = test_scan.select(columns).collect(engine="streaming")
                model = self._new_model(name).fit(
                    train,
                    validation,
                    features,
                    self.config.labels.horizons,
                    self.config.labels.weights,
                    signature,
                )
                predicted = (
                    model.predict(test)
                    .join(
                        test.select(
                            "trade_date",
                            "symbol",
                            *[f"target_{h}" for h in self.config.labels.horizons],
                        ),
                        on=["trade_date", "symbol"],
                        how="left",
                    )
                    .with_columns(pl.lit(fold.fold).alias("fold"))
                )
                predicted.sort(["trade_date", "symbol"]).write_parquet(cache, compression="zstd")
            cache_paths.append(cache)
            metrics = multi_horizon_metrics(predicted, self.config.labels.horizons)
            fold_metrics.append({"fold": fold.fold, **metrics})
        output = self.config.paths.predictions_dir / "walk_forward_predictions.parquet"
        pl.concat([pl.scan_parquet(path) for path in cache_paths]).sink_parquet(
            output, compression="zstd"
        )
        metric_frame = (
            pl.scan_parquet(output)
            .select(
                "trade_date",
                "model_alpha",
                *[f"prediction_{horizon}" for horizon in self.config.labels.horizons],
                *[f"target_{horizon}" for horizon in self.config.labels.horizons],
            )
            .collect(engine="streaming")
        )
        summary = {
            "model": name,
            "folds": fold_metrics,
            "overall": multi_horizon_metrics(metric_frame, self.config.labels.horizons),
            "universe_quality": self._aggregate_universe_quality(),
        }
        self.store.atomic_json(
            summary, self.config.paths.predictions_dir / "walk_forward_metrics.json"
        )
        return {"predictions": str(output), **summary}

    def backtest(
        self, predictions_path: str | Path | None = None, capital: float | None = None
    ) -> dict[str, object]:
        source = Path(
            predictions_path
            or self.config.paths.predictions_dir / "walk_forward_predictions.parquet"
        )
        if not source.exists():
            raise FileNotFoundError(
                "walk-forward predictions missing; run alpha-ashare walk-forward"
            )
        predictions = pl.read_parquet(source)
        start = str(predictions["trade_date"].min())
        end = str(predictions["trade_date"].max() + timedelta(days=10))
        raw_market = self.store.read_bars(start_date=start, end_date=end)
        benchmark = raw_market.filter(pl.col("symbol") == self.config.tickflow.benchmark).sort(
            "trade_date"
        )
        benchmark_return: float | None = (
            float(benchmark["close"][-1] / benchmark["close"][0] - 1.0)
            if benchmark.height >= 2
            else None
        )
        enriched, _ = self._attach_universe(raw_market)
        market = build_eligibility(reconstruct_pit_prices(enriched), self.config.data)
        capitals = (
            [capital]
            if capital is not None
            else [
                self.config.portfolio.initial_capital,
                self.config.portfolio.research_capital,
            ]
        )
        summaries: dict[str, object] = {}
        for amount in capitals:
            result: BacktestResult = run_backtest(
                market, predictions, self.config.portfolio, initial_capital=amount
            )
            if benchmark_return is not None:
                result.metrics["benchmark_total_return"] = benchmark_return
                result.metrics["excess_total_return"] = (
                    result.metrics["total_return"] - benchmark_return
                )
            label = str(int(amount))
            directory = self.config.paths.backtests_dir / label
            directory.mkdir(parents=True, exist_ok=True)
            result.nav.write_parquet(directory / "nav.parquet")
            result.trades.write_parquet(directory / "trades.parquet")
            result.holdings.write_parquet(directory / "holdings.parquet")
            self.store.atomic_json(result.metrics, directory / "metrics.json")
            summaries[label] = result.metrics
        return summaries

    def _load_champion(self):
        artifact = self.registry.champion_artifact()
        return joblib.load(artifact)

    def daily(self, skip_update: bool = False) -> dict[str, object]:
        if not skip_update:
            self.update()
        latest = self.store.latest_bar_date()
        if latest is None:
            raise FileNotFoundError("no market data")
        self.build(date.fromisoformat(latest).year)
        latest_date = date.fromisoformat(latest)
        latest_frame = self._research(latest_date, latest_date).filter(pl.col("eligible"))
        model = self._load_champion()
        predictions = model.predict(latest_frame)
        output = self.config.paths.predictions_dir / f"daily_{latest}.parquet"
        predictions.write_parquet(output, compression="zstd")
        quality = self._latest_universe_quality()
        write_reports(
            self.config.paths.docs_dir,
            self.config.reports.title,
            latest,
            predictions=predictions,
            metrics=self._latest_backtest_metrics(),
            universe_quality=quality,
        )
        return {"date": latest, "predictions": str(output), "rows": predictions.height}

    def weekly(self) -> dict[str, object]:
        latest = self.store.latest_bar_date() or "无数据"
        daily_files = sorted(self.config.paths.predictions_dir.glob("daily_*.parquet"))
        predictions = pl.read_parquet(daily_files[-1]) if daily_files else pl.DataFrame()
        pages = write_reports(
            self.config.paths.docs_dir,
            self.config.reports.title,
            latest,
            predictions,
            self._latest_backtest_metrics(),
            self._latest_universe_quality(),
        )
        return {"date": latest, "pages": [str(path) for path in pages]}

    def _latest_universe_quality(self) -> str:
        manifests = sorted(self.store.manifest_dir.glob("build_*.json"))
        if not manifests:
            return "UNKNOWN"
        with manifests[-1].open(encoding="utf-8") as handle:
            return json.load(handle).get("universe_quality", "UNKNOWN")

    def _aggregate_universe_quality(self) -> str:
        statuses: list[str] = []
        for path in sorted(self.store.manifest_dir.glob("build_*.json")):
            with path.open(encoding="utf-8") as handle:
                statuses.append(json.load(handle).get("universe_quality", "UNKNOWN"))
        if "DEGRADED" in statuses:
            return "DEGRADED"
        return "PIT" if statuses else "UNKNOWN"

    def _latest_backtest_metrics(self) -> dict[str, float]:
        path = (
            self.config.paths.backtests_dir
            / str(int(self.config.portfolio.initial_capital))
            / "metrics.json"
        )
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
