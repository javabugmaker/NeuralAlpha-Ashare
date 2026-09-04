from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    derived_dir: Path = Path("data/derived")
    cache_dir: Path = Path("data/cache")
    models_dir: Path = Path("models")
    predictions_dir: Path = Path("predictions")
    backtests_dir: Path = Path("backtests")
    logs_dir: Path = Path("logs")
    docs_dir: Path = Path("docs")


@dataclass(frozen=True)
class RuntimeConfig:
    memory_soft_limit_gb: float = 11.5
    cpu_threads: int = 6
    seed: int = 20260905


@dataclass(frozen=True)
class TickFlowConfig:
    exchanges: tuple[str, ...] = ("SH", "SZ", "BJ")
    instrument_types: tuple[str, ...] = ("stock", "etf", "fund")
    benchmark: str = "000300.SH"
    history_start: str = "2005-01-01"
    overlap_calendar_days: int = 21
    max_workers: int = 2
    batch_size: int = 100
    timeout_seconds: int = 45
    max_retries: int = 4
    period: str = "1d"


@dataclass(frozen=True)
class DataConfig:
    strict_pit: bool = True
    strict_survivorship: bool = True
    min_history_sessions: int = 240
    min_feature_coverage: float = 0.95
    min_median_amount_20: float = 20_000_000.0
    allowed_boards: tuple[str, ...] = ("MAIN", "CHINEXT", "STAR")
    include_bse: bool = False
    exclude_st: bool = True
    exclude_one_price_limit: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    max_lookback_sessions: int = 240
    min_cross_section: int = 30


@dataclass(frozen=True)
class LabelConfig:
    horizons: tuple[int, ...] = (5, 20, 60)
    weights: tuple[float, ...] = (0.2, 0.5, 0.3)
    liquidity_buckets: int = 5


@dataclass(frozen=True)
class LightGBMConfig:
    objective: str = "huber"
    learning_rate: float = 0.03
    n_estimators: int = 2000
    num_leaves: int = 31
    max_depth: int = 8
    min_child_samples: int = 1000
    max_bin: int = 63
    subsample: float = 0.8
    colsample_bytree: float = 0.75
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 100


@dataclass(frozen=True)
class CatBoostConfig:
    iterations: int = 3000
    depth: int = 8
    learning_rate: float = 0.03
    l2_leaf_reg: float = 5.0
    random_strength: float = 0.5
    border_count: int = 64
    early_stopping_rounds: int = 100
    devices: str = "0"


@dataclass(frozen=True)
class ModelConfig:
    primary: str = "lightgbm"
    rolling_train_years: int = 8
    max_train_rows: int = 1_800_000
    max_validation_rows: int = 400_000
    ridge_alpha: float = 10.0
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train_sessions: int = 756
    validation_sessions: int = 126
    test_sessions: int = 126
    step_sessions: int = 126
    purge_sessions: int = 60
    embargo_sessions: int = 5


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 200_000.0
    research_capital: float = 1_000_000.0
    entry_rank: int = 15
    exit_rank: int = 60
    rebalance_every_sessions: int = 5
    max_weight: float = 0.10
    max_holding_sessions: int = 60
    lot_size: int = 100
    max_participation: float = 0.02
    slippage_bps: float = 5.0
    impact_coefficient: float = 0.10
    impact_exponent: float = 0.5
    max_impact_bps: float = 30.0
    stock_commission: float = 0.00008499999
    fund_commission: float = 0.00005000001
    minimum_commission: float = 0.0
    stock_sell_stamp_duty: float = 0.0005
    max_exit_wait_sessions: int = 10


@dataclass(frozen=True)
class ReportConfig:
    title: str = "NeuralAlpha A股研究台"
    timezone: str = "Asia/Shanghai"


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tickflow: TickFlowConfig = field(default_factory=TickFlowConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)

    def ensure_directories(self) -> None:
        for path in vars(self.paths).values():
            Path(path).mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if len(self.labels.horizons) != len(self.labels.weights):
            raise ValueError("labels.horizons and labels.weights must have equal length")
        if abs(sum(self.labels.weights) - 1.0) > 1e-8:
            raise ValueError("labels.weights must sum to one")
        if self.walk_forward.purge_sessions < max(self.labels.horizons):
            raise ValueError("purge_sessions must cover the longest label horizon")
        if self.portfolio.entry_rank >= self.portfolio.exit_rank:
            raise ValueError("entry_rank must be smaller than exit_rank for hysteresis")
        if not 0 < self.portfolio.max_weight <= 1:
            raise ValueError("portfolio.max_weight must be in (0, 1]")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _tuple_values(section: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    values = dict(section)
    for key in keys:
        if key in values:
            values[key] = tuple(values[key])
    return values


def _resolve_paths(section: Mapping[str, Any], root: Path) -> PathsConfig:
    resolved: dict[str, Path] = {}
    for key, value in section.items():
        path = Path(value)
        resolved[key] = path if path.is_absolute() else (root / path).resolve()
    return PathsConfig(**resolved)


def load_config(
    path: str | Path = "config/default.yaml", overrides: Mapping[str, Any] | None = None
) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = _deep_merge(yaml.safe_load(handle) or {}, overrides or {})
    root = config_path.parent.parent
    model_raw = dict(raw.get("model", {}))
    lgb_raw = model_raw.pop("lightgbm", {})
    catboost_raw = model_raw.pop("catboost", {})
    config = AppConfig(
        paths=_resolve_paths(raw.get("paths", {}), root),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        tickflow=TickFlowConfig(
            **_tuple_values(raw.get("tickflow", {}), "exchanges", "instrument_types")
        ),
        data=DataConfig(**_tuple_values(raw.get("data", {}), "allowed_boards")),
        features=FeatureConfig(**raw.get("features", {})),
        labels=LabelConfig(**_tuple_values(raw.get("labels", {}), "horizons", "weights")),
        model=ModelConfig(
            **model_raw,
            lightgbm=LightGBMConfig(**lgb_raw),
            catboost=CatBoostConfig(**catboost_raw),
        ),
        walk_forward=WalkForwardConfig(**raw.get("walk_forward", {})),
        portfolio=PortfolioConfig(**raw.get("portfolio", {})),
        reports=ReportConfig(**raw.get("reports", {})),
    )
    config.validate()
    return config
