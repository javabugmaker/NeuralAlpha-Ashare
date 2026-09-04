from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from ..config import CatBoostConfig
from .base import ModelMetadata, add_ranked_alpha


def _matrix(frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    values = frame.select(features).to_numpy().astype(np.float32, copy=False)
    values[~np.isfinite(values)] = np.nan
    return values


class CatBoostGPUAlphaModel:
    """Optional RTX challenger; the default pipeline never requires CUDA."""

    def __init__(
        self,
        config: CatBoostConfig,
        seed: int = 20260905,
        max_train_rows: int = 1_200_000,
        max_validation_rows: int = 300_000,
    ) -> None:
        self.config = config
        self.seed = seed
        self.max_train_rows = max_train_rows
        self.max_validation_rows = max_validation_rows
        self.models: dict[int, Any] = {}
        self.metadata: ModelMetadata | None = None

    def _parameters(self) -> dict[str, Any]:
        return {
            "iterations": self.config.iterations,
            "depth": self.config.depth,
            "learning_rate": self.config.learning_rate,
            "l2_leaf_reg": self.config.l2_leaf_reg,
            "random_strength": self.config.random_strength,
            "border_count": self.config.border_count,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "task_type": "GPU",
            "devices": self.config.devices,
            "random_seed": self.seed,
            "allow_writing_files": False,
            "verbose": False,
        }

    @staticmethod
    def _bounded(frame: pl.DataFrame, maximum: int) -> pl.DataFrame:
        return frame if frame.height <= maximum else frame.tail(maximum)

    def fit(
        self,
        train: pl.DataFrame,
        validation: pl.DataFrame,
        feature_names: list[str],
        horizons: tuple[int, ...],
        weights: tuple[float, ...],
        data_signature: str,
    ) -> CatBoostGPUAlphaModel:
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('CatBoost is missing; run pip install -e ".[gpu]"') from exc
        for horizon in horizons:
            target = f"target_{horizon}"
            train_h = self._bounded(train.filter(pl.col(target).is_not_null()), self.max_train_rows)
            validation_h = self._bounded(
                validation.filter(pl.col(target).is_not_null()), self.max_validation_rows
            )
            if train_h.is_empty() or validation_h.is_empty():
                raise ValueError(f"insufficient mature rows for horizon {horizon}")
            model = CatBoostRegressor(**self._parameters())
            model.fit(
                _matrix(train_h, feature_names),
                train_h[target].to_numpy(),
                eval_set=(
                    _matrix(validation_h, feature_names),
                    validation_h[target].to_numpy(),
                ),
                use_best_model=True,
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose=False,
            )
            self.models[horizon] = model
        self.metadata = ModelMetadata(
            model_type="catboost-gpu",
            created_at=datetime.now(UTC).isoformat(),
            train_start=str(train["trade_date"].min()),
            train_end=str(train["trade_date"].max()),
            validation_start=str(validation["trade_date"].min()),
            validation_end=str(validation["trade_date"].max()),
            feature_names=tuple(feature_names),
            horizons=horizons,
            weights=weights,
            parameters=self._parameters(),
            data_signature=data_signature,
        )
        return self

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self.metadata is None:
            raise RuntimeError("model is not fitted")
        values = _matrix(frame, list(self.metadata.feature_names))
        output = frame.select(
            [
                column
                for column in (
                    "trade_date",
                    "symbol",
                    "eligible",
                    "median_amount_20",
                    "instrument_type",
                )
                if column in frame.columns
            ]
        )
        prediction_columns = []
        for horizon in self.metadata.horizons:
            column = f"prediction_{horizon}"
            output = output.with_columns(
                pl.Series(column, self.models[horizon].predict(values).astype(np.float32))
            )
            prediction_columns.append(column)
        return add_ranked_alpha(output, prediction_columns, self.metadata.weights)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
        os.close(fd)
        try:
            joblib.dump(self, temporary, compress=3)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def load(path: str | Path) -> CatBoostGPUAlphaModel:
        model = joblib.load(path)
        if not isinstance(model, CatBoostGPUAlphaModel):
            raise TypeError("artifact is not a CatBoostGPUAlphaModel")
        return model
