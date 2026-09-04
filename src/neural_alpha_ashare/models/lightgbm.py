from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from ..config import LightGBMConfig
from .base import ModelMetadata, add_ranked_alpha


def _finite_matrix(frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    matrix = frame.select(features).to_numpy().astype(np.float32, copy=False)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


class LightGBMAlphaModel:
    def __init__(
        self,
        config: LightGBMConfig,
        cpu_threads: int = 6,
        seed: int = 20260905,
        max_train_rows: int = 1_800_000,
        max_validation_rows: int = 400_000,
    ) -> None:
        self.config = config
        self.cpu_threads = cpu_threads
        self.seed = seed
        self.max_train_rows = max_train_rows
        self.max_validation_rows = max_validation_rows
        self.models: dict[int, Any] = {}
        self.metadata: ModelMetadata | None = None

    def _parameters(self) -> dict[str, Any]:
        values = vars(self.config).copy()
        values.pop("early_stopping_rounds")
        parameters = {
            **values,
            "n_jobs": self.cpu_threads,
            "random_state": self.seed,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        if self.config.subsample < 1.0:
            parameters["subsample_freq"] = 1
        return parameters

    @staticmethod
    def _bounded(frame: pl.DataFrame, maximum: int) -> pl.DataFrame:
        if frame.height <= maximum:
            return frame
        # Chronology is preserved; sampling never creates a random time split.
        return frame.sort(["trade_date", "symbol"]).tail(maximum)

    def fit(
        self,
        train: pl.DataFrame,
        validation: pl.DataFrame,
        feature_names: list[str],
        horizons: tuple[int, ...],
        weights: tuple[float, ...],
        data_signature: str,
    ) -> LightGBMAlphaModel:
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("LightGBM is missing; run pip install -e .") from exc
        for horizon in horizons:
            target = f"target_{horizon}"
            train_h = self._bounded(train.filter(pl.col(target).is_not_null()), self.max_train_rows)
            validation_h = self._bounded(
                validation.filter(pl.col(target).is_not_null()), self.max_validation_rows
            )
            if train_h.is_empty() or validation_h.is_empty():
                raise ValueError(f"insufficient mature rows for horizon {horizon}")
            model = lgb.LGBMRegressor(**self._parameters())
            model.fit(
                _finite_matrix(train_h, feature_names),
                train_h[target].to_numpy(),
                eval_X=_finite_matrix(validation_h, feature_names),
                eval_y=validation_h[target].to_numpy(),
                eval_metric="l2",
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            self.models[horizon] = model
        self.metadata = ModelMetadata(
            model_type="lightgbm",
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
        matrix = _finite_matrix(frame, list(self.metadata.feature_names))
        metadata_columns = [
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
        output = frame.select(metadata_columns)
        prediction_columns: list[str] = []
        for horizon in self.metadata.horizons:
            column = f"prediction_{horizon}"
            prediction = self.models[horizon].predict(matrix).astype(np.float32, copy=False)
            output = output.with_columns(pl.Series(column, prediction))
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
    def load(path: str | Path) -> LightGBMAlphaModel:
        model = joblib.load(path)
        if not isinstance(model, LightGBMAlphaModel):
            raise TypeError("artifact is not a LightGBMAlphaModel")
        return model
