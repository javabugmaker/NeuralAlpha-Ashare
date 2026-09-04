from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import ModelMetadata, add_ranked_alpha


class RidgeAlphaModel:
    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.models: dict[int, Pipeline] = {}
        self.metadata: ModelMetadata | None = None

    def fit(
        self,
        train: pl.DataFrame,
        validation: pl.DataFrame,
        feature_names: list[str],
        horizons: tuple[int, ...],
        weights: tuple[float, ...],
        data_signature: str,
    ) -> RidgeAlphaModel:
        del validation
        for horizon in horizons:
            subset = train.filter(pl.col(f"target_{horizon}").is_not_null())
            x = subset.select(feature_names).to_numpy().astype(np.float32, copy=False)
            y = subset[f"target_{horizon}"].to_numpy().astype(np.float32, copy=False)
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=self.alpha)),
                ]
            )
            model.fit(x, y)
            self.models[horizon] = model
        self.metadata = ModelMetadata(
            model_type="ridge",
            created_at=datetime.now(UTC).isoformat(),
            train_start=str(train["trade_date"].min()),
            train_end=str(train["trade_date"].max()),
            validation_start="",
            validation_end="",
            feature_names=tuple(feature_names),
            horizons=horizons,
            weights=weights,
            parameters={"alpha": self.alpha},
            data_signature=data_signature,
        )
        return self

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self.metadata is None:
            raise RuntimeError("model is not fitted")
        x = frame.select(self.metadata.feature_names).to_numpy().astype(np.float32, copy=False)
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
        columns: list[str] = []
        for horizon in self.metadata.horizons:
            column = f"prediction_{horizon}"
            output = output.with_columns(
                pl.Series(column, self.models[horizon].predict(x).astype(np.float32))
            )
            columns.append(column)
        return add_ranked_alpha(output, columns, self.metadata.weights)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
        os.close(fd)
        try:
            joblib.dump(self, temporary)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def load(path: str | Path) -> RidgeAlphaModel:
        model = joblib.load(path)
        if not isinstance(model, RidgeAlphaModel):
            raise TypeError("artifact is not a RidgeAlphaModel")
        return model
