from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import polars as pl


@dataclass(frozen=True)
class ModelMetadata:
    model_type: str
    created_at: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    feature_names: tuple[str, ...]
    horizons: tuple[int, ...]
    weights: tuple[float, ...]
    parameters: dict[str, Any]
    data_signature: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlphaModel(Protocol):
    metadata: ModelMetadata | None

    def fit(
        self,
        train: pl.DataFrame,
        validation: pl.DataFrame,
        feature_names: list[str],
        horizons: tuple[int, ...],
        weights: tuple[float, ...],
        data_signature: str,
    ) -> AlphaModel: ...

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    def save(self, path: str) -> None: ...


def add_ranked_alpha(
    frame: pl.DataFrame, prediction_columns: list[str], weights: tuple[float, ...]
) -> pl.DataFrame:
    if len(prediction_columns) != len(weights):
        raise ValueError("prediction columns and weights must have equal length")
    ranked = frame.with_columns(
        [
            (
                (pl.col(column).rank(method="average", descending=False).over("trade_date") - 1)
                / (pl.col(column).count().over("trade_date") - 1).clip(lower_bound=1)
            )
            .cast(pl.Float32)
            .alias(column.replace("prediction", "rank"))
            for column in prediction_columns
        ]
    )
    rank_columns = [column.replace("prediction", "rank") for column in prediction_columns]
    return ranked.with_columns(
        pl.sum_horizontal(
            [
                pl.col(column) * float(weight)
                for column, weight in zip(rank_columns, weights, strict=True)
            ]
        )
        .cast(pl.Float32)
        .alias("model_alpha")
    ).with_columns(
        pl.col("model_alpha")
        .rank(method="ordinal", descending=True)
        .over("trade_date")
        .cast(pl.Int32)
        .alias("model_rank")
    )
