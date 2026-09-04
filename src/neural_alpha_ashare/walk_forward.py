from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from .config import WalkForwardConfig


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date


def make_folds(
    trading_dates: Sequence[date],
    config: WalkForwardConfig,
    rolling_train_sessions: int | None = None,
) -> list[WalkForwardFold]:
    dates = sorted(set(trading_dates))
    train_end = config.initial_train_sessions
    folds: list[WalkForwardFold] = []
    while True:
        validation_start = train_end + config.purge_sessions
        validation_end = validation_start + config.validation_sessions
        test_start = validation_end + config.embargo_sessions
        test_end = test_start + config.test_sessions
        if test_end > len(dates):
            break
        train_start = max(0, train_end - rolling_train_sessions) if rolling_train_sessions else 0
        folds.append(
            WalkForwardFold(
                fold=len(folds),
                train_start=dates[train_start],
                train_end=dates[train_end - 1],
                validation_start=dates[validation_start],
                validation_end=dates[validation_end - 1],
                test_start=dates[test_start],
                test_end=dates[test_end - 1],
            )
        )
        train_end += config.step_sessions
    return folds


def assert_fold_isolation(fold: WalkForwardFold) -> None:
    if not (
        fold.train_start
        <= fold.train_end
        < fold.validation_start
        <= fold.validation_end
        < fold.test_start
        <= fold.test_end
    ):
        raise ValueError(f"invalid walk-forward chronology: {fold}")
