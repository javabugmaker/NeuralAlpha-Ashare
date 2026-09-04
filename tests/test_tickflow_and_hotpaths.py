from __future__ import annotations

from pathlib import Path

from neural_alpha_ashare.config import TickFlowConfig
from neural_alpha_ashare.data.tickflow import TickFlowFreeClient


class _Exchanges:
    def get_instruments(self, exchange: str, instrument_type: str):
        return [
            {
                "symbol": "600000.SH",
                "code": "600000",
                "name": "浦发银行",
                "exchange": exchange,
                "type": instrument_type,
                "ext": {"listing_date": "1999-11-10"},
            }
        ]


class _Klines:
    def batch(self, symbols, **kwargs):
        assert kwargs["adjust"] == "none"
        return {
            symbol: {
                "timestamp": [1704157200000],
                "open": [10.0],
                "high": [10.2],
                "low": [9.9],
                "close": [10.1],
                "prev_close": [10.0],
                "volume": [100.0],
                "amount": [1000.0],
            }
            for symbol in symbols
        }


class _SDK:
    exchanges = _Exchanges()
    klines = _Klines()


def test_tickflow_boundary_forces_unadjusted_data() -> None:
    config = TickFlowConfig(exchanges=("SH",), instrument_types=("stock",), benchmark="600000.SH")
    result = TickFlowFreeClient(config, sdk=_SDK()).update()
    assert result.rows_received == 1
    assert result.catalog.height == 1


def test_vectorized_hotpaths_have_no_rowwise_pandas() -> None:
    root = Path("src/neural_alpha_ashare")
    for filename in ("features.py", "labels.py", "market.py", "backtest.py"):
        source = (root / filename).read_text(encoding="utf-8")
        assert "iterrows(" not in source
        assert "apply(axis=1" not in source
