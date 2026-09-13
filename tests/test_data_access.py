# ============================================================
# Unit Tests — Unified Data Access Layer (Phase 6)
# ============================================================

from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline.data_access import DataAccessLayer, get_data_access
from src.data_pipeline.storage_backend import ParquetBackend


@pytest.fixture
def sample_ohlcv_df():
    dates = pd.date_range("2021-01-01", "2021-01-10", freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * len(dates),
            "high": [105.0] * len(dates),
            "low": [95.0] * len(dates),
            "close": [102.0] * len(dates),
            "volume": [1_000_000] * len(dates),
            "ticker": "TEST",
        }
    )


def test_data_access_save_and_get_ohlcv(tmp_path: Path, sample_ohlcv_df):
    backend = ParquetBackend(root_dir=tmp_path)
    dal = DataAccessLayer(backend=backend)

    # Save
    dal.save_ohlcv("TEST", sample_ohlcv_df)

    # List
    assert "TEST" in dal.list_available_tickers()

    # Query all
    df_all = dal.get_ohlcv("TEST")
    assert len(df_all) == len(sample_ohlcv_df)

    # Query range
    df_slice = dal.get_ohlcv("TEST", start="2021-01-03", end="2021-01-07")
    assert len(df_slice) == 5

    # Query columns
    df_cols = dal.get_ohlcv("TEST", columns=["date", "close"])
    assert list(df_cols.columns) == ["date", "close"]


def test_data_access_universe_integration():
    dal = DataAccessLayer()
    universe_2015 = dal.get_universe("2015-01-01")
    assert "YHOO" in universe_2015
    assert "AAPL" in universe_2015

    constituents = dal.get_constituents("2015-01-01")
    assert any(c.ticker == "YHOO" and c.is_delisted for c in constituents)


def test_data_access_delisted_info_integration():
    dal = DataAccessLayer()
    info = dal.get_delisted_info("YHOO")
    assert info is not None
    assert info.ticker == "YHOO"
    assert info.reason == "acquired"


def test_data_access_corporate_actions_integration(tmp_path: Path):
    dates = pd.date_range("2021-01-01", periods=4, freq="D")
    df_split = pd.DataFrame(
        {
            "date": dates,
            "close": [200.0, 200.0, 100.0, 100.0],  # 2:1 split
            "open": [199.0, 199.0, 99.0, 99.0],
            "volume": [1_000_000, 1_000_000, 2_000_000, 2_000_000],
        }
    )
    backend = ParquetBackend(root_dir=tmp_path)
    dal = DataAccessLayer(backend=backend)
    dal.save_ohlcv("SPLIT_TEST", df_split)

    actions = dal.get_corporate_actions("SPLIT_TEST", use_authoritative=False)
    assert len(actions) == 1
    assert actions[0].ratio == 2.0
    assert actions[0].ratio_str == "2:1"


def test_get_data_access_singleton():
    dal1 = get_data_access()
    dal2 = get_data_access()
    assert dal1 is dal2
