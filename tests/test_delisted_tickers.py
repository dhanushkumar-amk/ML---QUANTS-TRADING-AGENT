# ============================================================
# Unit Tests — Delisted Tickers Registry (Phase 5)
# ============================================================

import datetime
from pathlib import Path

import pytest

from src.data_pipeline.delisted_tickers import DelistedInfo, DelistedRegistry


@pytest.fixture
def sample_registry(tmp_path: Path):
    """Create a temporary registry CSV for isolated testing."""
    csv_file = tmp_path / "test_delisted.csv"
    csv_file.write_text(
        "ticker,name,delisting_date,reason,acquirer,last_price,notes\n"
        "TEST_ACQ,Test Acquired Inc.,2018-06-01,acquired,MegaCorp,50.0,Acquired for cash\n"
        "TEST_BNK,Test Bankrupt Corp,2020-03-15,bankruptcy,None,0.10,Chapter 11 liquidation\n"
        "TEST_PVT,Test Private Co,2022-10-01,privatized,Founder,100.0,Taken private\n"
    )
    return DelistedRegistry(data_path=csv_file)


def test_registry_loading(sample_registry):
    assert len(sample_registry.all_delisted_tickers()) == 3
    assert "TEST_ACQ" in sample_registry.all_delisted_tickers()
    assert "TEST_BNK" in sample_registry.all_delisted_tickers()


def test_get_delisting_info(sample_registry):
    info = sample_registry.get_delisting_info("TEST_ACQ")
    assert info is not None
    assert info.ticker == "TEST_ACQ"
    assert info.name == "Test Acquired Inc."
    assert info.delisting_date == datetime.date(2018, 6, 1)
    assert info.reason == "acquired"
    assert info.acquirer == "MegaCorp"
    assert info.last_price == 50.0

    # Non-existent ticker
    assert sample_registry.get_delisting_info("UNKNOWN") is None


def test_is_delisted_point_in_time(sample_registry):
    # Before delisting date: should be False
    assert sample_registry.is_delisted("TEST_ACQ", as_of_date="2018-05-31") is False
    assert sample_registry.is_delisted("TEST_ACQ", as_of_date=datetime.date(2017, 1, 1)) is False

    # On delisting date: should be True
    assert sample_registry.is_delisted("TEST_ACQ", as_of_date="2018-06-01") is True

    # After delisting date: should be True
    assert sample_registry.is_delisted("TEST_ACQ", as_of_date="2019-01-01") is True

    # Without as_of_date: checks if in registry at all
    assert sample_registry.is_delisted("TEST_ACQ") is True
    assert sample_registry.is_delisted("AAPL") is False


def test_get_delisted_between(sample_registry):
    results = sample_registry.get_delisted_between("2018-01-01", "2020-12-31")
    tickers = [r.ticker for r in results]
    assert "TEST_ACQ" in tickers  # 2018-06-01
    assert "TEST_BNK" in tickers  # 2020-03-15
    assert "TEST_PVT" not in tickers  # 2022-10-01


def test_add_delisted_ticker():
    registry = DelistedRegistry(data_path=Path("/non/existent/path.csv"))
    assert len(registry.all_delisted_tickers()) == 0

    new_info = DelistedInfo(
        ticker="CUSTOM",
        name="Custom Delisted Co",
        delisting_date=datetime.date(2021, 5, 20),
        reason="acquired",
    )
    registry.add_delisted_ticker(new_info)

    assert registry.is_delisted("CUSTOM") is True
    assert registry.is_delisted("CUSTOM", as_of_date="2020-01-01") is False
    assert registry.is_delisted("CUSTOM", as_of_date="2022-01-01") is True


def test_to_dataframe(sample_registry):
    df = sample_registry.to_dataframe()
    assert len(df) == 3
    assert set(df.columns) == {
        "ticker",
        "name",
        "delisting_date",
        "reason",
        "acquirer",
        "last_price",
        "notes",
    }
