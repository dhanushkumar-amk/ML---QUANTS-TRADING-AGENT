# ============================================================
# Unit Tests — Universe Builder & Survivorship Bias (Phase 5)
# ============================================================

import datetime
from pathlib import Path

import pytest

from src.data_pipeline.delisted_tickers import DelistedInfo, DelistedRegistry
from src.data_pipeline.universe_builder import UniverseBuilder


@pytest.fixture
def synthetic_universe(tmp_path: Path):
    """Set up synthetic index data with known additions and removals."""
    # 1. Base modern constituents snapshot
    base_file = tmp_path / "base_constituents.csv"
    base_file.write_text(
        "ticker,name,sector\n"
        "ALPHA,Alpha Corp,Tech\n"
        "BETA,Beta Corp,Tech\n"
        "GAMMA,Gamma Corp,Finance\n"
    )

    # 2. Historical changes:
    # On 2020-01-01: GAMMA was added, DELISTED_1 was removed (acquired)
    # On 2017-06-01: BETA was added, DELISTED_2 was removed (bankruptcy)
    changes_file = tmp_path / "historical_changes.csv"
    changes_file.write_text(
        "date,ticker_added,name_added,ticker_removed,name_removed,reason\n"
        "2017-06-01,BETA,Beta Corp,DELISTED_2,Delisted Two Corp,Bankruptcy liquidation\n"
        "2020-01-01,GAMMA,Gamma Corp,DELISTED_1,Delisted One Corp,Acquired by MegaCorp\n"
    )

    # 3. Delisted registry
    registry = DelistedRegistry(data_path=Path("/non/existent/path.csv"))
    registry.add_delisted_ticker(
        DelistedInfo(
            ticker="DELISTED_1",
            name="Delisted One Corp",
            delisting_date=datetime.date(2020, 1, 1),
            reason="acquired",
            acquirer="MegaCorp",
        )
    )
    registry.add_delisted_ticker(
        DelistedInfo(
            ticker="DELISTED_2",
            name="Delisted Two Corp",
            delisting_date=datetime.date(2017, 6, 1),
            reason="bankruptcy",
        )
    )

    return UniverseBuilder(
        constituents_path=base_file,
        changes_path=changes_file,
        delisted_registry=registry,
    )


def test_point_in_time_modern_universe(synthetic_universe):
    """As of 2021-01-01, modern universe has ALPHA, BETA, GAMMA."""
    universe = synthetic_universe.get_universe("2021-01-01")
    assert universe == ["ALPHA", "BETA", "GAMMA"]


def test_point_in_time_intermediate_universe(synthetic_universe):
    """As of 2018-01-01:
    - GAMMA was not yet added (added in 2020).
    - DELISTED_1 was still in the index (removed in 2020).
    - BETA was added in 2017.
    - DELISTED_2 was removed in 2017.
    Expected: ALPHA, BETA, DELISTED_1.
    """
    universe = synthetic_universe.get_universe("2018-01-01")
    assert universe == ["ALPHA", "BETA", "DELISTED_1"]


def test_point_in_time_early_universe_survivorship_fix(synthetic_universe):
    """As of 2016-01-01:
    - BETA was not yet added (added in 2017).
    - GAMMA was not yet added (added in 2020).
    - DELISTED_1 was still active.
    - DELISTED_2 was still active.
    Expected: ALPHA, DELISTED_1, DELISTED_2.
    """
    universe = synthetic_universe.get_universe("2016-01-01")
    assert universe == ["ALPHA", "DELISTED_1", "DELISTED_2"]


def test_constituent_delisting_metadata(synthetic_universe):
    """Verify that returned constituents contain delisting flags and metadata."""
    constituents = synthetic_universe.get_constituents("2016-01-01")
    const_dict = {c.ticker: c for c in constituents}

    assert "DELISTED_1" in const_dict
    c1 = const_dict["DELISTED_1"]
    assert c1.is_delisted is True
    assert c1.delisting_date == datetime.date(2020, 1, 1)
    assert c1.delisting_reason == "acquired"

    assert "DELISTED_2" in const_dict
    c2 = const_dict["DELISTED_2"]
    assert c2.is_delisted is True
    assert c2.delisting_date == datetime.date(2017, 6, 1)
    assert c2.delisting_reason == "bankruptcy"

    assert "ALPHA" in const_dict
    assert const_dict["ALPHA"].is_delisted is False


def test_get_universe_history(synthetic_universe):
    dates = ["2016-01-01", "2018-01-01", "2021-01-01"]
    history = synthetic_universe.get_universe_history(dates)

    assert len(history) == 3
    assert history[datetime.date(2016, 1, 1)] == ["ALPHA", "DELISTED_1", "DELISTED_2"]
    assert history[datetime.date(2018, 1, 1)] == ["ALPHA", "BETA", "DELISTED_1"]
    assert history[datetime.date(2021, 1, 1)] == ["ALPHA", "BETA", "GAMMA"]


def test_to_dataframe(synthetic_universe):
    df = synthetic_universe.to_dataframe("2016-01-01")
    assert len(df) == 3
    assert set(df["ticker"]) == {"ALPHA", "DELISTED_1", "DELISTED_2"}
    assert "is_delisted" in df.columns
    assert "delisting_date" in df.columns


def test_production_sp500_historical_survivorship():
    """Verify that in the default S&P 500 dataset, YHOO appears in 2015 and not in 2024."""
    builder = UniverseBuilder()
    universe_2015 = builder.get_universe("2015-01-01")
    universe_2024 = builder.get_universe("2024-01-01")

    # Yahoo (YHOO) was active in 2015 before 2017 acquisition by Verizon
    assert "YHOO" in universe_2015
    assert "YHOO" not in universe_2024

    # Celgene (CELG) was active in 2015 before 2019 acquisition by BMY
    assert "CELG" in universe_2015
    assert "CELG" not in universe_2024

    # Monsanto (MON) was active in 2015 before 2018 acquisition by Bayer
    assert "MON" in universe_2015
    assert "MON" not in universe_2024

    # Tesla (TSLA) was added to S&P 500 in Dec 2020: absent in 2015, present in 2024
    assert "TSLA" not in universe_2015
    assert "TSLA" in universe_2024
