# ============================================================
# Unit Tests — Storage Backend Abstraction (Phase 6)
# ============================================================

from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline.storage_backend import ParquetBackend, TimescaleDBBackend


@pytest.fixture
def sample_multiyear_df():
    """Create a synthetic 3-year OHLCV dataset (2020, 2021, 2022)."""
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i * 0.1 for i in range(len(dates))],
            "high": [102.0 + i * 0.1 for i in range(len(dates))],
            "low": [98.0 + i * 0.1 for i in range(len(dates))],
            "close": [101.0 + i * 0.1 for i in range(len(dates))],
            "volume": [1_000_000 + i * 1000 for i in range(len(dates))],
            "ticker": "TEST",
        }
    )


def test_parquet_backend_save_and_load(tmp_path: Path, sample_multiyear_df):
    backend = ParquetBackend(root_dir=tmp_path)
    written = backend.save(sample_multiyear_df, key="TEST")

    # Should have written 3 partitions (2020, 2021, 2022)
    assert len(written) == 3
    partition_names = [p.name for p in written]
    assert "2020.parquet" in partition_names
    assert "2021.parquet" in partition_names
    assert "2022.parquet" in partition_names

    # Load all
    loaded = backend.load("TEST")
    assert len(loaded) == len(sample_multiyear_df)
    assert "date" in loaded.columns
    assert "close" in loaded.columns


def test_parquet_backend_query_date_range(tmp_path: Path, sample_multiyear_df):
    backend = ParquetBackend(root_dir=tmp_path)
    backend.save(sample_multiyear_df, key="TEST")

    # Query single year 2021
    df_2021 = backend.query("TEST", start="2021-01-01", end="2021-12-31")
    assert not df_2021.empty
    assert df_2021["date"].dt.year.unique().tolist() == [2021]

    # Query cross-year boundary (2020-06-01 to 2021-06-01)
    df_cross = backend.query("TEST", start="2020-06-01", end="2021-06-01")
    assert not df_cross.empty
    assert df_cross["date"].min() >= pd.Timestamp("2020-06-01")
    assert df_cross["date"].max() <= pd.Timestamp("2021-06-01")


def test_parquet_backend_query_column_projection(tmp_path: Path, sample_multiyear_df):
    backend = ParquetBackend(root_dir=tmp_path)
    backend.save(sample_multiyear_df, key="TEST")

    df_proj = backend.query("TEST", columns=["close", "volume"])
    assert list(df_proj.columns) == ["close", "volume"]
    assert len(df_proj) == len(sample_multiyear_df)


def test_parquet_backend_list_and_delete(tmp_path: Path, sample_multiyear_df):
    backend = ParquetBackend(root_dir=tmp_path)
    backend.save(sample_multiyear_df, key="AAPL")
    backend.save(sample_multiyear_df, key="MSFT")

    keys = backend.list_keys()
    assert keys == ["AAPL", "MSFT"]

    # Filter by prefix
    assert backend.list_keys(prefix="AA") == ["AAPL"]

    # Delete AAPL
    assert backend.delete("AAPL") is True
    assert backend.list_keys() == ["MSFT"]

    # Delete non-existent
    assert backend.delete("UNKNOWN") is False


def test_parquet_backend_empty_or_missing(tmp_path: Path):
    backend = ParquetBackend(root_dir=tmp_path)

    # Empty df save
    assert backend.save(pd.DataFrame(), key="EMPTY") == []

    # Non-existent ticker load raises
    with pytest.raises(FileNotFoundError):
        backend.load("NONEXISTENT")

    # Non-existent query returns empty df
    res = backend.query("NONEXISTENT")
    assert res.empty


def test_timescaledb_backend_stub_graceful():
    """Verify TimescaleDB backend handles absence of a live DB gracefully without crashing."""
    backend = TimescaleDBBackend(
        connection_url="postgresql://invalid:invalid@localhost:5432/invalid"
    )
    assert backend.check_connection() is False
    assert backend.list_keys() == []
    assert backend.load("AAPL").empty
    assert backend.query("AAPL").empty
    assert backend.delete("AAPL") is False
    assert backend.save(pd.DataFrame(), "AAPL") is False


def test_db_schema_export(tmp_path: Path):
    from src.data_pipeline.db_schema import export_migration_file, generate_ddl_sql

    ddl = generate_ddl_sql()
    assert "CREATE TABLE IF NOT EXISTS ohlcv_data" in ddl
    assert "create_hypertable" in ddl
    assert "corporate_actions" in ddl
    assert "universe_membership" in ddl

    out_file = tmp_path / "migration.sql"
    written = export_migration_file(out_file)
    assert written.exists()
    assert "ohlcv_data" in written.read_text(encoding="utf-8")
