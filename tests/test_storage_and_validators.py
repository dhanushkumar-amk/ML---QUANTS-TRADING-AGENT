# ============================================================
# Unit Tests — Storage & Validators (Phase 2)
# ============================================================

from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline.storage import (
    load_dataframe,
    save_dataframe,
)
from src.data_pipeline.validators import (
    build_summary_table,
    print_summary,
    validate_ohlcv,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    dates = pd.date_range("2026-09-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [102.0, 103.0, 104.0, 105.0, 106.0],
            "volume": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        }
    )


def test_storage_save_and_load(tmp_path: Path, sample_df: pd.DataFrame):
    """Test saving dataframe as parquet and reloading it."""
    saved_path = save_dataframe(
        sample_df,
        source="test_source",
        name="test_ticker",
        raw_dir=tmp_path,
    )
    assert saved_path.exists()
    assert saved_path.suffix == ".parquet"

    # Reload
    loaded_df = load_dataframe(
        source="test_source",
        name="test_ticker",
        raw_dir=tmp_path,
    )
    assert len(loaded_df) == len(sample_df)
    assert "close" in loaded_df.columns

    # Test load non-existent
    with pytest.raises(FileNotFoundError):
        load_dataframe("test_source", "non_existent", raw_dir=tmp_path)


def test_validators_detect_duplicate_timestamps(sample_df: pd.DataFrame):
    """Test validator detects duplicate dates."""
    df_dupes = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    issues = validate_ohlcv(df_dupes, "TEST")
    assert any("duplicate timestamps" in issue for issue in issues)


def test_validators_detect_negative_values(sample_df: pd.DataFrame):
    """Test validator detects negative price and volume."""
    df_neg = sample_df.copy()
    df_neg.loc[0, "close"] = -10.0
    df_neg.loc[1, "volume"] = -500.0

    issues = validate_ohlcv(df_neg, "TEST")
    assert any("negative values in 'close'" in issue for issue in issues)
    assert any("negative values in 'volume'" in issue for issue in issues)


def test_validators_empty_dataframe():
    """Test validator on empty dataframe."""
    issues = validate_ohlcv(pd.DataFrame(), "EMPTY")
    assert any("DataFrame is empty" in issue for issue in issues)


def test_build_and_print_summary(sample_df: pd.DataFrame, capsys):
    """Test building summary table and printing it."""
    results = {"AAPL": sample_df, "MSFT": sample_df}
    summary_df = build_summary_table(results, source="test_source")
    assert len(summary_df) == 2
    assert "name" in summary_df.columns
    assert "rows" in summary_df.columns

    # Test print_summary does not raise
    print_summary(summary_df, title="Test Ingestion Summary")
    captured = capsys.readouterr()
    assert "Test Ingestion Summary" in captured.out
