# ============================================================
# Unit Tests — Data Cleaner & Quality Report (Phase 4)
# ============================================================

import pandas as pd
import pytest

from src.data_pipeline.data_cleaner import DataCleaner, DataQualityReport


@pytest.fixture
def clean_ohlcv_df():
    # 5 clean trading days: Mon Aug 17 to Fri Aug 21, 2026
    dates = pd.date_range("2026-08-17", periods=5, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000],
        }
    )


def test_cleaner_clean_series(clean_ohlcv_df):
    cleaner = DataCleaner(strategy="flag_only")
    df_res, report = cleaner.clean(clean_ohlcv_df, ticker="CLEAN")

    assert report.total_issues == 0
    assert report.total_raw_rows == 5
    assert report.cleaned_rows == 5
    assert "flag_duplicate" in df_res.columns
    assert df_res["flag_duplicate"].sum() == 0


def test_cleaner_detect_duplicates(clean_ohlcv_df):
    # Append duplicate row
    df_dupe = pd.concat([clean_ohlcv_df, clean_ohlcv_df.iloc[[0]]], ignore_index=True)

    # 1. flag_only
    cleaner_flag = DataCleaner(strategy="flag_only")
    df_res, report = cleaner_flag.clean(df_dupe, ticker="DUPE")
    assert report.duplicate_rows == 1
    assert len(df_res) == 6
    assert df_res["flag_duplicate"].sum() == 1

    # 2. drop
    cleaner_drop = DataCleaner(strategy="drop")
    df_dropped, report_drop = cleaner_drop.clean(df_dupe, ticker="DUPE")
    assert report_drop.duplicate_rows == 1
    assert len(df_dropped) == 5

    # 3. forward_fill
    cleaner_ffill = DataCleaner(strategy="forward_fill")
    df_ffill, report_ffill = cleaner_ffill.clean(df_dupe, ticker="DUPE")
    assert report_ffill.duplicate_rows == 1
    assert len(df_ffill) == 5


def test_cleaner_detect_bad_prices_and_zero_volume(clean_ohlcv_df):
    df_bad = clean_ohlcv_df.copy()
    df_bad.loc[1, "close"] = -5.0  # negative price
    df_bad.loc[2, "volume"] = 0  # zero volume

    cleaner = DataCleaner(strategy="flag_only")
    df_res, report = cleaner.clean(df_bad, ticker="BAD")

    assert report.zero_or_negative_prices == 1
    assert report.zero_volumes == 1
    assert df_res["flag_bad_price"].sum() == 1
    assert df_res["flag_zero_volume"].sum() == 1


def test_cleaner_detect_reverting_spike(clean_ohlcv_df):
    df_spike = clean_ohlcv_df.copy()
    # Day 3 spikes up to $150 and immediately reverts to $104 on Day 4
    df_spike.loc[2, "close"] = 150.0

    cleaner = DataCleaner(strategy="flag_only")
    df_res, report = cleaner.clean(df_spike, ticker="SPIKE")

    assert report.reverting_spikes == 1
    assert df_res["flag_spike"].sum() == 1


def test_cleaner_remediation_forward_fill(clean_ohlcv_df):
    df_bad = clean_ohlcv_df.copy()
    # Bad tick on row 2
    df_bad.loc[2, "close"] = 150.0  # reverting spike
    df_bad.loc[3, "open"] = -10.0  # negative price

    cleaner = DataCleaner(strategy="forward_fill")
    df_res, report = cleaner.clean(df_bad, ticker="FFILL")

    assert report.reverting_spikes == 1
    assert report.zero_or_negative_prices == 1
    assert len(df_res) == 5
    # The negative price should have been replaced with row 2's valid value
    assert df_res.loc[3, "open"] > 0


def test_cleaner_remediation_drop(clean_ohlcv_df):
    df_bad = clean_ohlcv_df.copy()
    df_bad.loc[2, "close"] = -1.0  # negative price

    cleaner = DataCleaner(strategy="drop")
    df_res, report = cleaner.clean(df_bad, ticker="DROP")

    assert report.zero_or_negative_prices == 1
    assert len(df_res) == 4


def test_cleaner_detect_trading_day_gaps(clean_ohlcv_df):
    # Drop Wednesday Aug 19 from a 5-day trading week
    df_gap = clean_ohlcv_df[clean_ohlcv_df["date"] != pd.Timestamp("2026-08-19")].reset_index(drop=True)

    cleaner = DataCleaner(strategy="flag_only")
    _, report = cleaner.clean(df_gap, ticker="GAP")

    assert len(report.missing_trading_days) == 1
    assert report.missing_trading_days[0].strftime("%Y-%m-%d") == "2026-08-19"


def test_data_quality_report_summary_table():
    report = DataQualityReport(
        ticker="AAPL",
        total_raw_rows=1000,
        cleaned_rows=998,
        duplicate_rows=2,
        zero_or_negative_prices=1,
        zero_volumes=1,
        reverting_spikes=1,
        strategy_used="flag_only",
    )
    table_str = report.summary_table()
    assert "DATA QUALITY REPORT: AAPL" in table_str
    assert "Total Raw Rows" in table_str
    assert "Total Issues Found" in table_str
    assert report.total_issues == 5
    assert report.affected_percentage == 0.5
