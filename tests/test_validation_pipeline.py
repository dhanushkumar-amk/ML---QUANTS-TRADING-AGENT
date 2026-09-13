# ============================================================
# Unit Tests: Validation Pipeline (Phase 7)
# ============================================================
"""
Tests for ValidationPipeline, Policy enforcement, ValidationReport,
and trend analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline.validation_pipeline import (
    DataValidationError,
    Policy,
    ValidationPipeline,
    get_validation_trends,
)
from src.data_pipeline.validation_rules import (
    ContinuityCheckRule,
    RangeCheckRule,
    SchemaValidationRule,
    ValidationResult,
    ValidationRule,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture providing valid daily OHLCV data."""
    dates = pd.bdate_range(start="2024-01-08", periods=10, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)],
            "low": [98.0 + i for i in range(10)],
            "close": [103.0 + i for i in range(10)],
            "volume": [1_000_000 + i * 10_000 for i in range(10)],
        },
        index=dates,
    )


class MockFailingRule(ValidationRule):
    def __init__(self, name: str = "mock_failing", severity: str = "critical") -> None:
        super().__init__(name=name)
        self.severity = severity

    def validate(self, df: pd.DataFrame, ticker: str = "") -> ValidationResult:
        return ValidationResult(
            rule_name=self.name,
            passed=False,
            severity=self.severity,
            affected_rows=1,
            message="Mock failure for testing.",
        )


# ============================================================
# ValidationPipeline Execution Tests
# ============================================================


def test_pipeline_valid_dataset(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    pipeline = ValidationPipeline(
        rules=[SchemaValidationRule(), RangeCheckRule(), ContinuityCheckRule()],
        reports_dir=tmp_path / "reports",
        save_reports=True,
    )
    report = pipeline.validate(sample_df, ticker="TEST")

    assert report.is_valid is True
    assert report.total_rules == 3
    assert report.passed_rules == 3
    assert report.failed_rules == 0
    assert report.critical_failures == 0
    assert report.warnings == 0

    # Test report table representation
    table = report.summary_table()
    assert "TEST" in table
    assert "schema_validation" in table
    assert "[PASS]" in table

    # Verify JSON file was created
    report_files = list((tmp_path / "reports").glob("*.json"))
    assert len(report_files) == 1


def test_pipeline_policy_block_raises_critical(sample_df: pd.DataFrame) -> None:
    pipeline = ValidationPipeline(
        rules=[MockFailingRule(severity="critical")],
        policy_map={"mock_failing": Policy.BLOCK},
        raise_on_critical=True,
    )
    with pytest.raises(DataValidationError) as excinfo:
        pipeline.validate(sample_df, ticker="AAPL")
    assert "Validation failed with critical errors" in str(excinfo.value)


def test_pipeline_policy_block_no_raise_if_configured(sample_df: pd.DataFrame) -> None:
    pipeline = ValidationPipeline(
        rules=[MockFailingRule(severity="critical")],
        policy_map={"mock_failing": Policy.BLOCK},
        raise_on_critical=False,
    )
    report = pipeline.validate(sample_df, ticker="AAPL")
    assert report.is_valid is False
    assert report.failed_rules == 1
    assert report.critical_failures == 1


def test_pipeline_policy_warn(sample_df: pd.DataFrame) -> None:
    pipeline = ValidationPipeline(
        rules=[MockFailingRule(severity="critical")],
        policy_map={"mock_failing": Policy.WARN},
        raise_on_critical=True,  # Even with raise_on_critical=True, WARN policy should not raise
    )
    report = pipeline.validate(sample_df, ticker="MSFT")
    assert report.is_valid is False
    assert report.warnings == 1
    assert report.critical_failures == 0


def test_pipeline_policy_ignore(sample_df: pd.DataFrame) -> None:
    pipeline = ValidationPipeline(
        rules=[MockFailingRule(severity="critical")],
        policy_map={"mock_failing": Policy.IGNORE},
    )
    report = pipeline.validate(sample_df, ticker="SPY")
    # When IGNORED, the rule result is omitted or treated as passed
    assert report.total_rules == 0
    assert report.failed_rules == 0
    assert report.is_valid is True


# ============================================================
# Serialization & Trend Tests
# ============================================================


def test_report_serialization(sample_df: pd.DataFrame) -> None:
    pipeline = ValidationPipeline(
        rules=[SchemaValidationRule()],
        save_reports=False,
    )
    report = pipeline.validate(sample_df, ticker="AAPL")

    data = report.to_dict()
    assert data["ticker"] == "AAPL"
    assert data["total_rules"] == 1
    assert data["is_valid"] is True

    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["ticker"] == "AAPL"
    assert parsed["passed_rules"] == 1


def test_validation_trends(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Write 3 mock report files
    r1 = {
        "timestamp": "2026-09-01T10:00:00",
        "ticker": "AAPL",
        "is_valid": True,
        "total_rules": 5,
        "passed_rules": 5,
        "failed_rules": 0,
        "critical_failures": 0,
        "warnings": 0,
        "results": [],
    }
    r2 = {
        "timestamp": "2026-09-02T10:00:00",
        "ticker": "MSFT",
        "is_valid": False,
        "total_rules": 5,
        "passed_rules": 4,
        "failed_rules": 1,
        "critical_failures": 0,
        "warnings": 1,
        "results": [
            {
                "rule_name": "freshness_check",
                "passed": False,
                "severity": "warning",
                "message": "Data stale",
            }
        ],
    }
    r3 = {
        "timestamp": "2026-09-03T10:00:00",
        "ticker": "SPY",
        "is_valid": False,
        "total_rules": 5,
        "passed_rules": 3,
        "failed_rules": 2,
        "critical_failures": 1,
        "warnings": 1,
        "results": [
            {
                "rule_name": "range_check",
                "passed": False,
                "severity": "critical",
                "message": "Negative price",
            }
        ],
    }

    for idx, r in enumerate([r1, r2, r3]):
        with open(reports_dir / f"2026090{idx+1}_report.json", "w") as f:
            json.dump(r, f)

    trends = get_validation_trends(reports_dir=reports_dir)
    assert trends["total_reports"] == 3
    assert trends["valid_count"] == 1
    assert trends["total_critical_failures"] == 1
    assert trends["total_warnings"] == 2
    assert "freshness_check" in trends["failures_by_rule"]
    assert "range_check" in trends["failures_by_rule"]
    assert not trends["history_df"].empty
