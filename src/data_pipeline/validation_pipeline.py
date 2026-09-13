# ============================================================
# Validation Pipeline Orchestrator (Phase 7)
# ============================================================
"""
Centralized automated data validation pipeline and gatekeeper.
Coordinates evaluation of composable rules, policy enforcement,
structured reporting, and trend analysis.
"""

from __future__ import annotations

import datetime
import enum
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_pipeline.validation_rules import (
    ContinuityCheckRule,
    FreshnessCheckRule,
    RangeCheckRule,
    SchemaValidationRule,
    ValidationResult,
    ValidationRule,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPORTS_DIR = _PROJECT_ROOT / "data" / "validation_reports"


class Policy(str, enum.Enum):
    """Enforcement policy per validation rule."""

    BLOCK = "block"  # Halt pipeline and raise DataValidationError on failure
    WARN = "warn"  # Log warning, record in report, but proceed
    IGNORE = "ignore"  # Suppress and treat as passed


class DataValidationError(Exception):
    """Raised when a validation rule with Policy.BLOCK fails."""

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class ValidationReport:
    """Consolidated report summarizing validation results for a dataset."""

    ticker: str
    timestamp: datetime.datetime
    total_rows: int
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if all evaluated rules passed."""
        return all(r.passed for r in self.results)

    @property
    def is_valid(self) -> bool:
        """Alias for passed."""
        return self.passed

    @property
    def total_rules(self) -> int:
        return len(self.results)

    @property
    def passed_rules(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_rules(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def critical_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "critical")

    @property
    def critical_failures(self) -> int:
        return self.critical_count

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "warning")

    @property
    def warnings(self) -> int:
        return self.warning_count

    @property
    def info_count(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "info")

    def summary_table(self) -> str:
        """Render a formatted ASCII validation report."""
        border = "-" * 88
        header = f"\n{border}\n  DATA VALIDATION REPORT: {self.ticker}\n{border}"
        meta = (
            f"  Evaluated At    : {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Total Rows      : {self.total_rows:>8}\n"
            f"  Overall Status  : {'[PASSED]' if self.passed else '[FAILED]'}\n"
            f"  Rules Evaluated : {self.total_rules:>8} ({self.passed_rules} passed, {self.failed_rules} failed)\n"
            f"  Critical Issues : {self.critical_count:>8}\n"
            f"  Warning Issues  : {self.warning_count:>8}\n"
            f"{border}\n"
            f"  {'Rule Name':<28} {'Status':<10} {'Severity':<10} {'Affected Rows':<14} {'Message'}\n"
            f"{border}"
        )

        rows = []
        for r in self.results:
            status = "[PASS]" if r.passed else "[FAIL]"
            rows.append(
                f"  {r.rule_name:<28} {status:<10} {r.severity:<10} {r.affected_rows:<14} {r.message[:50]}"
            )

        footer = f"{border}\n"
        return "\n".join([header, meta] + rows + [footer])

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "total_rows": self.total_rows,
            "is_valid": self.is_valid,
            "passed": self.passed,
            "total_rules": self.total_rules,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "critical_failures": self.critical_failures,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "warnings": self.warnings,
            "info_count": self.info_count,
            "results": [
                {
                    "rule_name": r.rule_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "affected_rows": r.affected_rows,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

    def to_json(self, output_dir: Path | str | None = None) -> str:
        """Return JSON string representation. If output_dir is provided, also save to disk."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)

        if output_dir is not None:
            self.save_json(output_dir=output_dir)

        return json_str

    def save_json(self, output_dir: Path | str | None = None) -> Path:
        """Serialize report to a machine-readable JSON file."""
        target_dir = Path(output_dir) if output_dir else _DEFAULT_REPORTS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        ts_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts_str}_{self.ticker}.json"
        filepath = target_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info("[%s] Saved validation report JSON -> %s", self.ticker, filepath)
        return filepath


class ValidationPipeline:
    """Orchestrator for running composable validation rules with policy enforcement."""

    def __init__(
        self,
        rules: Sequence[ValidationRule] | None = None,
        rule_policies: dict[str, Policy] | None = None,
        policy_map: dict[str, Policy] | None = None,
        raise_on_critical: bool = True,
        reports_dir: Path | str | None = None,
        save_reports: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        rules : Sequence[ValidationRule] | None
            Rules to evaluate. Defaults to Schema, Range, Continuity, and Freshness.
        rule_policies / policy_map : dict[str, Policy] | None
            Custom policy per rule name ('block', 'warn', 'ignore').
        raise_on_critical : bool
            Whether failing a rule with Policy.BLOCK or critical severity raises DataValidationError.
        reports_dir : Path | str | None
            Directory to persist JSON reports.
        save_reports : bool
            Whether to write JSON reports to disk by default.
        """
        self.rules = (
            list(rules)
            if rules is not None
            else [
                SchemaValidationRule(),
                RangeCheckRule(),
                ContinuityCheckRule(),
                FreshnessCheckRule(),
            ]
        )
        self.policy_map = policy_map if policy_map is not None else (rule_policies or {})
        self.raise_on_critical = raise_on_critical
        self.reports_dir = Path(reports_dir) if reports_dir else _DEFAULT_REPORTS_DIR
        self.save_reports = save_reports

    def _resolve_policy(self, rule: ValidationRule, result: ValidationResult) -> Policy:
        """Determine applicable enforcement policy for a rule."""
        if rule.name in self.policy_map:
            return self.policy_map[rule.name]

        # Default policy resolution: critical -> BLOCK, warning -> WARN, info -> IGNORE
        if result.severity == "critical":
            return Policy.BLOCK
        elif result.severity == "warning":
            return Policy.WARN
        return Policy.IGNORE

    def validate(
        self,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
        context: dict[str, Any] | None = None,
        save_report: bool | None = None,
    ) -> ValidationReport:
        """Run validation rules against a DataFrame and enforce policies."""
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        results: list[ValidationResult] = []
        blocking_failures: list[ValidationResult] = []

        ctx = context.copy() if context else {}
        ctx["ticker"] = ticker

        for rule in self.rules:
            # Check configured policy before or after
            policy = self.policy_map.get(rule.name, None)
            if policy == Policy.IGNORE:
                continue

            try:
                res = rule.validate(df, context=ctx, ticker=ticker)
            except Exception as exc:
                logger.error("Error running rule '%s' on %s: %s", rule.name, ticker, exc)
                res = ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    severity="critical",
                    affected_rows=len(df),
                    message=f"Rule crashed during execution: {exc}",
                )

            if not res.passed:
                # Apply policy adjustments
                if policy == Policy.WARN:
                    res.severity = "warning"
                elif policy == Policy.BLOCK:
                    res.severity = "critical"

                resolved_policy = self._resolve_policy(rule, res)
                if resolved_policy == Policy.BLOCK and res.severity == "critical":
                    blocking_failures.append(res)
                elif resolved_policy == Policy.WARN:
                    logger.warning(
                        "[%s] Data validation WARNING (%s): %s",
                        ticker,
                        rule.name,
                        res.message,
                    )

            results.append(res)

        report = ValidationReport(
            ticker=ticker,
            timestamp=timestamp,
            total_rows=len(df),
            results=results,
        )

        should_save = self.save_reports if save_report is None else save_report
        if should_save:
            report.save_json(output_dir=self.reports_dir)

        if blocking_failures and self.raise_on_critical:
            reasons = "; ".join(f"[{f.rule_name}] {f.message}" for f in blocking_failures)
            raise DataValidationError(
                f"Validation failed with critical errors for {ticker} under BLOCK policy: {reasons}",
                report=report,
            )

        return report

    def validate_ticker(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        save_report: bool | None = None,
    ) -> ValidationReport:
        """Load ticker data via DataAccessLayer and validate."""
        from src.data_pipeline.data_access import get_data_access

        dal = get_data_access()
        df = dal.get_ohlcv(ticker, start=start, end=end)
        return self.validate(df, ticker=ticker, save_report=save_report)


def get_validation_trends(reports_dir: Path | str | None = None) -> dict[str, Any]:
    """Aggregate historical JSON reports to track data quality trends over time."""
    p = Path(reports_dir) if reports_dir else _DEFAULT_REPORTS_DIR
    if not p.exists():
        return {
            "total_reports": 0,
            "valid_count": 0,
            "total_critical_failures": 0,
            "total_warnings": 0,
            "failures_by_rule": {},
            "history_df": pd.DataFrame(),
        }

    records: list[dict[str, Any]] = []
    failures_by_rule: dict[str, int] = {}
    valid_count = 0
    total_critical = 0
    total_warnings = 0

    for f in sorted(p.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)

            is_valid = data.get("is_valid", data.get("passed", False))
            if is_valid:
                valid_count += 1

            crit = data.get("critical_failures", data.get("critical_count", 0))
            warn = data.get("warnings", data.get("warning_count", 0))
            total_critical += crit
            total_warnings += warn

            # Collect failure counts by rule
            for res in data.get("results", []):
                if not res.get("passed", True):
                    r_name = res.get("rule_name", "unknown")
                    failures_by_rule[r_name] = failures_by_rule.get(r_name, 0) + 1

            records.append(
                {
                    "timestamp": pd.to_datetime(data["timestamp"]),
                    "ticker": data.get("ticker", "UNKNOWN"),
                    "total_rows": data.get("total_rows", 0),
                    "is_valid": is_valid,
                    "critical_count": crit,
                    "warning_count": warn,
                }
            )
        except Exception as exc:
            logger.debug("Failed to read report %s: %s", f.name, exc)

    df_trends = (
        pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
        if records
        else pd.DataFrame()
    )

    return {
        "total_reports": len(records),
        "valid_count": valid_count,
        "total_critical_failures": total_critical,
        "total_warnings": total_warnings,
        "failures_by_rule": failures_by_rule,
        "history_df": df_trends,
    }
