# ============================================================
# Unit Tests — Portfolio Risk Engine (Phase 39)
# ============================================================
"""
Unit tests for the Portfolio Risk Engine and safety gates.
Verifies Drawdown Kill Switch, Daily Loss Limit, Exposure/Sector Caps,
Volatility-Based De-risking, Non-Bypassable Execution Gate, and Audit Logging.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.portfolio.risk_engine import (
    DecisionStatus,
    OrderAction,
    OrderIntent,
    PortfolioState,
    RiskConfig,
    RiskEngine,
    RiskRule,
)


@pytest.fixture
def base_config() -> RiskConfig:
    """Standard risk configuration for tests."""
    return RiskConfig(
        max_drawdown_pct=0.15,
        daily_loss_limit_pct=0.03,
        max_position_size_pct=0.25,
        max_gross_exposure=1.00,
        max_net_exposure=1.00,
        max_sector_concentration=0.40,
        vol_spike_threshold=1.50,
        min_vol_derisk_multiplier=0.20,
    )


@pytest.fixture
def risk_engine(base_config: RiskConfig) -> RiskEngine:
    """RiskEngine initialized with base config."""
    return RiskEngine(config=base_config)


def test_drawdown_kill_switch(risk_engine: RiskEngine):
    """Test Drawdown Kill Switch blocks new positions when DD >= 15% but permits closing trades."""
    # Peak = $100,000, Current = $82,000 -> Drawdown = 18% (> 15% limit)
    portfolio = PortfolioState(
        current_equity=82000.0,
        peak_equity=100000.0,
        daily_start_equity=85000.0,
        current_date="2024-05-10",
        positions={"AAPL": 10000.0},  # currently holding $10k AAPL long
    )

    # 1. Attempt new position opening (Buy MSFT) -> MUST BE BLOCKED
    new_order = OrderIntent(
        ticker="MSFT",
        action=OrderAction.BUY,
        quantity=50,
        price=100.0,
    )
    decision_new = risk_engine.evaluate_order(new_order, portfolio)
    assert decision_new.status == DecisionStatus.BLOCKED
    assert decision_new.rule_triggered == RiskRule.DRAWDOWN_KILL_SWITCH
    assert decision_new.approved_quantity == 0.0

    # 2. Attempt closing/reducing existing position (Sell AAPL) -> MUST BE APPROVED
    reducing_order = OrderIntent(
        ticker="AAPL",
        action=OrderAction.SELL,
        quantity=50,
        price=100.0,
    )
    decision_reduce = risk_engine.evaluate_order(reducing_order, portfolio)
    assert decision_reduce.status == DecisionStatus.APPROVED
    assert decision_reduce.approved_quantity == 50.0


def test_daily_loss_limit_circuit_breaker(risk_engine: RiskEngine):
    """Test Daily Loss Limit halts new orders when intra-day loss >= 3%, and resets next day."""
    # Start equity = $100,000, Total loss today = $1,500 + $2,000 = $3,500 (3.5% > 3.0%)
    portfolio = PortfolioState(
        current_equity=96500.0,
        peak_equity=100000.0,
        daily_start_equity=100000.0,
        current_date="2024-05-10",
        positions={},
        realized_pnl_today=-1500.0,
        unrealized_pnl_today=-2000.0,
    )

    order = OrderIntent(
        ticker="GOOGL",
        action=OrderAction.BUY,
        quantity=10,
        price=150.0,
    )

    # Breach today -> BLOCKED
    decision = risk_engine.evaluate_order(order, portfolio)
    assert decision.status == DecisionStatus.BLOCKED
    assert decision.rule_triggered == RiskRule.DAILY_LOSS_LIMIT

    # Next trading day: start equity resets, circuit breaker cleared -> APPROVED
    next_day_portfolio = PortfolioState(
        current_equity=96500.0,
        peak_equity=100000.0,
        daily_start_equity=96500.0,
        current_date="2024-05-13",
        positions={},
        realized_pnl_today=0.0,
        unrealized_pnl_today=0.0,
    )
    decision_next_day = risk_engine.evaluate_order(order, next_day_portfolio)
    assert decision_next_day.status == DecisionStatus.APPROVED
    assert decision_next_day.approved_quantity == 10.0


def test_single_asset_kelly_cap(risk_engine: RiskEngine):
    """Test single-asset position size cap (25% max) resizes and blocks over-sized orders."""
    # Equity = $100,000 -> Max single asset = $25,000
    portfolio = PortfolioState(
        current_equity=100000.0,
        peak_equity=100000.0,
        daily_start_equity=100000.0,
        current_date="2024-05-10",
        positions={},
    )

    # Request $35,000 of SPY ($350/share x 100 shares) -> Should resize to 71.42 shares ($25,000)
    oversized_order = OrderIntent(
        ticker="SPY",
        action=OrderAction.BUY,
        quantity=100,
        price=350.0,
    )
    decision = risk_engine.evaluate_order(oversized_order, portfolio)
    assert decision.status == DecisionStatus.RESIZED
    assert decision.rule_triggered == RiskRule.POSITION_LIMIT
    assert pytest.approx(decision.approved_dollar_value, abs=1e-2) == 25000.0

    # If already at $25,000 in SPY, another buy order must be BLOCKED
    portfolio_at_limit = PortfolioState(
        current_equity=100000.0,
        peak_equity=100000.0,
        daily_start_equity=100000.0,
        current_date="2024-05-10",
        positions={"SPY": 25000.0},
    )
    decision_block = risk_engine.evaluate_order(
        OrderIntent(ticker="SPY", action=OrderAction.BUY, quantity=10, price=350.0),
        portfolio_at_limit,
    )
    assert decision_block.status == DecisionStatus.BLOCKED
    assert decision_block.rule_triggered == RiskRule.POSITION_LIMIT


def test_sector_concentration_cap(risk_engine: RiskEngine):
    """Test sector concentration cap (40% max tech) prevents excessive clustering."""
    # Equity = $100,000 -> Max Tech = $40,000. Already have AAPL $25,000 (Tech).
    portfolio = PortfolioState(
        current_equity=100000.0,
        peak_equity=100000.0,
        daily_start_equity=100000.0,
        current_date="2024-05-10",
        positions={"AAPL": 25000.0},  # Technology
    )

    # Propose buying MSFT $20,000 (Technology): 25k + 20k = 45k > 40k max -> Resized to $15k
    order_msft = OrderIntent(
        ticker="MSFT",
        action=OrderAction.BUY,
        quantity=200,
        price=100.0,
    )
    decision = risk_engine.evaluate_order(order_msft, portfolio)
    assert decision.status == DecisionStatus.RESIZED
    assert decision.rule_triggered == RiskRule.SECTOR_CONCENTRATION_LIMIT
    assert pytest.approx(decision.approved_dollar_value, abs=1e-2) == 15000.0


def test_gross_and_net_exposure_limits(risk_engine: RiskEngine):
    """Test maximum gross portfolio leverage cap (1.0 = 100% equity)."""
    # Equity = $100,000. Current gross = $85,000.
    portfolio = PortfolioState(
        current_equity=100000.0,
        peak_equity=100000.0,
        daily_start_equity=100000.0,
        current_date="2024-05-10",
        positions={"AAPL": 25000.0, "GOOGL": 25000.0, "AMZN": 20000.0, "TSLA": 15000.0},
    )

    # Order requesting $25,000 of SPY: 85k + 25k = 110k > 100k -> Resized to $15,000
    order_spy = OrderIntent(
        ticker="SPY",
        action=OrderAction.BUY,
        quantity=250,
        price=100.0,
    )
    decision = risk_engine.evaluate_order(order_spy, portfolio)
    assert decision.status == DecisionStatus.RESIZED
    assert decision.rule_triggered == RiskRule.GROSS_EXPOSURE_LIMIT
    assert pytest.approx(decision.approved_dollar_value, abs=1e-2) == 15000.0


def test_volatility_based_derisking(risk_engine: RiskEngine):
    """Test GARCH volatility spike downscales order sizes inversely with volatility."""
    portfolio = PortfolioState(
        current_equity=100000.0,
        peak_equity=100000.0,
        daily_start_equity=100000.0,
        current_date="2024-05-10",
        positions={},
    )

    order = OrderIntent(
        ticker="NVDA",
        action=OrderAction.BUY,
        quantity=100,
        price=100.0,
    )

    # GARCH forecast vol = 0.40 (40%), Baseline vol = 0.20 (20%) -> Ratio 2.0x (> 1.5x threshold)
    # Multiplier = 0.20 / 0.40 = 0.50 -> 50 shares approved
    market_vol = {"NVDA": (0.40, 0.20)}
    decision = risk_engine.evaluate_order(order, portfolio, market_volatility=market_vol)

    assert decision.status == DecisionStatus.RESIZED
    assert decision.rule_triggered == RiskRule.VOLATILITY_DERISKING
    assert pytest.approx(decision.approved_quantity, abs=1e-2) == 50.0
    assert pytest.approx(decision.approved_dollar_value, abs=1e-2) == 5000.0


def test_risk_engine_cannot_be_bypassed(risk_engine: RiskEngine):
    """Test hard architectural rule: order execution strictly raises PermissionError when blocked."""
    portfolio = PortfolioState(
        current_equity=80000.0,
        peak_equity=100000.0,  # 20% drawdown -> kill switch active
        daily_start_equity=85000.0,
        current_date="2024-05-10",
        positions={},
    )

    broker_mock = MagicMock()

    intent = OrderIntent(ticker="SPY", action=OrderAction.BUY, quantity=10, price=100.0)

    # Calling execute_gated_order must raise PermissionError and NEVER call broker_mock
    with pytest.raises(PermissionError, match="BLOCKED by RiskEngine"):
        risk_engine.execute_gated_order(intent, portfolio, broker_mock)

    broker_mock.assert_not_called()


def test_audit_logging_completeness(risk_engine: RiskEngine):
    """Test all risk interventions are recorded into the audit trail with structured fields."""
    portfolio = PortfolioState(
        current_equity=100000.0,
        peak_equity=100000.0,
        daily_start_equity=100000.0,
        current_date="2024-05-10",
    )

    # 1. Normal order
    risk_engine.evaluate_order(
        OrderIntent(ticker="AAPL", action=OrderAction.BUY, quantity=10, price=100.0),
        portfolio,
    )
    # 2. Oversized order
    risk_engine.evaluate_order(
        OrderIntent(ticker="AAPL", action=OrderAction.BUY, quantity=400, price=100.0),
        portfolio,
    )

    audit_df = risk_engine.get_audit_log()
    assert len(audit_df) == 2
    assert list(audit_df["status"].values) == ["APPROVED", "RESIZED"]
    assert "ticker" in audit_df.columns
    assert "rule_triggered" in audit_df.columns
    assert "reasons" in audit_df.columns
