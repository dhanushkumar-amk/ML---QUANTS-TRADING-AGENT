# ============================================================
# tests.test_api — Unit Tests for FastAPI Dashboard Endpoints
# ============================================================
"""
Unit tests for the FastAPI REST API layer.
Validates JSON responses, status codes, and schema conformance across all 7 endpoints.
"""

from fastapi.testclient import TestClient
import pytest

from src.api.server import app

client = TestClient(app)


def test_overview_endpoint() -> None:
    """Test /api/overview returns KPIs, equity curve, and recent trades."""
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()

    assert "metrics" in data
    assert data["metrics"]["current_equity"] > 0
    assert "sharpe_ratio" in data["metrics"]
    assert "max_drawdown_pct" in data["metrics"]

    assert "equity_curve" in data
    assert len(data["equity_curve"]) > 0
    first_pt = data["equity_curve"][0]
    assert "date" in first_pt
    assert "strategy" in first_pt
    assert "drawdown" in first_pt

    assert "recent_trades" in data
    assert isinstance(data["recent_trades"], list)


def test_chart_endpoint() -> None:
    """Test /api/chart/{ticker} returns candlestick bars, indicators, and markers."""
    res = client.get("/api/chart/AAPL?timeframe=1M")
    assert res.status_code == 200
    data = res.json()

    assert data["ticker"] == "AAPL"
    assert data["timeframe"] == "1M"
    assert "bars" in data
    assert len(data["bars"]) > 0

    first_bar = data["bars"][0]
    assert "time" in first_bar
    assert "open" in first_bar
    assert "high" in first_bar
    assert "low" in first_bar
    assert "close" in first_bar
    assert "volume" in first_bar
    assert "sma20" in first_bar
    assert "bb_upper" in first_bar

    assert "markers" in data
    assert isinstance(data["markers"], list)


def test_explain_endpoint() -> None:
    """Test /api/chart/{ticker}/explain returns inference signal and SHAP features."""
    res = client.get("/api/chart/AAPL/explain")
    assert res.status_code == 200
    data = res.json()

    assert data["ticker"] == "AAPL"
    assert "signal" in data
    assert data["signal"] in {"BUY", "SELL", "HOLD"}
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0

    assert "features" in data
    assert len(data["features"]) > 0
    first_feat = data["features"][0]
    assert "feature" in first_feat
    assert "shap" in first_feat


def test_backtest_endpoint() -> None:
    """Test /api/backtest returns tearsheet, rolling metrics, and Monte Carlo bands."""
    res = client.get("/api/backtest")
    assert res.status_code == 200
    data = res.json()

    assert "tearsheet" in data
    assert data["tearsheet"]["sharpe_ratio"] > 0
    assert "rolling_metrics" in data
    assert len(data["rolling_metrics"]) > 0
    assert "trade_pnl_distribution" in data
    assert "regime_breakdown" in data
    assert "monte_carlo_bands" in data
    assert len(data["monte_carlo_bands"]) > 0


def test_risk_endpoint() -> None:
    """Test /api/risk returns risk engine limits and exposure status."""
    res = client.get("/api/risk")
    assert res.status_code == 200
    data = res.json()

    assert "engine_state" in data
    assert "current_drawdown_pct" in data
    assert "max_drawdown_limit_pct" in data
    assert "gross_exposure_pct" in data
    assert "positions" in data
    assert data["kill_switch_active"] is False


def test_audit_endpoint() -> None:
    """Test /api/audit returns paginated and filtered audit events."""
    res = client.get("/api/audit?limit=10")
    assert res.status_code == 200
    data = res.json()

    assert "total" in data
    assert "events" in data
    assert len(data["events"]) <= 10

    # Test filtering by event_type
    filtered_res = client.get("/api/audit?event_type=ORDER_SUBMITTED")
    assert filtered_res.status_code == 200
    filtered_data = filtered_res.json()
    for ev in filtered_data["events"]:
        assert ev["event_type"] == "ORDER_SUBMITTED"


def test_models_endpoint() -> None:
    """Test /api/models returns model comparison matrix and feature importance."""
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()

    assert "models" in data
    assert len(data["models"]) >= 5
    model_names = [m["name"] for m in data["models"]]
    assert any("Ensemble" in name for name in model_names)

    assert "feature_importance" in data
    assert len(data["feature_importance"]) >= 10
