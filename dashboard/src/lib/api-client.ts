// API client fetching from FastAPI with embedded static fallback for resiliency

import {
  AuditResponse,
  BacktestData,
  ChartData,
  ModelExplanation,
  ModelsData,
  OverviewData,
  RiskStatus,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

async function fetchWithFallback<T>(endpoint: string, fallback: T): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      next: { revalidate: 10 },
      headers: { Accept: "application/json" },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    return fallback;
  }
}

// -------------------------------------------------------------
// Fallback Data Fixtures
// -------------------------------------------------------------

export const fallbackOverview: OverviewData = {
  metrics: {
    current_equity: 102450.8,
    daily_pnl: 1245.8,
    daily_pnl_pct: 1.23,
    sharpe_ratio: 2.14,
    sortino_ratio: 2.86,
    max_drawdown_pct: -8.42,
    win_rate_pct: 58.6,
    profit_factor: 1.85,
    sparklines: {
      equity: [100000, 100400, 100200, 101100, 100900, 101800, 102450],
      pnl: [0, 400, 200, 1100, 900, 1800, 2450],
      sharpe: [1.95, 1.98, 2.02, 2.05, 2.09, 2.12, 2.14],
      drawdown: [-2.1, -3.4, -4.5, -3.2, -5.1, -4.0, -2.4],
      win_rate: [56.0, 56.5, 57.1, 57.4, 58.0, 58.3, 58.6],
    },
  },
  equity_curve: Array.from({ length: 45 }).map((_, i) => {
    const d = new Date(2026, 6, 1 + i);
    const s = 100000 + i * 55 + Math.sin(i / 3) * 600;
    const b = 100000 + i * 32 + Math.cos(i / 4) * 400;
    const dd = -Math.abs(Math.sin(i / 5) * 4.2);
    return {
      date: d.toISOString().split("T")[0],
      strategy: Math.round(s),
      benchmark: Math.round(b),
      drawdown: Number(dd.toFixed(2)),
    };
  }),
  recent_trades: [
    {
      id: "trd_001",
      timestamp: "2026-09-16 15:21:32",
      ticker: "AAPL",
      side: "BUY",
      quantity: 94.68,
      price: 178.45,
      total_value: 16895.65,
      realized_pnl: 0.0,
      status: "FILLED",
    },
    {
      id: "trd_002",
      timestamp: "2026-09-16 15:21:32",
      ticker: "MSFT",
      side: "BUY",
      quantity: 59.14,
      price: 412.3,
      total_value: 24383.42,
      realized_pnl: 0.0,
      status: "FILLED",
    },
    {
      id: "trd_003",
      timestamp: "2026-09-15 14:15:00",
      ticker: "NVDA",
      side: "SELL",
      quantity: 45.0,
      price: 118.9,
      total_value: 5350.5,
      realized_pnl: 420.75,
      status: "FILLED",
    },
    {
      id: "trd_004",
      timestamp: "2026-09-15 11:30:00",
      ticker: "SPY",
      side: "SELL",
      quantity: 30.0,
      price: 548.2,
      total_value: 16446.0,
      realized_pnl: -115.2,
      status: "FILLED",
    },
    {
      id: "trd_005",
      timestamp: "2026-09-14 10:05:00",
      ticker: "GOOGL",
      side: "BUY",
      quantity: 50.0,
      price: 162.1,
      total_value: 8105.0,
      realized_pnl: 285.4,
      status: "FILLED",
    },
  ],
};

export const fallbackChart = (ticker: string, timeframe: string): ChartData => {
  const bars = Array.from({ length: 30 }).map((_, i) => {
    const d = new Date(2026, 7, 1 + i);
    const base = ticker === "MSFT" ? 410 : ticker === "NVDA" ? 120 : 175;
    const open = base + Math.sin(i / 2) * 5 + i * 0.3;
    const close = open + (Math.sin(i) > 0 ? 1.5 : -1.2);
    const high = Math.max(open, close) + 1.2;
    const low = Math.min(open, close) - 1.0;
    return {
      time: d.toISOString().split("T")[0],
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: 12000000 + Math.floor(Math.random() * 8000000),
      sma20: Number((open - 1).toFixed(2)),
      sma60: Number((open - 2).toFixed(2)),
      bb_upper: Number((open + 4).toFixed(2)),
      bb_lower: Number((open - 4).toFixed(2)),
    };
  });

  return {
    ticker,
    timeframe,
    bars,
    markers: [
      {
        time: bars[10].time,
        position: "belowBar",
        color: "#10b981",
        shape: "arrowUp",
        text: `BUY @ ${bars[10].close}`,
      },
      {
        time: bars[22].time,
        position: "aboveBar",
        color: "#f43f5e",
        shape: "arrowDown",
        text: `SELL @ ${bars[22].close}`,
      },
    ],
  };
};

export const fallbackExplanation = (ticker: string): ModelExplanation => ({
  ticker,
  model_name: "Ensemble (LightGBM + Temporal Transformer)",
  prediction_time: "2026-09-16 15:21:32 UTC",
  signal: "BUY",
  confidence: 0.842,
  regime: "BULL_MOMENTUM",
  features: [
    {
      feature: "rsi_14",
      name: "14-day RSI",
      value: 28.4,
      shap: 0.342,
      impact: "positive",
      description: "Oversold momentum reversal signal",
    },
    {
      feature: "macd_hist",
      name: "MACD Histogram",
      value: 0.45,
      shap: 0.215,
      impact: "positive",
      description: "Bullish divergence expansion",
    },
    {
      feature: "finbert_sentiment",
      name: "News Sentiment",
      value: 0.78,
      shap: 0.184,
      impact: "positive",
      description: "Positive NLP sentiment polarity",
    },
    {
      feature: "garch_vol",
      name: "GARCH(1,1) Volatility",
      value: 0.142,
      shap: -0.075,
      impact: "negative",
      description: "Slight volatility contraction drag",
    },
    {
      feature: "parkinson_vol",
      name: "Parkinson Volatility",
      value: 0.128,
      shap: 0.062,
      impact: "positive",
      description: "Intraday compression breakout",
    },
  ],
});

export const fallbackBacktest: BacktestData = {
  tearsheet: {
    annualized_return_pct: 24.8,
    benchmark_annualized_return_pct: 14.2,
    sharpe_ratio: 2.14,
    sortino_ratio: 2.86,
    calmar_ratio: 2.94,
    max_drawdown_pct: -8.42,
    volatility_annualized_pct: 11.6,
    win_rate_pct: 58.6,
    profit_factor: 1.85,
    alpha: 0.106,
    beta: 0.68,
    information_ratio: 1.28,
    tracking_error_pct: 8.3,
    var_95_daily_pct: -1.12,
    cvar_95_daily_pct: -1.68,
  },
  rolling_metrics: Array.from({ length: 30 }).map((_, i) => ({
    date: `2025-${String(Math.floor(i / 2.5) + 1).padStart(2, "0")}-${String((i % 28) + 1).padStart(2, "0")}`,
    rolling_sharpe: Number((1.8 + Math.sin(i / 3) * 0.4).toFixed(2)),
    rolling_volatility: Number((11.5 + Math.cos(i / 4) * 1.8).toFixed(2)),
  })),
  trade_pnl_distribution: [
    { bin: "<-4%", count: 12 },
    { bin: "-4% to -2%", count: 28 },
    { bin: "-2% to 0%", count: 45 },
    { bin: "0% to +2%", count: 68 },
    { bin: "+2% to +4%", count: 42 },
    { bin: "+4% to +6%", count: 24 },
    { bin: ">+6%", count: 16 },
  ],
  regime_breakdown: [
    { regime: "Bull Trend", strategy_return: 32.4, benchmark_return: 22.1 },
    { regime: "Bear Trend", strategy_return: 8.2, benchmark_return: -18.5 },
    { regime: "High Vol Mean-Rev", strategy_return: 18.6, benchmark_return: -6.2 },
    { regime: "Low Vol Chop", strategy_return: 12.1, benchmark_return: 9.4 },
  ],
  monte_carlo_bands: Array.from({ length: 40 }).map((_, i) => {
    const t = (i + 1) / 40;
    const p50 = 100000 * (1 + 0.22 * t);
    return {
      day: `Day +${i + 1}`,
      p5: Math.round(p50 * (1 - 0.14 * Math.sqrt(t))),
      p50: Math.round(p50),
      p95: Math.round(p50 * (1 + 0.16 * Math.sqrt(t))),
    };
  }),
};

export const fallbackRisk: RiskStatus = {
  engine_state: "ACTIVE",
  circuit_breaker_status: "NORMAL",
  current_drawdown_pct: 2.45,
  max_drawdown_limit_pct: 15.0,
  drawdown_headroom_pct: 12.55,
  gross_exposure_value: 41279.07,
  gross_exposure_pct: 40.29,
  max_gross_exposure_pct: 100.0,
  net_exposure_value: 41279.07,
  net_exposure_pct: 40.29,
  max_position_size_pct: 35.0,
  positions: [
    { ticker: "AAPL", market_value: 16895.65, weight_pct: 16.49, limit_pct: 35.0 },
    { ticker: "MSFT", market_value: 24383.42, weight_pct: 23.8, limit_pct: 35.0 },
  ],
  kill_switch_active: false,
  last_reconciliation_time: "2026-09-16 15:21:35",
  reconciliation_status: "SYNCHRONIZED",
};

export const fallbackAudit: AuditResponse = {
  total: 8,
  offset: 0,
  limit: 50,
  events: [
    {
      timestamp: "2026-09-16T15:21:35Z",
      event_type: "SYSTEM_EVENT",
      cycle_id: "cycle_002",
      details: { action: "SHUTDOWN", reason: "COMPLETED", cancelled_open_orders: 2 },
    },
    {
      timestamp: "2026-09-16T15:21:34Z",
      event_type: "ORDER_CANCELLED",
      ticker: "MSFT",
      cycle_id: "cycle_002",
      details: { order_id: "8a220e03-...", reason: "SHUTDOWN_CLEANUP" },
    },
    {
      timestamp: "2026-09-16T15:21:34Z",
      event_type: "ORDER_CANCELLED",
      ticker: "AAPL",
      cycle_id: "cycle_002",
      details: { order_id: "10b464c9-...", reason: "SHUTDOWN_CLEANUP" },
    },
    {
      timestamp: "2026-09-16T15:21:32Z",
      event_type: "ORDER_SUBMITTED",
      ticker: "MSFT",
      cycle_id: "cycle_001",
      details: { side: "BUY", quantity: 59.14, order_type: "MARKET" },
    },
    {
      timestamp: "2026-09-16T15:21:32Z",
      event_type: "RISK_DECISION",
      ticker: "MSFT",
      cycle_id: "cycle_001",
      details: { status: "APPROVED", approved_quantity: 59.14, rule_triggered: "NONE" },
    },
    {
      timestamp: "2026-09-16T15:21:32Z",
      event_type: "ORDER_SUBMITTED",
      ticker: "AAPL",
      cycle_id: "cycle_001",
      details: { side: "BUY", quantity: 94.68, order_type: "MARKET" },
    },
    {
      timestamp: "2026-09-16T15:21:32Z",
      event_type: "RISK_DECISION",
      ticker: "AAPL",
      cycle_id: "cycle_001",
      details: { status: "APPROVED", approved_quantity: 94.68, rule_triggered: "NONE" },
    },
    {
      timestamp: "2026-09-16T15:21:31Z",
      event_type: "SIGNAL_GENERATED",
      ticker: "AAPL",
      cycle_id: "cycle_001",
      details: { direction: "BUY", confidence: 0.84, target_weight: 0.3 },
    },
  ],
};

export const fallbackModels: ModelsData = {
  models: [
    {
      id: "naive_momentum",
      name: "Naive Momentum",
      family: "Heuristic Rule",
      directional_accuracy: 51.2,
      annualized_return: 11.4,
      sharpe_ratio: 0.65,
      sortino_ratio: 0.82,
      max_drawdown: -24.5,
      calmar_ratio: 0.47,
      win_rate: 49.1,
      profit_factor: 1.12,
      status: "Archived",
    },
    {
      id: "baseline_logistic",
      name: "Baseline Logistic Regression",
      family: "Linear ML",
      directional_accuracy: 53.1,
      annualized_return: 13.8,
      sharpe_ratio: 0.82,
      sortino_ratio: 1.05,
      max_drawdown: -21.2,
      calmar_ratio: 0.65,
      win_rate: 51.4,
      profit_factor: 1.25,
      status: "Baseline",
    },
    {
      id: "xgboost_lightgbm",
      name: "XGBoost / LightGBM (Phase 21)",
      family: "Gradient Boosted Trees",
      directional_accuracy: 58.4,
      annualized_return: 21.2,
      sharpe_ratio: 1.68,
      sortino_ratio: 2.15,
      max_drawdown: -12.4,
      calmar_ratio: 1.71,
      win_rate: 56.2,
      profit_factor: 1.64,
      status: "Production Ready",
    },
    {
      id: "bilstm_dl",
      name: "Bidirectional LSTM (Phase 26)",
      family: "Recurrent Deep Learning",
      directional_accuracy: 59.2,
      annualized_return: 22.8,
      sharpe_ratio: 1.82,
      sortino_ratio: 2.38,
      max_drawdown: -10.8,
      calmar_ratio: 2.11,
      win_rate: 57.0,
      profit_factor: 1.72,
      status: "Candidate",
    },
    {
      id: "transformer_dl",
      name: "Temporal Transformer (Phase 27)",
      family: "Attention Mechanism",
      directional_accuracy: 60.5,
      annualized_return: 24.1,
      sharpe_ratio: 2.05,
      sortino_ratio: 2.72,
      max_drawdown: -9.5,
      calmar_ratio: 2.54,
      win_rate: 58.1,
      profit_factor: 1.8,
      status: "Candidate",
    },
    {
      id: "production_ensemble",
      name: "Multi-Model Ensemble (Phase 34)",
      family: "Stacked Meta-Learner",
      directional_accuracy: 61.8,
      annualized_return: 26.5,
      sharpe_ratio: 2.24,
      sortino_ratio: 2.98,
      max_drawdown: -8.1,
      calmar_ratio: 3.27,
      win_rate: 59.4,
      profit_factor: 1.92,
      status: "Live Active",
    },
  ],
  feature_importance: [
    { feature: "RSI (14d)", category: "Momentum", importance: 0.385 },
    { feature: "FinBERT Sentiment", category: "NLP / Alternative", importance: 0.342 },
    { feature: "MACD Histogram", category: "Trend", importance: 0.298 },
    { feature: "Parkinson Volatility", category: "Microstructure", importance: 0.264 },
    { feature: "GARCH(1,1) Sigma", category: "Volatility", importance: 0.241 },
    { feature: "OBV Volume Flow", category: "Volume", importance: 0.218 },
    { feature: "Bollinger %B", category: "Mean-Reversion", importance: 0.195 },
    { feature: "ATR (14d)", category: "Volatility", importance: 0.182 },
    { feature: "ADX Trend Strength", category: "Trend", importance: 0.165 },
    { feature: "Garman-Klass Vol", category: "Microstructure", importance: 0.154 },
    { feature: "EMA-9 / SMA-21 Ratio", category: "Momentum", importance: 0.141 },
    { feature: "VWAP Deviation", category: "Execution", importance: 0.128 },
  ],
};

// -------------------------------------------------------------
// Exported API Methods
// -------------------------------------------------------------

export async function getOverview(): Promise<OverviewData> {
  return fetchWithFallback<OverviewData>("/overview", fallbackOverview);
}

export async function getChart(ticker: string, timeframe: string = "1M"): Promise<ChartData> {
  return fetchWithFallback<ChartData>(
    `/chart/${ticker}?timeframe=${timeframe}`,
    fallbackChart(ticker, timeframe)
  );
}

export async function getExplanation(ticker: string): Promise<ModelExplanation> {
  return fetchWithFallback<ModelExplanation>(
    `/chart/${ticker}/explain`,
    fallbackExplanation(ticker)
  );
}

export async function getBacktest(): Promise<BacktestData> {
  return fetchWithFallback<BacktestData>("/backtest", fallbackBacktest);
}

export async function getRisk(): Promise<RiskStatus> {
  return fetchWithFallback<RiskStatus>("/risk", fallbackRisk);
}

export async function getAudit(eventType?: string, ticker?: string): Promise<AuditResponse> {
  const query = new URLSearchParams();
  if (eventType) query.set("event_type", eventType);
  if (ticker) query.set("ticker", ticker);
  const qStr = query.toString() ? `?${query.toString()}` : "";
  return fetchWithFallback<AuditResponse>(`/audit${qStr}`, fallbackAudit);
}

export async function getModels(): Promise<ModelsData> {
  return fetchWithFallback<ModelsData>("/models", fallbackModels);
}
