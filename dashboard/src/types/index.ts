// TypeScript interfaces for Quant Trading Terminal

export interface OverviewMetrics {
  current_equity: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown_pct: number;
  win_rate_pct: number;
  profit_factor: number;
  sparklines: {
    equity: number[];
    pnl: number[];
    sharpe: number[];
    drawdown: number[];
    win_rate: number[];
  };
}

export interface EquityCurvePoint {
  date: string;
  strategy: number;
  benchmark: number;
  drawdown: number;
}

export interface TradeRecord {
  id: string;
  timestamp: string;
  ticker: string;
  side: "BUY" | "SELL";
  quantity: number;
  price: number;
  total_value: number;
  realized_pnl: number;
  status: "FILLED" | "ACCEPTED" | "CANCELLED" | "REJECTED";
}

export interface OverviewData {
  metrics: OverviewMetrics;
  equity_curve: EquityCurvePoint[];
  recent_trades: TradeRecord[];
}

export interface CandlestickBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sma20: number;
  sma60: number;
  bb_upper: number;
  bb_lower: number;
}

export interface ChartMarker {
  time: string;
  position: "aboveBar" | "belowBar" | "inBar";
  color: string;
  shape: "circle" | "square" | "arrowUp" | "arrowDown";
  text: string;
}

export interface ChartData {
  ticker: string;
  timeframe: string;
  bars: CandlestickBar[];
  markers: ChartMarker[];
}

export interface ModelFeatureDriver {
  feature: string;
  name: string;
  value: number;
  shap: number;
  impact: "positive" | "negative";
  description: string;
}

export interface ModelExplanation {
  ticker: string;
  model_name: string;
  prediction_time: string;
  signal: "BUY" | "SELL" | "HOLD";
  confidence: number;
  regime: string;
  features: ModelFeatureDriver[];
}

export interface TearsheetMetrics {
  annualized_return_pct: number;
  benchmark_annualized_return_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  max_drawdown_pct: number;
  volatility_annualized_pct: number;
  win_rate_pct: number;
  profit_factor: number;
  alpha: number;
  beta: number;
  information_ratio: number;
  tracking_error_pct: number;
  var_95_daily_pct: number;
  cvar_95_daily_pct: number;
}

export interface RollingMetricPoint {
  date: string;
  rolling_sharpe: number;
  rolling_volatility: number;
}

export interface TradePnlBin {
  bin: string;
  count: number;
}

export interface RegimePerformance {
  regime: string;
  strategy_return: number;
  benchmark_return: number;
}

export interface MonteCarloBand {
  day: string;
  p5: number;
  p50: number;
  p95: number;
}

export interface BacktestData {
  tearsheet: TearsheetMetrics;
  rolling_metrics: RollingMetricPoint[];
  trade_pnl_distribution: TradePnlBin[];
  regime_breakdown: RegimePerformance[];
  monte_carlo_bands: MonteCarloBand[];
}

export interface RiskPosition {
  ticker: string;
  market_value: number;
  weight_pct: number;
  limit_pct: number;
}

export interface RiskStatus {
  engine_state: string;
  circuit_breaker_status: string;
  current_drawdown_pct: number;
  max_drawdown_limit_pct: number;
  drawdown_headroom_pct: number;
  gross_exposure_value: number;
  gross_exposure_pct: number;
  max_gross_exposure_pct: number;
  net_exposure_value: number;
  net_exposure_pct: number;
  max_position_size_pct: number;
  positions: RiskPosition[];
  kill_switch_active: boolean;
  last_reconciliation_time: string;
  reconciliation_status: string;
}

export interface AuditEventRecord {
  timestamp: string;
  event_type: string;
  ticker?: string | null;
  cycle_id: string;
  details: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface AuditResponse {
  total: number;
  offset: number;
  limit: number;
  events: AuditEventRecord[];
}

export interface ModelMatrixEntry {
  id: string;
  name: string;
  family: string;
  directional_accuracy: number;
  annualized_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  win_rate: number;
  profit_factor: number;
  status: string;
}

export interface FeatureImportanceItem {
  feature: string;
  category: string;
  importance: number;
}

export interface ModelsData {
  models: ModelMatrixEntry[];
  feature_importance: FeatureImportanceItem[];
}
