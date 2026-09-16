"use client";

import * as React from "react";
import { getBacktest } from "@/lib/api-client";
import { BacktestData } from "@/types";
import { RollingMetricsChart } from "@/components/charts/RollingMetricsChart";
import { MonteCarloChart } from "@/components/charts/MonteCarloChart";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  Cell,
} from "recharts";
import { formatPercent } from "@/lib/utils";
import { Award, Compass, BarChart3, ShieldCheck, TrendingUp, Activity } from "lucide-react";

export default function BacktestAnalysisView() {
  const [data, setData] = React.useState<BacktestData | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    async function load() {
      try {
        const res = await getBacktest();
        setData(res);
      } catch (err) {
        console.error("Failed to load backtest analytics:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading || !data) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-44 w-full rounded-[4px] bg-[#141414]" />
          ))}
        </div>
        <Skeleton className="h-72 w-full rounded-[4px] bg-[#141414]" />
        <Skeleton className="h-72 w-full rounded-[4px] bg-[#141414]" />
      </div>
    );
  }

  const { tearsheet, rolling_metrics, trade_pnl_distribution, regime_breakdown, monte_carlo_bands } = data;

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* 1. Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#1F1F1F] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold tracking-tight text-[#FAFAFA]">
              Quantitative Tearsheet & Backtest Verification
            </h1>
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-[4px] bg-[#141414] text-[#888888] border border-[#1F1F1F]">
              Phase 42 / 43
            </span>
          </div>
          <p className="text-xs text-[#666666] mt-0.5">
            Standardized institutional walk-forward tearsheet, rolling alpha stability, and regime-conditional stress testing.
          </p>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="px-3 py-1.5 rounded-[4px] bg-[#0C0C0C] border border-[#1F1F1F]">
            <span className="text-[#666666] mr-2">ANNUAL ALPHA</span>
            <span className="text-[#00B386] font-bold">+{tearsheet.alpha.toFixed(3)}</span>
          </div>
          <div className="px-3 py-1.5 rounded-[4px] bg-[#0C0C0C] border border-[#1F1F1F]">
            <span className="text-[#666666] mr-2">SHARPE</span>
            <span className="text-[#FAFAFA] font-bold">{tearsheet.sharpe_ratio.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* 2. Tearsheet Metrics Table Matrix */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Card 1: Return & Alpha */}
        <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
          <CardHeader className="p-4 pb-2 border-b border-[#141414]">
            <CardTitle className="text-[11px] uppercase tracking-wider text-[#666666] font-mono flex items-center gap-1.5">
              <TrendingUp className="h-3.5 w-3.5 text-[#00B386]" />
              Return & Benchmark Alpha
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-2 space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Annualized Return</span>
              <span className="font-bold text-[#00B386] tabular-nums">
                {formatPercent(tearsheet.annualized_return_pct)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Benchmark (SPY)</span>
              <span className="text-[#CCCCCC] tabular-nums">
                {formatPercent(tearsheet.benchmark_annualized_return_pct)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Jensen&apos;s Alpha</span>
              <span className="font-bold text-[#00B386] tabular-nums">
                +{tearsheet.alpha.toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Beta to SPY</span>
              <span className="text-[#CCCCCC] tabular-nums">{tearsheet.beta.toFixed(2)}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#777777]">Information Ratio</span>
              <span className="font-bold text-[#00B386] tabular-nums">
                {tearsheet.information_ratio.toFixed(2)}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Card 2: Risk-Adjusted Ratios */}
        <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
          <CardHeader className="p-4 pb-2 border-b border-[#141414]">
            <CardTitle className="text-[11px] uppercase tracking-wider text-[#666666] font-mono flex items-center gap-1.5">
              <Activity className="h-3.5 w-3.5 text-[#FAFAFA]" />
              Risk-Adjusted Performance
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-2 space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Sharpe Ratio (Rf=0%)</span>
              <span className="font-bold text-[#00B386] tabular-nums">
                {tearsheet.sharpe_ratio.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Sortino Ratio</span>
              <span className="font-bold text-[#00B386] tabular-nums">
                {tearsheet.sortino_ratio.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Calmar Ratio</span>
              <span className="font-bold text-[#00B386] tabular-nums">
                {tearsheet.calmar_ratio.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Annualized Volatility</span>
              <span className="text-[#CCCCCC] tabular-nums">
                {formatPercent(tearsheet.volatility_annualized_pct, false)}
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#777777]">Tracking Error</span>
              <span className="text-[#CCCCCC] tabular-nums">
                {formatPercent(tearsheet.tracking_error_pct, false)}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Card 3: Downside Tail Risk */}
        <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
          <CardHeader className="p-4 pb-2 border-b border-[#141414]">
            <CardTitle className="text-[11px] uppercase tracking-wider text-[#666666] font-mono flex items-center gap-1.5">
              <Award className="h-3.5 w-3.5 text-[#EB5757]" />
              Tail Risk & Execution Stats
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-2 space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Maximum Drawdown</span>
              <span className="font-bold text-[#EB5757] tabular-nums">
                {tearsheet.max_drawdown_pct.toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Daily 95% VaR</span>
              <span className="text-[#EB5757] tabular-nums">
                {tearsheet.var_95_daily_pct.toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Daily 95% CVaR</span>
              <span className="text-[#EB5757] tabular-nums">
                {tearsheet.cvar_95_daily_pct.toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Win Rate</span>
              <span className="font-bold text-[#00B386] tabular-nums">
                {tearsheet.win_rate_pct.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#777777]">Profit Factor</span>
              <span className="font-bold text-[#00B386] tabular-nums">
                {tearsheet.profit_factor.toFixed(2)}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 3. Rolling Sharpe & Volatility Analysis */}
      <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-[#141414]">
          <CardTitle className="text-xs font-semibold text-[#FAFAFA] flex items-center justify-between">
            <span>Rolling 6-Month Sharpe Ratio & Volatility Trajectory</span>
            <span className="text-[10px] text-[#666666] font-normal font-mono">Rolling Window: 126 Trading Days</span>
          </CardTitle>
          <CardDescription className="text-[11px] text-[#666666]">
            Monitors alpha consistency and detects Sharpe decay across multi-quarter market regimes.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-4">
          <RollingMetricsChart data={rolling_metrics} height={260} />
        </CardContent>
      </Card>

      {/* 4. Regime Conditional Performance & Trade Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Regime Breakdown */}
        <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
          <CardHeader className="p-4 pb-2 border-b border-[#141414]">
            <CardTitle className="text-xs font-semibold flex items-center gap-1.5 text-[#FAFAFA]">
              <Compass className="h-4 w-4 text-[#888888]" />
              Regime-Conditional Breakdown (Phase 43)
            </CardTitle>
            <CardDescription className="text-[11px] text-[#666666]">
              Strategy alpha vs SPY across HMM/GMM detected macroeconomic regimes.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-4">
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={regime_breakdown} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke="#161616" vertical={false} />
                  <XAxis dataKey="regime" stroke="#555555" fontSize={10} tickLine={false} axisLine={{ stroke: "#1F1F1F" }} />
                  <YAxis stroke="#555555" fontSize={10} tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0C0C0C",
                      borderColor: "#1F1F1F",
                      borderRadius: "4px",
                      fontSize: "11px",
                      color: "#FAFAFA",
                      boxShadow: "0 8px 24px rgba(0,0,0,0.8)",
                    }}
                    formatter={(v: any) => [`${v}%`, ""]}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
                  <Bar dataKey="strategy_return" name="Strategy Return (%)" fill="#00B386" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="benchmark_return" name="SPY Return (%)" fill="#2E2E2E" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Trade P&L Distribution */}
        <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
          <CardHeader className="p-4 pb-2 border-b border-[#141414]">
            <CardTitle className="text-xs font-semibold flex items-center gap-1.5 text-[#FAFAFA]">
              <BarChart3 className="h-4 w-4 text-[#888888]" />
              Trade P&L Distribution Histogram
            </CardTitle>
            <CardDescription className="text-[11px] text-[#666666]">
              Frequency distribution of individual trade returns confirming positive right-tail skew.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-4">
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={trade_pnl_distribution} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke="#161616" vertical={false} />
                  <XAxis dataKey="bin" stroke="#555555" fontSize={10} tickLine={false} axisLine={{ stroke: "#1F1F1F" }} />
                  <YAxis stroke="#555555" fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0C0C0C",
                      borderColor: "#1F1F1F",
                      borderRadius: "4px",
                      fontSize: "11px",
                      color: "#FAFAFA",
                      boxShadow: "0 8px 24px rgba(0,0,0,0.8)",
                    }}
                    formatter={(v: any) => [`${v} executions`, "Frequency"]}
                  />
                  <Bar dataKey="count" name="Executed Trades" radius={[2, 2, 0, 0]}>
                    {trade_pnl_distribution.map((entry, index) => {
                      const isNegative = entry.bin.includes("-");
                      return (
                        <Cell
                          key={`bar-${index}`}
                          fill={isNegative ? "#EB5757" : "#00B386"}
                        />
                      );
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 5. Monte Carlo Simulation Bands */}
      <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-[#141414]">
          <CardTitle className="text-xs font-semibold flex items-center gap-1.5 text-[#FAFAFA]">
            <ShieldCheck className="h-4 w-4 text-[#00B386]" />
            Forward Monte Carlo Simulation Bands (1,000 Paths)
          </CardTitle>
          <CardDescription className="text-[11px] text-[#666666]">
            Synthetic paths projected forward: 5th percentile (stress boundary), 50th (median), and 95th percentile (bullish trajectory).
          </CardDescription>
        </CardHeader>
        <CardContent className="p-4">
          <MonteCarloChart data={monte_carlo_bands} height={260} />
        </CardContent>
      </Card>
    </div>
  );
}
