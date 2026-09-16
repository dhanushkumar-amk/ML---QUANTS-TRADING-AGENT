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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";
import { formatPercent } from "@/lib/utils";
import { Award, Compass, BarChart3, ShieldCheck } from "lucide-react";

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
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24 w-full rounded-lg" />
          ))}
        </div>
        <Skeleton className="h-72 w-full rounded-lg" />
        <Skeleton className="h-72 w-full rounded-lg" />
      </div>
    );
  }

  const { tearsheet, rolling_metrics, trade_pnl_distribution, regime_breakdown, monte_carlo_bands } = data;

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* 1. Page Header */}
      <div>
        <h1 className="text-xl font-bold tracking-tight text-foreground flex items-center gap-2">
          <Award className="h-5 w-5 text-emerald-400" />
          Quantitative Tearsheet & Backtest Analysis
        </h1>
        <p className="text-xs text-muted-foreground mt-0.5">
          Standardized institutional tearsheet (Phase 42) & regime-conditional stress testing (Phase 43).
        </p>
      </div>

      {/* 2. Tearsheet Metrics Table Matrix */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1: Return & Alpha */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs uppercase text-muted-foreground">
              Return & Benchmark Alpha
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Annualized Return</span>
              <span className="font-bold text-emerald-400">
                {formatPercent(tearsheet.annualized_return_pct)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Benchmark (SPY)</span>
              <span className="text-slate-300">
                {formatPercent(tearsheet.benchmark_annualized_return_pct)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Jensen&apos;s Alpha</span>
              <span className="font-bold text-emerald-400">
                +{tearsheet.alpha.toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Beta to SPY</span>
              <span className="text-slate-300">{tearsheet.beta.toFixed(2)}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-muted-foreground">Information Ratio</span>
              <span className="font-bold text-emerald-400">
                {tearsheet.information_ratio.toFixed(2)}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Card 2: Risk-Adjusted Ratios */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs uppercase text-muted-foreground">
              Risk-Adjusted Ratios
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Sharpe Ratio (Rf=0%)</span>
              <span className="font-bold text-emerald-400">
                {tearsheet.sharpe_ratio.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Sortino Ratio</span>
              <span className="font-bold text-emerald-400">
                {tearsheet.sortino_ratio.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Calmar Ratio</span>
              <span className="font-bold text-emerald-400">
                {tearsheet.calmar_ratio.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Annualized Volatility</span>
              <span className="text-slate-300">
                {formatPercent(tearsheet.volatility_annualized_pct, false)}
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-muted-foreground">Tracking Error</span>
              <span className="text-slate-300">
                {formatPercent(tearsheet.tracking_error_pct, false)}
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Card 3: Downside Tail Risk */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs uppercase text-muted-foreground">
              Tail Risk & Trade Statistics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Maximum Drawdown</span>
              <span className="font-bold text-rose-400">
                {tearsheet.max_drawdown_pct.toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Daily 95% VaR</span>
              <span className="text-rose-400">
                {tearsheet.var_95_daily_pct.toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Daily 95% CVaR (Expected Shortfall)</span>
              <span className="text-rose-400">
                {tearsheet.cvar_95_daily_pct.toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Win Rate</span>
              <span className="font-bold text-emerald-400">
                {tearsheet.win_rate_pct.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-muted-foreground">Profit Factor</span>
              <span className="font-bold text-emerald-400">
                {tearsheet.profit_factor.toFixed(2)}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 3. Rolling Sharpe & Volatility Analysis */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-semibold text-foreground">
            Rolling 6-Month Sharpe Ratio & Volatility Trajectory
          </CardTitle>
          <CardDescription className="text-xs">
            Evaluates alpha stability over changing market dynamics to detect alpha decay or regime sensitivity.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RollingMetricsChart data={rolling_metrics} height={260} />
        </CardContent>
      </Card>

      {/* 4. Regime Conditional Performance & Trade Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Regime Breakdown */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold flex items-center gap-1.5 text-foreground">
              <Compass className="h-4 w-4 text-cyan-400" />
              Regime-Conditional Breakdown (Phase 43)
            </CardTitle>
            <CardDescription className="text-xs">
              Strategy vs SPY performance across HMM/GMM detected macroeconomic regimes.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={regime_breakdown} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <XAxis dataKey="regime" stroke="#64748b" fontSize={10} tickLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={10} tickFormatter={(v) => `${v}%`} tickLine={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "rgba(51, 65, 85, 0.8)",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                    formatter={(v: any) => [`${v}%`, ""]}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
                  <Bar dataKey="strategy_return" name="Strategy Return (%)" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="benchmark_return" name="SPY Return (%)" fill="#64748b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Trade P&L Distribution */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold flex items-center gap-1.5 text-foreground">
              <BarChart3 className="h-4 w-4 text-emerald-400" />
              Trade P&L Distribution Histogram
            </CardTitle>
            <CardDescription className="text-xs">
              Frequency distribution of individual trade returns verifying positive right-tail skew.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={trade_pnl_distribution} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <XAxis dataKey="bin" stroke="#64748b" fontSize={10} tickLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={10} tickLine={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "rgba(51, 65, 85, 0.8)",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                    formatter={(v: any) => [`${v} trades`, "Frequency"]}
                  />
                  <Bar dataKey="count" name="Executed Trades" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 5. Monte Carlo Simulation Bands */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-semibold flex items-center gap-1.5 text-foreground">
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
            Forward Monte Carlo Simulation Bands (Phase 43)
          </CardTitle>
          <CardDescription className="text-xs">
            1,000 synthetic paths projected forward: 5th percentile (stress boundary), 50th (median), and 95th percentile.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <MonteCarloChart data={monte_carlo_bands} height={260} />
        </CardContent>
      </Card>
    </div>
  );
}
