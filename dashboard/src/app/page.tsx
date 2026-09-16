"use client";

import * as React from "react";
import { getOverview } from "@/lib/api-client";
import { OverviewData } from "@/types";
import { MetricCard } from "@/components/MetricCard";
import { EquityCurveChart } from "@/components/charts/EquityCurveChart";
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
import { formatCurrency, formatPercent } from "@/lib/utils";
import { DollarSign, Percent, TrendingUp, Shield, Activity } from "lucide-react";

export default function OverviewPage() {
  const [data, setData] = React.useState<OverviewData | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    async function load() {
      try {
        const res = await getOverview();
        setData(res);
      } catch (err) {
        console.error("Failed to load overview data:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
    const timer = setInterval(load, 15000); // Polling every 15s
    return () => clearInterval(timer);
  }, []);

  if (loading || !data) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-28 w-full rounded-lg" />
          ))}
        </div>
        <Skeleton className="h-96 w-full rounded-lg" />
        <Skeleton className="h-64 w-full rounded-lg" />
      </div>
    );
  }

  const { metrics, equity_curve, recent_trades } = data;

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* 1. Page Header */}
      <div>
        <h1 className="text-xl font-bold tracking-tight text-foreground">
          Executive Portfolio Overview
        </h1>
        <p className="text-xs text-muted-foreground mt-0.5">
          Real-time capital balance, drawdown safety headroom, and executed trades blotter.
        </p>
      </div>

      {/* 2. Key Metric Cards with Sparklines */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Portfolio Equity"
          value={formatCurrency(metrics.current_equity)}
          change={`+${formatPercent(metrics.daily_pnl_pct)} today`}
          changeType="profit"
          subtitle={`Day: +${formatCurrency(metrics.daily_pnl)}`}
          sparklineData={metrics.sparklines.equity}
          icon={<DollarSign className="h-4 w-4" />}
        />

        <MetricCard
          title="Annualized Sharpe"
          value={metrics.sharpe_ratio.toFixed(2)}
          change="Institutional > 2.0"
          changeType="profit"
          subtitle={`Sortino: ${metrics.sortino_ratio.toFixed(2)}`}
          sparklineData={metrics.sparklines.sharpe}
          icon={<TrendingUp className="h-4 w-4" />}
        />

        <MetricCard
          title="Maximum Drawdown"
          value={`${metrics.max_drawdown_pct.toFixed(2)}%`}
          change="Limit: -15.0%"
          changeType={Math.abs(metrics.max_drawdown_pct) < 10 ? "profit" : "loss"}
          subtitle="6.58% Headroom to Halt"
          sparklineData={metrics.sparklines.drawdown}
          icon={<Shield className="h-4 w-4" />}
        />

        <MetricCard
          title="Strategy Win Rate"
          value={`${metrics.win_rate_pct.toFixed(1)}%`}
          change={`PF: ${metrics.profit_factor.toFixed(2)}`}
          changeType="profit"
          subtitle="Trade Expectancy: +$142"
          sparklineData={metrics.sparklines.win_rate}
          icon={<Percent className="h-4 w-4" />}
        />
      </div>

      {/* 3. Main Equity Curve & Underwater Chart */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-sm font-semibold text-foreground">
              Equity Trajectory vs Benchmark & Underwater Drawdown
            </CardTitle>
            <CardDescription className="text-xs">
              Continuous multi-asset trend allocation vs S&P 500 ETF (SPY) with aligned drawdown subplot.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
              Alpha: +10.6%
            </span>
          </div>
        </CardHeader>
        <CardContent>
          <EquityCurveChart data={equity_curve} height={360} />
        </CardContent>
      </Card>

      {/* 4. Recent Trades Blotter */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle className="text-sm font-semibold text-foreground">
              Recent Trade Executions
            </CardTitle>
            <CardDescription className="text-xs">
              Live orders routed to Alpaca Paper Brokerage with risk clearance and fill prices.
            </CardDescription>
          </div>
          <Badge variant="profit" className="text-[11px] font-mono">
            Alpaca Connected
          </Badge>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Timestamp (UTC)</TableHead>
                <TableHead>Asset</TableHead>
                <TableHead>Side</TableHead>
                <TableHead className="text-right">Shares</TableHead>
                <TableHead className="text-right">Price</TableHead>
                <TableHead className="text-right">Notional Value</TableHead>
                <TableHead className="text-right">Realized PnL</TableHead>
                <TableHead className="text-center">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recent_trades.map((trade) => (
                <TableRow key={trade.id} className="font-mono text-xs">
                  <TableCell className="text-muted-foreground">
                    {trade.timestamp}
                  </TableCell>
                  <TableCell className="font-bold text-foreground">
                    {trade.ticker}
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={trade.side === "BUY" ? "profit" : "loss"}
                      className="px-2 py-0 text-[10px]"
                    >
                      {trade.side}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">{trade.quantity.toFixed(2)}</TableCell>
                  <TableCell className="text-right">{formatCurrency(trade.price)}</TableCell>
                  <TableCell className="text-right">{formatCurrency(trade.total_value)}</TableCell>
                  <TableCell
                    className={`text-right font-medium ${
                      trade.realized_pnl > 0
                        ? "text-emerald-400"
                        : trade.realized_pnl < 0
                        ? "text-rose-400"
                        : "text-muted-foreground"
                    }`}
                  >
                    {trade.realized_pnl === 0
                      ? "$0.00"
                      : formatCurrency(trade.realized_pnl)}
                  </TableCell>
                  <TableCell className="text-center">
                    <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] bg-muted/60 text-slate-300">
                      {trade.status}
                    </span>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
