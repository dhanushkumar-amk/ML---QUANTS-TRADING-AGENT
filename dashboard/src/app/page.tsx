"use client";

import * as React from "react";
import { getOverview } from "@/lib/api-client";
import { OverviewData } from "@/types";
import { HeroSection } from "@/components/landing/HeroSection";
import { MetricCard } from "@/components/MetricCard";
import { EquityCurveChart } from "@/components/charts/EquityCurveChart";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { formatCurrency, formatPercent, cn } from "@/lib/utils";
import {
  DollarSign,
  Percent,
  TrendingUp,
  Shield,
  ArrowUpRight,
  ArrowDownRight,
  Layers,
  Download,
  Calendar,
  ChevronDown,
  PieChart,
  ShieldCheck,
} from "lucide-react";

export default function OverviewPage() {
  const [data, setData] = React.useState<OverviewData | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [activeFilter, setActiveFilter] = React.useState("ALL");

  const dashboardRef = React.useRef<HTMLDivElement>(null);

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
    const timer = setInterval(load, 15000);
    return () => clearInterval(timer);
  }, []);

  const scrollToDashboard = () => {
    dashboardRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  if (loading || !data) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-96 w-full rounded-[4px] bg-[#111]" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-32 w-full rounded-[4px] bg-[#111]" />
          ))}
        </div>
        <Skeleton className="h-[380px] w-full rounded-[4px] bg-[#111]" />
      </div>
    );
  }

  const { metrics, equity_curve, recent_trades } = data;

  const filteredTrades =
    activeFilter === "ALL"
      ? recent_trades
      : recent_trades.filter((t) => t.ticker === activeFilter);

  return (
    <div className="space-y-8 pb-12 font-mono">
      {/* 1. Landing Hero with 3D Centerpiece */}
      <div className="-mx-6 md:-mx-8 -mt-6 md:-mt-8">
        <HeroSection onExploreDashboard={scrollToDashboard} />
      </div>

      {/* 2. Executive Portfolio Overview Section */}
      <div ref={dashboardRef} className="space-y-6 pt-4">
        {/* Header Strip */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-[#1f1f1f]">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold tracking-tight text-white uppercase">
                Portfolio Performance
              </h2>
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-[3px] text-[10px] font-semibold bg-[#00B386]/10 text-[#00B386] border border-[#00B386]/25">
                <span className="h-1.5 w-1.5 rounded-full bg-[#00B386] animate-pulse" />
                ALPACA PAPER
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 font-sans">
              Algorithmic execution metrics, risk engine headroom, and broker trade fills.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-[4px] bg-[#0c0c0c] border border-[#1f1f1f] text-xs text-slate-300">
              <Calendar className="h-3.5 w-3.5 text-slate-500" />
              <span>Session: 2026-09-16</span>
            </div>

            <Button
              variant="outline"
              size="sm"
              className="h-8 rounded-[4px] bg-[#0c0c0c] border-[#1f1f1f] text-xs text-slate-300 hover:text-white hover:bg-[#141414] gap-1.5"
              onClick={() => alert("Exporting trades blotter...")}
            >
              <Download className="h-3.5 w-3.5" />
              <span>Export Blotter</span>
            </Button>
          </div>
        </div>

        {/* 4 Primary Metric Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="PORTFOLIO EQUITY"
            value={formatCurrency(metrics.current_equity)}
            change={`+${formatPercent(metrics.daily_pnl_pct)}`}
            changeType="profit"
            subtitle={`Day: +${formatCurrency(metrics.daily_pnl)}`}
            sparklineData={metrics.sparklines.equity}
            icon={<DollarSign className="h-4 w-4" />}
          />

          <MetricCard
            title="ANNUALIZED SHARPE"
            value={metrics.sharpe_ratio.toFixed(2)}
            change="Target &gt; 2.0"
            changeType="profit"
            subtitle={`Sortino: ${metrics.sortino_ratio.toFixed(2)}`}
            sparklineData={metrics.sparklines.sharpe}
            icon={<TrendingUp className="h-4 w-4" />}
          />

          <MetricCard
            title="MAX DRAWDOWN"
            value={`${metrics.max_drawdown_pct.toFixed(2)}%`}
            change="Limit: -15.0%"
            changeType={Math.abs(metrics.max_drawdown_pct) < 10 ? "profit" : "loss"}
            subtitle="Headroom: +6.58%"
            sparklineData={metrics.sparklines.drawdown}
            icon={<Shield className="h-4 w-4" />}
          />

          <MetricCard
            title="STRATEGY WIN RATE"
            value={`${metrics.win_rate_pct.toFixed(1)}%`}
            change={`PF: ${metrics.profit_factor.toFixed(2)}`}
            changeType="profit"
            subtitle="Exp: +$142/trade"
            sparklineData={metrics.sparklines.win_rate}
            icon={<Percent className="h-4 w-4" />}
          />
        </div>

        {/* Hero Equity Curve Chart */}
        <div className="rounded-[4px] border border-[#1f1f1f] bg-[#0c0c0c] p-4 sm:p-5 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-[#1a1a1a]">
            <div>
              <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Layers className="h-4 w-4 text-[#00B386]" />
                Equity Trajectory vs Benchmark &amp; Drawdown
              </h3>
              <p className="text-[11px] text-slate-500 font-sans mt-0.5">
                Daily marked-to-market performance against SPY index.
              </p>
            </div>
            <span className="text-xs text-[#00B386] bg-[#00B386]/10 px-2.5 py-1 rounded-[3px] border border-[#00B386]/25 font-bold">
              Alpha: +18.6%
            </span>
          </div>

          <EquityCurveChart data={equity_curve} height={370} />
        </div>

        {/* Split Section: Order Blotter & Asset Allocation */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Order Blotter (Zerodha Kite Style) */}
          <div className="lg:col-span-8 rounded-[4px] border border-[#1f1f1f] bg-[#0c0c0c] overflow-hidden flex flex-col justify-between">
            <div>
              <div className="p-3.5 border-b border-[#1a1a1a] flex items-center justify-between">
                <div>
                  <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                    Executed Order Blotter
                  </h3>
                  <p className="text-[11px] text-slate-500 font-sans mt-0.5">
                    Recent broker fills on Alpaca Paper Rail.
                  </p>
                </div>

                {/* Quick Filter */}
                <div className="flex items-center gap-1 bg-[#141414] p-0.5 rounded-[3px] border border-[#222]">
                  {["ALL", "AAPL", "MSFT"].map((sym) => (
                    <button
                      key={sym}
                      onClick={() => setActiveFilter(sym)}
                      className={cn(
                        "px-2 py-0.5 text-[10px] rounded-[2px] transition-colors",
                        activeFilter === sym
                          ? "bg-white text-black font-bold"
                          : "text-slate-400 hover:text-white"
                      )}
                    >
                      {sym}
                    </button>
                  ))}
                </div>
              </div>

              <Table>
                <TableHeader>
                  <TableRow className="border-b border-[#1a1a1a] bg-[#080808] hover:bg-transparent">
                    <TableHead className="text-slate-500 text-[11px]">Time (UTC)</TableHead>
                    <TableHead className="text-slate-500 text-[11px]">Symbol</TableHead>
                    <TableHead className="text-slate-500 text-[11px]">Type</TableHead>
                    <TableHead className="text-right text-slate-500 text-[11px]">Qty</TableHead>
                    <TableHead className="text-right text-slate-500 text-[11px]">Price</TableHead>
                    <TableHead className="text-right text-slate-500 text-[11px]">Value</TableHead>
                    <TableHead className="text-right text-slate-500 text-[11px]">P&amp;L</TableHead>
                    <TableHead className="text-center text-slate-500 text-[11px]">Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredTrades.map((trade) => {
                    const isBuy = trade.side === "BUY";
                    return (
                      <TableRow
                        key={trade.id}
                        className="border-b border-[#141414] text-xs hover:bg-[#111111] transition-colors"
                      >
                        <TableCell className="text-slate-400 tabular-nums">
                          {trade.timestamp}
                        </TableCell>
                        <TableCell>
                          <span className="font-bold text-white bg-[#141414] px-1.5 py-0.5 rounded-[3px] border border-[#222]">
                            {trade.ticker}
                          </span>
                        </TableCell>
                        <TableCell>
                          <span
                            className={cn(
                              "inline-flex items-center gap-1 px-1.5 py-0.2 rounded-[2px] text-[10px] font-bold border",
                              isBuy
                                ? "bg-[#00B386]/10 text-[#00B386] border-[#00B386]/30"
                                : "bg-[#EB5757]/10 text-[#EB5757] border-[#EB5757]/30"
                            )}
                          >
                            {isBuy ? (
                              <ArrowUpRight className="h-3 w-3" />
                            ) : (
                              <ArrowDownRight className="h-3 w-3" />
                            )}
                            {trade.side}
                          </span>
                        </TableCell>
                        <TableCell className="text-right text-slate-300 tabular-nums">{trade.quantity.toFixed(2)}</TableCell>
                        <TableCell className="text-right text-slate-300 tabular-nums">{formatCurrency(trade.price)}</TableCell>
                        <TableCell className="text-right text-slate-300 tabular-nums">{formatCurrency(trade.total_value)}</TableCell>
                        <TableCell
                          className={cn(
                            "text-right font-bold tabular-nums",
                            trade.realized_pnl > 0
                              ? "text-[#00B386]"
                              : trade.realized_pnl < 0
                              ? "text-[#EB5757]"
                              : "text-slate-500"
                          )}
                        >
                          {trade.realized_pnl === 0
                            ? "$0.00"
                            : formatCurrency(trade.realized_pnl)}
                        </TableCell>
                        <TableCell className="text-center">
                          <span className="inline-flex items-center px-1.5 py-0.2 rounded-[2px] text-[10px] bg-[#141414] border border-[#222] text-slate-400">
                            {trade.status}
                          </span>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </div>

          {/* Allocation & Risk Guardrails (Groww Style) */}
          <div className="lg:col-span-4 rounded-[4px] border border-[#1f1f1f] bg-[#0c0c0c] p-4 space-y-4 flex flex-col justify-between">
            <div className="space-y-3.5">
              <div className="flex items-center justify-between pb-2.5 border-b border-[#1a1a1a]">
                <div className="flex items-center gap-2">
                  <PieChart className="h-4 w-4 text-[#00B386]" />
                  <h3 className="text-xs font-bold text-white uppercase tracking-wider">Asset Allocation</h3>
                </div>
                <span className="text-[10px] text-slate-500">4 Assets</span>
              </div>

              {/* Progress bars */}
              <div className="space-y-3 text-xs">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-300 font-medium">AAPL (Apple)</span>
                    <span className="text-[#00B386] font-bold tabular-nums">35.4%</span>
                  </div>
                  <Progress value={35.4} className="h-1.5 bg-[#181818]" />
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-300 font-medium">MSFT (Microsoft)</span>
                    <span className="text-slate-200 font-bold tabular-nums">28.2%</span>
                  </div>
                  <Progress value={28.2} className="h-1.5 bg-[#181818]" />
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-300 font-medium">NVDA (NVIDIA)</span>
                    <span className="text-slate-200 font-bold tabular-nums">21.8%</span>
                  </div>
                  <Progress value={21.8} className="h-1.5 bg-[#181818]" />
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-400 font-medium">USD Cash Reserve</span>
                    <span className="text-slate-500 font-bold tabular-nums">14.6%</span>
                  </div>
                  <Progress value={14.6} className="h-1.5 bg-[#181818]" />
                </div>
              </div>
            </div>

            {/* Risk Box */}
            <div className="p-3 rounded-[4px] bg-[#111111] border border-[#1f1f1f] space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <ShieldCheck className="h-3.5 w-3.5 text-[#00B386]" />
                  Risk Engine Gating
                </span>
                <span className="text-[#00B386] font-bold text-[10px]">ACTIVE</span>
              </div>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-500">Max Drawdown Limit</span>
                <span className="text-slate-300 tabular-nums">-15.0%</span>
              </div>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-500">Current Safety Headroom</span>
                <span className="text-[#00B386] font-bold tabular-nums">+6.58%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
