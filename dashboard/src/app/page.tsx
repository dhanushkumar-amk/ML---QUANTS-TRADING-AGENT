"use client";

import * as React from "react";
import { getOverview } from "@/lib/api-client";
import { OverviewData, TradeRecord } from "@/types";
import { HeroSection } from "@/components/landing/HeroSection";
import { MetricCard } from "@/components/MetricCard";
import { EquityCurveChart } from "@/components/charts/EquityCurveChart";
import { SystemArchitectureModal } from "@/components/architecture/SystemArchitectureModal";
import { TradeReasoningModal } from "@/components/trades/TradeReasoningModal";
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
  TrendingUp,
  Shield,
  ArrowUpRight,
  ArrowDownRight,
  Download,
  Calendar,
  PieChart,
  ShieldCheck,
  Cpu,
  Sparkles,
  ArrowRight,
  Database,
  Binary,
  Zap,
} from "lucide-react";

export default function OverviewPage() {
  const [data, setData] = React.useState<OverviewData | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [activeFilter, setActiveFilter] = React.useState("ALL");
  const [selectedTradeForReasoning, setSelectedTradeForReasoning] = React.useState<TradeRecord | null>(null);
  const [isArchModalOpen, setIsArchModalOpen] = React.useState(false);

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
        <HeroSection
          onExploreDashboard={scrollToDashboard}
          onOpenArchitecture={() => setIsArchModalOpen(true)}
        />
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
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-[3px] text-[10px] font-semibold bg-[#E5A93B]/10 text-[#E5A93B] border border-[#E5A93B]/30">
                <span className="h-1.5 w-1.5 rounded-full bg-[#E5A93B] animate-pulse" />
                ALPACA PAPER ENVIRONMENT
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 font-sans">
              Continuous walk-forward execution, real-time risk engine gating, and broker order fills.
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
              onClick={() => {
                const blob = new Blob([JSON.stringify(recent_trades, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "paper_trades_blotter.json";
                a.click();
              }}
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
            change="Target > 2.0"
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
            title="WIN RATE"
            value={`${metrics.win_rate_pct.toFixed(1)}%`}
            change={`PF: ${metrics.profit_factor.toFixed(2)}`}
            changeType="profit"
            subtitle={`${recent_trades.length} Recent Fills`}
            sparklineData={metrics.sparklines.equity}
            icon={<ShieldCheck className="h-4 w-4" />}
          />
        </div>

        {/* Recharts Equity Curve & Underwater Drawdown Chart */}
        <div className="rounded-[4px] border border-[#1f1f1f] bg-[#0c0c0c] p-4 sm:p-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 mb-2 border-b border-[#181818]">
            <div>
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                Cumulative Equity &amp; Underwater Drawdown
              </h3>
              <p className="text-[11px] text-slate-400 font-sans mt-0.5">
                Strategy equity trajectory vs S&amp;P 500 (SPY) buy-and-hold benchmark.
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5 text-slate-300">
                <span className="h-2 w-2 rounded-full bg-[#00B386]" /> Strategy Alpha
              </span>
              <span className="flex items-center gap-1.5 text-slate-400">
                <span className="h-2 w-2 rounded-full bg-[#555555]" /> SPY Benchmark
              </span>
              <span className="flex items-center gap-1.5 text-slate-400">
                <span className="h-2 w-2 rounded-full bg-[#EB5757]" /> Peak Drawdown
              </span>
            </div>
          </div>
          <EquityCurveChart data={equity_curve} height={320} />
        </div>

        {/* 3. Interactive 5-Stage System Pipeline Visualizer Bar */}
        <div className="rounded-[4px] border border-[#1f1f1f] bg-[#0C0C0C] p-4 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#161616] pb-2.5">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-[#00B386]" />
              <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                End-to-End System Pipeline (5-Stage Institutional Flow)
              </h3>
            </div>
            <button
              onClick={() => setIsArchModalOpen(true)}
              className="text-[11px] text-[#00B386] hover:text-[#00D49F] flex items-center gap-1 font-semibold"
            >
              <span>Explore Pipeline Details</span>
              <ArrowRight className="h-3 w-3" />
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs">
            <button
              onClick={() => setIsArchModalOpen(true)}
              className="p-3 rounded-[3px] bg-[#080808] hover:bg-[#121212] border border-[#181818] hover:border-[#333] transition-all text-left group"
            >
              <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                <span>STAGE 01</span>
                <span className="text-[#00B386]">&lt;2ms</span>
              </div>
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-sky-400 shrink-0" />
                <span className="font-bold text-slate-200 group-hover:text-white">Data Ingestion</span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-1">Alpaca Websockets</span>
            </button>

            <button
              onClick={() => setIsArchModalOpen(true)}
              className="p-3 rounded-[3px] bg-[#080808] hover:bg-[#121212] border border-[#181818] hover:border-[#333] transition-all text-left group"
            >
              <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                <span>STAGE 02</span>
                <span className="text-[#00B386]">&lt;4ms</span>
              </div>
              <div className="flex items-center gap-2">
                <Binary className="h-4 w-4 text-indigo-400 shrink-0" />
                <span className="font-bold text-slate-200 group-hover:text-white">Features</span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-1">FinBERT + Volatility</span>
            </button>

            <button
              onClick={() => setIsArchModalOpen(true)}
              className="p-3 rounded-[3px] bg-[#080808] hover:bg-[#121212] border border-[#181818] hover:border-[#333] transition-all text-left group"
            >
              <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                <span>STAGE 03</span>
                <span className="text-[#00B386]">12.4ms</span>
              </div>
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-[#00B386] shrink-0" />
                <span className="font-bold text-slate-200 group-hover:text-white">ML Ensemble</span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-1">Stacked Meta-Learner</span>
            </button>

            <button
              onClick={() => setIsArchModalOpen(true)}
              className="p-3 rounded-[3px] bg-[#080808] hover:bg-[#121212] border border-[#181818] hover:border-[#333] transition-all text-left group"
            >
              <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                <span>STAGE 04</span>
                <span className="text-[#00B386]">&lt;1ms</span>
              </div>
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-amber-400 shrink-0" />
                <span className="font-bold text-slate-200 group-hover:text-white">Risk Engine</span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-1">-15% DD Circuit Breaker</span>
            </button>

            <button
              onClick={() => setIsArchModalOpen(true)}
              className="col-span-2 sm:col-span-1 p-3 rounded-[3px] bg-[#080808] hover:bg-[#121212] border border-[#181818] hover:border-[#333] transition-all text-left group"
            >
              <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                <span>STAGE 05</span>
                <span className="text-[#00B386]">&lt;14ms</span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-emerald-400 shrink-0" />
                <span className="font-bold text-slate-200 group-hover:text-white">Paper Execution</span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-1">Smart TWAP Router</span>
            </button>
          </div>
        </div>

        {/* 4. Zerodha Style Order Blotter + Groww Style Asset Allocation Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* Order Blotter (Zerodha Kite inspired) */}
          <div className="lg:col-span-8 rounded-[4px] border border-[#1f1f1f] bg-[#0c0c0c] overflow-hidden">
            <div className="p-4 border-b border-[#1a1a1a] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                    Recent Executed Orders (Blotter)
                  </h3>
                  <span className="text-[10px] text-slate-500">
                    Click &ldquo;SHAP&rdquo; to view trade reasoning
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 font-sans mt-0.5">
                  Real-time broker order status, execution prices, and trade P&amp;L.
                </p>
              </div>

              {/* Symbol Filters */}
              <div className="flex items-center gap-1 bg-[#141414] p-0.5 rounded-[3px] border border-[#222]">
                {["ALL", "AAPL", "MSFT", "NVDA", "SPY"].map((sym) => (
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

            <div className="overflow-x-auto">
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
                    <TableHead className="text-right text-slate-500 text-[11px]">Why?</TableHead>
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
                        <TableCell className="text-right">
                          <button
                            onClick={() => setSelectedTradeForReasoning(trade)}
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-[3px] bg-[#141414] hover:bg-[#1F1F1F] border border-[#2A2A2A] text-[#00B386] hover:text-white text-[10px] font-mono transition-colors shadow-sm"
                            title="Inspect AI/SHAP reasoning for this trade"
                          >
                            <Sparkles className="h-2.5 w-2.5" />
                            <span>Why?</span>
                          </button>
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

      {/* Embedded Modals */}
      <TradeReasoningModal
        trade={selectedTradeForReasoning}
        onClose={() => setSelectedTradeForReasoning(null)}
      />

      <SystemArchitectureModal
        isOpen={isArchModalOpen}
        onClose={() => setIsArchModalOpen(false)}
      />
    </div>
  );
}
