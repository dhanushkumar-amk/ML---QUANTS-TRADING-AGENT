"use client";

import * as React from "react";
import { useParams } from "next/navigation";
import { getChart, getExplanation } from "@/lib/api-client";
import { ChartData, ModelExplanation } from "@/types";
import { TradingViewChart } from "@/components/charts/TradingViewChart";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  TrendingUp,
  TrendingDown,
  Brain,
  Sliders,
  CheckCircle2,
} from "lucide-react";

const TIMEFRAMES = ["1D", "1W", "1M", "3M", "1Y", "ALL"];

export default function LiveChartView() {
  const params = useParams();
  const ticker = (params?.ticker as string)?.toUpperCase() || "AAPL";

  const [timeframe, setTimeframe] = React.useState("1M");
  const [chartData, setChartData] = React.useState<ChartData | null>(null);
  const [explanation, setExplanation] = React.useState<ModelExplanation | null>(null);
  const [loading, setLoading] = React.useState(true);

  // Indicator Toggles
  const [showSMA20, setShowSMA20] = React.useState(true);
  const [showSMA60, setShowSMA60] = React.useState(true);
  const [showBollinger, setShowBollinger] = React.useState(false);

  React.useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const [cData, exp] = await Promise.all([
          getChart(ticker, timeframe),
          getExplanation(ticker),
        ]);
        setChartData(cData);
        setExplanation(exp);
      } catch (err) {
        console.error("Failed to load chart view data:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [ticker, timeframe]);

  const latestBar = chartData?.bars[chartData.bars.length - 1];
  const prevBar = chartData?.bars[chartData.bars.length - 2];
  const priceChange = latestBar && prevBar ? latestBar.close - prevBar.close : 0;
  const priceChangePct = latestBar && prevBar ? (priceChange / prevBar.close) * 100 : 0;

  return (
    <div className="space-y-5 animate-in fade-in duration-150 font-mono">
      {/* 1. Header Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-[#1f1f1f]">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-white font-mono">
              {ticker}
            </h1>
            <span className="text-[10px] px-1.5 py-0.5 rounded-[2px] bg-[#141414] border border-[#222] text-slate-400">
              NASDAQ
            </span>
            {latestBar && (
              <div className="flex items-baseline gap-2">
                <span className="text-xl font-bold text-white tabular-nums">
                  ${latestBar.close.toFixed(2)}
                </span>
                <span
                  className={`text-xs font-semibold flex items-center gap-0.5 tabular-nums ${
                    priceChange >= 0 ? "text-[#00B386]" : "text-[#EB5757]"
                  }`}
                >
                  {priceChange >= 0 ? (
                    <TrendingUp className="h-3 w-3" />
                  ) : (
                    <TrendingDown className="h-3 w-3" />
                  )}
                  {priceChange >= 0 ? "+" : ""}
                  {priceChange.toFixed(2)} ({priceChangePct.toFixed(2)}%)
                </span>
              </div>
            )}
          </div>
          <p className="text-xs text-slate-500 font-sans mt-0.5">
            TradingView lightweight-charts with technical overlays and execution markers.
          </p>
        </div>

        {/* Timeframe Selector */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 bg-[#0c0c0c] p-1 rounded-[4px] border border-[#1f1f1f]">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-2.5 py-1 text-xs rounded-[3px] transition-colors ${
                  timeframe === tf
                    ? "bg-white text-black font-bold"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 2. Main Content Grid: Chart (8 cols) + Model Sidebar (4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Candlestick Chart Area */}
        <div className="lg:col-span-8 space-y-3">
          {/* Overlay Controls */}
          <div className="flex items-center justify-between p-2.5 rounded-[4px] bg-[#0c0c0c] border border-[#1f1f1f] text-xs">
            <div className="flex items-center gap-2">
              <Sliders className="h-3.5 w-3.5 text-slate-500" />
              <span className="text-slate-400 font-medium">Overlays:</span>
              <button
                onClick={() => setShowSMA20(!showSMA20)}
                className={`px-2 py-0.5 rounded-[2px] border text-[11px] font-semibold transition-colors ${
                  showSMA20
                    ? "bg-[#f59e0b]/15 text-[#f59e0b] border-[#f59e0b]/30"
                    : "bg-[#141414] text-slate-400 border-[#222] hover:text-white"
                }`}
              >
                SMA 20
              </button>
              <button
                onClick={() => setShowSMA60(!showSMA60)}
                className={`px-2 py-0.5 rounded-[2px] border text-[11px] font-semibold transition-colors ${
                  showSMA60
                    ? "bg-[#06b6d4]/15 text-[#06b6d4] border-[#06b6d4]/30"
                    : "bg-[#141414] text-slate-400 border-[#222] hover:text-white"
                }`}
              >
                SMA 60
              </button>
              <button
                onClick={() => setShowBollinger(!showBollinger)}
                className={`px-2 py-0.5 rounded-[2px] border text-[11px] font-semibold transition-colors ${
                  showBollinger
                    ? "bg-[#8b5cf6]/15 text-[#8b5cf6] border-[#8b5cf6]/30"
                    : "bg-[#141414] text-slate-400 border-[#222] hover:text-white"
                }`}
              >
                Bollinger Bands
              </button>
            </div>

            <div className="text-[11px] text-slate-400 flex items-center gap-3">
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-[#00B386]" /> BUY Marker
              </span>
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-[#EB5757]" /> SELL Marker
              </span>
            </div>
          </div>

          {/* Chart Canvas */}
          {loading || !chartData ? (
            <Skeleton className="h-[480px] w-full rounded-[4px] bg-[#111]" />
          ) : (
            <TradingViewChart
              bars={chartData.bars}
              markers={chartData.markers}
              showSMA20={showSMA20}
              showSMA60={showSMA60}
              showBollinger={showBollinger}
              height={480}
            />
          )}

          {/* Bar Metrics Bar */}
          {latestBar && (
            <div className="grid grid-cols-5 gap-2 p-3 rounded-[4px] bg-[#0c0c0c] border border-[#1f1f1f] text-center text-xs">
              <div>
                <span className="text-[10px] text-slate-500 block">OPEN</span>
                <span className="font-semibold text-white tabular-nums">${latestBar.open.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 block">HIGH</span>
                <span className="font-semibold text-[#00B386] tabular-nums">${latestBar.high.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 block">LOW</span>
                <span className="font-semibold text-[#EB5757] tabular-nums">${latestBar.low.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 block">CLOSE</span>
                <span className="font-semibold text-white tabular-nums">${latestBar.close.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 block">VOLUME</span>
                <span className="font-semibold text-slate-300 tabular-nums">
                  {(latestBar.volume / 1000000).toFixed(2)}M
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Explainability & SHAP Sidebar */}
        <div className="lg:col-span-4 space-y-4">
          <div className="rounded-[4px] border border-[#1f1f1f] bg-[#0c0c0c] p-4 space-y-4">
            <div className="pb-3 border-b border-[#1a1a1a] flex items-center justify-between">
              <div>
                <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                  <Brain className="h-4 w-4 text-[#00B386]" />
                  Production Inference
                </h3>
                <p className="text-[11px] text-slate-500 font-sans mt-0.5">
                  Stacked Meta-Learner with real-time SHAP.
                </p>
              </div>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-[2px] bg-[#00B386]/10 text-[#00B386] border border-[#00B386]/30">
                LIVE SIGNAL
              </span>
            </div>

            {/* Signal & Confidence Gauge */}
            <div className="p-3 rounded-[4px] bg-[#111] border border-[#1a1a1a]">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-slate-400 font-medium font-sans">
                  Signal Direction
                </span>
                <span
                  className={`text-xs font-bold px-2.5 py-0.5 rounded-[2px] border ${
                    explanation?.signal === "BUY"
                      ? "bg-[#00B386]/10 text-[#00B386] border-[#00B386]/30"
                      : "bg-[#EB5757]/10 text-[#EB5757] border-[#EB5757]/30"
                  }`}
                >
                  {explanation?.signal || "BUY"}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs mt-2">
                <span className="text-slate-400 font-sans">Confidence Score</span>
                <span className="font-bold text-white tabular-nums">
                  {((explanation?.confidence || 0.84) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-[#1c1c1c] h-1.5 rounded-full mt-2 overflow-hidden">
                <div
                  className="bg-[#00B386] h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${(explanation?.confidence || 0.84) * 100}%`,
                  }}
                />
              </div>
            </div>

            {/* Top Feature Drivers (SHAP) */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-white uppercase tracking-wider text-[11px]">
                  Feature Drivers
                </span>
                <span className="text-[10px] text-slate-500 font-mono">
                  |SHAP| Impact
                </span>
              </div>

              <div className="space-y-1.5">
                {explanation?.features.map((feat) => (
                  <div
                    key={feat.feature}
                    className="p-2.5 rounded-[3px] bg-[#111] border border-[#1a1a1a]"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-white">
                        {feat.name}
                      </span>
                      <span
                        className={`font-semibold tabular-nums ${
                          feat.shap >= 0 ? "text-[#00B386]" : "text-[#EB5757]"
                        }`}
                      >
                        {feat.shap >= 0 ? "+" : ""}
                        {feat.shap.toFixed(3)}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-slate-400 mt-1 font-sans">
                      <span>{feat.description}</span>
                      <span className="font-mono text-slate-500">
                        val: {feat.value}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model Diagnostics Footnote */}
            <div className="text-[10px] text-slate-500 flex items-start gap-1.5 pt-2 border-t border-[#1a1a1a]">
              <CheckCircle2 className="h-3 w-3 text-[#00B386] shrink-0 mt-0.5" />
              <span>
                SHAP additive feature attribution: f(x) = φ0 + Σ φi
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
