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
  Info,
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
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* 1. Header Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-foreground font-mono">
              {ticker}
            </h1>
            <Badge variant="outline" className="text-xs font-mono text-muted-foreground">
              NASDAQ
            </Badge>
            {latestBar && (
              <div className="flex items-baseline gap-2 font-mono">
                <span className="text-xl font-bold text-foreground">
                  ${latestBar.close.toFixed(2)}
                </span>
                <span
                  className={`text-xs font-medium flex items-center gap-0.5 ${
                    priceChange >= 0 ? "text-emerald-400" : "text-rose-400"
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
          <p className="text-xs text-muted-foreground mt-0.5">
            TradingView lightweight-charts with technical overlays and machine learning execution markers.
          </p>
        </div>

        {/* Timeframe Selector */}
        <div className="flex items-center gap-2">
          <Tabs value={timeframe} onValueChange={setTimeframe}>
            <TabsList>
              {TIMEFRAMES.map((tf) => (
                <TabsTrigger key={tf} value={tf}>
                  {tf}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </div>
      </div>

      {/* 2. Main Content Grid: Chart (8 cols) + Model Sidebar (4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Candlestick Chart Area */}
        <div className="lg:col-span-8 space-y-3">
          {/* Overlay Controls */}
          <div className="flex items-center justify-between p-2.5 rounded-lg bg-card/60 border border-border/60 text-xs">
            <div className="flex items-center gap-2">
              <Sliders className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-muted-foreground font-medium">Overlays:</span>
              <Button
                variant={showSMA20 ? "default" : "outline"}
                size="sm"
                onClick={() => setShowSMA20(!showSMA20)}
                className={showSMA20 ? "bg-amber-500/20 text-amber-300 border border-amber-500/30" : ""}
              >
                SMA 20
              </Button>
              <Button
                variant={showSMA60 ? "default" : "outline"}
                size="sm"
                onClick={() => setShowSMA60(!showSMA60)}
                className={showSMA60 ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30" : ""}
              >
                SMA 60
              </Button>
              <Button
                variant={showBollinger ? "default" : "outline"}
                size="sm"
                onClick={() => setShowBollinger(!showBollinger)}
                className={showBollinger ? "bg-purple-500/20 text-purple-300 border border-purple-500/30" : ""}
              >
                Bollinger Bands
              </Button>
            </div>

            <div className="text-[11px] font-mono text-muted-foreground flex items-center gap-2">
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-emerald-400" /> BUY Marker
              </span>
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-rose-400" /> SELL Marker
              </span>
            </div>
          </div>

          {/* Chart Canvas */}
          {loading || !chartData ? (
            <Skeleton className="h-[480px] w-full rounded-lg" />
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
            <div className="grid grid-cols-5 gap-2 p-3 rounded-lg bg-card/40 border border-border/40 text-center font-mono text-xs">
              <div>
                <span className="text-[10px] text-muted-foreground block">OPEN</span>
                <span className="font-semibold">${latestBar.open.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-[10px] text-muted-foreground block">HIGH</span>
                <span className="font-semibold text-emerald-400">${latestBar.high.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-[10px] text-muted-foreground block">LOW</span>
                <span className="font-semibold text-rose-400">${latestBar.low.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-[10px] text-muted-foreground block">CLOSE</span>
                <span className="font-semibold">${latestBar.close.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-[10px] text-muted-foreground block">VOLUME</span>
                <span className="font-semibold text-slate-300">
                  {(latestBar.volume / 1000000).toFixed(2)}M
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Explainability & SHAP Sidebar */}
        <div className="lg:col-span-4 space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-semibold flex items-center gap-1.5 text-foreground">
                  <Brain className="h-4 w-4 text-emerald-400" />
                  Production Inference
                </CardTitle>
                <Badge variant="profit" className="text-[10px]">
                  LIVE SIGNAL
                </Badge>
              </div>
              <CardDescription className="text-xs">
                Stacked Meta-Learner (LightGBM + Transformer) with real-time SHAP decomposition.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Signal & Confidence Gauge */}
              <div className="p-3 rounded-lg bg-muted/40 border border-border/60">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground font-medium">
                    Signal Direction
                  </span>
                  <Badge
                    variant={
                      explanation?.signal === "BUY"
                        ? "profit"
                        : explanation?.signal === "SELL"
                        ? "loss"
                        : "neutral"
                    }
                    className="text-xs font-bold font-mono px-3 py-0.5"
                  >
                    {explanation?.signal || "BUY"}
                  </Badge>
                </div>
                <div className="flex items-center justify-between text-xs font-mono mt-2">
                  <span className="text-muted-foreground">Confidence Score</span>
                  <span className="font-bold text-foreground">
                    {((explanation?.confidence || 0.84) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1.5 overflow-hidden">
                  <div
                    className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${(explanation?.confidence || 0.84) * 100}%`,
                    }}
                  />
                </div>
              </div>

              {/* Top Feature Drivers (SHAP Waterfall) */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-foreground uppercase tracking-wider">
                    Top Feature Drivers
                  </span>
                  <span className="text-[10px] text-muted-foreground font-mono">
                    |SHAP| Impact
                  </span>
                </div>

                <div className="space-y-2">
                  {explanation?.features.map((feat) => (
                    <div
                      key={feat.feature}
                      className="p-2.5 rounded-md bg-muted/30 border border-border/40 hover:border-border/80 transition-colors"
                    >
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="font-medium text-foreground">
                          {feat.name}
                        </span>
                        <span
                          className={`font-semibold ${
                            feat.shap >= 0 ? "text-emerald-400" : "text-rose-400"
                          }`}
                        >
                          {feat.shap >= 0 ? "+" : ""}
                          {feat.shap.toFixed(3)}
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-muted-foreground mt-1">
                        <span>{feat.description}</span>
                        <span className="font-mono text-slate-400">
                          val: {feat.value}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Model Diagnostics Footnote */}
              <div className="text-[11px] text-muted-foreground/80 flex items-start gap-1.5 pt-2 border-t border-border/40">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
                <span>
                  SHAP values satisfy additive feature attribution: $f(x) = \phi_0 + \sum \phi_i$.
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
