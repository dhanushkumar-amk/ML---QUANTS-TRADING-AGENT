"use client";

import * as React from "react";
import { getModels } from "@/lib/api-client";
import { ModelsData, ModelMatrixEntry } from "@/types";
import { FeatureImportanceChart } from "@/components/charts/FeatureImportanceChart";
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
import { Skeleton } from "@/components/ui/skeleton";
import {
  Zap,
  Layers,
  BarChart3,
  TrendingUp,
  Award,
  Filter,
  CheckCircle2,
  Sparkles,
} from "lucide-react";

export default function ModelComparisonView() {
  const [data, setData] = React.useState<ModelsData | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [filterStatus, setFilterStatus] = React.useState<string>("ALL");

  React.useEffect(() => {
    async function load() {
      try {
        const res = await getModels();
        setData(res);
      } catch (err) {
        console.error("Failed to load models data:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading || !data) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-36 w-full rounded-[4px] bg-[#141414]" />
        <Skeleton className="h-80 w-full rounded-[4px] bg-[#141414]" />
        <Skeleton className="h-80 w-full rounded-[4px] bg-[#141414]" />
      </div>
    );
  }

  const { models, feature_importance } = data;

  // Find the active production champion
  const productionModel = models.find((m) => m.status === "Live Active") || models[models.length - 1];

  // Filter and sort models so the Live Active ensemble is always row 1 (highest Sharpe)
  const filteredModels = [...models]
    .filter((m) => {
      if (filterStatus === "ALL") return true;
      if (filterStatus === "ACTIVE") return m.status === "Live Active";
      if (filterStatus === "CANDIDATE") return m.status === "Candidate";
      if (filterStatus === "BASELINES") return m.status === "Baseline" || m.status === "Archived";
      return true;
    })
    .sort((a, b) => {
      if (a.status === "Live Active") return -1;
      if (b.status === "Live Active") return 1;
      return b.sharpe_ratio - a.sharpe_ratio;
    });

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* 1. Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#1F1F1F] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold tracking-tight text-[#FAFAFA]">
              AI Model Suite & Predictive Alpha
            </h1>
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-[4px] bg-[#141414] text-[#888888] border border-[#1F1F1F]">
              Phase 28 / 34
            </span>
          </div>
          <p className="text-xs text-[#666666] mt-0.5">
            Walk-forward financial evaluation across 6 ML/DL architectures and global SHAP predictive factor attribution.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-[4px] bg-[#0C0C0C] border border-[#1F1F1F]">
            <span className="h-2 w-2 rounded-full bg-[#00B386] animate-pulse" />
            <span className="text-[#888888]">CHAMPION:</span>
            <span className="text-[#00B386] font-bold">Stacked Meta-Learner</span>
          </div>
        </div>
      </div>

      {/* 2. Production Model Champion Spotlight Card */}
      <div className="rounded-[4px] bg-[#0C0C0C] border border-[#1F1F1F] p-5 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-1 h-full bg-[#00B386]" />
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#161616] pb-4">
          <div className="flex items-start gap-3">
            <div className="h-10 w-10 rounded-[4px] bg-[#00B386]/10 border border-[#00B386]/20 flex items-center justify-center shrink-0">
              <Zap className="h-5 w-5 text-[#00B386]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-[#FAFAFA]">{productionModel.name}</h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-[2px] bg-[#00B386]/10 text-[#00B386] border border-[#00B386]/30 font-semibold">
                  LIVE IN PROD
                </span>
              </div>
              <p className="text-xs text-[#666666] mt-0.5 font-mono">
                Architecture: {productionModel.family} • Ensemble of XGBoost + LightGBM + BiLSTM with Ridge Meta-Model
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono text-[#888888]">
            <span className="px-2.5 py-1 rounded-[3px] bg-[#141414] border border-[#1F1F1F]">
              Inference: <span className="text-[#FAFAFA] font-bold">12.4ms</span>
            </span>
            <span className="px-2.5 py-1 rounded-[3px] bg-[#141414] border border-[#1F1F1F]">
              Folds: <span className="text-[#FAFAFA] font-bold">5-Fold Purged CV</span>
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 pt-4 font-mono">
          <div className="p-2.5 rounded-[3px] bg-[#080808] border border-[#161616]">
            <span className="text-[10px] text-[#666666] uppercase block">Ann. Return</span>
            <span className="text-sm font-bold text-[#00B386] tabular-nums">
              +{productionModel.annualized_return.toFixed(1)}%
            </span>
          </div>
          <div className="p-2.5 rounded-[3px] bg-[#080808] border border-[#161616]">
            <span className="text-[10px] text-[#666666] uppercase block">Sharpe Ratio</span>
            <span className="text-sm font-bold text-[#FAFAFA] tabular-nums">
              {productionModel.sharpe_ratio.toFixed(2)}
            </span>
          </div>
          <div className="p-2.5 rounded-[3px] bg-[#080808] border border-[#161616]">
            <span className="text-[10px] text-[#666666] uppercase block">Sortino Ratio</span>
            <span className="text-sm font-bold text-[#00B386] tabular-nums">
              {productionModel.sortino_ratio.toFixed(2)}
            </span>
          </div>
          <div className="p-2.5 rounded-[3px] bg-[#080808] border border-[#161616]">
            <span className="text-[10px] text-[#666666] uppercase block">Max Drawdown</span>
            <span className="text-sm font-bold text-[#EB5757] tabular-nums">
              {productionModel.max_drawdown.toFixed(1)}%
            </span>
          </div>
          <div className="p-2.5 rounded-[3px] bg-[#080808] border border-[#161616]">
            <span className="text-[10px] text-[#666666] uppercase block">Win Rate</span>
            <span className="text-sm font-bold text-[#00B386] tabular-nums">
              {productionModel.win_rate.toFixed(1)}%
            </span>
          </div>
          <div className="p-2.5 rounded-[3px] bg-[#080808] border border-[#161616]">
            <span className="text-[10px] text-[#666666] uppercase block">OOS Accuracy</span>
            <span className="text-sm font-bold text-[#FAFAFA] tabular-nums">
              {productionModel.directional_accuracy.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* 3. Cross-Architecture Performance Matrix */}
      <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
        <CardHeader className="p-4 pb-3 border-b border-[#141414]">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <CardTitle className="text-xs font-semibold text-[#FAFAFA] flex items-center gap-2">
                <Layers className="h-3.5 w-3.5 text-[#888888]" />
                Cross-Architecture Benchmark Matrix
              </CardTitle>
              <CardDescription className="text-[11px] text-[#666666]">
                Identical test window evaluation with transaction cost model (10 bps slippage + commissions).
              </CardDescription>
            </div>

            {/* Filter Pills */}
            <div className="flex items-center gap-1.5 font-mono text-[11px]">
              {(["ALL", "ACTIVE", "CANDIDATE", "BASELINES"] as const).map((filter) => (
                <button
                  key={filter}
                  onClick={() => setFilterStatus(filter)}
                  className={`px-2.5 py-1 rounded-[3px] border transition-colors ${
                    filterStatus === filter
                      ? "bg-[#1C1C1C] text-[#FAFAFA] border-[#333333]"
                      : "bg-transparent text-[#666666] border-transparent hover:text-[#AAAAAA] hover:bg-[#111111]"
                  }`}
                >
                  {filter}
                </button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-b border-[#141414] hover:bg-transparent">
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9">Model Name</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9">Family</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9 text-right">Accuracy</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9 text-right">Ann. Return</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9 text-right">Sharpe</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9 text-right">Sortino</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9 text-right">Max DD</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9 text-right">Calmar</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9 text-right">Win Rate</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9 text-center">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredModels.map((m) => {
                  const isProduction = m.status === "Live Active";
                  return (
                    <TableRow
                      key={m.id}
                      className={`font-mono text-xs border-b border-[#141414] transition-colors ${
                        isProduction
                          ? "bg-[#00B386]/10 hover:bg-[#00B386]/15 border-l-4 border-l-[#00B386] shadow-[inset_0_0_12px_rgba(0,179,134,0.06)]"
                          : "hover:bg-[#111111]"
                      }`}
                    >
                      <TableCell className="font-bold text-[#FAFAFA] py-3.5">
                        <div className="flex items-center gap-2">
                          {isProduction ? (
                            <>
                              <div className="h-5 w-5 rounded-[2px] bg-[#00B386] flex items-center justify-center shrink-0">
                                <Zap className="h-3 w-3 text-black fill-black" />
                              </div>
                              <span className="text-white font-bold">{m.name}</span>
                              <span className="text-[9px] px-1.5 py-0.2 rounded-[2px] bg-[#00B386]/20 text-[#00B386] border border-[#00B386]/40 font-extrabold uppercase tracking-wide">
                                RANK 1 • PROD
                              </span>
                            </>
                          ) : (
                            <span>{m.name}</span>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="text-[#777777] text-[11px] py-3">
                        {m.family}
                      </TableCell>
                      <TableCell className="text-right tabular-nums py-3 text-[#CCCCCC]">
                        {m.directional_accuracy.toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-right font-bold text-[#00B386] tabular-nums py-3">
                        +{m.annualized_return.toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-right font-semibold text-[#FAFAFA] tabular-nums py-3">
                        {m.sharpe_ratio.toFixed(2)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums py-3 text-[#CCCCCC]">
                        {m.sortino_ratio.toFixed(2)}
                      </TableCell>
                      <TableCell className="text-right text-[#EB5757] tabular-nums py-3">
                        {m.max_drawdown.toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-right tabular-nums py-3 text-[#CCCCCC]">
                        {m.calmar_ratio.toFixed(2)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums py-3 text-[#00B386]">
                        {m.win_rate.toFixed(1)}%
                      </TableCell>
                      <TableCell className="text-center py-3">
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-[2px] font-mono ${
                            isProduction
                              ? "bg-[#00B386]/10 text-[#00B386] border border-[#00B386]/30 font-bold"
                              : m.status === "Candidate"
                              ? "bg-[#1A1A1A] text-[#CCCCCC] border border-[#2A2A2A]"
                              : "bg-[#141414] text-[#666666] border border-[#1F1F1F]"
                          }`}
                        >
                          {m.status}
                        </span>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* 4. Global SHAP Feature Importance (Vertical Bar Chart) */}
      <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
        <CardHeader className="p-4 pb-2 border-b border-[#141414]">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <CardTitle className="text-xs font-semibold text-[#FAFAFA] flex items-center gap-2">
                <BarChart3 className="h-3.5 w-3.5 text-[#00B386]" />
                Global Feature Attribution (|SHAP| Importance Ranking)
              </CardTitle>
              <CardDescription className="text-[11px] text-[#666666]">
                Mean absolute SHAP value across out-of-sample test folds. Bars stand vertically from highest to lowest alpha drivers.
              </CardDescription>
            </div>
            <div className="flex items-center gap-3 text-[10px] font-mono text-[#777777]">
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-[1px] bg-[#00B386]" /> Top Drivers (RSI, FinBERT, MACD)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-[1px] bg-[#3A3A3A]" /> Secondary Signals
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-[1px] bg-[#222222]" /> Baseline Features
              </span>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-4 pt-6">
          <FeatureImportanceChart data={feature_importance} height={320} />
        </CardContent>
      </Card>
    </div>
  );
}
