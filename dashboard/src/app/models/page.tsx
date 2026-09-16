"use client";

import * as React from "react";
import { getModels } from "@/lib/api-client";
import { ModelsData } from "@/types";
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
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { formatPercent } from "@/lib/utils";
import { Cpu, Award, Zap, Layers } from "lucide-react";

export default function ModelComparisonView() {
  const [data, setData] = React.useState<ModelsData | null>(null);
  const [loading, setLoading] = React.useState(true);

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
        <Skeleton className="h-80 w-full rounded-lg" />
        <Skeleton className="h-80 w-full rounded-lg" />
      </div>
    );
  }

  const { models, feature_importance } = data;

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* 1. Page Header */}
      <div>
        <h1 className="text-xl font-bold tracking-tight text-foreground flex items-center gap-2">
          <Cpu className="h-5 w-5 text-emerald-400" />
          Model Comparison Matrix & Interpretability (Phase 28)
        </h1>
        <p className="text-xs text-muted-foreground mt-0.5">
          Side-by-side financial evaluation across ML/DL architectures and global SHAP feature attribution.
        </p>
      </div>

      {/* 2. Side-by-Side Model Comparison Table */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm font-semibold text-foreground flex items-center gap-2">
                <Layers className="h-4 w-4 text-emerald-400" />
                Cross-Architecture Performance Evaluation
              </CardTitle>
              <CardDescription className="text-xs">
                Walk-forward verified metrics across heuristics, gradient boosted trees, recurrent, and attention models.
              </CardDescription>
            </div>
            <Badge variant="profit" className="text-[11px] font-mono">
              Ensemble Active in Prod
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model Architecture</TableHead>
                <TableHead>Family</TableHead>
                <TableHead className="text-right">Accuracy</TableHead>
                <TableHead className="text-right">Ann. Return</TableHead>
                <TableHead className="text-right">Sharpe</TableHead>
                <TableHead className="text-right">Sortino</TableHead>
                <TableHead className="text-right">Max DD</TableHead>
                <TableHead className="text-right">Calmar</TableHead>
                <TableHead className="text-right">Win Rate</TableHead>
                <TableHead className="text-center">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {models.map((m) => {
                const isProduction = m.status === "Live Active";
                return (
                  <TableRow
                    key={m.id}
                    className={`font-mono text-xs ${
                      isProduction
                        ? "bg-emerald-500/10 hover:bg-emerald-500/15 border-l-2 border-l-emerald-400"
                        : ""
                    }`}
                  >
                    <TableCell className="font-bold text-foreground">
                      <div className="flex items-center gap-1.5">
                        {isProduction && (
                          <Zap className="h-3 w-3 text-emerald-400" />
                        )}
                        <span>{m.name}</span>
                      </div>
                    </TableCell>
                    <TableCell className="text-muted-foreground text-[11px]">
                      {m.family}
                    </TableCell>
                    <TableCell className="text-right">
                      {m.directional_accuracy.toFixed(1)}%
                    </TableCell>
                    <TableCell className="text-right font-bold text-emerald-400">
                      +{m.annualized_return.toFixed(1)}%
                    </TableCell>
                    <TableCell className="text-right font-semibold text-foreground">
                      {m.sharpe_ratio.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-right">{m.sortino_ratio.toFixed(2)}</TableCell>
                    <TableCell className="text-right text-rose-400">
                      {m.max_drawdown.toFixed(1)}%
                    </TableCell>
                    <TableCell className="text-right">{m.calmar_ratio.toFixed(2)}</TableCell>
                    <TableCell className="text-right">{m.win_rate.toFixed(1)}%</TableCell>
                    <TableCell className="text-center">
                      <Badge
                        variant={
                          isProduction
                            ? "profit"
                            : m.status === "Candidate"
                            ? "secondary"
                            : "neutral"
                        }
                        className="text-[10px]"
                      >
                        {m.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* 3. Global SHAP Feature Importance Chart */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm font-semibold text-foreground flex items-center gap-2">
                <Award className="h-4 w-4 text-emerald-400" />
                Global Feature Attribution (|SHAP| Importance Ranking)
              </CardTitle>
              <CardDescription className="text-xs">
                Mean absolute SHAP value across out-of-sample validation folds colored by feature category.
              </CardDescription>
            </div>
            <div className="flex items-center gap-3 text-[10px] font-mono text-muted-foreground">
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-emerald-400" /> Momentum
              </span>
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-cyan-400" /> Trend
              </span>
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-purple-400" /> NLP Sentiment
              </span>
              <span className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full bg-amber-400" /> Volatility
              </span>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <FeatureImportanceChart data={feature_importance} height={380} />
        </CardContent>
      </Card>
    </div>
  );
}
