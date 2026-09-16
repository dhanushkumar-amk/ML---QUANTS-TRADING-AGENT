"use client";

import * as React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from "recharts";
import { FeatureImportanceItem } from "@/types";

interface FeatureImportanceChartProps {
  data: FeatureImportanceItem[];
  height?: number;
}

const CATEGORY_COLORS: Record<string, string> = {
  Momentum: "#10b981", // Emerald
  Trend: "#06b6d4", // Cyan
  "NLP / Alternative": "#8b5cf6", // Purple
  Volatility: "#f59e0b", // Amber
  Microstructure: "#ec4899", // Pink
  Volume: "#3b82f6", // Blue
  "Mean-Reversion": "#14b8a6", // Teal
  Execution: "#64748b", // Slate
};

export function FeatureImportanceChart({
  data,
  height = 360,
}: FeatureImportanceChartProps) {
  if (!data || data.length === 0) return null;

  // Sort ascending so highest appears at top in horizontal layout
  const sorted = [...data].sort((a, b) => a.importance - b.importance);

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={sorted}
          layout="vertical"
          margin={{ top: 5, right: 25, left: 90, bottom: 5 }}
        >
          <XAxis
            type="number"
            stroke="#64748b"
            fontSize={10}
            tickLine={false}
            axisLine={{ stroke: "rgba(51, 65, 85, 0.4)" }}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <YAxis
            type="category"
            dataKey="feature"
            stroke="#94a3b8"
            fontSize={11}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0f172a",
              borderColor: "rgba(51, 65, 85, 0.8)",
              borderRadius: "8px",
              fontSize: "12px",
            }}
            formatter={(val: any, _, item: any) => [
              `${Number(val).toFixed(3)} (Mean |SHAP|)`,
              `Category: ${item?.payload?.category || "Core"}`,
            ]}
          />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
            {sorted.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={CATEGORY_COLORS[entry.category] || "#10b981"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
