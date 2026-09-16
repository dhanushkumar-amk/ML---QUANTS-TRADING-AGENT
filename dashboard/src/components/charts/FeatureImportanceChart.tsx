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
  CartesianGrid,
} from "recharts";
import { FeatureImportanceItem } from "@/types";

interface FeatureImportanceChartProps {
  data: FeatureImportanceItem[];
  height?: number;
}

export function FeatureImportanceChart({
  data,
  height = 360,
}: FeatureImportanceChartProps) {
  if (!data || data.length === 0) return null;

  // Sort ascending so highest appears at top in horizontal layout
  const sorted = [...data].sort((a, b) => a.importance - b.importance);
  const maxImportance = Math.max(...sorted.map((s) => s.importance));

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={sorted}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 110, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="2 2" stroke="#161616" horizontal={false} />
          <XAxis
            type="number"
            stroke="#555555"
            fontSize={10}
            tickLine={false}
            axisLine={{ stroke: "#1F1F1F" }}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <YAxis
            type="category"
            dataKey="feature"
            stroke="#888888"
            fontSize={11}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0C0C0C",
              borderColor: "#1F1F1F",
              borderRadius: "4px",
              fontSize: "11px",
              color: "#FAFAFA",
              boxShadow: "0 8px 24px rgba(0,0,0,0.8)",
            }}
            formatter={(val: any, _, item: any) => [
              `${Number(val).toFixed(3)} Mean |SHAP|`,
              `Category: ${item?.payload?.category || "Alpha Factor"}`,
            ]}
          />
          <Bar dataKey="importance" radius={[0, 2, 2, 0]}>
            {sorted.map((entry, index) => {
              // Highlight top 3 alpha features with Zerodha teal-green, others in monochrome grayscale steps
              const isTop = entry.importance >= maxImportance * 0.75;
              return (
                <Cell
                  key={`cell-${index}`}
                  fill={isTop ? "#00B386" : "#2A2A2A"}
                />
              );
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
