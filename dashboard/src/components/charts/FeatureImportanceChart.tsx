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
  height = 320,
}: FeatureImportanceChartProps) {
  if (!data || data.length === 0) return null;

  // Sort descending so highest alpha drivers appear first on the left
  const sorted = [...data].sort((a, b) => b.importance - a.importance);
  const maxImportance = sorted[0]?.importance || 1;

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={sorted}
          margin={{ top: 15, right: 15, left: -15, bottom: 55 }}
        >
          <CartesianGrid strokeDasharray="2 2" stroke="#161616" vertical={false} />
          <XAxis
            dataKey="feature"
            stroke="#555555"
            fontSize={10}
            interval={0}
            angle={-35}
            textAnchor="end"
            tickLine={false}
            axisLine={{ stroke: "#1F1F1F" }}
            height={50}
          />
          <YAxis
            stroke="#555555"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <Tooltip
            cursor={{ fill: "rgba(255, 255, 255, 0.03)" }}
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
          <Bar dataKey="importance" radius={[3, 3, 0, 0]} maxBarSize={36}>
            {sorted.map((entry, index) => {
              // Top 3 primary alpha features highlighted in Zerodha teal-green
              const isTopTier = index < 3;
              const isMidTier = index < 6;
              return (
                <Cell
                  key={`bar-${index}`}
                  fill={isTopTier ? "#00B386" : isMidTier ? "#3A3A3A" : "#222222"}
                  stroke={isTopTier ? "#00D49F" : "transparent"}
                  strokeWidth={isTopTier ? 1 : 0}
                />
              );
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
