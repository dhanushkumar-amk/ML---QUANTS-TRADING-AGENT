"use client";

import * as React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";
import { RollingMetricPoint } from "@/types";

interface RollingMetricsChartProps {
  data: RollingMetricPoint[];
  height?: number;
}

export function RollingMetricsChart({ data, height = 260 }: RollingMetricsChartProps) {
  if (!data || data.length === 0) return null;

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="date"
            stroke="#64748b"
            fontSize={10}
            tickLine={false}
            axisLine={{ stroke: "rgba(51, 65, 85, 0.4)" }}
            tickFormatter={(str) => str.slice(5)}
          />
          <YAxis
            yAxisId="sharpe"
            domain={["auto", "auto"]}
            stroke="#10b981"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => v.toFixed(1)}
          />
          <YAxis
            yAxisId="vol"
            orientation="right"
            domain={["auto", "auto"]}
            stroke="#06b6d4"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0f172a",
              borderColor: "rgba(51, 65, 85, 0.8)",
              borderRadius: "8px",
              fontSize: "12px",
            }}
          />
          <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
          <Line
            yAxisId="sharpe"
            type="monotone"
            dataKey="rolling_sharpe"
            name="Rolling 6M Sharpe"
            stroke="#10b981"
            strokeWidth={2}
            dot={false}
          />
          <Line
            yAxisId="vol"
            type="monotone"
            dataKey="rolling_volatility"
            name="Rolling Volatility (%)"
            stroke="#06b6d4"
            strokeWidth={1.5}
            strokeDasharray="3 3"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
