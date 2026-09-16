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
  CartesianGrid,
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
          <CartesianGrid strokeDasharray="2 2" stroke="#161616" vertical={false} />
          <XAxis
            dataKey="date"
            stroke="#555555"
            fontSize={10}
            tickLine={false}
            axisLine={{ stroke: "#1F1F1F" }}
            tickFormatter={(str) => str.slice(5)}
          />
          <YAxis
            yAxisId="sharpe"
            domain={["auto", "auto"]}
            stroke="#00B386"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => v.toFixed(1)}
          />
          <YAxis
            yAxisId="vol"
            orientation="right"
            domain={["auto", "auto"]}
            stroke="#888888"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `${v}%`}
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
          />
          <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
          <Line
            yAxisId="sharpe"
            type="monotone"
            dataKey="rolling_sharpe"
            name="Rolling 6M Sharpe"
            stroke="#00B386"
            strokeWidth={1.5}
            dot={false}
          />
          <Line
            yAxisId="vol"
            type="monotone"
            dataKey="rolling_volatility"
            name="Rolling Volatility (%)"
            stroke="#888888"
            strokeWidth={1.2}
            strokeDasharray="3 3"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
