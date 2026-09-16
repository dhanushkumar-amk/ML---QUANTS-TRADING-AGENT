"use client";

import * as React from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";
import { EquityCurvePoint } from "@/types";

interface EquityCurveChartProps {
  data: EquityCurvePoint[];
  height?: number;
}

export function EquityCurveChart({ data, height = 360 }: EquityCurveChartProps) {
  if (!data || data.length === 0) return null;

  return (
    <div className="w-full space-y-2">
      {/* 1. Main Equity Area Chart */}
      <div className="w-full" style={{ height: height * 0.72 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <defs>
              <linearGradient id="stratGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="date"
              hide
              stroke="#64748b"
              tickLine={false}
              axisLine={{ stroke: "rgba(51, 65, 85, 0.4)" }}
            />
            <YAxis
              domain={["auto", "auto"]}
              stroke="#94a3b8"
              fontSize={11}
              tickFormatter={(val) => `$${(val / 1000).toFixed(0)}k`}
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
              formatter={(value: any) => [`$${Number(value).toLocaleString()}`, ""]}
            />
            <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: "12px" }} />
            <Area
              type="monotone"
              dataKey="strategy"
              name="Strategy Equity"
              stroke="#10b981"
              strokeWidth={2}
              fill="url(#stratGrad)"
            />
            <Line
              type="monotone"
              dataKey="benchmark"
              name="SPY Benchmark"
              stroke="#94a3b8"
              strokeWidth={1.5}
              strokeDasharray="4 4"
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* 2. Underwater Drawdown Area Subplot */}
      <div className="w-full" style={{ height: height * 0.28 }}>
        <div className="text-[11px] font-mono text-muted-foreground px-2">
          Drawdown Underwater (%)
        </div>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
            <defs>
              <linearGradient id="drawdownGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="date"
              stroke="#64748b"
              fontSize={10}
              tickLine={false}
              axisLine={{ stroke: "rgba(51, 65, 85, 0.4)" }}
              tickFormatter={(str) => str.slice(5)}
            />
            <YAxis
              domain={["dataMin", 0]}
              stroke="#94a3b8"
              fontSize={10}
              tickFormatter={(val) => `${val}%`}
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
              formatter={(value: any) => [`${value}%`, "Drawdown"]}
            />
            <Area
              type="monotone"
              dataKey="drawdown"
              stroke="#f43f5e"
              strokeWidth={1.5}
              fill="url(#drawdownGrad)"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
