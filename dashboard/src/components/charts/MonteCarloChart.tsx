"use client";

import * as React from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";
import { MonteCarloBand } from "@/types";

interface MonteCarloChartProps {
  data: MonteCarloBand[];
  height?: number;
}

export function MonteCarloChart({ data, height = 260 }: MonteCarloChartProps) {
  if (!data || data.length === 0) return null;

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="mcGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="day"
            stroke="#64748b"
            fontSize={10}
            tickLine={false}
            axisLine={{ stroke: "rgba(51, 65, 85, 0.4)" }}
          />
          <YAxis
            domain={["auto", "auto"]}
            stroke="#94a3b8"
            fontSize={10}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
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
            formatter={(v: any) => [`$${Number(v).toLocaleString()}`, ""]}
          />
          <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
          <Area
            type="monotone"
            dataKey="p95"
            name="95th Percentile (Bullish)"
            stroke="#06b6d4"
            strokeDasharray="2 2"
            fill="url(#mcGrad)"
          />
          <Line
            type="monotone"
            dataKey="p50"
            name="50th Percentile (Median)"
            stroke="#10b981"
            strokeWidth={2}
            dot={false}
          />
          <Area
            type="monotone"
            dataKey="p5"
            name="5th Percentile (Stress)"
            stroke="#f43f5e"
            strokeDasharray="2 2"
            fill="transparent"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
