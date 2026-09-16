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
  CartesianGrid,
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
            <linearGradient id="mcGradZerodha" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#00B386" stopOpacity={0.12} />
              <stop offset="95%" stopColor="#00B386" stopOpacity={0.01} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="2 2" stroke="#161616" vertical={false} />
          <XAxis
            dataKey="day"
            stroke="#555555"
            fontSize={10}
            tickLine={false}
            axisLine={{ stroke: "#1F1F1F" }}
          />
          <YAxis
            domain={["auto", "auto"]}
            stroke="#555555"
            fontSize={10}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
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
            formatter={(v: any) => [`$${Number(v).toLocaleString()}`, ""]}
          />
          <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
          <Area
            type="monotone"
            dataKey="p95"
            name="95th Pct (Bullish)"
            stroke="#444444"
            strokeDasharray="2 2"
            fill="url(#mcGradZerodha)"
          />
          <Line
            type="monotone"
            dataKey="p50"
            name="50th Pct (Median)"
            stroke="#00B386"
            strokeWidth={1.5}
            dot={false}
          />
          <Area
            type="monotone"
            dataKey="p5"
            name="5th Pct (Stress Boundary)"
            stroke="#EB5757"
            strokeDasharray="2 2"
            fill="transparent"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
