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
} from "recharts";
import { EquityCurvePoint } from "@/types";

interface EquityCurveChartProps {
  data: EquityCurvePoint[];
  height?: number;
}

function formatDate(dateStr: string) {
  if (!dateStr) return "";
  const parts = dateStr.split("-");
  if (parts.length === 3) {
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const m = parseInt(parts[1], 10) - 1;
    return `${months[m] || parts[1]} ${parts[2]}`;
  }
  return dateStr;
}

// Monochrome Glassmorphism Tooltip
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload || !payload.length) return null;

  const strat = payload.find((p: any) => p.dataKey === "strategy")?.value;
  const bench = payload.find((p: any) => p.dataKey === "benchmark")?.value;
  const dd = payload.find((p: any) => p.dataKey === "drawdown")?.value;

  return (
    <div className="rounded-[4px] border border-[#262626] bg-[#0c0c0c] p-3 shadow-xl text-xs font-mono space-y-1.5 min-w-[190px]">
      <div className="text-[11px] font-semibold text-slate-400 pb-1 border-b border-[#1f1f1f]">
        {formatDate(label)} ({label})
      </div>
      {strat !== undefined && (
        <div className="flex items-center justify-between text-white">
          <span className="flex items-center gap-1.5 text-[#00B386]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#00B386]" />
            Strategy Equity:
          </span>
          <span className="font-bold tabular-nums">${Number(strat).toLocaleString()}</span>
        </div>
      )}
      {bench !== undefined && (
        <div className="flex items-center justify-between text-slate-300">
          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
            SPY Index:
          </span>
          <span className="tabular-nums">${Number(bench).toLocaleString()}</span>
        </div>
      )}
      {dd !== undefined && (
        <div className="flex items-center justify-between text-[#EB5757] pt-1 border-t border-[#1f1f1f]">
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-[#EB5757]" />
            Drawdown:
          </span>
          <span className="font-bold tabular-nums">{dd}%</span>
        </div>
      )}
    </div>
  );
}

export function EquityCurveChart({ data, height = 370 }: EquityCurveChartProps) {
  if (!data || data.length === 0) return null;

  return (
    <div className="w-full space-y-2 font-mono">
      {/* Chart Legend */}
      <div className="flex items-center justify-between px-1 text-xs">
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-[#00B386]" />
            <span className="font-medium text-white">Strategy Equity</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-slate-500" />
            <span className="text-slate-400">S&amp;P 500 ETF (SPY)</span>
          </div>
        </div>
        <span className="text-slate-500 text-[11px]">Daily Close</span>
      </div>

      {/* 1. Main Equity Area Chart */}
      <div className="w-full" style={{ height: height * 0.72 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <defs>
              <linearGradient id="monochromeStratGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#00B386" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#00B386" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="date"
              hide
              stroke="#333"
              tickLine={false}
              axisLine={{ stroke: "#1f1f1f" }}
            />
            <YAxis
              domain={["dataMin - 1500", "dataMax + 1500"]}
              stroke="#555"
              fontSize={11}
              tickFormatter={(val) => `$${(val / 1000).toFixed(0)}k`}
              tickLine={false}
              axisLine={false}
              tickCount={5}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="strategy"
              name="Strategy Equity"
              stroke="#00B386"
              strokeWidth={2}
              fill="url(#monochromeStratGrad)"
              activeDot={{ r: 4, fill: "#00B386", stroke: "#080808", strokeWidth: 2 }}
            />
            <Line
              type="monotone"
              dataKey="benchmark"
              name="SPY Benchmark"
              stroke="#555555"
              strokeWidth={1.5}
              strokeDasharray="4 4"
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* 2. Underwater Drawdown Area Subplot */}
      <div className="w-full pt-2 border-t border-[#181818]" style={{ height: height * 0.28 }}>
        <div className="flex items-center justify-between text-[11px] text-slate-500 px-1 pb-1">
          <span className="flex items-center gap-1.5 text-[#EB5757]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#EB5757]" />
            Drawdown Underwater (%)
          </span>
          <span>Risk Limit: -15.0%</span>
        </div>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 2, right: 10, left: 10, bottom: 2 }}>
            <defs>
              <linearGradient id="monochromeDrawdownGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#EB5757" stopOpacity={0.25} />
                <stop offset="100%" stopColor="#EB5757" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="date"
              stroke="#555"
              fontSize={10}
              tickLine={false}
              axisLine={{ stroke: "#1f1f1f" }}
              tickFormatter={formatDate}
              interval="preserveStartEnd"
              minTickGap={50}
            />
            <YAxis
              domain={["dataMin - 1", 0]}
              stroke="#555"
              fontSize={10}
              tickFormatter={(val) => `${val}%`}
              tickLine={false}
              axisLine={false}
              tickCount={3}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="drawdown"
              stroke="#EB5757"
              strokeWidth={1.5}
              fill="url(#monochromeDrawdownGrad)"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
