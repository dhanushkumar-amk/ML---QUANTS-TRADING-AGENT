"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  title: string;
  value: string;
  change?: string;
  changeType?: "profit" | "loss" | "neutral";
  subtitle?: string;
  sparklineData?: number[];
  icon?: React.ReactNode;
}

export function MetricCard({
  title,
  value,
  change,
  changeType = "neutral",
  subtitle,
  sparklineData,
  icon,
}: MetricCardProps) {
  const renderSparkline = () => {
    if (!sparklineData || sparklineData.length < 2) return null;
    const min = Math.min(...sparklineData);
    const max = Math.max(...sparklineData);
    const range = max - min || 1;
    const width = 100;
    const height = 30;

    const points = sparklineData.map((val, idx) => {
      const x = (idx / (sparklineData.length - 1)) * width;
      const y = height - ((val - min) / range) * (height - 6) - 3;
      return { x: Number(x.toFixed(1)), y: Number(y.toFixed(1)) };
    });

    const isProfit = changeType === "profit" || sparklineData[sparklineData.length - 1] >= sparklineData[0];
    const strokeColor = isProfit ? "#00B386" : "#EB5757";
    const gradientId = React.useId().replace(/:/g, "-");

    const pathD = points.reduce((acc, p, i) => {
      return i === 0 ? `M ${p.x},${p.y}` : `${acc} L ${p.x},${p.y}`;
    }, "");

    const areaD = `${pathD} L ${width},${height} L 0,${height} Z`;

    return (
      <div className="shrink-0">
        <svg className="w-24 h-8 overflow-visible" viewBox={`0 0 ${width} ${height}`}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={strokeColor} stopOpacity={0.18} />
              <stop offset="100%" stopColor={strokeColor} stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <path d={areaD} fill={`url(#${gradientId})`} />
          <polyline
            fill="none"
            stroke={strokeColor}
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={points.map((p) => `${p.x},${p.y}`).join(" ")}
          />
        </svg>
      </div>
    );
  };

  const isProfit = changeType === "profit";
  const isLoss = changeType === "loss";

  return (
    <div className="rounded-[4px] border border-[#1f1f1f] bg-[#0c0c0c] p-4 hover:border-[#2f2f2f] transition-colors">
      {/* Title & Icon Header */}
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 font-mono">
          {title}
        </span>
        {icon && <div className="text-slate-500">{icon}</div>}
      </div>

      {/* Main Metric Value & Sparkline */}
      <div className="mt-2.5 flex items-baseline justify-between gap-3">
        <span className="text-2xl font-bold tracking-tight text-white font-mono tabular-nums">
          {value}
        </span>
        {renderSparkline()}
      </div>

      {/* Footer Pill & Subtitle */}
      <div className="mt-3 pt-2.5 border-t border-[#181818] flex items-center justify-between text-xs font-mono">
        {change && (
          <span
            className={cn(
              "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-[3px] text-[11px] font-medium border",
              isProfit && "bg-[#00B386]/10 text-[#00B386] border-[#00B386]/25",
              isLoss && "bg-[#EB5757]/10 text-[#EB5757] border-[#EB5757]/25",
              !isProfit && !isLoss && "bg-[#181818] text-slate-300 border-[#262626]"
            )}
          >
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                isProfit && "bg-[#00B386]",
                isLoss && "bg-[#EB5757]",
                !isProfit && !isLoss && "bg-slate-400"
              )}
            />
            {change}
          </span>
        )}

        {subtitle && (
          <span className="text-[11px] text-slate-500 tabular-nums ml-auto truncate">
            {subtitle}
          </span>
        )}
      </div>
    </div>
  );
}
