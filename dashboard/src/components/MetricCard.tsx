"use client";

import * as React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
  // Generate simple SVG path from sparkline points
  const renderSparkline = () => {
    if (!sparklineData || sparklineData.length < 2) return null;
    const min = Math.min(...sparklineData);
    const max = Math.max(...sparklineData);
    const range = max - min || 1;
    const width = 100;
    const height = 28;

    const points = sparklineData.map((val, idx) => {
      const x = (idx / (sparklineData.length - 1)) * width;
      const y = height - ((val - min) / range) * (height - 6) - 3;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    const isProfit = changeType === "profit" || sparklineData[sparklineData.length - 1] >= sparklineData[0];
    const strokeColor = isProfit ? "#10b981" : "#f43f5e";

    return (
      <svg className="w-24 h-7 overflow-visible" viewBox={`0 0 ${width} ${height}`}>
        <polyline
          fill="none"
          stroke={strokeColor}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points.join(" ")}
        />
      </svg>
    );
  };

  return (
    <Card className="hover:border-slate-700/80 hover:shadow-lg hover:shadow-black/20">
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            {title}
          </span>
          {icon && <div className="text-muted-foreground/70">{icon}</div>}
        </div>

        <div className="mt-2 flex items-baseline justify-between">
          <span className="text-2xl font-bold tracking-tight text-foreground">
            {value}
          </span>
          {renderSparkline()}
        </div>

        <div className="mt-2 flex items-center justify-between text-xs">
          {change && (
            <Badge
              variant={
                changeType === "profit"
                  ? "profit"
                  : changeType === "loss"
                  ? "loss"
                  : "neutral"
              }
            >
              {change}
            </Badge>
          )}
          {subtitle && (
            <span className="text-muted-foreground text-[11px] ml-auto">
              {subtitle}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
