import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/react";
import React from "react";
import { EquityCurveChart } from "@/components/charts/EquityCurveChart";
import { RollingMetricsChart } from "@/components/charts/RollingMetricsChart";
import { FeatureImportanceChart } from "@/components/charts/FeatureImportanceChart";

// Mock resize observer for Recharts / DOM canvas
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

describe("Chart Components Render Check", () => {
  it("renders EquityCurveChart without crashing", () => {
    const sampleEquity = [
      { date: "2026-09-01", strategy: 100000, benchmark: 100000, drawdown: 0 },
      { date: "2026-09-02", strategy: 101200, benchmark: 100400, drawdown: -0.5 },
    ];

    const { container } = render(<EquityCurveChart data={sampleEquity} height={300} />);
    expect(container).toBeDefined();
  });

  it("renders RollingMetricsChart without crashing", () => {
    const sampleRolling = [
      { date: "2026-09-01", rolling_sharpe: 2.1, rolling_volatility: 11.4 },
      { date: "2026-09-08", rolling_sharpe: 2.15, rolling_volatility: 11.2 },
    ];

    const { container } = render(<RollingMetricsChart data={sampleRolling} height={200} />);
    expect(container).toBeDefined();
  });

  it("renders FeatureImportanceChart without crashing", () => {
    const sampleFeatures = [
      { feature: "RSI (14d)", category: "Momentum", importance: 0.38 },
      { feature: "FinBERT Sentiment", category: "NLP / Alternative", importance: 0.34 },
    ];

    const { container } = render(<FeatureImportanceChart data={sampleFeatures} height={200} />);
    expect(container).toBeDefined();
  });
});
