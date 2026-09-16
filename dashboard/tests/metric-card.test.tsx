import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { MetricCard } from "@/components/MetricCard";

describe("MetricCard Component", () => {
  it("renders title and formatted value correctly", () => {
    render(
      <MetricCard
        title="Portfolio Equity"
        value="$102,450.80"
        change="+1.23% today"
        changeType="profit"
        subtitle="Day: +$1,245.80"
      />
    );

    expect(screen.getByText("Portfolio Equity")).toBeInTheDocument();
    expect(screen.getByText("$102,450.80")).toBeInTheDocument();
    expect(screen.getByText("+1.23% today")).toBeInTheDocument();
    expect(screen.getByText("Day: +$1,245.80")).toBeInTheDocument();
  });

  it("renders sparkline polyline when sparklineData is provided", () => {
    const { container } = render(
      <MetricCard
        title="Sharpe Ratio"
        value="2.14"
        sparklineData={[1.9, 2.0, 2.05, 2.1, 2.14]}
      />
    );

    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
    const polyline = container.querySelector("polyline");
    expect(polyline).toBeInTheDocument();
  });
});
