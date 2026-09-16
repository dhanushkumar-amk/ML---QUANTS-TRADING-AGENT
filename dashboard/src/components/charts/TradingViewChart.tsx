"use client";

import * as React from "react";
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  HistogramData,
  LineData,
  SeriesMarker,
  Time,
} from "lightweight-charts";
import { CandlestickBar, ChartMarker } from "@/types";

interface TradingViewChartProps {
  bars: CandlestickBar[];
  markers?: ChartMarker[];
  showSMA20?: boolean;
  showSMA60?: boolean;
  showBollinger?: boolean;
  height?: number;
}

export function TradingViewChart({
  bars,
  markers = [],
  showSMA20 = true,
  showSMA60 = true,
  showBollinger = false,
  height = 480,
}: TradingViewChartProps) {
  const chartContainerRef = React.useRef<HTMLDivElement>(null);
  const chartRef = React.useRef<IChartApi | null>(null);
  const candleSeriesRef = React.useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = React.useRef<ISeriesApi<"Histogram"> | null>(null);
  const sma20SeriesRef = React.useRef<ISeriesApi<"Line"> | null>(null);
  const sma60SeriesRef = React.useRef<ISeriesApi<"Line"> | null>(null);
  const bbUpperSeriesRef = React.useRef<ISeriesApi<"Line"> | null>(null);
  const bbLowerSeriesRef = React.useRef<ISeriesApi<"Line"> | null>(null);

  React.useEffect(() => {
    if (!chartContainerRef.current) return;

    // Initialize Lightweight-Charts canvas with terminal theme
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height,
      layout: {
        background: { color: "transparent" },
        textColor: "#94a3b8",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(51, 65, 85, 0.25)" },
        horzLines: { color: "rgba(51, 65, 85, 0.25)" },
      },
      crosshair: {
        mode: 1,
        vertLine: { color: "#64748b", width: 1, style: 2 },
        horzLine: { color: "#64748b", width: 1, style: 2 },
      },
      timeScale: {
        borderColor: "rgba(51, 65, 85, 0.5)",
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: "rgba(51, 65, 85, 0.5)",
        autoScale: true,
      },
    });

    chartRef.current = chart;

    // 1. Volume Histogram series (docked at bottom 20%)
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });
    volumeSeriesRef.current = volumeSeries;

    // 2. Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#10b981", // Emerald green
      downColor: "#f43f5e", // Rose red
      borderVisible: false,
      wickUpColor: "#10b981",
      wickDownColor: "#f43f5e",
    });
    candleSeriesRef.current = candleSeries;

    // 3. Technical Indicator Overlays
    const sma20 = chart.addLineSeries({
      color: "#f59e0b", // Amber
      lineWidth: 1,
      title: "SMA 20",
    });
    sma20SeriesRef.current = sma20;

    const sma60 = chart.addLineSeries({
      color: "#06b6d4", // Cyan
      lineWidth: 1,
      title: "SMA 60",
    });
    sma60SeriesRef.current = sma60;

    const bbUpper = chart.addLineSeries({
      color: "#a855f7", // Purple
      lineWidth: 1,
      lineStyle: 2,
      title: "BB Upper",
    });
    bbUpperSeriesRef.current = bbUpper;

    const bbLower = chart.addLineSeries({
      color: "#a855f7",
      lineWidth: 1,
      lineStyle: 2,
      title: "BB Lower",
    });
    bbLowerSeriesRef.current = bbLower;

    // Resize observer to ensure responsive reflow
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [height]);

  // Update Data and Series
  React.useEffect(() => {
    if (!bars || bars.length === 0 || !candleSeriesRef.current) return;

    // Map candlesticks
    const candleData: CandlestickData<Time>[] = bars.map((b) => ({
      time: b.time as Time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }));
    candleSeriesRef.current.setData(candleData);

    // Map volume histogram
    if (volumeSeriesRef.current) {
      const volData: HistogramData<Time>[] = bars.map((b) => ({
        time: b.time as Time,
        value: b.volume,
        color: b.close >= b.open ? "rgba(16, 185, 129, 0.35)" : "rgba(244, 63, 94, 0.35)",
      }));
      volumeSeriesRef.current.setData(volData);
    }

    // Indicators
    if (sma20SeriesRef.current) {
      sma20SeriesRef.current.applyOptions({ visible: showSMA20 });
      if (showSMA20) {
        const d: LineData<Time>[] = bars.map((b) => ({ time: b.time as Time, value: b.sma20 }));
        sma20SeriesRef.current.setData(d);
      }
    }

    if (sma60SeriesRef.current) {
      sma60SeriesRef.current.applyOptions({ visible: showSMA60 });
      if (showSMA60) {
        const d: LineData<Time>[] = bars.map((b) => ({ time: b.time as Time, value: b.sma60 }));
        sma60SeriesRef.current.setData(d);
      }
    }

    if (bbUpperSeriesRef.current && bbLowerSeriesRef.current) {
      bbUpperSeriesRef.current.applyOptions({ visible: showBollinger });
      bbLowerSeriesRef.current.applyOptions({ visible: showBollinger });
      if (showBollinger) {
        const up: LineData<Time>[] = bars.map((b) => ({ time: b.time as Time, value: b.bb_upper }));
        const low: LineData<Time>[] = bars.map((b) => ({ time: b.time as Time, value: b.bb_lower }));
        bbUpperSeriesRef.current.setData(up);
        bbLowerSeriesRef.current.setData(low);
      }
    }

    // Set Buy/Sell Trade Execution Markers directly on candles
    if (markers && markers.length > 0 && candleSeriesRef.current) {
      const tvMarkers: SeriesMarker<Time>[] = markers.map((m) => ({
        time: m.time as Time,
        position: m.position,
        color: m.color,
        shape: m.shape,
        text: m.text,
      }));
      candleSeriesRef.current.setMarkers(tvMarkers);
    }

    // Fit content into viewport
    chartRef.current?.timeScale().fitContent();
  }, [bars, markers, showSMA20, showSMA60, showBollinger]);

  return (
    <div className="relative w-full rounded-lg overflow-hidden border border-border/60 bg-card/40">
      <div ref={chartContainerRef} className="w-full" style={{ height }} />
    </div>
  );
}
