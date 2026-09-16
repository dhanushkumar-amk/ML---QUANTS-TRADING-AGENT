"use client";

import * as React from "react";
import Link from "next/link";
import { ParticleHero3D } from "@/components/3d/ParticleHero3D";
import { ArrowRight, Terminal, Activity, Cpu, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeroSectionProps {
  onExploreDashboard?: () => void;
  onOpenArchitecture?: () => void;
}

export function HeroSection({ onExploreDashboard, onOpenArchitecture }: HeroSectionProps) {
  return (
    <section className="relative w-full border-b border-[#1f1f1f] bg-[#080808] text-white overflow-hidden pt-8 pb-12">
      {/* Background Grid Pattern */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.03] bg-[linear-gradient(to_right,#fff_1px,transparent_1px),linear-gradient(to_bottom,#fff_1px,transparent_1px)] bg-[size:32px_32px]" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
        {/* Top Prominent Paper Trading Simulation Disclaimer Tag */}
        <div className="flex items-center justify-center mb-5">
          <div className="inline-flex items-center gap-2.5 px-3.5 py-1.5 rounded-[4px] border border-[#2E2412] bg-[#141008] text-xs font-mono">
            <span className="h-2 w-2 rounded-full bg-[#E5A93B] animate-pulse shrink-0" />
            <span className="text-[#E5A93B] font-bold tracking-wide uppercase text-[11px]">
              SIMULATED / PAPER TRADING
            </span>
            <span className="text-[#554326]">•</span>
            <span className="text-slate-300 text-[11px]">ALPACA API SANDBOX • ZERO REAL CAPITAL AT RISK</span>
          </div>
        </div>

        {/* Hero Headline */}
        <div className="text-center space-y-3 max-w-3xl mx-auto">
          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white uppercase font-mono leading-tight">
            Pure Signal. Zero Noise.
          </h1>
          <p className="text-sm sm:text-base text-slate-400 font-normal leading-relaxed">
            Institutional-grade quantitative momentum models, risk gating, and continuous paper trade execution on Alpaca.
          </p>
        </div>

        {/* 3D Visual Centerpiece */}
        <div className="my-4">
          <ParticleHero3D />
        </div>

        {/* Primary CTAs including Architecture Pipeline */}
        <div className="flex flex-wrap items-center justify-center gap-3 mt-2">
          <Link href="/chart/AAPL">
            <Button className="h-10 px-5 rounded-[4px] bg-white text-black font-semibold text-xs hover:bg-slate-200 transition-colors gap-2 font-mono">
              <Terminal className="h-4 w-4" />
              <span>Launch Live Terminal</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Button>
          </Link>

          <Button
            variant="outline"
            onClick={onExploreDashboard}
            className="h-10 px-5 rounded-[4px] bg-[#111] border-[#222] text-xs text-white hover:bg-[#181818] hover:border-[#333] transition-colors gap-2 font-mono"
          >
            <Activity className="h-4 w-4 text-[#00B386]" />
            <span>View Executive Portfolio</span>
          </Button>

          {onOpenArchitecture && (
            <Button
              variant="outline"
              onClick={onOpenArchitecture}
              className="h-10 px-4 rounded-[4px] bg-[#0E0E0E] border-[#222] text-xs text-slate-300 hover:text-white hover:bg-[#161616] hover:border-[#333] transition-colors gap-2 font-mono"
            >
              <Cpu className="h-4 w-4 text-sky-400" />
              <span>System Pipeline (5-Stage)</span>
            </Button>
          )}
        </div>

        {/* Below-the-fold Key Metrics Ticker Bar */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2 sm:gap-4 mt-10 pt-8 border-t border-[#1a1a1a] text-center font-mono">
          <div className="p-3 rounded-[4px] bg-[#0d0d0d] border border-[#1a1a1a]">
            <span className="text-[11px] text-slate-500 block uppercase tracking-wider">ANNUALIZED SHARPE</span>
            <span className="text-xl sm:text-2xl font-bold text-white tabular-nums">2.14</span>
            <span className="text-[10px] text-[#00B386] block mt-0.5">Institutional &gt; 2.0</span>
          </div>

          <div className="p-3 rounded-[4px] bg-[#0d0d0d] border border-[#1a1a1a]">
            <span className="text-[11px] text-slate-500 block uppercase tracking-wider">ALPHA VS SPY</span>
            <span className="text-xl sm:text-2xl font-bold text-[#00B386] tabular-nums">+18.6%</span>
            <span className="text-[10px] text-slate-400 block mt-0.5">S&amp;P 500 Outperformance</span>
          </div>

          <div className="p-3 rounded-[4px] bg-[#0d0d0d] border border-[#1a1a1a]">
            <span className="text-[11px] text-slate-500 block uppercase tracking-wider">MAX DRAWDOWN</span>
            <span className="text-xl sm:text-2xl font-bold text-white tabular-nums">-8.42%</span>
            <span className="text-[10px] text-slate-400 block mt-0.5">Halt Limit: -15.0%</span>
          </div>

          <div className="p-3 rounded-[4px] bg-[#0d0d0d] border border-[#1a1a1a]">
            <span className="text-[11px] text-slate-500 block uppercase tracking-wider">WIN RATE</span>
            <span className="text-xl sm:text-2xl font-bold text-white tabular-nums">58.6%</span>
            <span className="text-[10px] text-[#00B386] block mt-0.5">Profit Factor: 1.85</span>
          </div>

          <div className="col-span-2 md:col-span-1 p-3 rounded-[4px] bg-[#0d0d0d] border border-[#1a1a1a]">
            <span className="text-[11px] text-slate-500 block uppercase tracking-wider">EXECUTION LATENCY</span>
            <span className="text-xl sm:text-2xl font-bold text-white tabular-nums">&lt;14ms</span>
            <span className="text-[10px] text-[#00B386] block mt-0.5">Direct Broker Rail</span>
          </div>
        </div>
      </div>
    </section>
  );
}
