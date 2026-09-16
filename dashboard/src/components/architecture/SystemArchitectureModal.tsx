"use client";

import * as React from "react";
import {
  X,
  Database,
  Binary,
  Cpu,
  ShieldCheck,
  Zap,
  ArrowRight,
  ExternalLink,
  CheckCircle2,
  Lock,
} from "lucide-react";

interface SystemArchitectureModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface Stage {
  num: string;
  phase: string;
  name: string;
  icon: React.ReactNode;
  summary: string;
  details: string[];
  latency: string;
  status: string;
}

const PIPELINE_STAGES: Stage[] = [
  {
    num: "01",
    phase: "Phases 1–4",
    name: "Data Ingestion & Feed",
    icon: <Database className="h-5 w-5 text-sky-400" />,
    summary: "Real-time websocket streaming and historical bar aggregation.",
    details: [
      "Polygon.io & Alpaca real-time market data websockets",
      "NYSE market hours calendar & trading holiday gating",
      "1-minute OHLCV bar builder with volume-weighted timestamps",
      "Corporate actions adjustment and survivorship bias filtering",
    ],
    latency: "< 2.1 ms",
    status: "Active Feed",
  },
  {
    num: "02",
    phase: "Phases 17 & 34",
    name: "Feature Pipeline",
    icon: <Binary className="h-5 w-5 text-indigo-400" />,
    summary: "Multi-timeframe technical, microstructure, and NLP sentiment.",
    details: [
      "Technical momentum: Multi-period RSI, MACD, Bollinger %B",
      "Microstructure: Parkinson volatility & Garman-Klass estimator",
      "Alternative NLP: FinBERT news & SEC 10-K sentiment scoring",
      "Stationarity: Fractional differentiation (d=0.35) preservation",
    ],
    latency: "< 4.3 ms",
    status: "Real-time Calc",
  },
  {
    num: "03",
    phase: "Phases 28–34",
    name: "Production Model Inference",
    icon: <Cpu className="h-5 w-5 text-[#00B386]" />,
    summary: "Stacked meta-learner combining tree ensembles and recurrent models.",
    details: [
      "Base models: XGBoost (Tree), LightGBM, and BiLSTM",
      "Meta-model: Ridge logistic stacking with cross-validated weights",
      "Walk-forward purged cross-validation (5 test folds)",
      "Generates directional probability & target positioning vector",
    ],
    latency: "< 12.4 ms",
    status: "Live Active",
  },
  {
    num: "04",
    phase: "Phase 39",
    name: "Risk Engine Gating",
    icon: <ShieldCheck className="h-5 w-5 text-amber-400" />,
    summary: "Deterministic pre-trade safety controls and circuit breakers.",
    details: [
      "Drawdown circuit breaker: Hard halt at -15.0% peak drawdown",
      "Concentration limit: Max 15% portfolio allocation per single ticker",
      "Volatility scaling: Inversely scaled by rolling ATR & GARCH sigma",
      "Daily 95% Value-at-Risk (VaR) and Expected Shortfall limits",
    ],
    latency: "< 0.8 ms",
    status: "Pre-Trade Gated",
  },
  {
    num: "05",
    phase: "Phases 44–47",
    name: "Execution & Audit Trail",
    icon: <Zap className="h-5 w-5 text-emerald-400" />,
    summary: "Smart order routing to Alpaca paper broker with immutable logging.",
    details: [
      "Smart Order Router: TWAP/VWAP slices to minimize price slippage",
      "Direct broker connectivity via Alpaca Markets Paper API",
      "Two-way ledger reconciliation matching broker fills vs local state",
      "Forensic JSONL event audit logging answering 'Why trade X at time T?'",
    ],
    latency: "< 13.8 ms",
    status: "Paper Live",
  },
];

export function SystemArchitectureModal({ isOpen, onClose }: SystemArchitectureModalProps) {
  const [selectedStage, setSelectedStage] = React.useState<number>(2); // Default to ML Model

  if (!isOpen) return null;

  const current = PIPELINE_STAGES[selectedStage];

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-3 sm:p-6 animate-in fade-in duration-150">
      <div className="bg-[#0C0C0C] border border-[#222222] rounded-[4px] max-w-4xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden font-mono">
        {/* Modal Top Header */}
        <div className="p-4 border-b border-[#1A1A1A] flex items-center justify-between bg-[#080808]">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-[4px] bg-[#00B386]/10 border border-[#00B386]/20 flex items-center justify-center">
              <Cpu className="h-4 w-4 text-[#00B386]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-bold text-[#FAFAFA]">
                  Quantitative End-to-End System Architecture
                </h2>
                <span className="text-[10px] px-2 py-0.2 rounded-[2px] bg-[#E5A93B]/10 text-[#E5A93B] border border-[#E5A93B]/30 font-semibold">
                  SIMULATED / PAPER DEMO
                </span>
              </div>
              <p className="text-[11px] text-[#666666] font-sans">
                Modular 5-stage institutional pipeline from raw tick ingestion to broker paper execution.
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-[3px] text-[#666666] hover:text-[#FAFAFA] hover:bg-[#161616] transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Pipeline Progress Breadcrumb / Node Bar */}
        <div className="p-4 bg-[#090909] border-b border-[#1A1A1A] overflow-x-auto">
          <div className="flex items-center min-w-[620px] justify-between">
            {PIPELINE_STAGES.map((st, idx) => {
              const isSelected = selectedStage === idx;
              return (
                <React.Fragment key={st.num}>
                  <button
                    onClick={() => setSelectedStage(idx)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-[4px] border text-left transition-all ${
                      isSelected
                        ? "bg-[#141414] border-[#00B386] shadow-[0_0_12px_rgba(0,179,134,0.15)]"
                        : "bg-[#0C0C0C] border-[#1C1C1C] hover:border-[#2A2A2A] opacity-75 hover:opacity-100"
                    }`}
                  >
                    <div className="shrink-0">{st.icon}</div>
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] text-[#555555] font-bold">{st.num}</span>
                        <span className={`text-xs font-bold ${isSelected ? "text-white" : "text-slate-300"}`}>
                          {st.name}
                        </span>
                      </div>
                      <span className="text-[10px] text-[#777777] block">{st.phase}</span>
                    </div>
                  </button>
                  {idx < PIPELINE_STAGES.length - 1 && (
                    <ArrowRight className="h-4 w-4 text-[#333333] shrink-0 mx-1" />
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Selected Stage Detail Panel */}
        <div className="p-5 overflow-y-auto space-y-5 flex-1 bg-[#0A0A0A]">
          {/* Header of Stage */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-[4px] bg-[#0F0F0F] border border-[#1A1A1A]">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-[4px] bg-[#141414] border border-[#222222]">
                {current.icon}
              </div>
              <div>
                <span className="text-[10px] text-[#00B386] font-bold uppercase tracking-wider">
                  STAGE {current.num} • {current.phase}
                </span>
                <h3 className="text-base font-bold text-white mt-0.5">{current.name}</h3>
                <p className="text-xs text-slate-400 font-sans mt-0.5">{current.summary}</p>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <div className="px-3 py-1.5 rounded-[3px] bg-[#0A0A0A] border border-[#1F1F1F] text-right">
                <span className="text-[10px] text-[#555555] block">PIPELINE LATENCY</span>
                <span className="text-xs font-bold text-[#00B386] tabular-nums">{current.latency}</span>
              </div>
              <div className="px-3 py-1.5 rounded-[3px] bg-[#0A0A0A] border border-[#1F1F1F] text-right">
                <span className="text-[10px] text-[#555555] block">STATUS</span>
                <span className="text-xs font-bold text-slate-200">{current.status}</span>
              </div>
            </div>
          </div>

          {/* Core Subsystem Specifications */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-4 rounded-[4px] bg-[#0D0D0D] border border-[#1A1A1A] space-y-2.5">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block border-b border-[#161616] pb-1.5">
                Technical Components & Implementation
              </span>
              <ul className="space-y-2 text-xs text-slate-300 font-sans">
                {current.details.map((detail, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-[#00B386] shrink-0 mt-0.5" />
                    <span>{detail}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="p-4 rounded-[4px] bg-[#0D0D0D] border border-[#1A1A1A] space-y-3">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block border-b border-[#161616] pb-1.5">
                Risk & Verification Controls
              </span>
              <div className="space-y-2 text-xs font-mono">
                <div className="flex justify-between py-1 border-b border-[#161616]">
                  <span className="text-[#666666]">Lookahead Bias Protection:</span>
                  <span className="text-[#00B386] font-bold">Purged Cross-Validation</span>
                </div>
                <div className="flex justify-between py-1 border-b border-[#161616]">
                  <span className="text-[#666666]">Execution Environment:</span>
                  <span className="text-amber-400 font-bold">Alpaca Paper (Mock Fund)</span>
                </div>
                <div className="flex justify-between py-1 border-b border-[#161616]">
                  <span className="text-[#666666]">Audit Traceability:</span>
                  <span className="text-slate-300">Deterministic UUID Cycle Log</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-[#666666]">Fail-Safe Fallback:</span>
                  <span className="text-slate-300">Auto-Flatten on 3 Feed Drops</span>
                </div>
              </div>
            </div>
          </div>

          {/* Disclaimer Callout */}
          <div className="p-3 rounded-[4px] bg-[#111111] border border-[#222222] flex items-center gap-3">
            <Lock className="h-4 w-4 text-[#E5A93B] shrink-0" />
            <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
              <strong className="text-slate-200">Notice for Reviewers & Recruiters:</strong> All metrics, execution logs, and fills visualized in this system represent <strong>simulated paper trading</strong> on Alpaca&apos;s paper trading rail. No real institutional capital is deployed or claimed.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 bg-[#080808] border-t border-[#1A1A1A] flex items-center justify-between text-xs text-[#555555]">
          <span>ML + Quants Trading Agent • Architecture Suite</span>
          <button
            onClick={onClose}
            className="px-3 py-1 rounded-[3px] bg-[#141414] hover:bg-[#1A1A1A] text-slate-300 hover:text-white border border-[#222222] transition-colors"
          >
            Close Overview
          </button>
        </div>
      </div>
    </div>
  );
}
