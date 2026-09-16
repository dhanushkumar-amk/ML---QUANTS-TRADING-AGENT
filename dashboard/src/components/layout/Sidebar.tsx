"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  CandlestickChart,
  LineChart,
  ShieldCheck,
  Cpu,
  ChevronRight,
  Radio,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Overview", href: "/", icon: LayoutDashboard },
  { name: "Live Terminal", href: "/chart/AAPL", icon: CandlestickChart },
  { name: "Backtest Studio", href: "/backtest", icon: LineChart },
  { name: "Risk Guardrails", href: "/risk", icon: ShieldCheck },
  { name: "AI Models", href: "/models", icon: Cpu },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-[230px] border-r border-[#1a1a1a] bg-[#070707] flex flex-col justify-between shrink-0 min-h-screen select-none z-20">
      <div>
        {/* Brand Header - Zerodha/Groww Clean Flat Logo */}
        <div className="h-14 flex items-center px-4 border-b border-[#1a1a1a] gap-2.5">
          <div className="h-7 w-7 rounded-[4px] bg-white text-black flex items-center justify-center font-bold text-xs font-mono">
            AQ
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-xs tracking-wider text-white font-mono uppercase">
              AGY QUANT
            </span>
            <span className="text-[10px] text-slate-500 font-mono">
              Alpaca Paper Rail
            </span>
          </div>
        </div>

        {/* Navigation Section */}
        <div className="p-3">
          <div className="px-2 mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500 font-mono">
            Menu
          </div>
          <nav className="space-y-1">
            {navigation.map((item) => {
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href.split("/")[1]);
              const Icon = item.icon;

              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "flex items-center justify-between px-3 py-2 rounded-[4px] text-xs font-medium transition-colors group",
                    isActive
                      ? "bg-[#141414] text-white font-semibold border-l-2 border-[#00B386]"
                      : "text-slate-400 hover:text-white hover:bg-[#0f0f0f]"
                  )}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon
                      className={cn(
                        "h-4 w-4 transition-colors",
                        isActive ? "text-[#00B386]" : "text-slate-500 group-hover:text-slate-300"
                      )}
                    />
                    <span>{item.name}</span>
                  </div>
                  {isActive && (
                    <ChevronRight className="h-3 w-3 text-slate-500" />
                  )}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Footer System Status - Minimal Zerodha Kite Account Box */}
      <div className="p-3 border-t border-[#1a1a1a]">
        <div className="p-3 rounded-[4px] bg-[#0c0c0c] border border-[#1a1a1a] space-y-2 text-xs font-mono">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider">Simulated Equity</span>
            <span className="text-white font-bold tabular-nums">$102,450.80</span>
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-slate-500">Buying Power</span>
            <span className="text-slate-300 tabular-nums">$399,240.87</span>
          </div>
          <div className="pt-2 border-t border-[#181818] flex items-center justify-between text-[10px]">
            <span className="inline-flex items-center gap-1.5 text-[#E5A93B] font-bold">
              <span className="h-1.5 w-1.5 rounded-full bg-[#E5A93B] animate-pulse" />
              ALPACA PAPER
            </span>
            <span className="text-slate-500 font-mono text-[9px]">ID: SIM-79401</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
