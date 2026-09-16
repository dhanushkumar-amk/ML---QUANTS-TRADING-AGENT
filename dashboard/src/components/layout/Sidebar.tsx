"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  CandlestickChart,
  LineChart,
  ShieldAlert,
  Cpu,
  Activity,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Overview", href: "/", icon: LayoutDashboard },
  { name: "Live Chart", href: "/chart/AAPL", icon: CandlestickChart },
  { name: "Backtest Analysis", href: "/backtest", icon: LineChart },
  { name: "Risk & Audit", href: "/risk", icon: ShieldAlert },
  { name: "Model Comparison", href: "/models", icon: Cpu },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-border/70 bg-card/60 backdrop-blur flex flex-col justify-between shrink-0 min-h-screen">
      <div>
        {/* Brand Header */}
        <div className="h-16 flex items-center px-6 border-b border-border/60 gap-3">
          <div className="h-8 w-8 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <div className="font-bold text-sm tracking-tight text-foreground flex items-center gap-1.5">
              <span>AGY QUANT</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-400 font-mono">
                PAPER
              </span>
            </div>
            <div className="text-[10px] text-muted-foreground font-mono">
              v1.0 • Alpaca Live Rail
            </div>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="p-3 space-y-1">
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
                  "flex items-center gap-3 px-3 py-2.5 rounded-md text-xs font-medium transition-all duration-150",
                  isActive
                    ? "bg-emerald-500/15 text-emerald-400 font-semibold border border-emerald-500/30"
                    : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                )}
              >
                <Icon className={cn("h-4 w-4", isActive ? "text-emerald-400" : "text-muted-foreground")} />
                {item.name}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer System Diagnostics */}
      <div className="p-4 border-t border-border/60 text-xs">
        <div className="p-3 rounded-lg bg-muted/40 border border-border/40 space-y-2">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-muted-foreground">Execution Loop</span>
            <span className="flex items-center gap-1 text-emerald-400 font-mono">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              ONLINE
            </span>
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-muted-foreground">Market Feed</span>
            <span className="text-foreground font-mono">Alpaca IEX</span>
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-muted-foreground">Drawdown Limit</span>
            <span className="text-foreground font-mono">15.0% Kill-Switch</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
