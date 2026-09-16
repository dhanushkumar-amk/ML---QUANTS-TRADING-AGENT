"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Clock, TrendingUp } from "lucide-react";

const WATCHLIST_ITEMS = [
  { ticker: "AAPL", price: "224.50", delta: "+1.24%", isUp: true },
  { ticker: "MSFT", price: "432.10", delta: "+0.85%", isUp: true },
  { ticker: "NVDA", price: "118.90", delta: "+2.40%", isUp: true },
  { ticker: "SPY",  price: "562.80", delta: "+0.32%", isUp: true },
  { ticker: "GOOGL", price: "178.40", delta: "-0.15%", isUp: false },
];

export function Header() {
  const pathname = usePathname();
  const [timeStr, setTimeStr] = React.useState("");

  React.useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(now.toUTCString().slice(17, 25) + " UTC");
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-14 border-b border-[#1a1a1a] bg-[#070707] px-5 flex items-center justify-between sticky top-0 z-30 font-mono">
      {/* Watchlist Strip - Zerodha Marketwatch style */}
      <div className="flex items-center gap-2 overflow-x-auto py-1 scrollbar-none">
        <span className="text-[11px] text-slate-500 mr-2 flex items-center gap-1 font-semibold uppercase tracking-wider shrink-0">
          <TrendingUp className="h-3.5 w-3.5 text-slate-400" /> Watchlist
        </span>
        <div className="flex items-center gap-1.5">
          {WATCHLIST_ITEMS.map((item) => {
            const isSelected = pathname === `/chart/${item.ticker}`;
            return (
              <Link
                key={item.ticker}
                href={`/chart/${item.ticker}`}
                className={`flex items-center gap-2 px-2.5 py-1 rounded-[4px] text-xs transition-colors border ${
                  isSelected
                    ? "bg-[#141414] text-white border-white/20 font-bold"
                    : "bg-[#0c0c0c] text-slate-300 border-[#1a1a1a] hover:bg-[#121212] hover:border-[#2a2a2a]"
                }`}
              >
                <span className="text-white font-semibold">{item.ticker}</span>
                <span className="text-slate-400 tabular-nums">{item.price}</span>
                <span className={`text-[11px] font-medium tabular-nums ${item.isUp ? "text-[#00B386]" : "text-[#EB5757]"}`}>
                  {item.delta}
                </span>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Right Controls: Market Status, UTC clock */}
      <div className="flex items-center gap-3 shrink-0">
        {/* Live NYSE Status */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-[4px] bg-[#0c0c0c] border border-[#1a1a1a] text-xs">
          <span className="h-1.5 w-1.5 rounded-full bg-[#00B386]" />
          <span className="text-slate-300 font-medium">NYSE</span>
          <span className="text-[10px] text-[#00B386] font-semibold">PAPER</span>
        </div>

        {/* Live UTC Clock */}
        <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-[4px] bg-[#0c0c0c] border border-[#1a1a1a] text-xs text-slate-400">
          <Clock className="h-3 w-3 text-slate-500" />
          <span className="tabular-nums">{timeStr || "16:45:00 UTC"}</span>
        </div>
      </div>
    </header>
  );
}
