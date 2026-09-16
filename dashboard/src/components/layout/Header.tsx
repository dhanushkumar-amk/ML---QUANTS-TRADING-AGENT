"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Clock, TrendingUp, Cpu } from "lucide-react";
import { SystemArchitectureModal } from "@/components/architecture/SystemArchitectureModal";

interface WatchItem {
  ticker: string;
  price: number;
  delta: number;
  flash?: "up" | "down" | null;
}

const INITIAL_WATCHLIST: WatchItem[] = [
  { ticker: "AAPL",  price: 224.50, delta: 1.24 },
  { ticker: "MSFT",  price: 432.10, delta: 0.85 },
  { ticker: "NVDA",  price: 118.90, delta: 2.40 },
  { ticker: "SPY",   price: 562.80, delta: 0.32 },
  { ticker: "GOOGL", price: 178.40, delta: -0.15 },
];

interface HeaderProps {
  onOpenArchitecture?: () => void;
}

export function Header({ onOpenArchitecture }: HeaderProps) {
  const pathname = usePathname();
  const [watchlist, setWatchlist] = React.useState<WatchItem[]>(INITIAL_WATCHLIST);
  const [timeStr, setTimeStr] = React.useState("");
  const [isArchOpen, setIsArchOpen] = React.useState(false);

  // Clock interval
  React.useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(now.toUTCString().slice(17, 25) + " UTC");
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  // Subtle live price ticking animation (every 2.8 seconds)
  React.useEffect(() => {
    const tickInterval = setInterval(() => {
      setWatchlist((prev) => {
        // Pick a random ticker to tick
        const targetIdx = Math.floor(Math.random() * prev.length);
        const item = prev[targetIdx];
        
        // Random price fluctuation between -$0.30 and +$0.35
        const change = (Math.random() * 0.65 - 0.30);
        const newPrice = Math.max(10, +(item.price + change).toFixed(2));
        const priceDiff = newPrice - item.price;
        const newDelta = +(item.delta + (priceDiff / item.price) * 100).toFixed(2);
        const direction = priceDiff >= 0 ? "up" : "down";

        return prev.map((w, idx) => {
          if (idx !== targetIdx) return { ...w, flash: null };
          return {
            ...w,
            price: newPrice,
            delta: newDelta,
            flash: direction,
          };
        });
      });

      // Clear flash after 700ms
      setTimeout(() => {
        setWatchlist((prev) => prev.map((w) => ({ ...w, flash: null })));
      }, 700);
    }, 2800);

    return () => clearInterval(tickInterval);
  }, []);

  const handleOpenArch = () => {
    if (onOpenArchitecture) {
      onOpenArchitecture();
    } else {
      setIsArchOpen(true);
    }
  };

  return (
    <>
      <header className="h-14 border-b border-[#1a1a1a] bg-[#070707] px-4 sm:px-5 flex items-center justify-between sticky top-0 z-30 font-mono">
        {/* Watchlist Strip - Zerodha Marketwatch style with live ticks */}
        <div className="flex items-center gap-2 overflow-x-auto py-1 scrollbar-none">
          <span className="text-[11px] text-slate-500 mr-2 flex items-center gap-1 font-semibold uppercase tracking-wider shrink-0">
            <TrendingUp className="h-3.5 w-3.5 text-slate-400" /> Marketwatch
          </span>
          <div className="flex items-center gap-1.5">
            {watchlist.map((item) => {
              const isSelected = pathname === `/chart/${item.ticker}`;
              const isPositive = item.delta >= 0;
              return (
                <Link
                  key={item.ticker}
                  href={`/chart/${item.ticker}`}
                  className={`flex items-center gap-2 px-2.5 py-1 rounded-[4px] text-xs transition-all duration-300 border ${
                    item.flash === "up"
                      ? "bg-[#00B386]/25 border-[#00B386] text-white shadow-[0_0_8px_rgba(0,179,134,0.3)]"
                      : item.flash === "down"
                      ? "bg-[#EB5757]/25 border-[#EB5757] text-white shadow-[0_0_8px_rgba(235,87,87,0.3)]"
                      : isSelected
                      ? "bg-[#161616] text-white border-white/25 font-bold"
                      : "bg-[#0c0c0c] text-slate-300 border-[#1a1a1a] hover:bg-[#121212] hover:border-[#2a2a2a]"
                  }`}
                >
                  <span className="text-white font-semibold">{item.ticker}</span>
                  <span className="text-slate-300 tabular-nums">
                    {item.price.toFixed(2)}
                  </span>
                  <span
                    className={`text-[11px] font-medium tabular-nums ${
                      isPositive ? "text-[#00B386]" : "text-[#EB5757]"
                    }`}
                  >
                    {isPositive ? `+${item.delta.toFixed(2)}%` : `${item.delta.toFixed(2)}%`}
                  </span>
                </Link>
              );
            })}
          </div>
        </div>

        {/* Right Controls: Architecture Button, Paper Status, UTC clock */}
        <div className="flex items-center gap-2.5 shrink-0">
          {/* Architecture Modal Trigger */}
          <button
            onClick={handleOpenArch}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-[4px] bg-[#111111] hover:bg-[#181818] border border-[#222222] hover:border-[#333333] text-xs text-slate-300 hover:text-white transition-colors"
          >
            <Cpu className="h-3.5 w-3.5 text-sky-400" />
            <span className="hidden sm:inline">Pipeline Architecture</span>
            <span className="sm:hidden">Arch</span>
          </button>

          {/* Prominent Paper Trading Beacon */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-[4px] bg-[#121212] border border-[#252525] text-xs">
            <span className="h-2 w-2 rounded-full bg-[#E5A93B] animate-pulse" />
            <span className="text-slate-300 font-medium">ALPACA</span>
            <span className="text-[10px] text-[#E5A93B] font-bold tracking-wider">PAPER SIM</span>
          </div>

          {/* Live UTC Clock */}
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-[4px] bg-[#0c0c0c] border border-[#1a1a1a] text-xs text-slate-400">
            <Clock className="h-3 w-3 text-slate-500" />
            <span className="tabular-nums">{timeStr || "16:45:00 UTC"}</span>
          </div>
        </div>
      </header>

      {/* Embedded Global System Architecture Modal */}
      <SystemArchitectureModal isOpen={isArchOpen} onClose={() => setIsArchOpen(false)} />
    </>
  );
}
