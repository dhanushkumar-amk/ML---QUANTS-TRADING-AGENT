"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sun, Moon, Clock, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";

const POPULAR_TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "GOOGL"];

export function Header() {
  const pathname = usePathname();
  const [isDark, setIsDark] = React.useState(true);
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

  const toggleTheme = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle("light");
  };

  return (
    <header className="h-16 border-b border-border/70 bg-card/40 backdrop-blur px-6 flex items-center justify-between sticky top-0 z-20">
      {/* Ticker Quick Switcher */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground mr-1 flex items-center gap-1">
          <TrendingUp className="h-3.5 w-3.5" /> Watchlist:
        </span>
        {POPULAR_TICKERS.map((t) => {
          const isSelected = pathname === `/chart/${t}`;
          return (
            <Link
              key={t}
              href={`/chart/${t}`}
              className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                isSelected
                  ? "bg-emerald-500/20 text-emerald-400 font-semibold border border-emerald-500/40"
                  : "bg-muted/40 text-muted-foreground hover:bg-muted/70 hover:text-foreground"
              }`}
            >
              {t}
            </Link>
          );
        })}
      </div>

      {/* Right Controls: Market Status, UTC clock, Theme toggle */}
      <div className="flex items-center gap-4">
        {/* Live NYSE Status Badge */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping" />
          <span>NYSE LIVE (PAPER)</span>
        </div>

        {/* UTC Clock */}
        <div className="hidden sm:flex items-center gap-1 text-xs text-muted-foreground font-mono">
          <Clock className="h-3.5 w-3.5" />
          <span>{timeStr || "15:21:32 UTC"}</span>
        </div>

        {/* Theme Toggle */}
        <Button
          variant="outline"
          size="icon"
          onClick={toggleTheme}
          title="Toggle light/dark theme"
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
        >
          {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </div>
    </header>
  );
}
