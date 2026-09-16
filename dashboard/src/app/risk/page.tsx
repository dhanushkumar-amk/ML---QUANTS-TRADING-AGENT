"use client";

import * as React from "react";
import { getRisk, getAudit } from "@/lib/api-client";
import { RiskStatus, AuditResponse, AuditEventRecord } from "@/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { formatCurrency, formatPercent } from "@/lib/utils";
import {
  ShieldAlert,
  ShieldCheck,
  Search,
  Filter,
  FileText,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";

const EVENT_TYPE_OPTIONS = [
  "ALL",
  "SIGNAL_GENERATED",
  "RISK_DECISION",
  "ORDER_SUBMITTED",
  "ORDER_CANCELLED",
  "RECONCILIATION",
  "HEARTBEAT",
  "SYSTEM_EVENT",
];

export default function RiskAndAuditView() {
  const [riskData, setRiskData] = React.useState<RiskStatus | null>(null);
  const [auditData, setAuditData] = React.useState<AuditResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [selectedEventType, setSelectedEventType] = React.useState("ALL");
  const [searchQuery, setSearchQuery] = React.useState("");
  const [selectedEvent, setSelectedEvent] = React.useState<AuditEventRecord | null>(null);

  React.useEffect(() => {
    async function load() {
      try {
        const [r, a] = await Promise.all([
          getRisk(),
          getAudit(selectedEventType === "ALL" ? undefined : selectedEventType),
        ]);
        setRiskData(r);
        setAuditData(a);
      } catch (err) {
        console.error("Failed to load risk and audit data:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [selectedEventType]);

  if (loading || !riskData || !auditData) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-32 w-full rounded-lg" />
          ))}
        </div>
        <Skeleton className="h-96 w-full rounded-lg" />
      </div>
    );
  }

  // Filter audit events by search query (symbol or cycle ID)
  const filteredEvents = auditData.events.filter((ev) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      ev.ticker?.toLowerCase().includes(q) ||
      ev.cycle_id?.toLowerCase().includes(q) ||
      ev.event_type?.toLowerCase().includes(q)
    );
  });

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* 1. Page Header */}
      <div>
        <h1 className="text-xl font-bold tracking-tight text-foreground flex items-center gap-2">
          <ShieldAlert className="h-5 w-5 text-emerald-400" />
          Risk Engine Gating & Forensic Audit Trail
        </h1>
        <p className="text-xs text-muted-foreground mt-0.5">
          Real-time safety bounds (Phase 39) & centralized immutable event logging (Phase 47).
        </p>
      </div>

      {/* 2. Real-Time Risk Gating Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Drawdown Kill-Switch Gauge */}
        <Card className="border-border/80">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-xs uppercase text-muted-foreground">
                Drawdown Kill-Switch
              </CardTitle>
              <Badge variant="profit" className="text-[10px] font-mono">
                {riskData.circuit_breaker_status}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-baseline justify-between font-mono">
              <span className="text-2xl font-bold text-foreground">
                {riskData.current_drawdown_pct.toFixed(2)}%
              </span>
              <span className="text-xs text-muted-foreground">
                Max Limit: {riskData.max_drawdown_limit_pct.toFixed(1)}%
              </span>
            </div>
            <Progress
              value={riskData.current_drawdown_pct}
              max={riskData.max_drawdown_limit_pct}
              indicatorClassName={
                riskData.current_drawdown_pct > 10
                  ? "bg-rose-500"
                  : riskData.current_drawdown_pct > 5
                  ? "bg-amber-500"
                  : "bg-emerald-500"
              }
            />
            <div className="flex items-center justify-between text-[11px] text-muted-foreground">
              <span>Remaining Headroom:</span>
              <span className="font-mono text-emerald-400 font-bold">
                {riskData.drawdown_headroom_pct.toFixed(2)}%
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Gross Exposure Utilization */}
        <Card className="border-border/80">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-xs uppercase text-muted-foreground">
                Gross Portfolio Exposure
              </CardTitle>
              <Badge variant="secondary" className="text-[10px] font-mono">
                CAP: 100%
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-baseline justify-between font-mono">
              <span className="text-2xl font-bold text-foreground">
                {riskData.gross_exposure_pct.toFixed(1)}%
              </span>
              <span className="text-xs text-muted-foreground">
                {formatCurrency(riskData.gross_exposure_value)}
              </span>
            </div>
            <Progress value={riskData.gross_exposure_pct} max={100} />
            <div className="flex items-center justify-between text-[11px] text-muted-foreground">
              <span>Cash Reserve:</span>
              <span className="font-mono text-slate-300">
                {(100 - riskData.gross_exposure_pct).toFixed(1)}% Available
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Concentration & Ledger Status */}
        <Card className="border-border/80">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-xs uppercase text-muted-foreground">
                Reconciliation & Bounds
              </CardTitle>
              <Badge variant="profit" className="text-[10px] font-mono">
                {riskData.reconciliation_status}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-border/40">
              <span className="text-muted-foreground">Single-Asset Cap</span>
              <span className="text-foreground font-bold">
                {riskData.max_position_size_pct.toFixed(1)}%
              </span>
            </div>
            {riskData.positions.map((p) => (
              <div key={p.ticker} className="flex justify-between py-0.5">
                <span className="text-muted-foreground">{p.ticker} Allocation</span>
                <span className="text-emerald-400">
                  {p.weight_pct.toFixed(1)}% / {p.limit_pct.toFixed(0)}%
                </span>
              </div>
            ))}
            <div className="flex justify-between pt-1 text-[11px] text-muted-foreground">
              <span>Last Reconciled:</span>
              <span>{riskData.last_reconciliation_time.slice(11)}</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 3. Forensic Audit Trail Table */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <CardTitle className="text-sm font-semibold text-foreground">
                Structured Audit Trail (Phase 47)
              </CardTitle>
              <CardDescription className="text-xs">
                Immutable JSONL execution record answering &ldquo;Why did the bot do X at time T?&rdquo;
              </CardDescription>
            </div>

            {/* Filter Controls */}
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="h-3.5 w-3.5 absolute left-2.5 top-2.5 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Filter ticker / cycle..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-8 pl-8 pr-3 rounded-md bg-muted/40 border border-border text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-emerald-500 w-48 font-mono"
                />
              </div>

              {/* Event Type Filter Dropdown */}
              <select
                value={selectedEventType}
                onChange={(e) => setSelectedEventType(e.target.value)}
                className="h-8 px-2 rounded-md bg-muted/40 border border-border text-xs text-foreground font-mono focus:outline-none focus:ring-1 focus:ring-emerald-500"
              >
                {EVENT_TYPE_OPTIONS.map((opt) => (
                  <option key={opt} value={opt} className="bg-slate-900 text-slate-200">
                    {opt}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Timestamp (UTC)</TableHead>
                <TableHead>Event Type</TableHead>
                <TableHead>Asset</TableHead>
                <TableHead>Cycle ID</TableHead>
                <TableHead>Event Summary</TableHead>
                <TableHead className="text-right">Inspection</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredEvents.map((ev, idx) => (
                <TableRow key={idx} className="font-mono text-xs">
                  <TableCell className="text-muted-foreground text-[11px]">
                    {ev.timestamp.slice(0, 19).replace("T", " ")}
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        ev.event_type.includes("SIGNAL")
                          ? "profit"
                          : ev.event_type.includes("RISK")
                          ? "secondary"
                          : ev.event_type.includes("CANCEL")
                          ? "loss"
                          : "neutral"
                      }
                      className="text-[10px]"
                    >
                      {ev.event_type}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-bold text-foreground">
                    {ev.ticker || "—"}
                  </TableCell>
                  <TableCell className="text-muted-foreground text-[11px]">
                    {ev.cycle_id}
                  </TableCell>
                  <TableCell className="text-slate-300 max-w-md truncate text-[11px]">
                    {JSON.stringify(ev.details)}
                  </TableCell>
                  <TableCell className="text-right">
                    <button
                      onClick={() => setSelectedEvent(ev)}
                      className="px-2 py-1 rounded bg-muted/40 hover:bg-muted/80 text-muted-foreground hover:text-foreground text-[10px] font-medium transition-colors"
                    >
                      Raw JSON
                    </button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Detail Inspection Modal for forensic drill-down */}
      {selectedEvent && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-card border border-border rounded-lg max-w-xl w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-border/60 pb-3">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-emerald-400" />
                <h3 className="font-bold text-sm text-foreground">
                  Forensic Event Inspection
                </h3>
              </div>
              <button
                onClick={() => setSelectedEvent(null)}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                ✕ Close
              </button>
            </div>

            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Event Type:</span>
                <span className="text-emerald-400">{selectedEvent.event_type}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Timestamp:</span>
                <span className="text-foreground">{selectedEvent.timestamp}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Cycle ID:</span>
                <span className="text-slate-300">{selectedEvent.cycle_id}</span>
              </div>
              {selectedEvent.ticker && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Ticker:</span>
                  <span className="text-foreground font-bold">{selectedEvent.ticker}</span>
                </div>
              )}
            </div>

            <div>
              <span className="text-[11px] text-muted-foreground uppercase font-mono block mb-1">
                Full JSON Payload:
              </span>
              <pre className="p-3 rounded bg-slate-950 border border-border/60 text-emerald-300 font-mono text-[11px] overflow-auto max-h-60">
                {JSON.stringify(selectedEvent, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
