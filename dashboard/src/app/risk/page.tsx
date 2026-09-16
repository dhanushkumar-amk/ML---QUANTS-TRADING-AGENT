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
import { formatCurrency } from "@/lib/utils";
import {
  ShieldAlert,
  Search,
  FileText,
  Activity,
  Sliders,
  CheckCircle2,
  X,
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
            <Skeleton key={i} className="h-36 w-full rounded-[4px] bg-[#141414]" />
          ))}
        </div>
        <Skeleton className="h-96 w-full rounded-[4px] bg-[#141414]" />
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
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#1F1F1F] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold tracking-tight text-[#FAFAFA]">
              Risk Engine Gating & Forensic Audit Trail
            </h1>
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-[4px] bg-[#141414] text-[#888888] border border-[#1F1F1F]">
              Phase 39 / 47
            </span>
          </div>
          <p className="text-xs text-[#666666] mt-0.5">
            Real-time pre-trade safety controls, dynamic circuit breakers, and deterministic immutable JSONL event records.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-[4px] bg-[#0C0C0C] border border-[#1F1F1F]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#00B386]" />
            <span className="text-[#888888] text-[11px]">CIRCUIT BREAKER:</span>
            <span className="text-[#00B386] font-bold text-[11px]">{riskData.circuit_breaker_status}</span>
          </div>
        </div>
      </div>

      {/* 2. Real-Time Risk Gating Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Drawdown Kill-Switch Gauge */}
        <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
          <CardHeader className="p-4 pb-2 border-b border-[#141414]">
            <div className="flex items-center justify-between">
              <CardTitle className="text-[11px] uppercase tracking-wider text-[#666666] font-mono flex items-center gap-1.5">
                <ShieldAlert className="h-3.5 w-3.5 text-[#00B386]" />
                Drawdown Kill-Switch
              </CardTitle>
              <Badge variant="profit" className="text-[10px] font-mono rounded-[4px]">
                {riskData.circuit_breaker_status}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="p-4 space-y-3">
            <div className="flex items-baseline justify-between font-mono">
              <span className="text-2xl font-bold text-[#FAFAFA] tabular-nums">
                {riskData.current_drawdown_pct.toFixed(2)}%
              </span>
              <span className="text-xs text-[#666666] tabular-nums">
                Limit: {riskData.max_drawdown_limit_pct.toFixed(1)}%
              </span>
            </div>
            <Progress
              value={riskData.current_drawdown_pct}
              max={riskData.max_drawdown_limit_pct}
              className="h-1.5 bg-[#1A1A1A] rounded-[2px]"
              indicatorClassName={
                riskData.current_drawdown_pct > 10
                  ? "bg-[#EB5757]"
                  : riskData.current_drawdown_pct > 5
                  ? "bg-[#E5A93B]"
                  : "bg-[#00B386]"
              }
            />
            <div className="flex items-center justify-between text-[11px] text-[#666666] font-mono">
              <span>Remaining Headroom:</span>
              <span className="text-[#00B386] font-bold tabular-nums">
                {riskData.drawdown_headroom_pct.toFixed(2)}%
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Gross Exposure Utilization */}
        <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
          <CardHeader className="p-4 pb-2 border-b border-[#141414]">
            <div className="flex items-center justify-between">
              <CardTitle className="text-[11px] uppercase tracking-wider text-[#666666] font-mono flex items-center gap-1.5">
                <Sliders className="h-3.5 w-3.5 text-[#FAFAFA]" />
                Gross Portfolio Exposure
              </CardTitle>
              <Badge variant="outline" className="text-[10px] font-mono rounded-[4px] border-[#1F1F1F] text-[#888888]">
                CAP 100%
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="p-4 space-y-3">
            <div className="flex items-baseline justify-between font-mono">
              <span className="text-2xl font-bold text-[#FAFAFA] tabular-nums">
                {riskData.gross_exposure_pct.toFixed(1)}%
              </span>
              <span className="text-xs text-[#666666] tabular-nums">
                {formatCurrency(riskData.gross_exposure_value)}
              </span>
            </div>
            <Progress
              value={riskData.gross_exposure_pct}
              max={100}
              className="h-1.5 bg-[#1A1A1A] rounded-[2px]"
              indicatorClassName="bg-[#FAFAFA]"
            />
            <div className="flex items-center justify-between text-[11px] text-[#666666] font-mono">
              <span>Cash Cushion:</span>
              <span className="text-[#CCCCCC] tabular-nums">
                {(100 - riskData.gross_exposure_pct).toFixed(1)}% Available
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Concentration & Ledger Status */}
        <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
          <CardHeader className="p-4 pb-2 border-b border-[#141414]">
            <div className="flex items-center justify-between">
              <CardTitle className="text-[11px] uppercase tracking-wider text-[#666666] font-mono flex items-center gap-1.5">
                <CheckCircle2 className="h-3.5 w-3.5 text-[#00B386]" />
                Reconciliation & Bounds
              </CardTitle>
              <Badge variant="profit" className="text-[10px] font-mono rounded-[4px]">
                {riskData.reconciliation_status}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="p-4 space-y-2 text-xs font-mono">
            <div className="flex justify-between py-1 border-b border-[#141414]">
              <span className="text-[#777777]">Single-Asset Cap</span>
              <span className="text-[#FAFAFA] font-bold tabular-nums">
                {riskData.max_position_size_pct.toFixed(1)}%
              </span>
            </div>
            {riskData.positions.map((p) => (
              <div key={p.ticker} className="flex justify-between py-0.5">
                <span className="text-[#777777]">{p.ticker} Weight</span>
                <span className="text-[#00B386] tabular-nums">
                  {p.weight_pct.toFixed(1)}% / {p.limit_pct.toFixed(0)}%
                </span>
              </div>
            ))}
            <div className="flex justify-between pt-1 text-[11px] text-[#555555]">
              <span>Last Reconciled:</span>
              <span>{riskData.last_reconciliation_time.slice(11)} UTC</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 3. Forensic Audit Trail Table */}
      <Card className="bg-[#0C0C0C] border-[#1F1F1F] rounded-[4px] shadow-none">
        <CardHeader className="p-4 pb-3 border-b border-[#141414]">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <CardTitle className="text-xs font-semibold text-[#FAFAFA] flex items-center gap-2">
                <Activity className="h-3.5 w-3.5 text-[#888888]" />
                Structured Audit Trail (Phase 47)
              </CardTitle>
              <CardDescription className="text-[11px] text-[#666666]">
                Immutable cryptographic JSONL execution log answering: &ldquo;Why did the bot make decision X at timestamp T?&rdquo;
              </CardDescription>
            </div>

            {/* Filter Controls */}
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="h-3.5 w-3.5 absolute left-2.5 top-2.5 text-[#555555]" />
                <input
                  type="text"
                  placeholder="Filter ticker / cycle..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-8 pl-8 pr-3 rounded-[4px] bg-[#080808] border border-[#1F1F1F] text-xs text-[#FAFAFA] placeholder:text-[#555555] focus:outline-none focus:border-[#00B386] w-48 font-mono"
                />
              </div>

              {/* Event Type Filter Dropdown */}
              <select
                value={selectedEventType}
                onChange={(e) => setSelectedEventType(e.target.value)}
                className="h-8 px-2.5 rounded-[4px] bg-[#080808] border border-[#1F1F1F] text-xs text-[#FAFAFA] font-mono focus:outline-none focus:border-[#00B386]"
              >
                {EVENT_TYPE_OPTIONS.map((opt) => (
                  <option key={opt} value={opt} className="bg-[#0C0C0C] text-[#FAFAFA]">
                    {opt}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-b border-[#141414] hover:bg-transparent">
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9">Timestamp (UTC)</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9">Event Type</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9">Asset</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9">Cycle ID</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9">Event Context / Details</TableHead>
                  <TableHead className="text-[11px] font-mono uppercase text-[#666666] h-9 text-right">Inspection</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredEvents.map((ev, idx) => (
                  <TableRow
                    key={idx}
                    className="font-mono text-xs border-b border-[#141414] hover:bg-[#111111] transition-colors"
                  >
                    <TableCell className="text-[#666666] text-[11px] py-2.5 tabular-nums">
                      {ev.timestamp.slice(0, 19).replace("T", " ")}
                    </TableCell>
                    <TableCell className="py-2.5">
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded-[2px] font-mono ${
                          ev.event_type.includes("SIGNAL")
                            ? "bg-[#00B386]/10 text-[#00B386] border border-[#00B386]/30"
                            : ev.event_type.includes("RISK")
                            ? "bg-[#1A1A1A] text-[#CCCCCC] border border-[#2A2A2A]"
                            : ev.event_type.includes("CANCEL")
                            ? "bg-[#EB5757]/10 text-[#EB5757] border border-[#EB5757]/30"
                            : "bg-[#141414] text-[#888888] border border-[#1F1F1F]"
                        }`}
                      >
                        {ev.event_type}
                      </span>
                    </TableCell>
                    <TableCell className="font-bold text-[#FAFAFA] py-2.5">
                      {ev.ticker || "—"}
                    </TableCell>
                    <TableCell className="text-[#666666] text-[11px] py-2.5">
                      {ev.cycle_id}
                    </TableCell>
                    <TableCell className="text-[#999999] max-w-md truncate text-[11px] py-2.5">
                      {JSON.stringify(ev.details)}
                    </TableCell>
                    <TableCell className="text-right py-2.5">
                      <button
                        onClick={() => setSelectedEvent(ev)}
                        className="px-2 py-1 rounded-[3px] bg-[#141414] hover:bg-[#1F1F1F] text-[#888888] hover:text-[#FAFAFA] text-[10px] font-mono border border-[#1F1F1F] transition-colors"
                      >
                        Raw JSON
                      </button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Detail Inspection Modal for forensic drill-down */}
      {selectedEvent && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[#0C0C0C] border border-[#1F1F1F] rounded-[4px] max-w-xl w-full p-5 space-y-4 shadow-2xl animate-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between border-b border-[#141414] pb-3">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-[#00B386]" />
                <h3 className="font-semibold text-xs text-[#FAFAFA]">
                  Forensic Event Inspection
                </h3>
              </div>
              <button
                onClick={() => setSelectedEvent(null)}
                className="text-[#666666] hover:text-[#FAFAFA] p-1 rounded-[3px] hover:bg-[#141414] transition-colors"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between py-1 border-b border-[#141414]">
                <span className="text-[#666666]">Event Type:</span>
                <span className="text-[#00B386] font-bold">{selectedEvent.event_type}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[#141414]">
                <span className="text-[#666666]">Timestamp:</span>
                <span className="text-[#FAFAFA] tabular-nums">{selectedEvent.timestamp}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[#141414]">
                <span className="text-[#666666]">Cycle ID:</span>
                <span className="text-[#CCCCCC]">{selectedEvent.cycle_id}</span>
              </div>
              {selectedEvent.ticker && (
                <div className="flex justify-between py-1 border-b border-[#141414]">
                  <span className="text-[#666666]">Ticker:</span>
                  <span className="text-[#FAFAFA] font-bold">{selectedEvent.ticker}</span>
                </div>
              )}
            </div>

            <div>
              <span className="text-[10px] text-[#666666] uppercase font-mono block mb-1.5">
                Full Immutable JSON Payload:
              </span>
              <pre className="p-3 rounded-[3px] bg-[#050505] border border-[#141414] text-[#CCCCCC] font-mono text-[11px] overflow-auto max-h-64 leading-relaxed">
                {JSON.stringify(selectedEvent, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
