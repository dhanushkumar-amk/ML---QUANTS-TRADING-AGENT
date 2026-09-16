import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

describe("Table and Badge Components", () => {
  const sampleTrades = [
    { id: "1", ticker: "AAPL", side: "BUY", qty: 94.68, price: 178.45 },
    { id: "2", ticker: "MSFT", side: "SELL", qty: 50.0, price: 412.3 },
  ];

  it("renders table rows and badges accurately", () => {
    render(
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Ticker</TableHead>
            <TableHead>Side</TableHead>
            <TableHead>Quantity</TableHead>
            <TableHead>Price</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sampleTrades.map((t) => (
            <TableRow key={t.id}>
              <TableCell>{t.ticker}</TableCell>
              <TableCell>
                <Badge variant={t.side === "BUY" ? "profit" : "loss"}>
                  {t.side}
                </Badge>
              </TableCell>
              <TableCell>{t.qty}</TableCell>
              <TableCell>${t.price}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );

    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("MSFT")).toBeInTheDocument();
    expect(screen.getByText("BUY")).toBeInTheDocument();
    expect(screen.getByText("SELL")).toBeInTheDocument();
    expect(screen.getByText("94.68")).toBeInTheDocument();
  });
});
