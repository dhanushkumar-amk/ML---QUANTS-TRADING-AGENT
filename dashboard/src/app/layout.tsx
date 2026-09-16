import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "AGY QUANT | Live High-Frequency Trading Terminal",
  description: "Institutional quantitative portfolio analytics, TradingView charting, and forensic audit trail.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-[#0b0e14] text-[#f1f5f9] min-h-screen flex antialiased selection:bg-emerald-500/20 selection:text-emerald-300 relative overflow-x-hidden`}>
        {/* Subtle Ambient Background Gradient */}
        <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
          <div className="absolute top-0 left-1/4 w-[700px] h-[350px] bg-emerald-500/[0.03] rounded-full blur-[140px]" />
          <div className="absolute top-1/3 right-10 w-[550px] h-[350px] bg-blue-500/[0.025] rounded-full blur-[140px]" />
        </div>

        {/* Left Navigation Sidebar */}
        <Sidebar />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col min-w-0 z-10">
          <Header />
          <main className="flex-1 p-6 lg:p-8 overflow-y-auto max-w-[1700px] w-full mx-auto">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
