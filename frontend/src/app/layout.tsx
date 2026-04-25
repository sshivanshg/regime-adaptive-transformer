import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RAMT Live",
  description: "Monthly momentum strategy runner and research backtests.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-zinc-50 text-zinc-950">
        <header className="sticky top-0 z-10 border-b border-zinc-200 bg-white/80 backdrop-blur">
          <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
            <div className="flex items-center gap-3">
              <div className="h-7 w-7 rounded-md border border-zinc-200 bg-white" />
              <span className="text-sm font-semibold tracking-tight">RAMT Live</span>
            </div>
            <nav className="flex items-center gap-6 text-sm text-zinc-700">
              <Link className="hover:text-zinc-950" href="/">
                Dashboard
              </Link>
              <Link className="hover:text-zinc-950" href="/live">
                Live
              </Link>
              <Link className="hover:text-zinc-950" href="/backtest">
                Backtest
              </Link>
              <Link className="hover:text-zinc-950" href="/runs">
                Runs
              </Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">
          {children}
        </main>
        <footer className="border-t border-zinc-200 bg-white">
          <div className="mx-auto max-w-6xl px-4 py-6 text-xs text-zinc-500">
            Local runner UI. Strategy metrics are computed from your most recent succeeded
            monthly run.
          </div>
        </footer>
      </body>
    </html>
  );
}
