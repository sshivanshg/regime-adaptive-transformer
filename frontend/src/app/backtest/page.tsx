"use client";

import { useState } from "react";
import { apiPost } from "@/lib/api";
import Link from "next/link";

export default function BacktestPage() {
  const [year, setYear] = useState<number>(2024);
  const [jobId, setJobId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setErr(null);
    setJobId(null);
    try {
      const res = await apiPost<{ job_id: string }>("/jobs/backtest", { year });
      setJobId(res.job_id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Backtest research</h1>
        <p className="text-sm text-zinc-600">
          Run a backtest for a single calendar year. Results are stored under <span className="font-mono">results/runs/&lt;run_id&gt;</span>.
        </p>
      </div>

      <div className="rounded-xl border border-zinc-200 bg-white p-4">
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-xs font-medium uppercase tracking-wide text-zinc-500">
              Year
            </span>
            <input
              className="w-40 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm"
              type="number"
              value={year}
              onChange={(e) => setYear(Number(e.target.value))}
              min={1990}
              max={2100}
            />
          </label>
          <button
            className="rounded-xl bg-zinc-950 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800"
            onClick={() => void run()}
          >
            Run backtest
          </button>
        </div>

        {err ? (
          <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-800">
            {err}
          </div>
        ) : null}

        {jobId ? (
          <div className="mt-4 text-sm text-zinc-700">
            Started run:{" "}
            <Link className="font-mono text-xs underline" href={`/runs/${jobId}`}>
              {jobId}
            </Link>
          </div>
        ) : null}
      </div>
    </div>
  );
}

