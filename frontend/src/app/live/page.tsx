"use client";

import { useEffect, useMemo, useState } from "react";
import { apiGet, apiPost } from "@/lib/api";
import Link from "next/link";

type JobStatus = {
  id: string;
  type: string;
  status: "queued" | "running" | "succeeded" | "failed";
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  stdout_path?: string | null;
  stderr_path?: string | null;
  error_message?: string | null;
  artifacts?: Array<{ path: string; kind?: string | null; created_at: string }>;
};

export default function LivePage() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const busy = job?.status === "queued" || job?.status === "running";

  const pollMs = useMemo(() => (busy ? 1500 : 0), [busy]);

  async function startFetch() {
    setErr(null);
    setJob(null);
    const res = await apiPost<{ job_id: string }>("/jobs/fetch-latest", {});
    setJobId(res.job_id);
  }

  async function startMonthly() {
    setErr(null);
    setJob(null);
    const res = await apiPost<{ job_id: string }>("/jobs/run-monthly", {});
    setJobId(res.job_id);
  }

  useEffect(() => {
    if (!jobId) return;
    let timer: number | undefined;

    const tick = async () => {
      try {
        const r = await apiGet<JobStatus>(`/jobs/${jobId}`);
        setJob(r);
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      }
    };

    void tick();
    if (pollMs > 0) {
      timer = window.setInterval(() => void tick(), pollMs);
    }
    return () => {
      if (timer) window.clearInterval(timer);
    };
  }, [jobId, pollMs]);

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Live monthly run</h1>
        <p className="text-sm text-zinc-600">
          Fetch the latest prices, then run the monthly momentum strategy and persist a run log.
        </p>
      </div>

      <div className="flex flex-wrap gap-3">
        <button
          className="rounded-xl bg-zinc-950 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-50"
          onClick={() => void startFetch()}
          disabled={busy}
        >
          Fetch data to latest
        </button>
        <button
          className="rounded-xl border border-zinc-200 bg-white px-4 py-2 text-sm font-medium hover:bg-zinc-50 disabled:opacity-50"
          onClick={() => void startMonthly()}
          disabled={busy}
        >
          Run monthly rebalance
        </button>
      </div>

      {err ? (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          {err}
        </div>
      ) : null}

      {jobId ? (
        <div className="rounded-xl border border-zinc-200 bg-white p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-sm font-medium">Job</div>
            <Link className="text-xs text-zinc-600 underline" href={`/runs/${jobId}`}>
              Open run detail
            </Link>
          </div>
          <div className="mt-2 grid gap-2 text-sm sm:grid-cols-2">
            <div>
              <div className="text-xs uppercase tracking-wide text-zinc-500">Job ID</div>
              <div className="font-mono text-xs">{jobId}</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide text-zinc-500">Status</div>
              <div className="text-sm">{job?.status ?? "…"}</div>
            </div>
          </div>

          {job?.error_message ? (
            <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-800">
              {job.error_message}
            </div>
          ) : null}

          {job?.artifacts?.length ? (
            <div className="mt-4">
              <div className="text-xs uppercase tracking-wide text-zinc-500">Artifacts</div>
              <ul className="mt-2 space-y-1 text-xs text-zinc-700">
                {job.artifacts.map((a) => (
                  <li key={a.path} className="font-mono">
                    {a.path}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="rounded-xl border border-zinc-200 bg-white p-6 text-sm text-zinc-700">
          No job running. Start with “Fetch data to latest”, then “Run monthly rebalance”.
        </div>
      )}
    </div>
  );
}

