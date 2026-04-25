import { API_BASE } from "@/lib/api";
import Link from "next/link";

type LatestResponse = {
  run_id: string | null;
  metrics:
    | null
    | {
        strategy_sharpe: number | null;
        nifty_sharpe: number | null;
        strategy_cagr: number | null;
        nifty_cagr: number | null;
        strategy_max_dd: number | null;
        nifty_max_dd: number | null;
        strategy_win_rate: number | null;
        last_rebalance_date: string | null;
      };
};

async function getLatest(): Promise<LatestResponse> {
  const res = await fetch(`${API_BASE}/results/latest`, { cache: "no-store" });
  if (!res.ok) return { run_id: null, metrics: null };
  return (await res.json()) as LatestResponse;
}

function MetricCard({
  label,
  value,
  delta,
}: {
  label: string;
  value: string;
  delta?: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">
        {label}
      </div>
      <div className="mt-2 text-2xl font-semibold tracking-tight">{value}</div>
      {delta ? <div className="mt-1 text-xs text-zinc-500">{delta}</div> : null}
    </div>
  );
}

export default async function Home() {
  const latest = await getLatest();
  const m = latest.metrics;

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
        <p className="text-sm text-zinc-600">
          Strategy vs NIFTY based on the most recent succeeded monthly run.
        </p>
      </div>

      {!m ? (
        <div className="rounded-xl border border-zinc-200 bg-white p-6 text-sm text-zinc-700">
          No monthly run metrics yet. Go to{" "}
          <Link className="underline" href="/live">
            Live
          </Link>{" "}
          and run “Fetch data to latest” and “Run monthly rebalance”.
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Sharpe (net)"
            value={`${(m.strategy_sharpe ?? 0).toFixed(2)}`}
            delta={`vs NIFTY: ${((m.strategy_sharpe ?? 0) - (m.nifty_sharpe ?? 0)).toFixed(2)}`}
          />
          <MetricCard
            label="CAGR"
            value={`${(100 * (m.strategy_cagr ?? 0)).toFixed(1)}%`}
            delta={`vs NIFTY: ${(100 * ((m.strategy_cagr ?? 0) - (m.nifty_cagr ?? 0))).toFixed(1)} pp`}
          />
          <MetricCard
            label="Max drawdown"
            value={`${(100 * (m.strategy_max_dd ?? 0)).toFixed(1)}%`}
            delta={`vs NIFTY: ${(100 * ((m.strategy_max_dd ?? 0) - (m.nifty_max_dd ?? 0))).toFixed(1)} pp`}
          />
          <MetricCard
            label="Win rate"
            value={`${(100 * (m.strategy_win_rate ?? 0)).toFixed(0)}%`}
            delta={m.last_rebalance_date ? `Last rebalance: ${m.last_rebalance_date}` : undefined}
          />
        </div>
      )}

      {latest.run_id ? (
        <div className="text-xs text-zinc-500">
          Latest run:{" "}
          <Link className="underline" href={`/runs/${latest.run_id}`}>
            {latest.run_id}
          </Link>
        </div>
      ) : null}
    </div>
  );
}
