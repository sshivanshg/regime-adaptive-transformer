import { API_BASE } from "@/lib/api";
import Link from "next/link";

type RunRow = {
  id: string;
  type: string;
  status: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error_message?: string | null;
  params?: Record<string, unknown>;
};

async function getRuns(): Promise<{ runs: RunRow[] }> {
  const res = await fetch(`${API_BASE}/results/history?limit=100`, {
    cache: "no-store",
  });
  if (!res.ok) return { runs: [] };
  return (await res.json()) as { runs: RunRow[] };
}

function Badge({ status }: { status: string }) {
  const cls =
    status === "succeeded"
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : status === "failed"
        ? "bg-red-50 text-red-700 border-red-200"
        : "bg-zinc-50 text-zinc-700 border-zinc-200";
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs ${cls}`}>
      {status}
    </span>
  );
}

export default async function RunsPage() {
  const data = await getRuns();

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Runs</h1>
        <p className="text-sm text-zinc-600">Audit trail of all jobs triggered from the UI.</p>
      </div>

      <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white">
        <table className="w-full text-sm">
          <thead className="bg-zinc-50 text-xs uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="px-4 py-3 text-left">Run</th>
              <th className="px-4 py-3 text-left">Type</th>
              <th className="px-4 py-3 text-left">Status</th>
              <th className="px-4 py-3 text-left">Finished</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100">
            {data.runs.map((r) => (
              <tr key={r.id} className="hover:bg-zinc-50">
                <td className="px-4 py-3 font-mono text-xs">
                  <Link className="underline" href={`/runs/${r.id}`}>
                    {r.id}
                  </Link>
                </td>
                <td className="px-4 py-3">{r.type}</td>
                <td className="px-4 py-3">
                  <Badge status={r.status} />
                </td>
                <td className="px-4 py-3 text-zinc-600">
                  {r.finished_at ? new Date(r.finished_at).toLocaleString() : "—"}
                </td>
              </tr>
            ))}
            {!data.runs.length ? (
              <tr>
                <td className="px-4 py-6 text-zinc-600" colSpan={4}>
                  No runs yet. Go to{" "}
                  <Link className="underline" href="/live">
                    Live
                  </Link>{" "}
                  to start.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}

