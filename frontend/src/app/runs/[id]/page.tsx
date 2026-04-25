import { API_BASE } from "@/lib/api";

type Artifact = { path: string; kind?: string | null; created_at: string };

type RunDetail = {
  id: string;
  type: string;
  status: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  stdout_path?: string | null;
  stderr_path?: string | null;
  error_message?: string | null;
  params?: Record<string, unknown>;
  artifacts?: Artifact[];
};

async function getRun(id: string): Promise<RunDetail | null> {
  const res = await fetch(`${API_BASE}/jobs/${id}`, { cache: "no-store" });
  if (!res.ok) return null;
  return (await res.json()) as RunDetail;
}

async function getLogTail(id: string, stream: "stdout" | "stderr") {
  const res = await fetch(`${API_BASE}/jobs/${id}/logs?stream=${stream}&tail=200`, {
    cache: "no-store",
  });
  if (!res.ok) return "";
  return await res.text();
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-start justify-between gap-6 border-b border-zinc-100 py-2 text-sm">
      <div className="text-zinc-500">{k}</div>
      <div className="text-right font-mono text-xs text-zinc-800">{v}</div>
    </div>
  );
}

export default async function RunDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const r = await getRun(id);
  if (!r) {
    return (
      <div className="rounded-xl border border-zinc-200 bg-white p-6 text-sm">
        Run not found.
      </div>
    );
  }

  const [stdout, stderr] = await Promise.all([getLogTail(id, "stdout"), getLogTail(id, "stderr")]);

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Run detail</h1>
        <div className="font-mono text-xs text-zinc-600">{r.id}</div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-zinc-200 bg-white p-4">
          <div className="text-sm font-medium">Metadata</div>
          <div className="mt-3">
            <Row k="type" v={r.type} />
            <Row k="status" v={r.status} />
            <Row k="created_at" v={r.created_at ?? "—"} />
            <Row k="started_at" v={r.started_at ?? "—"} />
            <Row k="finished_at" v={r.finished_at ?? "—"} />
          </div>
          {r.error_message ? (
            <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-800">
              {r.error_message}
            </div>
          ) : null}
        </div>

        <div className="rounded-xl border border-zinc-200 bg-white p-4">
          <div className="text-sm font-medium">Artifacts</div>
          <div className="mt-3 space-y-2">
            {r.artifacts?.length ? (
              <ul className="space-y-1 text-xs text-zinc-700">
                {r.artifacts.map((a) => (
                  <li key={a.path} className="font-mono">
                    {a.path}
                  </li>
                ))}
              </ul>
            ) : (
              <div className="text-sm text-zinc-600">No artifacts recorded.</div>
            )}
          </div>
          <div className="mt-4 text-xs text-zinc-500">
            stdout: <span className="font-mono">{r.stdout_path ?? "—"}</span>
            <br />
            stderr: <span className="font-mono">{r.stderr_path ?? "—"}</span>
          </div>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-zinc-200 bg-white p-4">
          <div className="text-sm font-medium">stdout (tail)</div>
          <pre className="mt-3 max-h-[420px] overflow-auto rounded-lg bg-zinc-950 p-3 text-xs text-zinc-50">
            {stdout || "—"}
          </pre>
        </div>
        <div className="rounded-xl border border-zinc-200 bg-white p-4">
          <div className="text-sm font-medium">stderr (tail)</div>
          <pre className="mt-3 max-h-[420px] overflow-auto rounded-lg bg-zinc-950 p-3 text-xs text-zinc-50">
            {stderr || "—"}
          </pre>
        </div>
      </div>
    </div>
  );
}

