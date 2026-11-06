import { Bot } from "lucide-react";
import { api } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Badge, Loading, Notice } from "../components/ui";
import MissingDataAgent from "../components/MissingDataAgent";

/** Agent-only page: just the missing-data agent (served alone by scripts/run_backend.py --agent-only). */
export default function AgentOnly() {
  const status = useApi(() => api.qseqAgentStatus(), []);
  const health = useApi(() => api.health(), []);
  const s = status.data;
  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3">
          <Bot className="h-6 w-6 text-indigo-600" />
          <div className="min-w-0">
            <h1 className="text-lg font-semibold text-slate-900">Missing-Data Maintenance Agent</h1>
            <p className="text-xs text-slate-500">Imputes missing sensor values and forecasts remaining useful life with quantum sequence models (NASA C-MAPSS FD001)</p>
          </div>
          {health.data?.storage?.mode === "development" && <Badge tone="amber">development storage</Badge>}
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 pb-10">
        {status.loading ? <div className="mt-4"><Loading label="Checking the agent's models…" /></div>
          : status.error ? <div className="mt-4"><Notice tone="warn">{status.error}</Notice></div>
            : s && !s.ready ? (
              <div className="mt-4"><Notice tone="warn">
                <b>The agent is not ready yet.</b> It needs the trained quantum sequence models and their benchmark. {s.hint}
                <div className="mt-1 text-xs">Benchmark report: {s.benchmark_report ? "found" : "missing"} · RUL models: {s.rul_models.length ? s.rul_models.join(", ") : "none"}</div>
              </Notice></div>
            ) : s && (
              <div className="mt-4 text-xs text-slate-500">Ready · imputers: {s.imputers.join(", ")} · RUL models: {s.rul_models.join(", ")}</div>
            )}
        <MissingDataAgent />
      </main>
    </div>
  );
}
