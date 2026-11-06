import { useState } from "react";
import { ArrowRight, CheckCircle2, Circle, Loader2, Play, RotateCcw, XCircle } from "lucide-react";
import { api } from "../services/api";
import { useAction, useApi, useEventSource } from "../hooks/useApi";
import { Badge, Button, Card, ErrorBox, Field, Input, Notice, Select, StatusBadge, Table, cx, fmt } from "./ui";

// The single missing-data agent: run form, live step log, result and past runs.
// Used by the Quantum Sequence Lab page and by the agent-only page (run_backend.py --agent-only).

export const STEPS = ["data_quality", "plan", "impute", "check", "forecast", "report"];
const LABEL = { data_quality: "Data quality", plan: "Plan", impute: "Impute", check: "Check", forecast: "Forecast", report: "Report" };
export const MODEL_LABEL = { q_lstm: "Quantum LSTM", q_transformer: "Quantum Transformer", q_cnn_lstm: "Quantum CNN-LSTM",
  c_lstm: "LSTM (classical twin)", c_transformer: "Transformer (classical twin)", c_cnn_lstm: "CNN-LSTM (classical twin)",
  mean: "Mean fill", locf: "Last obs. carried forward", linear: "Linear interpolation" };
const DONE = ["completed", "failed", "insufficient_data", "completed_without_forecast"];
const IMPUTERS = ["mean", "locf", "linear", "q_lstm", "q_transformer", "q_cnn_lstm", "c_lstm", "c_transformer", "c_cnn_lstm"];

/** One agent, six steps. Live from "agent_step" events; after the run, from the saved step log. */
export function AgentFlow({ events, state }) {
  const log = state?.steps?.length ? state.steps
    : events.filter((e) => e.type === "agent_step").reduce((acc, e) => {
      if (e.status === "running") return [...acc, { step: e.step, status: "running" }];
      const k = acc.map((x) => x.step).lastIndexOf(e.step);
      return k < 0 ? [...acc, e] : acc.map((x, n) => (n === k ? { ...x, ...e } : x));
    }, []);
  const failed = state && state.trace?.[0]?.status === "failed" ? state.trace[0].summary : null;
  const icon = { running: <Loader2 className="h-4 w-4 animate-spin text-indigo-600" />, completed: <CheckCircle2 className="h-4 w-4 text-emerald-600" />,
    failed: <XCircle className="h-4 w-4 text-rose-600" /> };
  const visits = log.reduce((m, x) => ({ ...m, [x.step]: (m[x.step] || 0) + 1 }), {});
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-1.5 text-[11px] text-slate-500">
        <Badge tone="violet">Missing-Data Maintenance Agent</Badge>
        {STEPS.map((st, i) => (
          <span key={st} className="flex items-center gap-1.5">
            <span className={cx("rounded px-1.5 py-0.5", visits[st] ? "bg-emerald-50 text-emerald-700" : "bg-slate-100")}>
              {LABEL[st]}{visits[st] > 1 ? ` ×${visits[st]}` : ""}</span>
            {i < STEPS.length - 1 && <ArrowRight className="h-3 w-3 text-slate-300" />}
          </span>
        ))}
      </div>
      <ol className="space-y-1 text-xs">
        {log.map((x, i) => (
          <li key={i} className="flex items-start gap-2 rounded border border-slate-100 bg-slate-50 px-2 py-1">
            {icon[x.status] || <Circle className="h-4 w-4 text-slate-300" />}
            <span className="w-24 flex-none font-medium text-slate-700">{i + 1}. {LABEL[x.step] || x.step}</span>
            <span className="min-w-0 flex-1 text-slate-600">{x.summary || ""}</span>
          </li>
        ))}
      </ol>
      {failed && <div className="mt-2"><ErrorBox error={failed} /></div>}
    </div>
  );
}

export function RunResult({ record }) {
  const s = record?.state;
  if (!s) return null;
  const rep = s.report || {};
  const ev = rep.evaluation || {};
  const f = s.forecast || {};
  return (
    <div className="mt-4 space-y-3">
      <Notice tone={record.status === "completed" ? "ok" : "warn"}>{rep.text}</Notice>
      <div className="grid gap-3 md:grid-cols-2">
        <Table rows={s.attempts || []} rowKey={(a) => a.method} empty="No imputation attempts" columns={[
          { key: "method", label: "Attempt", render: (a) => MODEL_LABEL[a.method] || a.method },
          { key: "exp", label: "Benchmark RMSE", align: "right", render: (a) => fmt.n(a.expected_rmse, 3) },
          { key: "h", label: "Self-check RMSE", align: "right", render: (a) => fmt.n(a.holdout_rmse, 3) },
          { key: "lin", label: "Linear (same check)", align: "right", render: (a) => fmt.n(a.linear_holdout_rmse, 3) },
        ]} />
        <Table rows={[f.prediction, f.classical_twin].filter(Boolean)} rowKey={(p) => p.model} empty="No forecast" columns={[
          { key: "model", label: "RUL model", render: (p) => MODEL_LABEL[p.model] || p.model },
          { key: "rul", label: "RUL (cycles)", align: "right", render: (p) => fmt.n(p.rul_pred, 1) },
          { key: "int", label: "90% interval", align: "right", render: (p) => `${fmt.n(p.interval_90[0], 0)}–${fmt.n(p.interval_90[1], 0)}` },
        ]} />
      </div>
      {Object.keys(ev).length > 0 && (
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
          <div className="mb-1 font-semibold text-slate-700">Scoring against the hidden true values <Badge tone="violet">simulated missingness</Badge></div>
          Imputation RMSE {fmt.n(ev.imputation_rmse_true, 3)} (linear interpolation {fmt.n(ev.linear_rmse_true, 3)}) ·
          true RUL {fmt.n(ev.rul_true, 0)} · forecast error {fmt.n(ev.rul_abs_error, 1)} cycles ·
          forecast with complete data {fmt.n(ev.rul_pred_complete_data, 1)}. <span className="text-slate-400">{ev.note}</span>
        </div>
      )}
    </div>
  );
}


/** Agent form + live view + result + recent runs. */
export default function MissingDataAgent({ showRuns = true }) {
  const runs = useApi(() => api.qseqRuns(), []);
  const [form, setForm] = useState({ unit: 34, missing_rate: 0.3, pattern: "mcar", seed: 0, imputer: "", use_llm: true });
  const [runId, setRunId] = useState(null);
  const [events, setEvents] = useState([]);
  const [record, setRecord] = useState(null);
  const start = useAction(() => api.qseqStartRun({
    asset_id: `FD001-test-unit${Number(form.unit)}`, missing_rate: Number(form.missing_rate), pattern: form.pattern,
    seed: Number(form.seed), imputer: form.imputer || null, use_llm: form.use_llm,
  }));
  useEventSource(runId && !record ? api.qseqRunEventsUrl(runId) : null, async (ev) => {
    setEvents((xs) => [...xs, ev]);
    if (ev.type === "run_status" && DONE.includes(ev.status)) {
      setRecord(await api.qseqRun(runId));
      runs.reload();
    }
  }, [runId]);
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value });
  return (
    <>
      <Card title="Missing-data agent" className="mt-4"
        subtitle="One agent: data quality → plan → impute → check (retries the next-best imputer on rejection) → forecast → report">
        <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Field label="Test engine (FD001)"><Input type="number" min={1} max={100} value={form.unit} onChange={set("unit")} /></Field>
          <Field label="Missing rate"><Input type="number" step="0.05" min={0} max={0.95} value={form.missing_rate} onChange={set("missing_rate")} /></Field>
          <Field label="Pattern"><Select value={form.pattern} onChange={set("pattern")} options={[{ value: "mcar", label: "Random (MCAR)" }, { value: "block", label: "Sensor outages (block)" }]} /></Field>
          <Field label="Seed"><Input type="number" value={form.seed} onChange={set("seed")} /></Field>
          <Field label="Force first imputer" hint="the check may reject it"><Select value={form.imputer} onChange={set("imputer")} options={[{ value: "", label: "Agent decides" },
            ...IMPUTERS.map((m) => ({ value: m, label: MODEL_LABEL[m] }))]} /></Field>
          <Field label="LLM narrative"><label className="flex h-9 items-center gap-2 text-sm"><input type="checkbox" checked={form.use_llm} onChange={set("use_llm")} /> use if available</label></Field>
        </div>
        <div className="mt-3 flex gap-2">
          <Button icon={Play} loading={start.loading || (runId && !record)} onClick={async () => {
            setEvents([]); setRecord(null);
            const res = await start.run();
            if (res) setRunId(res.run_id);
          }}>Run agent</Button>
        </div>
        <div className="mt-3"><ErrorBox error={start.error} /></div>
        {runId && <div className="mt-4"><AgentFlow events={events} state={record?.state} /></div>}
        <RunResult record={record} />
        {record && ["failed", "interrupted"].includes(record.status) && <ErrorBox error={record.error || record.state?.trace?.[0]?.summary} />}
      </Card>
      {showRuns && (
        <Card title="Recent agent runs" className="mt-4" subtitle="Click a run to see its steps and result" actions={<Button variant="ghost" icon={RotateCcw} onClick={runs.reload}>Refresh</Button>}>
          {runs.error ? <ErrorBox error={runs.error} code={runs.code} /> : (
            <Table rows={runs.data || []} rowKey={(x) => x.run_id} maxHeight={320} empty="No runs yet"
              onRowClick={async (x) => { setRunId(x.run_id); setEvents([]); setRecord(await api.qseqRun(x.run_id)); }} columns={[
                { key: "t", label: "Started", render: (x) => fmt.time(x.created_at) },
                { key: "a", label: "Engine", render: (x) => x.request?.asset_id || "uploaded window" },
                { key: "m", label: "Missing", align: "right", render: (x) => (x.request?.values ? "uploaded" : `${fmt.pct(x.request?.missing_rate, 0)} ${x.request?.pattern}`) },
                { key: "i", label: "Imputer", render: (x) => x.summary?.imputer || "—" },
                { key: "rul", label: "RUL", align: "right", render: (x) => fmt.n(x.summary?.rul_pred, 1) },
                { key: "s", label: "Status", render: (x) => <StatusBadge status={x.status} /> },
              ]} />
          )}
        </Card>
      )}
    </>
  );
}
