import { useEffect, useMemo, useState } from "react";
import { ArrowRight, Bot, CheckCircle2, Circle, Loader2, MessageSquare, Play, ShieldCheck, XCircle } from "lucide-react";
import { api } from "../services/api";
import { useApi, useAction, useEventSource } from "../hooks/useApi";
import { Badge, Button, Card, Empty, ErrorBox, Field, Input, Loading, Notice, PageHeader, RiskBadge, StatusBadge, Table, cx, fmt } from "../components/ui";

const PIPELINE = ["data_monitoring", "predictive_maintenance", "risk_assessment", "digital_twin",
  "counterfactual_reasoning", "quantum_optimization", "decision_coordination", "human_approval"];
const LABEL = { data_monitoring: "Data Monitoring", predictive_maintenance: "Predictive Maintenance", risk_assessment: "Risk Assessment",
  digital_twin: "Digital Twin", counterfactual_reasoning: "Counterfactual Reasoning", quantum_optimization: "Quantum Optimization",
  decision_coordination: "Decision Coordination", human_approval: "Human Approval" };

function agentStatus(events, finalStatus) {
  const st = {};
  events.forEach((e) => {
    if (e.type === "agent_start") st[e.agent] = { s: "running" };
    if (e.type === "agent_end") st[e.agent] = { s: e.status, summary: e.summary || e.error, seconds: e.seconds };
    if (e.type === "approval_required") st.human_approval = { s: "waiting" };
  });
  if (["completed", "approved", "rejected", "failed"].includes(finalStatus)) {
    PIPELINE.forEach((a) => { if (!st[a]) st[a] = { s: "skipped" }; });
  }
  return st;
}

function PipelineView({ status, routes }) {
  const icon = { running: <Loader2 className="h-4 w-4 animate-spin text-indigo-600" />, completed: <CheckCircle2 className="h-4 w-4 text-emerald-600" />,
    failed: <XCircle className="h-4 w-4 text-rose-600" />, waiting: <ShieldCheck className="h-4 w-4 text-amber-600" />,
    skipped: <Circle className="h-4 w-4 text-slate-300" /> };
  return (
    <div>
      <div className="flex flex-wrap items-stretch gap-1.5">
        {PIPELINE.map((a, i) => {
          const s = status[a]?.s;
          return (
            <div key={a} className="flex items-center gap-1.5">
              <div className={cx("w-40 rounded-lg border p-2 text-xs", s === "running" ? "border-indigo-400 bg-indigo-50 ring-2 ring-indigo-200" :
                s === "completed" ? "border-emerald-200 bg-emerald-50/50" : s === "failed" ? "border-rose-300 bg-rose-50" :
                  s === "waiting" ? "border-amber-300 bg-amber-50" : s === "skipped" ? "border-dashed border-slate-200 bg-white opacity-60" : "border-slate-200 bg-white")}>
                <div className="flex items-center gap-1.5 font-medium text-slate-800">{icon[s] || <Circle className="h-4 w-4 text-slate-300" />}{LABEL[a]}</div>
                <div className="mt-1 line-clamp-3 text-[11px] text-slate-500">{status[a]?.summary || (s === "skipped" ? "not needed (routing)" : s === "waiting" ? "awaiting engineer decision" : "")}</div>
                {status[a]?.seconds !== undefined && <div className="mt-0.5 text-[10px] text-slate-400">{status[a].seconds}s</div>}
              </div>
              {i < PIPELINE.length - 1 && <ArrowRight className="h-4 w-4 flex-none text-slate-300" />}
            </div>
          );
        })}
      </div>
      {routes.length > 0 && <div className="mt-2 space-y-0.5 text-xs text-slate-500">{routes.map((r) => <div key={r.from}>↳ <b>{LABEL[r.from]}</b> routed to <b>{LABEL[r.to]}</b>: {r.reason}</div>)}</div>}
    </div>
  );
}

function EventLog({ events }) {
  const [open, setOpen] = useState(null);
  const shown = events.filter((e) => e.type !== "agent_start");
  if (!shown.length) return <Empty>Events appear here while the workflow runs.</Empty>;
  return (
    <div className="max-h-[420px] space-y-1 overflow-y-auto pr-1 font-mono text-xs">
      {shown.map((e) => (
        <div key={e.seq} className="rounded border border-slate-100 bg-slate-50 px-2 py-1">
          <button type="button" className="flex w-full items-start gap-2 text-left" onClick={() => setOpen(open === e.seq ? null : e.seq)}>
            <span className="text-slate-400">{new Date(e.time * 1000).toLocaleTimeString()}</span>
            <span className={cx("font-semibold", e.type.startsWith("tool") ? "text-sky-700" : e.type === "route" ? "text-violet-700" : e.type.includes("approval") ? "text-amber-700" : "text-slate-700")}>{e.type}</span>
            <span className="min-w-0 flex-1 truncate text-slate-600">
              {e.type === "tool_call" && `${e.tool}(${Object.entries(e.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})`}
              {e.type === "tool_result" && `${e.tool} → ${e.seconds}s`}
              {e.type === "tool_error" && `${e.tool}: ${e.error}`}
              {e.type === "agent_end" && `${LABEL[e.agent]}: ${e.status} — ${e.summary || e.error || ""}`}
              {e.type === "route" && `${LABEL[e.from]} → ${LABEL[e.to]} (${e.reason})`}
              {e.type === "run_status" && e.status}
              {e.type === "approval_required" && `${e.request?.actions?.length} action(s) need approval`}
              {e.type === "approval_received" && `${e.approved ? "approved" : "rejected"} by ${e.reviewer}`}
            </span>
          </button>
          {open === e.seq && e.result && <pre className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap rounded bg-white p-2 text-[11px] text-slate-700">{JSON.stringify(e.result, null, 1)}</pre>}
        </div>
      ))}
    </div>
  );
}

function ApprovalPanel({ run, onDone }) {
  const pending = (run.state?.decision?.actions || []).filter((a) => a.requires_approval);
  const [sel, setSel] = useState(() => new Set(pending.map((a) => a.asset_id)));
  const [reviewer, setReviewer] = useState("");
  const [comment, setComment] = useState("");
  const act = useAction((approved) => api.approve(run.id, { approved, reviewer, comment, asset_ids: [...sel] }));
  return (
    <Card title="Human approval required" subtitle={`Maintenance actions at or above the "${run.state?.decision?.approval_gate}" risk gate need an engineer's decision`} className="border-amber-300">
      <div className="space-y-1">
        {pending.map((a) => (
          <label key={a.asset_id} className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={sel.has(a.asset_id)} onChange={(e) => { const n = new Set(sel); e.target.checked ? n.add(a.asset_id) : n.delete(a.asset_id); setSel(n); }} />
            <span className="font-medium">{a.asset_id}</span><RiskBadge level={a.risk_level} />
            <span className="text-slate-600">{a.action.replace(/_/g, " ")}{a.scheduled_in !== undefined && a.scheduled_in !== null ? ` at t+${a.scheduled_in}` : ""} · basis: {a.basis}</span>
          </label>
        ))}
      </div>
      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        <Input placeholder="Reviewer name (required)" value={reviewer} onChange={(e) => setReviewer(e.target.value)} />
        <div className="sm:col-span-2"><Input placeholder="Comment (optional)" value={comment} onChange={(e) => setComment(e.target.value)} /></div>
      </div>
      <div className="mt-3 flex gap-2">
        <Button variant="success" disabled={!reviewer || !sel.size} loading={act.loading} onClick={async () => { if (await act.run(true)) onDone(); }}>Approve selected ({sel.size})</Button>
        <Button variant="danger" disabled={!reviewer} loading={act.loading} onClick={async () => { if (await act.run(false)) onDone(); }}>Reject all</Button>
      </div>
      <ErrorBox error={act.error} />
    </Card>
  );
}

function RunResults({ run }) {
  const s = run.state || {};
  const d = s.decision;
  const cur = "USD";
  return (
    <div className="space-y-4">
      {s.errors?.length > 0 && <Notice tone="warn">{s.errors.map((e) => <div key={e.agent + e.error}><b>{LABEL[e.agent] || e.agent}</b>: {e.error}</div>)}</Notice>}
      {d && (
        <Card title="Final maintenance recommendation" actions={<><Badge tone={d.narrative_source === "llm" ? "violet" : "slate"}>narrative: {d.narrative_source}</Badge><StatusBadge status={run.status} /></>}>
          <p className="text-sm font-medium text-slate-800">{d.narrative.summary}</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">{d.narrative.rationale.map((r) => <li key={r}>{r}</li>)}</ul>
          <ul className="mt-2 space-y-0.5 text-xs text-slate-500">{d.narrative.caveats.map((c) => <li key={c}>⚠ {c}</li>)}</ul>
          {d.llm && !d.llm.used && <div className="mt-2 text-xs text-slate-500">LLM not used: {d.llm.reason}{d.llm.ungrounded_numbers ? ` (unsupported numbers: ${d.llm.ungrounded_numbers.join(", ")})` : ""}</div>}
          {d.llm?.used && <div className="mt-2 text-xs text-slate-500">LLM {d.llm.status?.model} · {d.llm.latency_seconds}s · every number checked against tool outputs</div>}
          <div className="mt-3"><Table rows={d.actions} rowKey={(r) => r.asset_id} columns={[
            { key: "asset_id", label: "Asset" }, { key: "risk_level", label: "Risk", render: (r) => <RiskBadge level={r.risk_level} /> },
            { key: "pof", label: "PoF", align: "right", render: (r) => fmt.pct(r.probability_of_failure) },
            { key: "risk_score", label: "Risk score", align: "right", render: (r) => fmt.int(r.risk_score) },
            { key: "action", label: "Action", render: (r) => r.action.replace(/_/g, " ") },
            { key: "when", label: "When", render: (r) => (r.scheduled_in === undefined || r.scheduled_in === null ? "—" : `t+${r.scheduled_in}`) },
            { key: "basis", label: "Basis" }, { key: "approval", label: "Approval", render: (r) => (r.requires_approval ? "required" : "—") },
          ]} /></div>
          {run.approval && <div className="mt-2 text-sm text-slate-600">{run.approval.approved ? "Approved" : "Rejected"} by <b>{run.approval.reviewer}</b>{run.approval.comment ? ` — “${run.approval.comment}”` : ""}; {run.approval.work_orders.length} work order(s) created.</div>}
        </Card>
      )}
      <div className="grid gap-4 xl:grid-cols-2">
        {s.risk && <Card title="Risk assessment" subtitle="Risk Assessment agent · assess_risk tool">
          <Table rows={Object.values(s.risk)} rowKey={(r) => r.asset_id} columns={[
            { key: "asset_id", label: "Asset" }, { key: "rul_pred", label: "RUL", align: "right", render: (r) => fmt.n(r.rul_pred) },
            { key: "pof", label: "PoF", align: "right", render: (r) => fmt.pct(r.probability_of_failure) },
            { key: "risk_score", label: "Risk", align: "right", render: (r) => fmt.int(r.risk_score) },
            { key: "risk_level", label: "Level", render: (r) => <RiskBadge level={r.risk_level} /> }, { key: "urgency", label: "Urgency" }]} />
        </Card>}
        {s.counterfactual && <Card title="Counterfactual comparison" subtitle="Counterfactual Reasoning agent · best scenario per asset (simulated)">
          <Table rows={Object.entries(s.counterfactual).map(([id, r]) => ({ id, ...r }))} rowKey={(r) => r.id} columns={[
            { key: "id", label: "Asset" }, { key: "best", label: "Best scenario", render: (r) => r.best?.replace(/_/g, " ") || r.error },
            { key: "cost", label: "Expected cost", align: "right", render: (r) => fmt.int(r.scenarios?.find((x) => x.scenario === r.best)?.total_expected_cost) },
            { key: "cont", label: "If continued", align: "right", render: (r) => fmt.int(r.scenarios?.find((x) => x.scenario === "continue_operation")?.total_expected_cost) }]} />
        </Card>}
      </div>
      {s.optimization?.results && <Card title="Optimised schedule" subtitle={`Quantum Optimization agent · ${s.optimization.qubo_variables} QUBO variables`}>
        <Table rows={s.optimization.results} rowKey={(r) => r.solver} columns={[
          { key: "solver", label: "Solver" }, { key: "total_expected_cost", label: `Expected cost (${cur})`, align: "right", render: (r) => fmt.int(r.total_expected_cost) },
          { key: "gap_percent", label: "Gap to optimum", align: "right", render: (r) => (r.gap_percent === undefined ? "—" : `${fmt.n(r.gap_percent, 2)}%`) },
          { key: "feasible", label: "Feasible", render: (r) => (r.feasible ? "yes" : r.error ? "error" : "no") },
          { key: "runtime_seconds", label: "Runtime", align: "right", render: (r) => (r.runtime_seconds === undefined ? "—" : `${fmt.n(r.runtime_seconds, 3)} s`) },
          { key: "qubits", label: "Qubits", align: "right", render: (r) => r.qubits ?? "—" }]} />
        <div className="mt-2 text-xs text-slate-500">{s.optimization.note}</div>
      </Card>}
    </div>
  );
}

function AskPanel({ llm }) {
  const [q, setQ] = useState("What is the risk level of FD001-test-unit34 and which maintenance scenario is cheapest?");
  const ask = useAction(() => api.ask(q));
  const r = ask.result;
  return (
    <Card title="Ask the coordinator" subtitle="LangChain tool-calling agent over read-only tools; numbers are checked against tool results">
      {!llm?.available && <Notice tone="warn">No LLM available ({llm?.detail}). Free-form questions need Ollama or an API provider.</Notice>}
      <div className="mt-2 flex gap-2"><Input value={q} onChange={(e) => setQ(e.target.value)} /><Button icon={MessageSquare} loading={ask.loading} onClick={ask.run} disabled={q.length < 3}>Ask</Button></div>
      {ask.loading && <div className="mt-2 text-xs text-slate-500">The local LLM can take a minute or two…</div>}
      <ErrorBox error={ask.error} />
      {r && !r.available && <div className="mt-2"><Notice>{r.message}</Notice></div>}
      {r?.available && (
        <div className="mt-3 space-y-2 text-sm">
          <div className="whitespace-pre-wrap rounded-lg bg-slate-50 p-3 text-slate-800">{r.answer}</div>
          <div><div className="text-xs font-semibold text-slate-600">Verified from tool results</div>
            <ul className="mt-1 list-disc pl-5 text-slate-700">{r.verified_facts.map((f) => <li key={f}>{f}</li>)}</ul></div>
          <div className="text-xs text-slate-500">Tools called: {r.tool_calls.map((t) => t.tool).join(", ") || "none"} · {r.seconds}s
            {r.ungrounded_numbers.length > 0 && <span className="ml-2 font-medium text-rose-700">⚠ numbers not found in tool output: {r.ungrounded_numbers.join(", ")}</span>}</div>
        </div>
      )}
    </Card>
  );
}

export default function AgentControlCenter() {
  const specs = useApi(api.agentSpecs, []);
  const runs = useApi(api.runs, []);
  const [runId, setRunId] = useState(null);
  const [events, setEvents] = useState([]);
  const [form, setForm] = useState({ n_assets: 5, asset_ids: "", horizon: 60, optimize: true, use_llm: true, force_full_analysis: false });
  const run = useApi(() => api.run(runId), [runId], { enabled: !!runId });
  const start = useAction(() => api.startRun({
    n_assets: Number(form.n_assets), horizon: Number(form.horizon), optimize: form.optimize, use_llm: form.use_llm,
    force_full_analysis: form.force_full_analysis, asset_ids: form.asset_ids.trim() ? form.asset_ids.split(/[\s,]+/).filter(Boolean) : null,
  }));
  useEffect(() => { setEvents([]); }, [runId]);
  useEventSource(runId ? api.runEventsUrl(runId) : null, (ev) => {
    setEvents((list) => (list.some((e) => e.seq === ev.seq) ? list : [...list, ev]));
    if (ev.type === "run_status" || ev.type === "approval_required") { run.reload(); runs.reload(); }
  }, [runId]);
  const status = useMemo(() => agentStatus(events, run.data?.status), [events, run.data?.status]);
  const routes = events.filter((e) => e.type === "route");
  const active = PIPELINE.find((a) => status[a]?.s === "running");
  const traceStatus = useMemo(() => {   // for runs whose live events are gone (e.g. after a restart)
    if (events.length || !run.data?.state?.trace) return null;
    const st = {};
    run.data.state.trace.forEach((t) => { st[t.agent] = { s: t.status, summary: t.summary, seconds: t.seconds }; });
    if (run.data.status === "awaiting_approval") st.human_approval = { s: "waiting" };
    PIPELINE.forEach((a) => { if (!st[a]) st[a] = { s: "skipped" }; });
    return st;
  }, [events.length, run.data]);
  return (
    <>
      <PageHeader title="Agentic AI Control Center"
        description="LangGraph orchestrates seven agents over a shared state. Agents call LangChain tools backed by the real models; the LLM only writes explanations, and consequential maintenance needs human approval." />
      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="Start a workflow run" subtitle="Default scope: the highest-risk FD001 test engines">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Number of assets"><Input type="number" min={1} max={12} value={form.n_assets} onChange={(e) => setForm({ ...form, n_assets: e.target.value })} /></Field>
            <Field label="Planning horizon"><Input type="number" min={1} max={300} value={form.horizon} onChange={(e) => setForm({ ...form, horizon: e.target.value })} /></Field>
            <div className="col-span-2"><Field label="Or explicit asset ids" hint="e.g. FD001-test-unit34, IMS-1st_test-bearing3"><Input value={form.asset_ids} onChange={(e) => setForm({ ...form, asset_ids: e.target.value })} /></Field></div>
          </div>
          <div className="mt-3 space-y-1 text-sm">
            {[["optimize", "Joint QAOA scheduling when ≥2 assets need maintenance"], ["use_llm", "Use LLM for the narrative (falls back if unavailable)"], ["force_full_analysis", "Force twin & counterfactuals even for low risk"]].map(([k, l]) => (
              <label key={k} className="flex items-center gap-2"><input type="checkbox" checked={form[k]} onChange={(e) => setForm({ ...form, [k]: e.target.checked })} />{l}</label>))}
          </div>
          <div className="mt-3"><Button icon={Play} loading={start.loading} onClick={async () => { const r = await start.run(); if (r) { setRunId(r.run_id); runs.reload(); } }}>Run agents</Button></div>
          <ErrorBox error={start.error} />
        </Card>
        <Card title="Workflow" className="xl:col-span-2" subtitle={runId ? `Run ${runId}${active ? ` · active agent: ${LABEL[active]}` : ""}` : "Select or start a run"}
          actions={run.data && <StatusBadge status={run.data.status} />}>
          <PipelineView status={traceStatus || status} routes={routes.length ? routes : (run.data?.state?.route_log || [])} />
        </Card>
      </div>
      {run.data?.status === "awaiting_approval" && <div className="mt-4"><ApprovalPanel key={run.data.id} run={run.data} onDone={() => { run.reload(); runs.reload(); }} /></div>}
      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <div className="min-w-0 xl:col-span-2">{run.loading && !run.data ? <Loading /> : run.data ? <RunResults run={run.data} /> : <Empty>No run selected.</Empty>}</div>
        <Card title="Live events" subtitle="Agent steps, tool calls with results, routing (Server-Sent Events)"><EventLog events={events} /></Card>
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="Run history" actions={<Button variant="secondary" onClick={runs.reload}>Refresh</Button>}>
          <Table rows={runs.data} rowKey={(r) => r.id} onRowClick={(r) => setRunId(r.id)} selectedKey={runId} maxHeight={300} columns={[
            { key: "id", label: "Run", render: (r) => <span className="font-mono text-xs">{r.id}</span> }, { key: "created_at", label: "Started", render: (r) => fmt.time(r.created_at) },
            { key: "assets", label: "Assets", render: (r) => r.request?.asset_ids?.length }, { key: "status", label: "Status", render: (r) => <StatusBadge status={r.status} /> }]} />
        </Card>
        <AskPanel llm={specs.data?.llm} />
      </div>
      <Card title="Agent definitions" className="mt-4" subtitle="Inputs, outputs, tools, execution conditions and error handling of each LangGraph node">
        {specs.loading && <Loading />}
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {specs.data?.agents.map((a) => (
            <div key={a.key} className="rounded-lg border border-slate-200 p-3 text-xs">
              <div className="flex items-center gap-1.5 text-sm font-semibold text-slate-800"><Bot className="h-4 w-4 text-indigo-500" />{a.name}</div>
              <ul className="mt-1 list-disc pl-4 text-slate-600">{a.responsibilities.map((r) => <li key={r}>{r}</li>)}</ul>
              <div className="mt-1.5"><b>In:</b> {a.inputs.join(", ")} · <b>Out:</b> {a.outputs.join(", ")}</div>
              <div><b>Tools:</b> {a.tools.join(", ") || "—"}{a.uses_llm && " · LLM (narrative only)"}</div>
              <div><b>Runs when:</b> {a.runs_when}</div>
              <div className="text-slate-500"><b>On error:</b> {a.on_error}</div>
            </div>
          ))}
        </div>
      </Card>
    </>
  );
}
