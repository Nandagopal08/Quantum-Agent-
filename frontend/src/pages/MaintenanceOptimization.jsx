import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { Network } from "lucide-react";
import { api } from "../services/api";
import { useApi, useAction, useJob } from "../hooks/useApi";
import { Badge, Button, Card, Empty, ErrorBox, Field, Input, JobPanel, Loading, Notice, PageHeader, Table, fmt } from "../components/ui";
import { ChartBox, PALETTE, tooltipStyle } from "../components/charts";

function Result({ r }) {
  const q = r.results.find((x) => x.solver === "qaoa");
  const best = r.results.find((x) => x.solver === r.best_solver);
  const slots = r.problem.slot_times;
  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="Solver comparison" className="xl:col-span-2" subtitle={`${r.problem.assets.length} assets · ${slots.length} slots at t = ${slots.join(", ")} · capacity ${r.problem.capacity}/slot · ${r.qubo_variables} QUBO variables`}>
          <Table rows={r.results} rowKey={(x) => x.solver} columns={[
            { key: "solver", label: "Solver", render: (x) => <span>{x.solver}{x.solver === r.best_solver && <Badge tone="green" className="ml-1">best</Badge>}{x.optimal && <Badge className="ml-1">exact</Badge>}</span> },
            { key: "total_expected_cost", label: "Total expected cost", align: "right", render: (x) => fmt.int(x.total_expected_cost) },
            { key: "gap", label: "Gap to optimum", align: "right", render: (x) => (x.gap_percent === undefined ? "—" : `${fmt.n(x.gap_percent, 2)}%`) },
            { key: "expected_failures", label: "Expected failures", align: "right", render: (x) => fmt.n(x.expected_failures, 3) },
            { key: "planned_downtime", label: "Planned downtime", align: "right", render: (x) => fmt.n(x.planned_downtime, 0) },
            { key: "feasible", label: "Feasible", render: (x) => (x.error ? <span className="text-rose-600">{x.error}</span> : x.feasible ? "yes" : "no") },
            { key: "runtime_seconds", label: "Runtime", align: "right", render: (x) => (x.runtime_seconds === undefined ? "—" : `${fmt.n(x.runtime_seconds, 3)} s`) },
          ]} />
          <div className="mt-2 text-xs text-slate-500">{r.note}</div>
        </Card>
        <Card title="Total expected cost">
          <ChartBox height={240}>
            <BarChart data={r.results.filter((x) => x.total_expected_cost !== undefined)}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="solver" tick={{ fontSize: 10 }} /><YAxis /><Tooltip {...tooltipStyle} />
              <Bar dataKey="total_expected_cost" radius={[3, 3, 0, 0]}>{r.results.map((x, i) => <Cell key={x.solver} fill={x.solver === "qaoa" ? PALETTE[5] : PALETTE[i % 3]} />)}</Bar>
            </BarChart>
          </ChartBox>
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title={`Maintenance plan (${r.best_solver})`} subtitle="Which asset is maintained in which slot">
          <Table rows={best.plan.map((p) => ({ ...p, idx: r.problem.assets.findIndex((a) => a.asset_id === p.asset_id) }))} rowKey={(p) => p.asset_id} columns={[
            { key: "asset_id", label: "Asset" },
            { key: "rul", label: "Pred. RUL", align: "right", render: (p) => fmt.n(r.problem.assets[p.idx].rul_pred) },
            { key: "pof_none", label: "PoF if not maintained", align: "right", render: (p) => fmt.pct(r.problem.pof_none[p.idx]) },
            ...slots.map((t, si) => ({ key: `s${si}`, label: `slot t=${t}`, render: (p) => (p.slot === si ? <Badge tone="blue">maintain</Badge> : "") })),
            { key: "none", label: "no maintenance", render: (p) => (p.maintain ? "" : <Badge>—</Badge>) },
          ]} />
        </Card>
        {q && !q.error && (
          <Card title="QAOA run" subtitle={`${q.qubits} qubits · p = ${q.layers} · depth ${q.circuit_depth} · ${q.gate_count} gates · ${q.device}`}>
            <div className="mb-2 grid grid-cols-3 gap-2 text-sm">
              <div><div className="text-xs text-slate-500">Evaluations</div><b>{q.function_evaluations}</b></div>
              <div><div className="text-xs text-slate-500">P(selected plan)</div><b>{fmt.pct(q.probability_of_selected, 3)}</b></div>
              <div><div className="text-xs text-slate-500">Feasible mass (top-k)</div><b>{fmt.pct(q.feasible_probability_top_k)}</b></div>
            </div>
            <ChartBox height={180}>
              <LineChart data={q.expectation_history.map((v, i) => ({ i, v }))}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="i" hide /><YAxis domain={["auto", "auto"]} /><Tooltip {...tooltipStyle} />
                <Line dataKey="v" name="⟨H_cost⟩ (normalised)" stroke={PALETTE[5]} dot={false} /></LineChart>
            </ChartBox>
            <div className="mt-1 text-xs text-slate-500">COBYLA optimisation of the cost-Hamiltonian expectation (linear-ramp start + one random restart).</div>
          </Card>
        )}
      </div>
    </div>
  );
}

export default function MaintenanceOptimization() {
  const [form, setForm] = useState({ n_assets: 5, asset_ids: "", qaoa_layers: 2, qaoa_steps: 80, crew_capacity_per_slot: 2, slots: 2 });
  const history = useApi(api.optimizationResults, []);
  const [sel, setSel] = useState(0);
  const [job, setJob] = useJob(() => { history.reload(); setSel(0); });
  const start = useAction(() => api.optimize({
    n_assets: Number(form.n_assets), qaoa_layers: Number(form.qaoa_layers), qaoa_steps: Number(form.qaoa_steps),
    crew_capacity_per_slot: Number(form.crew_capacity_per_slot), slots: Number(form.slots),
    asset_ids: form.asset_ids.trim() ? form.asset_ids.split(/[\s,]+/).filter(Boolean) : null,
  }));
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  const results = history.data || [];
  return (
    <>
      <PageHeader title="Maintenance Optimization" description="Joint maintenance scheduling as a QUBO (binary decisions, maintenance and failure costs, downtime, criticality, crew capacity) solved with QAOA and classical baselines." />
      <Notice tone="warn">QAOA runs on a statevector simulator. Results are reported exactly as measured against the exact MILP optimum; QAOA is not assumed to be better.</Notice>
      <Card title="Run optimisation" className="mt-4" subtitle="Assets default to the highest-risk FD001 engines">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
          <Field label="Assets"><Input type="number" min={2} max={8} value={form.n_assets} onChange={set("n_assets")} /></Field>
          <Field label="Slots"><Input type="number" min={1} max={3} value={form.slots} onChange={set("slots")} /></Field>
          <Field label="Crew capacity / slot"><Input type="number" min={1} max={4} value={form.crew_capacity_per_slot} onChange={set("crew_capacity_per_slot")} /></Field>
          <Field label="QAOA layers (p)"><Input type="number" min={1} max={6} value={form.qaoa_layers} onChange={set("qaoa_layers")} /></Field>
          <Field label="QAOA evaluations"><Input type="number" min={10} max={1000} value={form.qaoa_steps} onChange={set("qaoa_steps")} /></Field>
          <div className="col-span-2 md:col-span-6"><Field label="Explicit asset ids (optional)" hint="qubits = assets × slots + slack bits; capped at 16 — the highest-risk assets that fit are kept"><Input value={form.asset_ids} onChange={set("asset_ids")} /></Field></div>
        </div>
        <div className="mt-3"><Button icon={Network} loading={start.loading} onClick={async () => { const j = await start.run(); if (j) setJob(j); }}>Optimise schedule</Button></div>
        <div className="mt-3 space-y-2"><ErrorBox error={start.error} /><JobPanel job={job} /></div>
      </Card>
      <div className="mt-4">
        {history.loading && <Loading />}
        <ErrorBox error={history.error} code={history.code} />
        {results.length > 1 && (
          <div className="mb-3 flex flex-wrap gap-1.5 text-xs">{results.map((r, i) => (
            <button type="button" key={r.generated_at} onClick={() => setSel(i)} className={`rounded-full px-2.5 py-1 ring-1 ${i === sel ? "bg-indigo-600 text-white ring-indigo-600" : "bg-white ring-slate-300"}`}>{r.generated_at}</button>))}</div>
        )}
        {results[sel] ? <Result r={results[sel]} /> : !history.loading && <Empty>No optimisation results yet.</Empty>}
      </div>
    </>
  );
}
