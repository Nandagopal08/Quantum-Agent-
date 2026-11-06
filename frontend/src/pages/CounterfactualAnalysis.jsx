import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { GitCompare } from "lucide-react";
import { api } from "../services/api";
import { useAction } from "../hooks/useApi";
import { Badge, Button, Card, ErrorBox, Field, Input, Notice, PageHeader, RiskBadge, SimulatedTag, Stat, Table, fmt } from "../components/ui";
import { ChartBox, PALETTE, tooltipStyle } from "../components/charts";

export default function CounterfactualAnalysis() {
  const [params] = useSearchParams();
  const [form, setForm] = useState({ asset_id: params.get("asset") || "FD001-test-unit34", horizon: 60, delay: "", reduced_load: 0.7, as_of: "" });
  const cf = useAction(() => api.counterfactual({
    asset_id: form.asset_id.trim(), horizon: Number(form.horizon) || null, delay: form.delay === "" ? null : Number(form.delay),
    reduced_load: Number(form.reduced_load), as_of: form.as_of === "" ? null : Number(form.as_of),
  }));
  const r = cf.result;
  const costData = r?.scenarios.map((s) => ({ name: s.label.replace(/ \(.*\)/, ""), maintenance: s.maintenance_cost, downtime: s.downtime_cost,
    "expected failure": s.expected_failure_cost, "lost output": s.production_loss }));
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  return (
    <>
      <PageHeader title="Counterfactual Analysis" description="What happens under each maintenance decision? Five scenarios are simulated from the model's RUL prediction and uncertainty with the configured cost assumptions." />
      <Card title="Scenario inputs">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
          <div className="col-span-2"><Field label="Asset id"><Input value={form.asset_id} onChange={set("asset_id")} /></Field></div>
          <Field label="Planning horizon"><Input type="number" value={form.horizon} onChange={set("horizon")} /></Field>
          <Field label="Delay (blank = auto)"><Input type="number" value={form.delay} onChange={set("delay")} /></Field>
          <Field label="Reduced load"><Input type="number" step="0.05" min={0.1} max={0.95} value={form.reduced_load} onChange={set("reduced_load")} /></Field>
          <Field label="Bearing snapshot (as of)" hint="bearings only"><Input type="number" value={form.as_of} onChange={set("as_of")} /></Field>
        </div>
        <div className="mt-3"><Button icon={GitCompare} loading={cf.loading} onClick={cf.run}>Compare scenarios</Button></div>
        <div className="mt-2"><ErrorBox error={cf.error} /></div>
      </Card>
      {r && (
        <div className="mt-4 space-y-4">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="Predicted RUL" value={`${fmt.n(r.inputs.rul_pred)} ${r.time_unit}s`} hint={`σ ${fmt.n(r.inputs.sigma)} · model ${r.prediction.model}`} />
            <Stat label="Current risk" value={<RiskBadge level={r.risk.risk_level} />} hint={`PoF ${fmt.pct(r.risk.probability_of_failure)}`} />
            <Stat label="Consequence of failure" value={fmt.money(r.consequence_of_failure)} />
            <Stat label="Best scenario" value={r.scenarios.find((s) => s.scenario === r.best).label} tone="good" />
          </div>
          <Card title="Scenario comparison" actions={<SimulatedTag />} subtitle={`${r.inputs.samples.toLocaleString()} Monte Carlo samples · horizon ${r.horizon} ${r.time_unit}s · seed ${r.inputs.seed}`}>
            <Table rows={[...r.scenarios].sort((a, b) => a.rank - b.rank)} rowKey={(s) => s.scenario} columns={[
              { key: "rank", label: "#", render: (s) => (s.rank === 1 ? <Badge tone="green">1</Badge> : s.rank) },
              { key: "label", label: "Scenario" },
              { key: "failure_probability", label: "Failure risk", align: "right", render: (s) => fmt.pct(s.failure_probability) },
              { key: "expected_rul_end_of_horizon", label: "Expected RUL left", align: "right", render: (s) => fmt.n(s.expected_rul_end_of_horizon) },
              { key: "maintenance_cost", label: "Maintenance", align: "right", render: (s) => fmt.int(s.maintenance_cost) },
              { key: "downtime_cost", label: "Downtime", align: "right", render: (s) => fmt.int(s.downtime_cost) },
              { key: "expected_failure_cost", label: "Expected failure", align: "right", render: (s) => fmt.int(s.expected_failure_cost) },
              { key: "production_loss", label: "Lost output", align: "right", render: (s) => fmt.int(s.production_loss) },
              { key: "total_expected_cost", label: "Total expected cost", align: "right", render: (s) => <b>{fmt.int(s.total_expected_cost)}</b> },
            ]} />
          </Card>
          <div className="grid gap-4 xl:grid-cols-2">
            <Card title="Cost breakdown by scenario" actions={<SimulatedTag />}>
              <ChartBox height={280}>
                <BarChart data={costData}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="name" interval={0} tick={{ fontSize: 10 }} /><YAxis /><Tooltip {...tooltipStyle} /><Legend />
                  {["maintenance", "downtime", "expected failure", "lost output"].map((k, i) => <Bar key={k} dataKey={k} stackId="c" fill={PALETTE[i]} />)}
                </BarChart>
              </ChartBox>
            </Card>
            <Card title="Expected cost vs maintenance timing" actions={<SimulatedTag />} subtitle="Used to find the optimal timing scenario">
              <ChartBox height={280}>
                <LineChart data={r.timing_curve}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="maintenance_at" label={{ value: `maintenance at (${r.time_unit})`, position: "insideBottom", offset: -2, fontSize: 11 }} />
                  <YAxis yAxisId="c" /><YAxis yAxisId="p" orientation="right" domain={[0, 1]} /><Tooltip {...tooltipStyle} /><Legend />
                  <Line yAxisId="c" dataKey="total_expected_cost" name="total expected cost" stroke={PALETTE[0]} dot={false} />
                  <Line yAxisId="p" dataKey="failure_probability" name="failure probability" stroke={PALETTE[4]} dot={false} strokeDasharray="4 3" />
                </LineChart>
              </ChartBox>
            </Card>
          </div>
          <Card title="Why this scenario?"><ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">{r.explanation.map((l) => <li key={l}>{l}</li>)}</ul></Card>
          <Notice><pre className="whitespace-pre-wrap font-sans text-xs">{r.assumptions}</pre></Notice>
        </div>
      )}
    </>
  );
}
