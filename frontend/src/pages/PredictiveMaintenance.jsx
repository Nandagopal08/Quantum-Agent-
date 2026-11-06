import { useState } from "react";
import { Link } from "react-router-dom";
import { CartesianGrid, Cell, ReferenceLine, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { api } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Card, ErrorBox, Field, Loading, Notice, PageHeader, RISK_COLORS, RiskBadge, Select, Stat, Table, fmt } from "../components/ui";
import { ChartBox, tooltipStyle } from "../components/charts";

function AssetDetail({ id }) {
  const p = useApi(() => api.prediction(id), [id], { enabled: !!id });
  if (!id) return <div className="text-sm text-slate-500">Select an asset to see its risk drivers.</div>;
  if (p.loading) return <Loading />;
  if (p.error) return <ErrorBox error={p.error} code={p.code} />;
  const r = p.data.risk;
  return (
    <div className="space-y-3 text-sm">
      <div className="flex items-center justify-between"><span className="font-medium">{id}</span><RiskBadge level={r.risk_level} /></div>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Probability of failure" value={fmt.pct(r.probability_of_failure)} hint={`within ${r.horizon} ${p.data.time_unit}s`} />
        <Stat label="Consequence of failure" value={fmt.int(r.consequence_of_failure)} hint="USD (configurable)" />
        <Stat label="Risk score = PoF × CoF" value={fmt.int(r.risk_score)} tone={r.risk_level === "low" ? "good" : "bad"} />
        <Stat label="Health score" value={fmt.n(r.health_score, 0)} hint={`urgency: ${r.urgency}`} />
      </div>
      <ul className="list-disc space-y-1 pl-5 text-slate-600">{r.drivers.map((d) => <li key={d}>{d}</li>)}</ul>
      <div className="text-xs text-slate-500">Model: {p.data.model} · method: {r.method}{r.requires_approval && " · maintenance needs human approval"}</div>
      <div className="flex gap-3 text-xs font-medium"><Link className="text-indigo-600" to={`/twin?asset=${id}`}>Open digital twin →</Link>
        <Link className="text-indigo-600" to={`/counterfactual?asset=${id}`}>Compare scenarios →</Link><Link className="text-indigo-600" to={`/explainability?asset=${id}`}>Explain →</Link></div>
    </div>
  );
}

export default function PredictiveMaintenance() {
  const [subset, setSubset] = useState("FD001");
  const [model, setModel] = useState("");
  const [test, setTest] = useState("2nd_test");
  const [asOf, setAsOf] = useState("");
  const [selected, setSelected] = useState(null);
  const models = useApi(api.models, []);
  const fleet = useApi(() => api.fleet(subset, model || undefined), [subset, model]);
  const bear = useApi(() => api.bearings(test, undefined, asOf === "" ? undefined : Number(asOf)), [test, asOf]);
  const trained = (models.data?.trained || []).filter((m) => m.name.startsWith("rul_") && m.name.endsWith(subset)).map((m) => m.name.replace("rul_", "").replace(`_${subset}`, ""));
  const f = fleet.data;
  return (
    <>
      <PageHeader title="Predictive Maintenance" description="Remaining useful life, probability of failure, health, risk (PoF × consequence) and urgency for every asset, computed by the trained models." />
      <div className="mb-4 flex flex-wrap gap-3">
        <div className="w-36"><Field label="C-MAPSS subset"><Select value={subset} onChange={(e) => { setSubset(e.target.value); setModel(""); }} options={["FD001", "FD002", "FD003", "FD004"]} /></Field></div>
        <div className="w-48"><Field label="RUL model"><Select value={model} onChange={(e) => setModel(e.target.value)} options={[{ value: "", label: "best (lowest RMSE)" }, ...trained]} /></Field></div>
      </div>
      <ErrorBox error={fleet.error} code={fleet.code} onRetry={fleet.reload} />
      {fleet.loading && <Loading />}
      {f && (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
            {["critical", "high", "medium", "low"].map((l) => <Stat key={l} label={`${l} risk`} value={f.risk_counts[l]} tone={l === "low" ? "good" : l === "medium" ? "warn" : "bad"} />)}
            <Stat label="Model test RMSE" value={fmt.n(f.model_metrics.rmse, 2)} hint={`${f.model} · σ = ${fmt.n(f.sigma, 1)} cycles`} />
          </div>
          <div className="mt-4 grid gap-4 xl:grid-cols-3">
            <Card title="Predicted vs true RUL" subtitle="Official test engines; true RUL from RUL_FDxxx.txt" className="xl:col-span-1">
              <ChartBox height={300}>
                <ScatterChart margin={{ left: -10 }}><CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="rul_true" name="true RUL" domain={[0, "auto"]} /><YAxis type="number" dataKey="rul_pred" name="predicted RUL" domain={[0, "auto"]} />
                  <ZAxis range={[30, 30]} /><Tooltip {...tooltipStyle} cursor={{ strokeDasharray: "3 3" }} />
                  <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 150, y: 150 }]} stroke="#94a3b8" strokeDasharray="4 4" />
                  <Scatter data={f.assets}>{f.assets.map((a) => <Cell key={a.asset_id} fill={RISK_COLORS[a.risk_level]} />)}</Scatter>
                </ScatterChart>
              </ChartBox>
            </Card>
            <Card title="Turbofan fleet" subtitle="Sorted by risk score · click a row" className="xl:col-span-2">
              <Table rows={f.assets} rowKey={(r) => r.asset_id} maxHeight={330} onRowClick={(r) => setSelected(r.asset_id)} selectedKey={selected}
                columns={[
                  { key: "asset_id", label: "Engine" }, { key: "current_cycle", label: "Cycle", align: "right" },
                  { key: "rul_pred", label: "Pred. RUL", align: "right", render: (r) => fmt.n(r.rul_pred) },
                  { key: "rul_true", label: "True RUL", align: "right", render: (r) => fmt.n(r.rul_true, 0) },
                  { key: "pof", label: "PoF", align: "right", render: (r) => fmt.pct(r.pof) },
                  { key: "health", label: "Health", align: "right", render: (r) => fmt.n(r.health, 0) },
                  { key: "risk_score", label: "Risk", align: "right", render: (r) => fmt.int(r.risk_score) },
                  { key: "risk_level", label: "Level", render: (r) => <RiskBadge level={r.risk_level} /> },
                  { key: "urgency", label: "Urgency" },
                ]} />
            </Card>
          </div>
        </>
      )}
      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card title="IMS bearings" subtitle="Risk from anomaly severity vs the alarm threshold (logistic mapping, a documented heuristic)" className="xl:col-span-2"
          actions={<>
            <div className="w-32"><Select value={test} onChange={(e) => { setTest(e.target.value); setAsOf(""); }} options={["1st_test", "2nd_test", "3rd_test"]} /></div>
          </>}>
          <ErrorBox error={bear.error} code={bear.code} />
          {bear.data && (
            <>
              <div className="mb-3 flex items-center gap-3 text-xs text-slate-600">
                <span>As of snapshot</span>
                <input type="range" min={10} max={bear.data.max_file_index} value={asOf === "" ? bear.data.max_file_index : asOf}
                  onChange={(e) => setAsOf(e.target.value)} className="flex-1" />
                <span className="w-40 text-right tabular-nums">{asOf === "" ? "latest" : asOf} · {bear.data.assets[0]?.timestamp}</span>
              </div>
              <Table rows={bear.data.assets} rowKey={(r) => r.asset_id} onRowClick={(r) => setSelected(r.asset_id)} selectedKey={selected} columns={[
                { key: "asset_id", label: "Bearing" },
                { key: "anomaly_score", label: "Score", align: "right", render: (r) => fmt.n(r.anomaly_score, 3) },
                { key: "threshold", label: "Threshold", align: "right", render: (r) => fmt.n(r.threshold, 3) },
                { key: "ttt", label: "Time to threshold", align: "right", render: (r) => (r.time_to_threshold_hours === null ? "no trend" : `${fmt.n(r.time_to_threshold_hours)} h`) },
                { key: "pof", label: "PoF", align: "right", render: (r) => fmt.pct(r.pof) },
                { key: "risk_level", label: "Level", render: (r) => <RiskBadge level={r.risk_level} /> },
                { key: "failed_in_dataset", label: "Failed in data", render: (r) => (r.failed_in_dataset ? "yes" : "no") },
              ]} />
              <div className="mt-2"><Notice>Time to threshold is extrapolated from the recent anomaly-score trend (simulated), not a measured RUL.</Notice></div>
            </>
          )}
        </Card>
        <Card title="Asset risk detail"><AssetDetail id={selected} /></Card>
      </div>
    </>
  );
}
