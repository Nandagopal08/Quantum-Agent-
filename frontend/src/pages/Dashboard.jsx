import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, AlertTriangle, BrainCircuit, Database, Wrench } from "lucide-react";
import { api } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Card, Empty, ErrorBox, Loading, Notice, PageHeader, RISK_COLORS, RiskBadge, Stat, StatusBadge, Table, fmt } from "../components/ui";
import { ChartBox, tooltipStyle } from "../components/charts";

export default function Dashboard() {
  const { data, error, code, loading, reload } = useApi(api.overview, []);
  const header = <PageHeader title="Dashboard Overview"
    description="Live state of datasets, trained models, fleet risk and agent activity. Every value comes from the backend's stored results and trained models." />;
  if (loading && !data) return <>{header}<Loading label="Loading overview from the backend…" /></>;
  if (error) return <>{header}<ErrorBox error={error} code={code} onRetry={reload} /></>;
  const ready = Object.values(data.datasets).filter((d) => d.ready).length;
  const fleet = data.fleet;
  const counts = fleet?.risk_counts || {};
  const riskData = ["low", "medium", "high", "critical"].map((k) => ({ level: k, count: counts[k] || 0 }));
  const best = data.models.rul?.[0];
  return (
    <>
      {header}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Datasets ready" value={`${ready} / ${Object.keys(data.datasets).length}`} icon={Database}
          tone={ready === Object.keys(data.datasets).length ? "good" : "warn"} hint="verified & extracted in storage" />
        <Stat label="Trained models" value={data.models.total} icon={BrainCircuit}
          hint={Object.entries(data.models.by_category).map(([k, v]) => `${k.replace("_ai", "")}: ${v}`).join(" · ") || "none yet"} />
        <Stat label="Best RUL model (FD001)" value={best ? fmt.n(best.rmse, 2) : "—"} hint={best ? `${best.name.replace("rul_", "")} · test RMSE (cycles)` : "train a model"} />
        <Stat label="Critical engines" value={fleet ? counts.critical : "—"} tone={counts.critical ? "bad" : "good"} icon={AlertTriangle}
          hint={fleet ? `of ${fleet.n_assets} FD001 test engines` : data.fleet_error} />
        <Stat label="Maintenance work orders" value={data.maintenance_events} icon={Wrench} hint="approved via agents or recorded" />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card title="Fleet risk distribution" subtitle={fleet ? `C-MAPSS FD001 test engines · model ${fleet.model}` : undefined}>
          {fleet ? (
            <ChartBox height={220}>
              <BarChart data={riskData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="level" /><YAxis allowDecimals={false} /><Tooltip {...tooltipStyle} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>{riskData.map((d) => <Cell key={d.level} fill={RISK_COLORS[d.level]} />)}</Bar>
              </BarChart>
            </ChartBox>
          ) : <Empty>{data.fleet_error || "No RUL model trained yet."} <Link className="text-indigo-600" to="/training">Train one</Link></Empty>}
        </Card>
        <Card title="Highest-risk engines" subtitle="Risk = P(failure within horizon) × consequence" className="xl:col-span-2"
          actions={<Link to="/predictive" className="text-xs font-medium text-indigo-600">All assets →</Link>}>
          <Table rows={fleet?.assets?.slice(0, 8)} rowKey={(r) => r.asset_id} empty="No predictions yet"
            columns={[
              { key: "asset_id", label: "Engine", render: (r) => <Link className="text-indigo-600" to={`/twin?asset=${r.asset_id}`}>{r.asset_id}</Link> },
              { key: "rul_pred", label: "Predicted RUL", align: "right", render: (r) => fmt.n(r.rul_pred) },
              { key: "rul_true", label: "True RUL", align: "right", render: (r) => fmt.n(r.rul_true, 0) },
              { key: "pof", label: "PoF", align: "right", render: (r) => fmt.pct(r.pof) },
              { key: "risk_score", label: "Risk score", align: "right", render: (r) => fmt.int(r.risk_score) },
              { key: "risk_level", label: "Level", render: (r) => <RiskBadge level={r.risk_level} /> },
              { key: "urgency", label: "Urgency" },
            ]} />
        </Card>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card title="IMS bearings" subtitle={data.bearings ? `${data.bearings.test} · detector ${data.bearings.model} · last snapshot` : undefined}>
          {data.bearings ? (
            <Table rows={data.bearings.assets} rowKey={(r) => r.asset_id} columns={[
              { key: "bearing", label: "Bearing" },
              { key: "anomaly_score", label: "Score / threshold", align: "right", render: (r) => `${fmt.n(r.anomaly_score, 3)} / ${fmt.n(r.threshold, 3)}` },
              { key: "risk_level", label: "Risk", render: (r) => <RiskBadge level={r.risk_level} /> },
              { key: "failed_in_dataset", label: "Failed in data", render: (r) => (r.failed_in_dataset ? "yes" : "no") },
            ]} />
          ) : <Empty>{data.bearings_error || "No detector trained"}</Empty>}
        </Card>
        <Card title="Recent jobs" actions={<Link to="/training" className="text-xs font-medium text-indigo-600">Training →</Link>}>
          <Table rows={data.recent_jobs} rowKey={(r) => r.id} empty="No jobs yet" columns={[
            { key: "type", label: "Job" }, { key: "status", label: "Status", render: (r) => <StatusBadge status={r.status} /> },
            { key: "created_at", label: "Started", render: (r) => fmt.time(r.created_at) },
          ]} />
        </Card>
        <Card title="Recent agent runs" actions={<Link to="/agents" className="text-xs font-medium text-indigo-600">Control center →</Link>}>
          <Table rows={data.recent_runs} rowKey={(r) => r.id} empty="No agent runs yet" columns={[
            { key: "id", label: "Run", render: (r) => <span className="font-mono text-xs">{r.id}</span> },
            { key: "assets", label: "Assets", render: (r) => r.request?.asset_ids?.length ?? "—" },
            { key: "status", label: "Status", render: (r) => <StatusBadge status={r.status} /> },
          ]} />
        </Card>
      </div>
      {!data.llm.available && (
        <div className="mt-4"><Notice>LLM unavailable ({data.llm.detail}). Agents run with deterministic, fully computed reports; configure Ollama or an API provider in <Link className="font-medium underline" to="/settings">System Settings</Link> for LLM narratives.</Notice></div>
      )}
      <div className="mt-2 flex items-center gap-1 text-xs text-slate-400"><Activity className="h-3 w-3" /> Storage: {data.storage.project_root}</div>
    </>
  );
}
