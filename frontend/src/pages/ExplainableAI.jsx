import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, Tooltip, XAxis, YAxis } from "recharts";
import { Lightbulb } from "lucide-react";
import { api } from "../services/api";
import { useApi, useAction, useJob } from "../hooks/useApi";
import { Button, Card, Empty, ErrorBox, Input, JobPanel, Loading, Notice, PageHeader, Select, Stat, Table, fmt } from "../components/ui";
import { ChartBox, PALETTE, tooltipStyle } from "../components/charts";

function HBar({ data, valueKey, labelKey, signed, height }) {
  return (
    <ChartBox height={height || Math.max(180, data.length * 24)}>
      <BarChart data={data} layout="vertical" margin={{ left: 10 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" /><YAxis type="category" dataKey={labelKey} width={120} tick={{ fontSize: 11 }} /><Tooltip {...tooltipStyle} />
        {signed && <ReferenceLine x={0} stroke="#64748b" />}
        <Bar dataKey={valueKey} radius={3}>{data.map((d) => <Cell key={d[labelKey]} fill={signed ? (d[valueKey] < 0 ? "#e11d48" : "#10b981") : PALETTE[0]} />)}</Bar>
      </BarChart>
    </ChartBox>
  );
}

function AssetExplanation({ assetId }) {
  const ex = useApi(() => api.explainAsset(assetId), [assetId], { enabled: !!assetId });
  if (ex.loading) return <Loading label="Computing attributions…" />;
  if (ex.error) return <ErrorBox error={ex.error} code={ex.code} />;
  const d = ex.data;
  if (!d) return null;
  if (d.deviation) {
    return (
      <Card title={`Why is ${assetId} flagged?`} subtitle={`${d.deviation.method} · snapshot ${d.deviation.file_index} (${d.deviation.timestamp})`}>
        <HBar data={d.deviation.top_deviations} valueKey="z" labelKey="feature" signed />
      </Card>
    );
  }
  const cf = d.counterfactual;
  const u = d.uncertainty;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Predicted RUL" value={fmt.n(u.prediction)} hint={`model ${d.model}`} />
        <Stat label="90% interval" value={`${fmt.n(u.interval_90[0], 0)}–${fmt.n(u.interval_90[1], 0)}`} hint="split-conformal" />
        <Stat label="σ (calibrated)" value={fmt.n(u.sigma)} hint="validation residuals" />
        <Stat label={`P(fail ≤ ${u.horizon})`} value={fmt.pct(u.pof)} hint={u.ensemble_std ? `RF tree spread ±${fmt.n(u.ensemble_std)}` : undefined} />
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Local attribution (this engine)" subtitle={d.local.note}>
          <HBar data={d.local.contributions.map((c) => ({ ...c, label: `${c.feature} ${c.sensor_name}`.slice(0, 26) }))} valueKey="contribution" labelKey="label" signed />
        </Card>
        <Card title="Counterfactual explanation" subtitle={cf.note}>
          <p className="text-sm text-slate-800">{cf.interpretation}</p>
          <div className="mt-3"><Table rows={cf.changes} rowKey={(c) => c.sensor} empty="No counterfactual found" columns={[
            { key: "sensor_name", label: "Sensor" },
            { key: "current_value", label: "Current", align: "right", render: (c) => fmt.n(c.current_value, 3) },
            { key: "counterfactual_value", label: "Counterfactual", align: "right", render: (c) => fmt.n(c.counterfactual_value, 3) },
            { key: "predicted_rul_after", label: "RUL after", align: "right", render: (c) => fmt.n(c.predicted_rul_after) },
          ]} /></div>
          <div className="mt-2 text-xs text-slate-500">Target RUL {fmt.n(cf.target_rul)} · {cf.target_reached ? "reached" : "not reached"} with {cf.changes.length} change(s)</div>
        </Card>
      </div>
    </div>
  );
}

export default function ExplainableAI() {
  const [params] = useSearchParams();
  const [asset, setAsset] = useState(params.get("asset") || "FD001-test-unit34");
  const [draft, setDraft] = useState(asset);
  const [model, setModel] = useState("xgboost");
  const reports = useApi(api.explainReports, []);
  const [job, setJob] = useJob(() => reports.reload());
  const start = useAction(() => api.explainModel({ model, subset: "FD001" }));
  const rep = (reports.data || []).find((r) => r.model === model);
  return (
    <>
      <PageHeader title="Explainable AI" description="SHAP for tree models, occlusion attributions for sequence models, permutation importance, counterfactual sensor explanations, and calibrated uncertainty for every prediction." />
      <Card title="Explain a single prediction" actions={<><Input value={draft} onChange={(e) => setDraft(e.target.value)} /><Button variant="secondary" onClick={() => setAsset(draft.trim())}>Explain</Button></>}>
        <AssetExplanation assetId={asset} />
      </Card>
      <Card className="mt-4" title="Global model explanation" subtitle="Computed on the official test engines (SHAP) and validation engines (permutation importance)"
        actions={<>
          <div className="w-40"><Select value={model} onChange={(e) => setModel(e.target.value)} options={["random_forest", "xgboost", "lightgbm", "catboost", "lstm", "gru", "cnn_lstm", "transformer"]} /></div>
          <Button icon={Lightbulb} loading={start.loading} onClick={async () => { const j = await start.run(); if (j) setJob(j); }}>{rep ? "Recompute" : "Compute"}</Button>
        </>}>
        <ErrorBox error={start.error} /><JobPanel job={job} />
        {reports.loading && <Loading />}
        {!rep && !reports.loading && <Empty>No saved explanation for {model} yet — click Compute.</Empty>}
        {rep && (
          <div className="grid gap-4 xl:grid-cols-3">
            <div>
              <div className="mb-1 text-xs font-semibold text-slate-600">{rep.shap_global ? "Mean |SHAP value| (cycles of RUL)" : "SHAP not applicable to sequence models"}</div>
              {rep.shap_global ? <HBar data={rep.shap_global.importance.slice(0, 12)} valueKey="mean_abs_shap" labelKey="feature" /> :
                <Notice>For {model}, attributions use occlusion (see the single-prediction panel) and permutation importance.</Notice>}
              {rep.shap_plot && <img className="mt-2 w-full rounded border" alt="SHAP summary" src={api.fileUrl("shap", rep.shap_plot.split("/").pop())} />}
            </div>
            <div>
              <div className="mb-1 text-xs font-semibold text-slate-600">Permutation importance (Δ validation RMSE)</div>
              <HBar data={rep.permutation_importance.slice(0, 12).map((p) => ({ ...p, label: p.feature_group }))} valueKey="rmse_increase" labelKey="label" />
            </div>
            <div>
              <div className="mb-1 text-xs font-semibold text-slate-600">Model-native importance</div>
              {rep.native_importance ? <HBar data={rep.native_importance.slice(0, 12)} valueKey="importance" labelKey="feature" /> : <Empty>Not available for this model type.</Empty>}
            </div>
          </div>
        )}
      </Card>
    </>
  );
}
