import { useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { Play } from "lucide-react";
import { api } from "../services/api";
import { useApi, useAction, useJob } from "../hooks/useApi";
import { Button, Card, Empty, ErrorBox, Field, Input, JobPanel, Loading, Notice, PageHeader, Select, StatusBadge, Table, fmt } from "../components/ui";
import { ChartBox, PALETTE, tooltipStyle } from "../components/charts";

function RulForm({ onDone }) {
  const [form, setForm] = useState({ model: "lstm", subset: "FD001", epochs: 40 });
  const [job, setJob] = useJob(onDone);
  const start = useAction(() => api.trainRul({ ...form, epochs: Number(form.epochs) }));
  return (
    <Card title="Train an RUL model (C-MAPSS)" subtitle="Evaluated on the official test set; sequence models checkpoint every epoch to Drive and resume after a disconnect">
      <div className="grid grid-cols-3 gap-3">
        <Field label="Model"><Select value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })}
          options={["random_forest", "xgboost", "lightgbm", "catboost", "lstm", "gru", "cnn_lstm", "transformer"]} /></Field>
        <Field label="Subset"><Select value={form.subset} onChange={(e) => setForm({ ...form, subset: e.target.value })} options={["FD001", "FD002", "FD003", "FD004"]} /></Field>
        <Field label="Epochs" hint="deep models only"><Input type="number" min={1} max={300} value={form.epochs} onChange={(e) => setForm({ ...form, epochs: e.target.value })} /></Field>
      </div>
      <div className="mt-3 flex items-center gap-2"><Button icon={Play} loading={start.loading} onClick={async () => { const j = await start.run(); if (j) setJob(j); }}>Start training</Button></div>
      <div className="mt-3 space-y-2"><ErrorBox error={start.error} /><JobPanel job={job} />
        {job?.status === "completed" && job.result && (
          <Notice tone="ok">{job.result.model} on {job.result.subset}: test RMSE {fmt.n(job.result.test.rmse, 2)}, MAE {fmt.n(job.result.test.mae, 2)},
            R² {fmt.n(job.result.test.r2, 3)}, NASA score {fmt.int(job.result.test.nasa_score)} ({fmt.n(job.result.train_seconds, 0)} s)</Notice>)}
      </div>
    </Card>
  );
}

function AnomalyForm({ onDone }) {
  const [form, setForm] = useState({ model: "isolation_forest", test: "2nd_test", threshold_quantile: 0.99 });
  const [job, setJob] = useJob(onDone);
  const start = useAction(() => api.trainAnomaly({ ...form, threshold_quantile: Number(form.threshold_quantile) }));
  const m = job?.result?.metrics;
  return (
    <Card title="Train a bearing anomaly detector (IMS)" subtitle="Fitted on the healthy baseline period; alarm threshold = quantile of baseline scores">
      <div className="grid grid-cols-3 gap-3">
        <Field label="Detector"><Select value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} options={["isolation_forest", "one_class_svm", "autoencoder", "lstm_autoencoder"]} /></Field>
        <Field label="IMS test"><Select value={form.test} onChange={(e) => setForm({ ...form, test: e.target.value })} options={["1st_test", "2nd_test", "3rd_test"]} /></Field>
        <Field label="Threshold quantile"><Input type="number" step="0.005" min={0.9} max={0.999} value={form.threshold_quantile} onChange={(e) => setForm({ ...form, threshold_quantile: e.target.value })} /></Field>
      </div>
      <div className="mt-3"><Button icon={Play} loading={start.loading} onClick={async () => { const j = await start.run(); if (j) setJob(j); }}>Start training</Button></div>
      <div className="mt-3 space-y-2"><ErrorBox error={start.error} /><JobPanel job={job} />
        {m && <Notice tone="ok">ROC-AUC {fmt.n(m.roc_auc, 3)}, F1 {fmt.n(m.f1, 3)}, false-alarm rate {fmt.pct(m.false_alarm_rate)};
          lead time {Object.entries(job.result.detection).map(([b, v]) => `bearing ${b}: ${v.lead_time_hours} h`).join(", ")}</Notice>}
      </div>
    </Card>
  );
}

function TrainingCurve({ run }) {
  const log = useApi(() => api.trainingLog(run), [run], { enabled: !!run });
  const rows = useMemo(() => (log.data || []).filter((r) => r.epoch !== undefined), [log.data]);
  if (!run) return <Empty>Select a deep-learning run to see its training curve.</Empty>;
  if (log.loading) return <Loading />;
  if (log.error) return <ErrorBox error={log.error} />;
  if (!rows.length) return <Empty>No per-epoch log for {run} (tree models train in one pass).</Empty>;
  const keys = Object.keys(rows[0]).filter((k) => !["epoch", "time"].includes(k));
  return (
    <ChartBox height={240}>
      <LineChart data={rows}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="epoch" /><YAxis /><Tooltip {...tooltipStyle} /><Legend />
        {keys.map((k, i) => <Line key={k} dataKey={k} stroke={PALETTE[i]} dot={false} />)}
      </LineChart>
    </ChartBox>
  );
}

export default function ModelTraining() {
  const models = useApi(api.models, []);
  const jobs = useApi(() => api.jobs({ limit: 30 }), []);
  const [run, setRun] = useState(null);
  const refresh = () => { models.reload(); jobs.reload(); };
  const trained = models.data?.trained || [];
  const rul = trained.filter((m) => m.name.startsWith("rul_"));
  const anomaly = trained.filter((m) => m.name.startsWith("anomaly_"));
  return (
    <>
      <PageHeader title="Model Training" description="Train classical models as background jobs. Progress streams live; every model, checkpoint and log is saved to Drive." />
      <div className="grid gap-4 xl:grid-cols-2"><RulForm onDone={refresh} /><AnomalyForm onDone={refresh} /></div>
      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card title="Trained RUL models" subtitle="Official C-MAPSS test set (last cycle of each engine) · click a row for its training curve" className="xl:col-span-2">
          {models.loading && <Loading />}<ErrorBox error={models.error} code={models.code} />
          <Table rows={[...rul].sort((a, b) => (a.metrics.rmse ?? 1e9) - (b.metrics.rmse ?? 1e9))} rowKey={(r) => r.name} onRowClick={(r) => setRun(r.name)} selectedKey={run}
            empty="No RUL model trained yet"
            columns={[
              { key: "name", label: "Model", render: (r) => r.name.replace("rul_", "") },
              { key: "rmse", label: "RMSE", align: "right", render: (r) => fmt.n(r.metrics.rmse, 2) },
              { key: "mae", label: "MAE", align: "right", render: (r) => fmt.n(r.metrics.mae, 2) },
              { key: "r2", label: "R²", align: "right", render: (r) => fmt.n(r.metrics.r2, 3) },
              { key: "nasa", label: "NASA score", align: "right", render: (r) => fmt.int(r.metrics.nasa_score) },
              { key: "auc", label: "PoF ROC-AUC", align: "right", render: (r) => fmt.n(r.metrics.roc_auc, 3) },
              { key: "saved_at", label: "Saved", render: (r) => r.saved_at },
            ]} />
        </Card>
        <Card title="Training curve" subtitle={run || "per-epoch train / validation RMSE"}><TrainingCurve run={run} /></Card>
      </div>
      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="Trained anomaly detectors" subtitle="IMS pseudo-labels: first 25% healthy, last 10% of failed bearings anomalous">
          <Table rows={anomaly} rowKey={(r) => r.name} empty="No detector trained yet" columns={[
            { key: "name", label: "Detector", render: (r) => r.name.replace("anomaly_", "") },
            { key: "roc_auc", label: "ROC-AUC", align: "right", render: (r) => fmt.n(r.metrics.roc_auc, 3) },
            { key: "f1", label: "F1", align: "right", render: (r) => fmt.n(r.metrics.f1, 3) },
            { key: "far", label: "False alarms", align: "right", render: (r) => fmt.pct(r.metrics.false_alarm_rate) },
          ]} />
        </Card>
        <Card title="Job history" actions={<Button variant="secondary" onClick={jobs.reload}>Refresh</Button>}>
          <Table rows={jobs.data} rowKey={(r) => r.id} maxHeight={320} columns={[
            { key: "type", label: "Type" }, { key: "params", label: "Parameters", render: (r) => <span className="text-xs text-slate-500">{Object.entries(r.params || {}).slice(0, 3).map(([k, v]) => `${k}=${Array.isArray(v) ? v.length : v}`).join(" ")}</span> },
            { key: "status", label: "Status", render: (r) => <StatusBadge status={r.status} /> },
            { key: "created_at", label: "Started", render: (r) => fmt.time(r.created_at) },
          ]} />
        </Card>
      </div>
    </>
  );
}
