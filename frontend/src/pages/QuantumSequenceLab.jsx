import { useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { FlaskConical } from "lucide-react";
import { api } from "../services/api";
import { useAction, useApi, useJob } from "../hooks/useApi";
import { Badge, Button, Card, ErrorBox, Field, Input, JobPanel, Loading, Notice, PageHeader, Select, Table, fmt } from "../components/ui";
import { ChartBox, PALETTE, tooltipStyle } from "../components/charts";
import MissingDataAgent, { MODEL_LABEL } from "../components/MissingDataAgent";

const METHOD_COLOR = { mean: "#94a3b8", locf: "#64748b", linear: "#0f172a", q_lstm: PALETTE[0], q_transformer: PALETTE[5],
  q_cnn_lstm: PALETTE[4], c_lstm: PALETTE[1], c_transformer: PALETTE[6], c_cnn_lstm: PALETTE[3] };
const pm = (v, sd, d = 2) => (sd === null || sd === undefined ? fmt.n(v, d) : `${fmt.n(v, d)} ± ${fmt.n(sd, d)}`);

const sci = (v) => (v === null || v === undefined ? "—" : Number(v).toExponential(1));
const ci = (c) => (c && c[0] !== null && c[0] !== undefined ? `[${fmt.n(c[0], 3)}, ${fmt.n(c[1], 3)}]` : "—");
const Unavailable = ({ children }) => <div className="rounded-lg border border-dashed border-slate-300 p-4 text-sm text-slate-500"><Badge>unavailable</Badge> <span className="ml-1">{children}</span></div>;

function StatsTable({ rows, unit = 3 }) {
  return (
    <Table rows={rows} rowKey={(x) => `${x.a}-${x.b}-${x.pattern || ""}-${x.rate || ""}`} empty="No paired comparisons in this report" columns={[
      { key: "c", label: "A vs B", render: (x) => `${MODEL_LABEL[x.a] || x.a} vs ${MODEL_LABEL[x.b] || x.b}` },
      { key: "d", label: "RMSE(A) − RMSE(B)", align: "right", render: (x) => fmt.n(x.rmse_diff, unit) },
      { key: "ci", label: "95% CI (engines)", align: "right", render: (x) => ci(x.bootstrap_ci95) },
      { key: "sci", label: "95% CI (seeds)", align: "right", render: (x) => ci(x.seed_t_ci95) },
      { key: "p", label: "Holm p", align: "right", render: (x) => fmt.n(x.wilcoxon_p_holm, 3) },
      { key: "v", label: "Verdict", render: (x) => <Badge tone={x.significant ? "violet" : "slate"}>{x.verdict}</Badge> },
    ]} />
  );
}

export default function QuantumSequenceLab() {
  const report = useApi(() => api.qseqReport(), []);
  const qiskit = useApi(() => api.qseqQiskit(), []);
  const [pattern, setPattern] = useState("mcar");
  const [downRate, setDownRate] = useState(0.5);
  const [bench, setBench] = useState({ seeds: "42", epochs: 10, imputation_epochs: 8 });
  const [job, setJob] = useJob(() => { report.reload(); qiskit.reload(); });

  const startBench = useAction(() => api.qseqBenchmark({ seeds: bench.seeds.split(",").map(Number),
    epochs: Number(bench.epochs), imputation_epochs: Number(bench.imputation_epochs) }));


  const r = report.data;
  const chart = useMemo(() => {
    if (!r) return { data: [], methods: [] };
    const rows = r.imputation.evaluation.filter((x) => x.pattern === pattern);
    const methods = [...new Set(rows.map((x) => x.method))];
    const rates = [...new Set(rows.map((x) => x.rate))].sort((a, b) => a - b);
    return { methods, data: rates.map((rate) => Object.fromEntries([["rate", `${Math.round(rate * 100)}%`],
      ...rows.filter((x) => x.rate === rate).map((x) => [x.method, x.rmse])])) };
  }, [r, pattern]);
  const downstream = useMemo(() => {
    if (!r) return { models: [], methods: [], cell: () => null, clean: {}, rates: [] };
    const rows = r.downstream_rul || [];
    const models = [...new Set(rows.map((x) => x.rul_model))];
    const rates = [...new Set(rows.filter((x) => x.pattern !== "none").map((x) => x.rate))].sort((a, b) => a - b);
    const methods = [...new Set(rows.filter((x) => x.method !== "complete_data").map((x) => x.method))];
    const key = (m, meth) => rows.find((x) => x.rul_model === m && x.method === meth && x.pattern === pattern && x.rate === downRate);
    const clean = Object.fromEntries(models.map((m) => [m, rows.find((x) => x.rul_model === m && x.method === "complete_data")]));
    return { models, methods, cell: key, clean, rates };
  }, [r, pattern, downRate]);
  const impStats = useMemo(() => (r?.statistics?.imputation || []).filter((x) => x.pattern === pattern && x.rate === 0.3), [r, pattern]);
  const perSeedMin = r && r.total_seconds && r.seeds?.length ? r.total_seconds / r.seeds.length / 60 : null;
  const qk = qiskit.data;

  return (
    <>
      <PageHeader title="Quantum Sequence Lab"
        description="Quantum LSTM, Quantum Transformer and Quantum CNN-LSTM (PennyLane circuits inside PyTorch sequence models) against classical twins on C-MAPSS sensor time series — complete data, then 10–50 % missing values — and a single LangGraph agent that imputes and forecasts." />
      <Notice tone="warn">Statevector-simulator experiments. Each classical twin has the same architecture with the circuit replaced by a same-width classical layer. Measured differences are not evidence of quantum advantage.</Notice>

      <MissingDataAgent />

      {report.loading ? <div className="mt-4"><Loading /></div> : !r ? (
        <div className="mt-4"><ErrorBox error={report.error} code={report.code} /></div>
      ) : (
        <>
          <Card title="Task 1 · RUL prediction, complete data" className="mt-4"
            subtitle={`C-MAPSS ${r.config.subset}: ${r.data.window}-cycle windows of ${r.sensors.length} sensors; official test engines (n=${r.data.n_test_engines}); ${r.seeds.length} seed(s), mean ± sd. RMSE is against RUL capped at ${r.data.rul_cap ?? 125}${r.data.test_engines_above_cap !== undefined ? ` (${r.data.test_engines_above_cap} engines exceed it)` : ""}.`}>
            <Table rows={r.rul.summary} rowKey={(x) => x.model} columns={[
              { key: "model", label: "Model", render: (x) => <span className="flex items-center gap-1.5">{MODEL_LABEL[x.model]}{x.quantum && <Badge tone="violet">quantum</Badge>}</span> },
              { key: "rmse", label: "RMSE", align: "right", render: (x) => pm(x.rmse, x.rmse_std) },
              { key: "mae", label: "MAE", align: "right", render: (x) => pm(x.mae, x.mae_std) },
              { key: "r2", label: "R²", align: "right", render: (x) => fmt.n(x.r2, 3) },
              { key: "nasa", label: "NASA score", align: "right", render: (x) => fmt.int(x.nasa_score) },
              { key: "unc", label: "RMSE uncapped", align: "right", render: (x) => (x.rmse_uncapped === undefined ? "—" : fmt.n(x.rmse_uncapped, 2)) },
              { key: "abl", label: "RMSE, circuits ablated", align: "right", render: (x) => (x.rmse_ablated === undefined ? "—" : fmt.n(x.rmse_ablated, 2)) },
              { key: "params", label: "Params (circuit)", align: "right", render: (x) => `${x.trainable_params}${x.quantum ? ` (${x.quantum_params})` : ""}` },
              { key: "circ", label: "Circuit", render: (x) => (x.quantum && x.circuit ? `${x.n_qubits}q × ${x.circuit_layers ?? "?"}L · depth ${x.circuit.depth ?? "—"} · ${x.circuit.gates ?? "—"} gates · ${x.circuits} circuits` : "—") },
              { key: "t", label: "Train / infer", align: "right", render: (x) => `${fmt.n(x.train_seconds, 0)} s / ${x.inference_seconds === undefined ? "—" : `${fmt.n(1000 * x.inference_seconds, 0)} ms`}` },
            ]} />
            <div className="mt-4 text-xs font-semibold text-slate-600">Accuracy — “fails within 30 cycles” (same predictions, 100 official test engines)</div>
            <div className="mt-1">{r.rul.accuracy ? (
              <Table rows={r.rul.accuracy} rowKey={(x) => x.model} columns={[
                { key: "m", label: "Model", render: (x) => <span className="flex items-center gap-1.5">{MODEL_LABEL[x.model]}{x.model.startsWith("q_") && <Badge tone="violet">quantum</Badge>}</span> },
                ...[["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["roc_auc", "ROC-AUC"],
                  ["false_alarm_rate", "False alarms"], ["within_10", "Within ±10 cyc"], ["within_20", "Within ±20 cyc"]].map(([k, l]) => (
                  { key: k, label: l, align: "right", render: (x) => pm(x[k], x[`${k}_std`], 3) })),
              ]} />
            ) : <Unavailable>The raw per-seed predictions for this report are not available, so accuracy cannot be computed.</Unavailable>}</div>
            <p className="mt-2 text-xs text-slate-500">“Circuits ablated”: every circuit (or the twin’s counterpart layer) outputs only its batch mean — the rise in error shows how much the model relies on it.</p>
          </Card>

          <div className="mt-4 grid gap-4 xl:grid-cols-2">
            <Card title="Quantum vs classical twin — paired statistics (RUL)" subtitle={r.statistics?.method || "Paired comparison on the same engines and seeds"}>
              {r.statistics ? <StatsTable rows={r.statistics.rul} unit={2} /> : <Unavailable>This report predates the paired statistics; re-run the benchmark.</Unavailable>}
            </Card>
            <Card title="Qiskit cross-check" subtitle="The same circuits executed by Qiskit’s exact Statevector instead of PennyLane (simulator vs simulator; no hardware)">
              {qiskit.loading ? <Loading /> : !qk ? <Unavailable>{qiskit.error || "No verification recorded."}</Unavailable> : (
                <>
                  {qk.circuit && <div className="mb-2 grid grid-cols-2 gap-2 text-xs text-slate-600 sm:grid-cols-4">
                    {[["forward", "Forward"], ["grad_inputs", "∂/∂inputs"], ["grad_weights", "∂/∂weights"], ["parameter_mapping", "Parameter map"]].map(([k, l]) => (
                      <div key={k} className="rounded border border-slate-200 p-2"><div className="font-medium text-slate-700">{l}</div>max {sci(qk.circuit[k]?.max_abs)} · mean {sci(qk.circuit[k]?.mean_abs)}</div>))}
                  </div>}
                  <Table rows={qk.rows || []} rowKey={(x) => `${x.task || "rul"}-${x.model}`} columns={[
                    { key: "m", label: "Model", render: (x) => `${MODEL_LABEL[x.model] || x.model} · ${x.task || "rul"}` },
                    { key: "w", label: "Windows", align: "right", render: (x) => x.windows },
                    { key: "e", label: "Circuit evals", align: "right", render: (x) => fmt.int(x.qiskit_circuit_evaluations) },
                    { key: "mx", label: "Max |Δ|", align: "right", render: (x) => sci(x.max_abs_output_diff) },
                    { key: "mn", label: "Mean |Δ|", align: "right", render: (x) => sci(x.mean_abs_output_diff) },
                  ]} />
                  <p className="mt-2 text-xs text-slate-500">Generated {qk.generated || "—"}.</p>
                </>
              )}
            </Card>
          </div>

          <Card title="Task 2 · Imputation error vs missing rate" className="mt-4"
            subtitle={`RMSE on the removed values only (standardised sensor units), ${r.data.n_eval_windows} windows from ${r.data.n_eval_engines ?? "held-out"} test engines, fixed seeded masks`}
            actions={<Select value={pattern} onChange={(e) => setPattern(e.target.value)} options={[{ value: "mcar", label: "Random (MCAR)" }, { value: "block", label: "Sensor outages (block)" }]} />}>
            <ChartBox height={300}>
              <LineChart data={chart.data}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="rate" /><YAxis domain={["auto", "auto"]} width={50} />
                <Tooltip {...tooltipStyle} formatter={(v, n) => [fmt.n(v, 4), MODEL_LABEL[n] || n]} /><Legend formatter={(n) => MODEL_LABEL[n] || n} />
                {chart.methods.map((m) => <Line key={m} dataKey={m} stroke={METHOD_COLOR[m]} strokeWidth={m.startsWith("q_") ? 2.5 : 1.5}
                  strokeDasharray={["mean", "locf", "linear"].includes(m) ? "4 3" : undefined} dot={{ r: 3 }} />)}
              </LineChart>
            </ChartBox>
            <div className="mt-4 text-xs font-semibold text-slate-600">Paired comparisons at 30 % missing ({pattern})</div>
            <div className="mt-1">{r.statistics ? <StatsTable rows={impStats} unit={4} /> : <Unavailable>Not in this report.</Unavailable>}</div>
            <div className="mt-4 text-xs font-semibold text-slate-600">Learned imputers — training diagnostics</div>
            <div className="mt-1"><Table rows={r.imputation.models || []} rowKey={(x) => x.model} columns={[
              { key: "m", label: "Imputer", render: (x) => MODEL_LABEL[x.model] || x.model },
              { key: "v", label: "Val RMSE", align: "right", render: (x) => pm(x.val_rmse, x.val_rmse_std, 4) },
              { key: "l", label: "Final train loss", align: "right", render: (x) => fmt.n(x.final_train_loss, 4) },
              { key: "g", label: "Grad norm (epoch 1 → last)", align: "right", render: (x) => (x.first_epoch_grad_norm === undefined ? "—" : `${fmt.n(x.first_epoch_grad_norm, 4)} → ${fmt.n(x.last_epoch_grad_norm, 4)}`) },
              { key: "a", label: "RMSE / ablated (MCAR 30%)", align: "right", render: (x) => (x.ablation_mcar30 ? `${fmt.n(x.ablation_mcar30.rmse, 4)} / ${fmt.n(x.ablation_mcar30.rmse_ablated, 4)}` : "—") },
              { key: "p", label: "Params (circuit)", align: "right", render: (x) => `${x.trainable_params}${x.quantum ? ` (${x.quantum_params})` : ""}` },
            ]} /></div>
          </Card>

          <Card title="Task 3 · Downstream RUL error after imputation" className="mt-4"
            subtitle={`Test-engine windows masked (${pattern}), imputed by each method, then forecast by each RUL model. Test RMSE in cycles, mean ± sd over seeds.`}
            actions={<Select value={String(downRate)} onChange={(e) => setDownRate(Number(e.target.value))} options={downstream.rates.map((x) => ({ value: String(x), label: `${Math.round(x * 100)}% missing` }))} />}>
            {!downstream.models.length ? <Unavailable>No downstream evaluation in this report.</Unavailable> : (
              <Table rows={[{ m: "complete_data" }, ...downstream.methods.map((m) => ({ m }))]} rowKey={(x) => x.m} columns={[
                { key: "m", label: "Imputer", render: (x) => (x.m === "complete_data" ? <b>complete data (no missing)</b> : MODEL_LABEL[x.m] || x.m) },
                ...downstream.models.map((rm) => ({ key: rm, label: MODEL_LABEL[rm] || rm, align: "right",
                  render: (x) => { const c = x.m === "complete_data" ? downstream.clean[rm] : downstream.cell(rm, x.m); return c ? pm(c.rul_rmse, c.rul_rmse_std) : "—"; } })),
              ]} />
            )}
          </Card>
        </>
      )}

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="Re-run the benchmark" subtitle={`Trains all 6 models for both tasks${perSeedMin ? ` (last run: ${fmt.n(perSeedMin, 0)} min per seed)` : ""}. The previous report and models are kept in history.`}>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Seeds"><Input value={bench.seeds} onChange={(e) => setBench({ ...bench, seeds: e.target.value })} /></Field>
            <Field label="RUL epochs"><Input type="number" value={bench.epochs} onChange={(e) => setBench({ ...bench, epochs: e.target.value })} /></Field>
            <Field label="Imputer epochs"><Input type="number" value={bench.imputation_epochs} onChange={(e) => setBench({ ...bench, imputation_epochs: e.target.value })} /></Field>
          </div>
          <div className="mt-3"><Button icon={FlaskConical} variant="secondary" loading={startBench.loading}
            onClick={async () => { const j = await startBench.run(); if (j) setJob(j); }}>Start benchmark job</Button></div>
          <div className="mt-3 space-y-2"><ErrorBox error={startBench.error} /><JobPanel job={job} /></div>
        </Card>
      </div>
    </>
  );
}
