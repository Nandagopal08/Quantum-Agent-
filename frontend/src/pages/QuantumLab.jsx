import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Tooltip, XAxis, YAxis } from "recharts";
import { Atom } from "lucide-react";
import { api } from "../services/api";
import { useApi, useAction, useJob } from "../hooks/useApi";
import { Badge, Button, Card, ErrorBox, Field, Input, JobPanel, Loading, Notice, PageHeader, Select, Table, fmt } from "../components/ui";
import { ChartBox, PALETTE, tooltipStyle } from "../components/charts";

export default function QuantumLab() {
  const [form, setForm] = useState({ model: "vqc", n_qubits: 4, layers: 2, n_train: 200, epochs: 20, lr: 0.05 });
  const exps = useApi(api.quantumExperiments, []);
  const [job, setJob] = useJob(() => exps.reload());
  const start = useAction(() => api.quantumRun(Object.fromEntries(Object.entries(form).map(([k, v]) => [k, k === "model" ? v : Number(v)]))));
  const rows = useMemo(() => [...(exps.data || [])].sort((a, b) => b.saved_at - a.saved_at), [exps.data]);
  const chart = rows.filter((r) => r.comparison.metric === "roc_auc").map((r) => ({
    run: `${r.model} ${r.quantum.qubits}q`, quantum: r.comparison.quantum, best_classical: r.comparison.best_classical,
  }));
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  return (
    <>
      <PageHeader title="Quantum AI Laboratory"
        description="Hybrid quantum-classical experiments in PennyLane on a statevector simulator (4–8 qubits). Every quantum model is compared with classical models trained on the same PCA-reduced inputs." />
      <Notice tone="warn">These are small simulator experiments. A better or worse score in one run is not evidence of quantum advantage, and no advantage is claimed.</Notice>
      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card title="Run an experiment" subtitle="Task: C-MAPSS “fails within 30 cycles” (classifiers) or RUL (QNN regressor)">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Model"><Select value={form.model} onChange={set("model")} options={[
              { value: "vqc", label: "Variational Quantum Classifier" }, { value: "quantum_kernel", label: "Quantum kernel SVM" },
              { value: "qnn_regressor", label: "Hybrid QNN regressor" }]} /></Field>
            <Field label="Qubits (= PCA components)"><Input type="number" min={2} max={8} value={form.n_qubits} onChange={set("n_qubits")} /></Field>
            <Field label="Layers / repeats"><Input type="number" min={1} max={6} value={form.layers} onChange={set("layers")} /></Field>
            <Field label="Training samples" hint="balanced subsample"><Input type="number" min={40} max={1000} value={form.n_train} onChange={set("n_train")} /></Field>
            <Field label="Epochs"><Input type="number" min={1} max={200} value={form.epochs} onChange={set("epochs")} /></Field>
            <Field label="Learning rate"><Input type="number" step="0.01" value={form.lr} onChange={set("lr")} /></Field>
          </div>
          <div className="mt-3"><Button icon={Atom} loading={start.loading} onClick={async () => { const j = await start.run(); if (j) setJob(j); }}>Run on simulator</Button></div>
          <div className="mt-3 space-y-2"><ErrorBox error={start.error} /><JobPanel job={job} /></div>
        </Card>
        <Card title="Quantum vs best classical (ROC-AUC)" subtitle="Same PCA features, official test engines" className="xl:col-span-2">
          {exps.loading ? <Loading /> : (
            <ChartBox height={280}>
              <BarChart data={chart}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="run" /><YAxis domain={[0.5, 1]} /><Tooltip {...tooltipStyle} /><Legend />
                <Bar dataKey="quantum" fill={PALETTE[5]} radius={[3, 3, 0, 0]} /><Bar dataKey="best_classical" name="best classical" fill={PALETTE[1]} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ChartBox>
          )}
        </Card>
      </div>
      <Card title="Recorded experiments" className="mt-4" subtitle="Circuit metrics are measured after decomposition to device gates">
        <ErrorBox error={exps.error} code={exps.code} />
        <Table rows={rows} rowKey={(r) => r.run + r.saved_at} empty="No quantum experiments recorded yet" columns={[
          { key: "model", label: "Model" }, { key: "task", label: "Task" },
          { key: "qubits", label: "Qubits", align: "right", render: (r) => r.quantum.qubits },
          { key: "depth", label: "Depth", align: "right", render: (r) => r.quantum.circuit_depth ?? "—" },
          { key: "gates", label: "Gates", align: "right", render: (r) => r.quantum.gate_count ?? "—" },
          { key: "params", label: "Params", align: "right", render: (r) => r.quantum.trainable_params },
          { key: "train", label: "Train time", align: "right", render: (r) => `${fmt.n(r.quantum.train_seconds, 1)} s` },
          { key: "metric", label: "Quantum", align: "right", render: (r) => `${r.comparison.metric} ${fmt.n(r.comparison.quantum, 3)}` },
          { key: "classical", label: "Best classical", align: "right", render: (r) => `${r.comparison.best_classical_model} ${fmt.n(r.comparison.best_classical, 3)}` },
          { key: "verdict", label: "Result", render: (r) => (r.comparison.quantum_better ? <Badge tone="violet">quantum higher this run</Badge> : <Badge>classical higher</Badge>) },
          { key: "pca", label: "PCA var.", align: "right", render: (r) => fmt.pct(r.data.pca_explained_variance, 0) },
        ]} />
      </Card>
    </>
  );
}
