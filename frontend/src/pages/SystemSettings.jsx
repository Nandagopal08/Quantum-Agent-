import { useEffect, useState } from "react";
import { Save } from "lucide-react";
import { api } from "../services/api";
import { useApi, useAction } from "../hooks/useApi";
import { Badge, Button, Card, ErrorBox, Field, Input, Loading, Notice, PageHeader, Select } from "../components/ui";

const num = (v) => (v === "" || v === null ? v : Number(v));

function NumberGrid({ obj, onChange, fields }) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      {fields.map(([k, label, hint]) => (
        <Field key={k} label={label} hint={hint}><Input type="number" step="any" value={obj[k]} onChange={(e) => onChange({ ...obj, [k]: num(e.target.value) })} /></Field>
      ))}
    </div>
  );
}

export default function SystemSettings() {
  const s = useApi(api.settings, []);
  const llm = useApi(api.llmStatus, []);
  const [draft, setDraft] = useState(null);
  useEffect(() => { if (s.data) setDraft(structuredClone(s.data)); }, [s.data]);
  const save = useAction(() => api.saveSettings(draft));
  const reset = useAction(api.resetSettings);
  if (s.error) return <><PageHeader title="System Settings" /><ErrorBox error={s.error} code={s.code} onRetry={s.reload} /></>;
  if (s.loading || !draft) return <><PageHeader title="System Settings" /><Loading /></>;
  const eq = (kind, v) => setDraft({ ...draft, equipment: { ...draft.equipment, [kind]: v } });
  const costFields = [["preventive_maintenance_cost", "Preventive maintenance cost"], ["corrective_maintenance_cost", "Corrective (failure) cost"],
    ["downtime_cost_per_unit", "Downtime cost per time unit"], ["preventive_downtime", "Preventive downtime"], ["failure_downtime", "Failure downtime"],
    ["lost_output_per_unit", "Lost output per time unit", "at 100% load reduction"], ["criticality", "Criticality", "multiplies consequence"],
    ["restoration_fraction", "Restoration fraction", "share of max RUL restored"]];
  return (
    <>
      <PageHeader title="System Settings" description="Cost model, risk thresholds, simulation, optimisation and LLM configuration. Saved to configuration/app_settings.json in Drive; API keys stay in environment variables."
        actions={<>
          <Button variant="secondary" loading={reset.loading} onClick={async () => { await reset.run(); s.reload(); }}>Reset to defaults</Button>
          <Button icon={Save} loading={save.loading} onClick={async () => { if (await save.run()) { s.reload(); llm.reload(); } }}>Save settings</Button>
        </>} />
      <ErrorBox error={save.error || reset.error} />
      {save.result && <Notice tone="ok">Settings saved. Predictions and risk use the new values immediately.</Notice>}
      <Notice tone="warn">All monetary values are assumptions for demonstration. Replace them with real figures before using recommendations.</Notice>
      <div className="mt-4 space-y-4">
        {Object.entries(draft.equipment).map(([kind, v]) => (
          <Card key={kind} title={v.label} subtitle={`Time unit: ${v.time_unit} · currency: ${draft.currency}`}><NumberGrid obj={v} onChange={(o) => eq(kind, o)} fields={costFields} /></Card>
        ))}
        <Card title="Risk thresholds" subtitle="Risk level from risk score as a fraction of the asset's consequence of failure">
          <NumberGrid obj={draft.risk.thresholds} onChange={(o) => setDraft({ ...draft, risk: { ...draft.risk, thresholds: o } })}
            fields={[["medium", "Medium ≥"], ["high", "High ≥"], ["critical", "Critical ≥"]]} />
          <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Field label="PoF horizon"><Input type="number" value={draft.risk.failure_horizon} onChange={(e) => setDraft({ ...draft, risk: { ...draft.risk, failure_horizon: num(e.target.value) } })} /></Field>
            {Object.entries(draft.risk.urgency_rul).map(([k, val]) => (
              <Field key={k} label={`Urgency “${k}” if RUL ≤`}><Input type="number" value={val} onChange={(e) => setDraft({ ...draft, risk: { ...draft.risk, urgency_rul: { ...draft.risk.urgency_rul, [k]: num(e.target.value) } } })} /></Field>))}
            <Field label="Human approval required from"><Select value={draft.risk.human_approval_level} onChange={(e) => setDraft({ ...draft, risk: { ...draft.risk, human_approval_level: e.target.value } })} options={["low", "medium", "high", "critical"]} /></Field>
          </div>
        </Card>
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Simulation"><NumberGrid obj={draft.simulation} onChange={(o) => setDraft({ ...draft, simulation: o })}
            fields={[["monte_carlo_samples", "Monte Carlo samples"], ["load_exponent", "Load exponent α", "wear ∝ load^α"], ["planning_horizon", "Planning horizon"], ["seed", "Random seed"]]} /></Card>
          <Card title="Optimisation"><NumberGrid obj={draft.optimization} onChange={(o) => setDraft({ ...draft, optimization: o })}
            fields={[["crew_capacity_per_slot", "Crew capacity / slot"], ["slots", "Slots"], ["qaoa_layers", "QAOA layers"], ["qaoa_steps", "QAOA evaluations"], ["max_qubits", "Max qubits (simulator)"]]} /></Card>
        </div>
        <Card title="LLM" subtitle="Used only for narratives and the chat panel; numbers always come from tools"
          actions={llm.data && <Badge tone={llm.data.available ? "green" : "slate"}>{llm.data.available ? "available" : "unavailable"}: {llm.data.detail}</Badge>}>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Field label="Provider"><Select value={draft.llm.provider} onChange={(e) => setDraft({ ...draft, llm: { ...draft.llm, provider: e.target.value } })}
              options={[{ value: "ollama", label: "Ollama (local)" }, { value: "openai", label: "OpenAI-compatible API" }, { value: "none", label: "None (deterministic only)" }]} /></Field>
            <Field label="Model"><Input value={draft.llm.model} onChange={(e) => setDraft({ ...draft, llm: { ...draft.llm, model: e.target.value } })} /></Field>
            <Field label="Ollama base URL"><Input value={draft.llm.base_url} onChange={(e) => setDraft({ ...draft, llm: { ...draft.llm, base_url: e.target.value } })} /></Field>
            <Field label="Timeout (s)"><Input type="number" value={draft.llm.timeout_seconds} onChange={(e) => setDraft({ ...draft, llm: { ...draft.llm, timeout_seconds: num(e.target.value) } })} /></Field>
          </div>
          <div className="mt-2 text-xs text-slate-500">API providers read <code>OPENAI_API_KEY</code> (and optional <code>OPENAI_BASE_URL</code>) from the backend's environment — never from this page.</div>
        </Card>
      </div>
    </>
  );
}
