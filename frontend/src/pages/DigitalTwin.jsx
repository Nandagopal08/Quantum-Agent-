import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Area, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, Tooltip, XAxis, YAxis } from "recharts";
import { Pause, Play, Wrench } from "lucide-react";
import { api } from "../services/api";
import { useApi, useAction } from "../hooks/useApi";
import { Button, Card, ErrorBox, Field, Input, Loading, MeasuredTag, Notice, PageHeader, RiskBadge, Select, SimulatedTag, Stat, Table, fmt } from "../components/ui";
import { ChartBox, PALETTE, tooltipStyle } from "../components/charts";

function usePlayer(length) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => { setIdx(Math.max(length - 1, 0)); setPlaying(false); }, [length]);
  useEffect(() => {
    if (!playing) return undefined;
    const t = setInterval(() => setIdx((i) => { if (i >= length - 1) { setPlaying(false); return i; } return i + 1; }), 80);
    return () => clearInterval(t);
  }, [playing, length]);
  return { idx, setIdx, playing, toggle: () => { if (idx >= length - 1) setIdx(0); setPlaying((p) => !p); } };
}

function SimControls({ unit, onRun, loading }) {
  const [p, setP] = useState({ horizon: 60, load: 1.0, maintenance_at: "" });
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Field label={`Horizon (${unit}s)`}><Input type="number" min={1} max={500} value={p.horizon} onChange={(e) => setP({ ...p, horizon: e.target.value })} /></Field>
      <Field label="Operating load" hint="1.0 = nominal; wear ∝ load²"><Input type="number" step="0.1" min={0.1} max={2} value={p.load} onChange={(e) => setP({ ...p, load: e.target.value })} /></Field>
      <Field label={`Maintenance at (${unit})`} hint="blank = none"><Input type="number" min={1} value={p.maintenance_at} onChange={(e) => setP({ ...p, maintenance_at: e.target.value })} /></Field>
      <div className="flex items-end"><Button loading={loading} onClick={() => onRun({ horizon: Number(p.horizon), load: Number(p.load), maintenance_at: p.maintenance_at === "" ? null : Number(p.maintenance_at) })}>Simulate future</Button></div>
    </div>
  );
}

function MaintenanceHistory({ assetId }) {
  const ev = useApi(() => api.maintenanceEvents(assetId), [assetId]);
  const [form, setForm] = useState({ action: "inspection", performed_by: "", notes: "" });
  const rec = useAction(() => api.recordMaintenance(assetId, form));
  return (
    <Card title="Maintenance history" subtitle="Work orders approved through the agents, plus manually recorded maintenance">
      <Table rows={ev.data} rowKey={(r) => r.id} empty="No maintenance recorded for this asset" columns={[
        { key: "created_at", label: "Recorded", render: (r) => fmt.time(r.created_at) }, { key: "action", label: "Action" },
        { key: "scheduled_for", label: "When" }, { key: "status", label: "Status" }, { key: "approved_by", label: "By" },
        { key: "run_id", label: "Agent run", render: (r) => r.run_id || "manual" },
      ]} />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Input placeholder="action" value={form.action} onChange={(e) => setForm({ ...form, action: e.target.value })} />
        <Input placeholder="performed by" value={form.performed_by} onChange={(e) => setForm({ ...form, performed_by: e.target.value })} />
        <Input placeholder="notes" value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} />
        <Button icon={Wrench} variant="secondary" disabled={!form.performed_by} loading={rec.loading} onClick={async () => { if (await rec.run()) ev.reload(); }}>Record</Button>
      </div>
      <ErrorBox error={rec.error} />
    </Card>
  );
}

function TurbofanTwin({ assetId }) {
  const pb = useApi(() => api.twinPlayback(assetId), [assetId]);
  const state = useApi(() => api.twinState(assetId), [assetId]);
  const sim = useAction((body) => api.twinSimulate(assetId, body));
  const series = pb.data?.series || [];
  const player = usePlayer(series.length);
  const cur = series[player.idx];
  const chart = useMemo(() => {
    const m = series.map((p) => ({ cycle: p.cycle, rul_pred: p.rul_pred, rul_true: p.rul_true }));
    const s = (sim.result?.series || []).map((p) => ({ cycle: p.cycle, sim_median: p.rul_median, sim_band: [p.rul_p10, p.rul_p90], sim_pof: p.cumulative_pof }));
    return [...m, ...s];
  }, [series, sim.result]);
  if (pb.loading) return <Loading label="Loading measured history…" />;
  if (pb.error) return <ErrorBox error={pb.error} code={pb.code} />;
  const st = state.data;
  return (
    <div className="space-y-4">
      {st && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          <Stat label="Current cycle" value={st.current_cycle} />
          <Stat label="Predicted RUL" value={fmt.n(st.rul_pred)} hint={`model ${st.model}`} />
          <Stat label="True RUL (dataset)" value={fmt.n(st.rul_true, 0)} />
          <Stat label="P(failure ≤ 30 cycles)" value={fmt.pct(st.risk.probability_of_failure)} tone={st.risk.risk_level === "low" ? "good" : "bad"} />
          <Stat label="Risk level" value={<RiskBadge level={st.risk.risk_level} />} hint={`urgency: ${st.risk.urgency}`} />
        </div>
      )}
      <Card title="RUL over the engine's life" subtitle="Solid: model prediction on measured data · grey: true RUL · dashed violet: simulated future"
        actions={<><MeasuredTag />{sim.result && <SimulatedTag />}</>}>
        <ChartBox height={300}>
          <ComposedChart data={chart}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="cycle" type="number" domain={["dataMin", "dataMax"]} /><YAxis /><Tooltip {...tooltipStyle} /><Legend />
            <Line dataKey="rul_true" name="true RUL (measured)" stroke="#94a3b8" dot={false} strokeWidth={1.5} />
            <Line dataKey="rul_pred" name="predicted RUL" stroke={PALETTE[0]} dot={false} strokeWidth={2} />
            {sim.result && <Area dataKey="sim_band" name="simulated p10–p90" stroke="none" fill={PALETTE[5]} fillOpacity={0.15} />}
            {sim.result && <Line dataKey="sim_median" name="simulated median RUL" stroke={PALETTE[5]} strokeDasharray="6 4" dot={false} strokeWidth={2} />}
            {cur && <ReferenceLine x={cur.cycle} stroke="#0f172a" strokeDasharray="2 2" />}
            {sim.result && <ReferenceLine x={sim.result.start_cycle} stroke={PALETTE[5]} label={{ value: "now → simulated", fontSize: 11, fill: PALETTE[5], position: "insideTopRight" }} />}
          </ComposedChart>
        </ChartBox>
        <div className="mt-3 flex items-center gap-3">
          <Button variant="secondary" icon={player.playing ? Pause : Play} onClick={player.toggle}>{player.playing ? "Pause" : "Play history"}</Button>
          <input type="range" className="flex-1" min={0} max={Math.max(series.length - 1, 0)} value={player.idx} onChange={(e) => player.setIdx(Number(e.target.value))} />
          <span className="w-28 text-right text-sm tabular-nums">cycle {cur?.cycle}</span>
        </div>
      </Card>
      <div className="grid gap-4 xl:grid-cols-3">
        <Card title={`Equipment state at cycle ${cur?.cycle ?? "—"}`} subtitle="Measured sensor readings (raw units)" actions={<MeasuredTag />}>
          {cur && (
            <>
              <div className="mb-2 grid grid-cols-3 gap-2 text-sm">
                <div><div className="text-xs text-slate-500">Predicted RUL</div><b>{fmt.n(cur.rul_pred)}</b></div>
                <div><div className="text-xs text-slate-500">PoF ≤ 30</div><b>{fmt.pct(cur.pof)}</b></div>
                <div><div className="text-xs text-slate-500">Health</div><b>{fmt.n(cur.health, 0)}</b></div>
              </div>
              <Table maxHeight={300} rows={Object.entries(cur.sensors).map(([k, v]) => ({ k, v }))} rowKey={(r) => r.k} columns={[
                { key: "k", label: "Sensor", render: (r) => `${r.k} · ${pb.data.sensor_names[r.k]}` }, { key: "v", label: "Value", align: "right", render: (r) => fmt.n(r.v, 3) }]} />
            </>
          )}
        </Card>
        <Card title="Simulate future operation" className="xl:col-span-2" actions={<SimulatedTag />}
          subtitle="Monte Carlo from the model's RUL and uncertainty; configurable load and maintenance timing">
          <SimControls unit="cycle" onRun={sim.run} loading={sim.loading} />
          <ErrorBox error={sim.error} />
          {sim.result && (
            <div className="mt-3 space-y-2">
              <div className="grid grid-cols-3 gap-2">
                <Stat label="Cumulative PoF at horizon" value={fmt.pct(sim.result.final_pof)} tone={sim.result.final_pof > 0.3 ? "bad" : "good"} />
                <Stat label="Wear-rate multiplier" value={`${fmt.n(sim.result.load_wear_rate, 2)}×`} />
                <Stat label="Median RUL at horizon" value={fmt.n(sim.result.series.at(-1).rul_median)} />
              </div>
              <Notice>{sim.result.assumptions}</Notice>
            </div>
          )}
        </Card>
      </div>
      <MaintenanceHistory assetId={assetId} />
    </div>
  );
}

function BearingTwin({ assetId }) {
  const pb = useApi(() => api.twinPlayback(assetId, { stride: 2 }), [assetId]);
  const state = useApi(() => api.twinState(assetId), [assetId]);
  const sim = useAction((body) => api.twinSimulate(assetId, body));
  const series = pb.data?.series || [];
  const player = usePlayer(series.length);
  const cur = series[player.idx];
  const chart = useMemo(() => {
    const last = series.at(-1)?.file_index ?? 0;
    const m = series.map((p) => ({ x: p.file_index, score: p.anomaly_score }));
    // simulated steps are hours ahead; IMS snapshots are ~10 min apart, so place them on the same axis
    const s = (sim.result?.series || []).map((p) => ({ x: last + p.hours_ahead * 6, sim_score: p.anomaly_score }));
    return [...m, ...s];
  }, [series, sim.result]);
  if (pb.loading) return <Loading />;
  if (pb.error) return <ErrorBox error={pb.error} code={pb.code} />;
  const st = state.data;
  return (
    <div className="space-y-4">
      {st && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat label="Anomaly score" value={fmt.n(st.anomaly_score, 3)} hint={`threshold ${fmt.n(pb.data.threshold, 3)}`} />
          <Stat label="Time to threshold" value={st.time_to_threshold_hours === null ? "no trend" : `${fmt.n(st.time_to_threshold_hours)} h`} hint="extrapolated (simulated)" />
          <Stat label="PoF" value={fmt.pct(st.risk.probability_of_failure)} />
          <Stat label="Risk" value={<RiskBadge level={st.risk.risk_level} />} hint={st.timestamp} />
        </div>
      )}
      <Card title="Anomaly score over the test" subtitle="Detector score on measured vibration snapshots vs the alarm threshold · x = snapshot index (~10 min apart; simulated hours are placed at 6 snapshots/hour)" actions={<><MeasuredTag />{sim.result && <SimulatedTag />}</>}>
        <ChartBox height={300}>
          <ComposedChart data={chart}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} /><YAxis /><Tooltip {...tooltipStyle} /><Legend />
            <Line dataKey="score" name="anomaly score (measured)" stroke={PALETTE[0]} dot={false} />
            {sim.result && <Line dataKey="sim_score" name="projected score (simulated)" stroke={PALETTE[5]} strokeDasharray="6 4" dot={false} />}
            <ReferenceLine y={pb.data.threshold} stroke="#e11d48" strokeDasharray="4 4" label={{ value: "alarm threshold", fontSize: 11, fill: "#e11d48" }} />
            {cur && <ReferenceLine x={cur.file_index} stroke="#0f172a" strokeDasharray="2 2" />}
          </ComposedChart>
        </ChartBox>
        <div className="mt-3 flex items-center gap-3">
          <Button variant="secondary" icon={player.playing ? Pause : Play} onClick={player.toggle}>{player.playing ? "Pause" : "Play history"}</Button>
          <input type="range" className="flex-1" min={0} max={Math.max(series.length - 1, 0)} value={player.idx} onChange={(e) => player.setIdx(Number(e.target.value))} />
          <span className="w-56 text-right text-xs tabular-nums">{cur?.timestamp}</span>
        </div>
        {cur && <div className="mt-2 grid grid-cols-4 gap-2 text-sm">
          {[["RMS", cur.rms], ["Kurtosis", cur.kurtosis], ["Envelope @ BPFO", cur.env_bpfo], ["Envelope @ BPFI", cur.env_bpfi]].map(([k, v]) =>
            <div key={k}><div className="text-xs text-slate-500">{k}</div><b className="tabular-nums">{fmt.n(v, 4)}</b></div>)}
        </div>}
      </Card>
      <Card title="Simulate future operation" actions={<SimulatedTag />} subtitle="Exponential trend of recent scores, scaled by load²; maintenance resets to the healthy baseline">
        <SimControls unit="hour" onRun={sim.run} loading={sim.loading} />
        <ErrorBox error={sim.error} />
        {sim.result && <div className="mt-2"><Notice>Projected threshold crossing: {sim.result.hours_to_threshold === null ? "not within the horizon" : `${sim.result.hours_to_threshold} h ahead`} (simulated).</Notice></div>}
      </Card>
      <MaintenanceHistory assetId={assetId} />
    </div>
  );
}

export default function DigitalTwin() {
  const [params, setParams] = useSearchParams();
  const asset = params.get("asset") || "FD001-test-unit34";
  const isBearing = asset.startsWith("IMS-");
  const [kind, setKind] = useState(isBearing ? "bearing" : "turbofan");
  const tf = asset.match(/^(FD00\d)-(test|train)-unit(\d+)$/);
  const br = asset.match(/^IMS-(\w+_test)-bearing(\d)$/);
  const go = (id) => setParams({ asset: id });
  return (
    <>
      <PageHeader title="Digital Twin" description="Software twin of each asset: measured history playback with model predictions, current equipment state, simulated future operation and maintenance history. Measured and simulated values are always labelled." />
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <div className="w-40"><Field label="Equipment"><Select value={kind} onChange={(e) => { setKind(e.target.value); go(e.target.value === "bearing" ? "IMS-2nd_test-bearing1" : "FD001-test-unit34"); }}
          options={[{ value: "turbofan", label: "Turbofan engine" }, { value: "bearing", label: "IMS bearing" }]} /></Field></div>
        {kind === "turbofan" ? (
          <>
            <div className="w-28"><Field label="Subset"><Select value={tf?.[1] || "FD001"} onChange={(e) => go(`${e.target.value}-test-unit1`)} options={["FD001", "FD002", "FD003", "FD004"]} /></Field></div>
            <div className="w-28"><Field label="Test engine"><Input type="number" min={1} max={260} value={tf?.[3] || 1} onChange={(e) => go(`${tf?.[1] || "FD001"}-test-unit${e.target.value}`)} /></Field></div>
          </>
        ) : (
          <>
            <div className="w-32"><Field label="Test"><Select value={br?.[1] || "2nd_test"} onChange={(e) => go(`IMS-${e.target.value}-bearing1`)} options={["1st_test", "2nd_test", "3rd_test"]} /></Field></div>
            <div className="w-28"><Field label="Bearing"><Select value={br?.[2] || "1"} onChange={(e) => go(`IMS-${br?.[1] || "2nd_test"}-bearing${e.target.value}`)} options={["1", "2", "3", "4"]} /></Field></div>
          </>
        )}
        <div className="pb-1 font-mono text-sm text-slate-600">{asset}</div>
      </div>
      {isBearing ? <BearingTwin key={asset} assetId={asset} /> : <TurbofanTwin key={asset} assetId={asset} />}
    </>
  );
}
