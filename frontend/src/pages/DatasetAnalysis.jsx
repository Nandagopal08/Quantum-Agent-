import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api";
import { useApi, useJob } from "../hooks/useApi";
import { Badge, Button, Card, ErrorBox, JobPanel, Loading, MeasuredTag, PageHeader, Select, Stat, Table, cx, fmt } from "../components/ui";
import { ChartBox, PALETTE, tooltipStyle } from "../components/charts";

function Cmapss() {
  const [subset, setSubset] = useState("FD001");
  const eda = useApi(() => api.cmapssEda(subset), [subset]);
  const pipe = useApi(() => api.cmapssPipeline(subset), [subset]);
  const d = eda.data;
  const hist = useMemo(() => d?.life_histogram.map((c, i) => ({ bin: `${Math.round(d.life_bin_edges[i])}`, engines: c })) || [], [d]);
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3"><div className="w-40"><Select value={subset} onChange={(e) => setSubset(e.target.value)} options={["FD001", "FD002", "FD003", "FD004"]} /></div><MeasuredTag /></div>
      {eda.loading && <Loading label="Computing analysis on the raw dataset…" />}
      <ErrorBox error={eda.error} code={eda.code} onRetry={eda.reload} />
      {d && (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
            <Stat label="Training engines" value={d.train_units} hint={`${d.train_rows.toLocaleString()} cycles`} />
            <Stat label="Test engines" value={d.test_units} hint={`${d.test_rows.toLocaleString()} cycles`} />
            <Stat label="Operating conditions" value={d.operating_conditions} />
            <Stat label="Engine life (cycles)" value={`${d.life_cycles.min}–${d.life_cycles.max}`} hint={`mean ${fmt.n(d.life_cycles.mean)}`} />
            <Stat label="Test true RUL" value={`${d.test_rul.min}–${d.test_rul.max}`} hint={`mean ${fmt.n(d.test_rul.mean)}`} />
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            <Card title="Engine-wise life distribution" subtitle="Run-to-failure length of each training engine">
              <ChartBox><BarChart data={hist}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="bin" /><YAxis allowDecimals={false} /><Tooltip {...tooltipStyle} /><Bar dataKey="engines" fill={PALETTE[0]} radius={[3, 3, 0, 0]} /></BarChart></ChartBox>
            </Card>
            <Card title="Mean degradation trajectory" subtitle={`Top RUL-correlated sensors averaged over engines, aligned on cycles to failure`}>
              <ChartBox>
                <LineChart data={d.degradation_trajectory}>
                  <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="rul_true" reversed label={{ value: "cycles to failure", position: "insideBottom", offset: -2, fontSize: 11 }} />
                  {d.top_sensors.map((s) => <YAxis key={s} yAxisId={s} hide domain={["auto", "auto"]} />)}
                  <Tooltip {...tooltipStyle} /><Legend />
                  {d.top_sensors.map((s, i) => <Line key={s} yAxisId={s} type="monotone" dataKey={s} stroke={PALETTE[i]} dot={false} strokeWidth={1.5} />)}
                </LineChart>
              </ChartBox>
            </Card>
          </div>
          <Card title={`Example engine ${d.example_unit.unit} (longest life)`} subtitle="Raw sensor readings over its life — each sensor on its own scale">
            <ChartBox height={220}>
              <LineChart data={d.example_unit.series}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="cycle" />
                {d.top_sensors.map((s) => <YAxis key={s} yAxisId={s} hide domain={["auto", "auto"]} />)}
                <Tooltip {...tooltipStyle} /><Legend />
                {d.top_sensors.map((s, i) => <Line key={s} yAxisId={s} dataKey={s} stroke={PALETTE[i]} dot={false} />)}
              </LineChart>
            </ChartBox>
          </Card>
          <div className="grid gap-4 xl:grid-cols-3">
            <Card title="Sensor analysis" className="xl:col-span-2" subtitle="Pearson correlation with true RUL; non-informative sensors are dropped by the pipeline">
              <Table maxHeight={420} rows={[...d.sensors].sort((a, b) => Math.abs(b.corr_with_rul) - Math.abs(a.corr_with_rul))} rowKey={(r) => r.sensor}
                columns={[
                  { key: "sensor", label: "Sensor" }, { key: "name", label: "Description" },
                  { key: "mean", label: "Mean", align: "right", render: (r) => fmt.n(r.mean, 3) },
                  { key: "std", label: "Std", align: "right", render: (r) => fmt.n(r.std, 4) },
                  { key: "corr_with_rul", label: "Corr. with RUL", align: "right", render: (r) => (
                    <span className={cx(Math.abs(r.corr_with_rul) > 0.5 && "font-semibold")}>{fmt.n(r.corr_with_rul, 3)}</span>) },
                  { key: "used", label: "Used", render: (r) => (pipe.data?.dropped_sensors?.includes(r.sensor) ? <Badge>dropped</Badge> : <Badge tone="green">used</Badge>) },
                ]} />
            </Card>
            <Card title="Preprocessing pipeline" subtitle="Fitted on training engines only (no leakage)">
              {pipe.loading && <Loading />}
              {pipe.data && (
                <div className="space-y-2 text-sm">
                  <div><span className="text-slate-500">RUL label:</span> piecewise linear, cap {pipe.data.config.rul_cap} cycles</div>
                  <div><span className="text-slate-500">Split:</span> {pipe.data.train_units.length} train / {pipe.data.val_units.length} validation engines (by engine, not by row)</div>
                  <div><span className="text-slate-500">Normalisation:</span> per operating condition ({d.operating_conditions})</div>
                  <div><span className="text-slate-500">Features ({pipe.data.feature_cols.length}):</span> sensors + {pipe.data.config.rolling}-cycle rolling mean/std + cycle</div>
                  <div><span className="text-slate-500">Dropped sensors:</span> {pipe.data.dropped_sensors.join(", ") || "none"}</div>
                  <div><span className="text-slate-500">Sequence window:</span> {pipe.data.config.window} cycles</div>
                  <div className="text-xs text-slate-400">Validation engines: {pipe.data.val_units.join(", ")}</div>
                </div>
              )}
              <div className="mt-3 text-xs text-slate-500">Condition counts: {Object.entries(d.condition_counts).map(([k, v]) => `#${k}: ${v}`).join(" · ")}</div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

function Ims() {
  const [test, setTest] = useState("2nd_test");
  const [feature, setFeature] = useState("rms");
  const eda = useApi(() => api.imsEda(test), [test]);
  const [job, setJob] = useJob(() => eda.reload());
  const d = eda.data;
  const series = useMemo(() => {
    if (!d) return [];
    const by = {};
    d.trend.forEach((r) => { (by[r.file_index] ||= { file_index: r.file_index })[`b${r.bearing}`] = r[feature]; });
    return Object.values(by);
  }, [d, feature]);
  const bearings = d ? [...new Set(d.trend.map((r) => r.bearing))] : [];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="w-40"><Select value={test} onChange={(e) => setTest(e.target.value)} options={["1st_test", "2nd_test", "3rd_test"]} /></div>
        <div className="w-48"><Select value={feature} onChange={(e) => setFeature(e.target.value)} options={["rms", "kurtosis", "env_bpfo", "env_bpfi", "spec_centroid"]} /></div>
        <MeasuredTag />
        <Button variant="secondary" onClick={async () => setJob(await api.imsFeatures(test))}>Re-extract features</Button>
      </div>
      <JobPanel job={job} />
      {eda.loading && <Loading />}
      <ErrorBox error={eda.error} code={eda.code} onRetry={eda.reload} />
      {eda.error && /not extracted/.test(eda.error) && <Button onClick={async () => setJob(await api.imsFeatures(test))}>Extract features for {test}</Button>}
      {d && (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="Snapshots" value={d.snapshots.toLocaleString()} hint="1 s @ 20 kHz each" />
            <Stat label="Bearings" value={d.bearings} />
            <Stat label="Duration" value={`${fmt.n(d.duration_hours / 24)} days`} hint={`${d.start} → ${d.end}`} />
            <Stat label="Failed bearings" value={Object.keys(d.failed_bearings).join(", ")} hint={Object.values(d.failed_bearings).join("; ")} tone="bad" />
          </div>
          <Card title={`${feature} per bearing over the test`} subtitle={`Averaged over each bearing's accelerometer channels${d.trend_stride > 1 ? ` · every ${d.trend_stride}th snapshot` : ""}`}>
            <ChartBox height={300}>
              <LineChart data={series}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="file_index" label={{ value: "snapshot", position: "insideBottom", offset: -2, fontSize: 11 }} /><YAxis /><Tooltip {...tooltipStyle} /><Legend />
                {bearings.map((b, i) => <Line key={b} dataKey={`b${b}`} name={`bearing ${b}${d.failed_bearings[b] ? " (failed)" : ""}`} stroke={PALETTE[i]} dot={false} strokeWidth={d.failed_bearings[b] ? 2 : 1} />)}
              </LineChart>
            </ChartBox>
          </Card>
          <Card title="Extracted features" subtitle="Time-domain, statistical, frequency-domain and envelope-spectrum features at the bearing defect frequencies (BPFO 236.4 Hz, BPFI 296.9 Hz, BSF 279.8 Hz, FTF 14.8 Hz)">
            <div className="flex flex-wrap gap-1.5">{d.feature_names.map((f) => <Badge key={f}>{f}</Badge>)}</div>
          </Card>
        </>
      )}
    </div>
  );
}

export default function DatasetAnalysis() {
  const [tab, setTab] = useState("cmapss");
  return (
    <>
      <PageHeader title="Dataset Analysis" description="Exploratory analysis computed from the measured data. The two datasets have different signals and labels, so each has its own pipeline." />
      <div className="mb-4 inline-flex rounded-lg bg-slate-100 p-1 text-sm">
        {[["cmapss", "NASA C-MAPSS turbofan"], ["ims", "NASA IMS bearings"]].map(([k, l]) => (
          <button key={k} type="button" onClick={() => setTab(k)} className={cx("rounded-md px-3 py-1.5", tab === k ? "bg-white font-medium shadow-sm" : "text-slate-600")}>{l}</button>
        ))}
      </div>
      {tab === "cmapss" ? <Cmapss /> : <Ims />}
    </>
  );
}
