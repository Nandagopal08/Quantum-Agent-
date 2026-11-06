import { useState } from "react";
import { Download, Upload } from "lucide-react";
import { api } from "../services/api";
import { useApi, useAction, useJob } from "../hooks/useApi";
import { Badge, Button, Card, ErrorBox, JobPanel, Loading, Notice, PageHeader, ProgressBar, fmt } from "../components/ui";

function DatasetCard({ d, onChanged }) {
  const [job, setJob] = useJob(onChanged);
  const [file, setFile] = useState(null);
  const [upPct, setUpPct] = useState(null);
  const dl = useAction((force) => api.downloadDataset(d.spec.name, force));
  const up = useAction(() => api.uploadDataset(d.spec.name, file, setUpPct));
  const m = d.metadata || {};
  const s = d.status;
  return (
    <Card title={d.spec.title} subtitle={d.spec.version}
      actions={s.ready ? <Badge tone="green">Ready</Badge> : <Badge tone="amber">Not ready</Badge>}>
      <dl className="grid grid-cols-1 gap-x-6 gap-y-1.5 text-sm sm:grid-cols-2">
        {[
          ["Source", <a key="s" className="break-all text-indigo-600" href={d.spec.url}>{d.spec.url}</a>],
          ["Downloaded", m.download_date || "—"],
          ["Archive size", fmt.bytes(m.file_size_bytes)],
          ["SHA-256", <span key="h" className="break-all font-mono text-xs">{m.sha256 || "—"}</span>],
          ["ZIP verified", m.zip_verified ? "yes (CRC of every member)" : "no"],
          ["Extraction", s.extraction_status],
          ["Extracted files", `${s.file_count.toLocaleString()} · ${fmt.bytes(s.size_bytes)}`],
          ["Contents found", d.parts?.length ? d.parts.join(", ") : "—"],
        ].map(([k, v]) => (<div key={k} className="min-w-0"><dt className="text-xs text-slate-500">{k}</dt><dd className="text-slate-800">{v}</dd></div>))}
      </dl>
      {m.nested_archives?.length > 0 && (
        <div className="mt-3 text-xs text-slate-500">Nested archives: {m.nested_archives.map((n) => `${n.archive.split("/").pop()} (${n.status})`).join(", ")}</div>
      )}
      {s.problems.length > 0 && <div className="mt-3"><Notice tone="warn">{s.problems.map((p) => <div key={p}>{p}</div>)}</Notice></div>}
      <div className="mt-3 space-y-1 text-xs text-slate-500">
        <div>Archive: <code className="break-all">{s.zip_path}</code></div>
        <div>Processed: <code className="break-all">{s.processed_path}</code></div>
      </div>
      <div className="mt-4 flex flex-wrap items-end gap-2 border-t border-slate-100 pt-3">
        <Button icon={Download} loading={dl.loading} onClick={async () => { const j = await dl.run(false); if (j) setJob(j); }}>
          {s.ready ? "Verify / re-check" : "Download & extract"}</Button>
        <Button variant="secondary" onClick={async () => { const j = await dl.run(true); if (j) setJob(j); }}>Force re-download</Button>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <input type="file" accept=".zip" className="text-xs" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <Button variant="secondary" icon={Upload} disabled={!file} loading={up.loading}
            onClick={async () => { const j = await up.run(); if (j) setJob(j); setUpPct(null); }}>Upload ZIP</Button>
        </div>
      </div>
      {upPct !== null && <div className="mt-2"><ProgressBar value={upPct} label="Uploading to storage" /></div>}
      <div className="mt-2 space-y-2">
        <ErrorBox error={dl.error || up.error} />
        <JobPanel job={job} />
        {job?.status === "completed" && <Notice tone="ok">Job finished: dataset {job.result?.ready ? "ready" : "not ready"}.</Notice>}
      </div>
    </Card>
  );
}

export default function DatasetManagement() {
  const { data, error, code, loading, reload } = useApi(api.datasets, []);
  return (
    <>
      <PageHeader title="Dataset Management"
        description="Download the NASA datasets straight into Drive (resumable, never duplicated), or upload the ZIP yourself. Archives are CRC-verified before extraction; nested .zip/.7z/.rar archives are unpacked." />
      {loading && !data && <Loading />}
      <ErrorBox error={error} code={code} onRetry={reload} />
      <div className="grid gap-4 xl:grid-cols-2">{data?.map((d) => <DatasetCard key={d.spec.name} d={d} onChanged={reload} />)}</div>
    </>
  );
}
