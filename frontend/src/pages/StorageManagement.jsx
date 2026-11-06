import { HardDrive } from "lucide-react";
import { api } from "../services/api";
import { useApi, useAction } from "../hooks/useApi";
import { Badge, Button, Card, ErrorBox, Loading, Notice, PageHeader, ProgressBar, Stat, Table, fmt } from "../components/ui";

export default function StorageManagement() {
  const s = useApi(api.storage, []);
  const init = useAction(api.initStorage);
  if (s.loading && !s.data) return <Loading label="Scanning storage…" />;
  if (s.error) {
    return (
      <>
        <PageHeader title="Storage Management" />
        <ErrorBox error={s.error} code={s.code} onRetry={s.reload} />
        <div className="mt-4"><Notice tone="warn">
          <b>How to connect Google Drive</b>
          <pre className="mt-2 whitespace-pre-wrap text-xs">{`In Colab (the notebook is only a terminal for the Python scripts):
    %run scripts/setup_environment.py        # mounts Drive, installs dependencies
    !python scripts/run_backend.py --tunnel  # starts this API, prints a private dashboard link

If the mount dropped ("Transport endpoint is not connected"), re-run:
    %run scripts/setup_environment.py

On a desktop with Google Drive for desktop:
    QADT_STORAGE_MODE=desktop QADT_DRIVE_MOUNT=~/Library/CloudStorage/GoogleDrive-<account>`}</pre>
        </Notice></div>
      </>
    );
  }
  const { info, report: r } = s.data;
  return (
    <>
      <PageHeader title="Storage Management" description="Everything the project persists: datasets, models, checkpoints, results and reports. Google Drive is the permanent store; local scratch is temporary only."
        actions={<><Button variant="secondary" onClick={s.reload}>Rescan</Button><Button variant="secondary" icon={HardDrive} loading={init.loading} onClick={async () => { await init.run(); s.reload(); }}>Create missing folders</Button></>} />
      {!info.is_google_drive && <Notice tone="warn">Development mode: data is stored in a local folder, not Google Drive.</Notice>}
      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Storage mode" value={info.mode} hint={info.is_google_drive ? "Google Drive" : "local folder"} tone={info.is_google_drive ? "good" : "warn"} />
        <Stat label="This project" value={fmt.bytes(r.project_size_bytes)} />
        <Stat label={info.is_google_drive ? "Drive free" : "Disk free"} value={fmt.bytes(r.drive_free_bytes)} hint={`of ${fmt.bytes(r.drive_total_bytes)}`} />
        <Stat label="Checkpointed runs" value={r.checkpoints.length} />
      </div>
      <div className="mt-3"><ProgressBar value={r.drive_used_bytes / r.drive_total_bytes} label="Used space" /></div>
      <Card className="mt-4" title="Paths">
        <dl className="grid gap-2 text-sm sm:grid-cols-3">
          <div><dt className="text-xs text-slate-500">Project root (persistent)</dt><dd className="break-all font-mono text-xs">{r.project_root}</dd></div>
          <div><dt className="text-xs text-slate-500">Drive mount</dt><dd className="break-all font-mono text-xs">{r.drive_mount}</dd></div>
          <div><dt className="text-xs text-slate-500">Local scratch (temporary)</dt><dd className="break-all font-mono text-xs">{r.local_scratch}</dd></div>
        </dl>
      </Card>
      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="Datasets">
          <Table rows={r.datasets} rowKey={(d) => d.name} columns={[
            { key: "title", label: "Dataset" }, { key: "ready", label: "Status", render: (d) => (d.ready ? <Badge tone="green">ready</Badge> : <Badge tone="amber">not ready</Badge>) },
            { key: "zip", label: "Archive", align: "right", render: (d) => fmt.bytes(d.file_size_bytes) },
            { key: "extracted", label: "Extracted", align: "right", render: (d) => `${d.file_count.toLocaleString()} files · ${fmt.bytes(d.size_bytes)}` },
            { key: "download_date", label: "Downloaded" }]} />
        </Card>
        <Card title="Model checkpoints" subtitle="Training resumes from the latest checkpoint after a runtime disconnect">
          <Table rows={r.checkpoints} rowKey={(c) => c.path} empty="No checkpoints yet" maxHeight={300} columns={[
            { key: "run_name", label: "Run" }, { key: "category", label: "Category" },
            { key: "count", label: "Kept", align: "right" }, { key: "latest_step", label: "Latest step", align: "right" },
            { key: "latest_saved_at", label: "Saved" }, { key: "size_bytes", label: "Size", align: "right", render: (c) => fmt.bytes(c.size_bytes) }]} />
        </Card>
      </div>
      <Card className="mt-4" title="Folders and files" subtitle="Most recently modified files per folder">
        <div className="divide-y divide-slate-100">
          {r.folders.map((f) => (
            <details key={f.path} className="py-1.5">
              <summary className="flex cursor-pointer items-center gap-3 text-sm"><code className="flex-1">{f.path}</code>
                <span className="text-xs text-slate-500">{f.file_count.toLocaleString()} files · {fmt.bytes(f.size_bytes)}</span></summary>
              {f.files.length ? <div className="mt-2"><Table rows={f.files} rowKey={(x) => x.name} columns={[
                { key: "name", label: "File" }, { key: "size_bytes", label: "Size", align: "right", render: (x) => fmt.bytes(x.size_bytes) }, { key: "modified", label: "Modified" }]} /></div>
                : <div className="py-2 text-xs text-slate-400">Empty</div>}
            </details>
          ))}
        </div>
      </Card>
    </>
  );
}
