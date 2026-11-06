import { AlertTriangle, CheckCircle2, Info, Loader2, XCircle } from "lucide-react";

export const cx = (...c) => c.filter(Boolean).join(" ");

export function Card({ title, subtitle, actions, children, className, bodyClass }) {
  return (
    <section className={cx("min-w-0 rounded-xl border border-slate-200 bg-white shadow-sm", className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-2 border-b border-slate-100 px-4 py-3">
          <div>
            {title && <h2 className="text-sm font-semibold text-slate-800">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cx("p-4", bodyClass)}>{children}</div>
    </section>
  );
}

export function PageHeader({ title, description, actions }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-slate-500">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

export function Stat({ label, value, hint, tone = "default", icon: Icon }) {
  const tones = { default: "text-slate-900", good: "text-emerald-700", bad: "text-rose-700", warn: "text-amber-700" };
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between text-xs font-medium text-slate-500">
        <span>{label}</span>{Icon && <Icon className="h-4 w-4 text-slate-400" />}
      </div>
      <div className={cx("mt-1 text-2xl font-semibold tabular-nums", tones[tone])}>{value ?? "—"}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

const RISK = {
  low: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  medium: "bg-amber-50 text-amber-700 ring-amber-200",
  high: "bg-orange-50 text-orange-700 ring-orange-200",
  critical: "bg-rose-50 text-rose-700 ring-rose-200",
};
export const RISK_COLORS = { low: "#10b981", medium: "#f59e0b", high: "#f97316", critical: "#e11d48" };

export function Badge({ children, tone = "slate", className }) {
  const tones = {
    slate: "bg-slate-100 text-slate-700 ring-slate-200", blue: "bg-blue-50 text-blue-700 ring-blue-200",
    green: "bg-emerald-50 text-emerald-700 ring-emerald-200", red: "bg-rose-50 text-rose-700 ring-rose-200",
    amber: "bg-amber-50 text-amber-700 ring-amber-200", violet: "bg-violet-50 text-violet-700 ring-violet-200",
  };
  return <span className={cx("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset", tones[tone], className)}>{children}</span>;
}

export const RiskBadge = ({ level }) => (
  <span className={cx("inline-flex rounded-full px-2 py-0.5 text-xs font-semibold capitalize ring-1 ring-inset", RISK[level] || RISK.low)}>{level}</span>
);

export const SimulatedTag = () => <Badge tone="violet">Simulated</Badge>;
export const MeasuredTag = () => <Badge tone="blue">Measured</Badge>;

export const StatusBadge = ({ status }) => {
  const map = { completed: "green", approved: "green", running: "blue", queued: "slate", resuming: "blue",
    awaiting_approval: "amber", failed: "red", rejected: "red", expired: "slate", interrupted: "amber" };
  return <Badge tone={map[status] || "slate"}>{String(status).replace(/_/g, " ")}</Badge>;
};

export function Button({ children, variant = "primary", loading, className, icon: Icon, ...props }) {
  const v = {
    primary: "bg-indigo-600 text-white hover:bg-indigo-700 disabled:bg-indigo-300",
    secondary: "bg-white text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 disabled:text-slate-400",
    danger: "bg-rose-600 text-white hover:bg-rose-700 disabled:bg-rose-300",
    success: "bg-emerald-600 text-white hover:bg-emerald-700 disabled:bg-emerald-300",
  };
  return (
    <button type="button" className={cx("inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed", v[variant], className)}
      disabled={loading || props.disabled} {...props}>
      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : Icon ? <Icon className="h-4 w-4" /> : null}
      {children}
    </button>
  );
}

export function Field({ label, hint, children }) {
  return (
    <label className="block text-xs font-medium text-slate-600">
      <span>{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <span className="mt-0.5 block font-normal text-slate-400">{hint}</span>}
    </label>
  );
}

const inputCls = "w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500";
export const Input = (props) => <input className={inputCls} {...props} />;
export const Select = ({ options, ...props }) => (
  <select className={inputCls} {...props}>
    {options.map((o) => (typeof o === "string"
      ? <option key={o} value={o}>{o}</option>
      : <option key={o.value} value={o.value}>{o.label}</option>))}
  </select>
);

export function Loading({ label = "Loading…" }) {
  return <div className="flex items-center gap-2 py-6 text-sm text-slate-500"><Loader2 className="h-4 w-4 animate-spin" />{label}</div>;
}

export function ErrorBox({ error, code, onRetry }) {
  if (!error) return null;
  const hint = {
    drive_not_mounted: "Mount Google Drive (see Storage Management) and reload.",
    model_not_trained: "Train a model on the Model Training page first.",
    dataset_not_ready: "Download and extract the dataset on the Dataset Management page.",
  }[code];
  return (
    <div className="flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
      <XCircle className="mt-0.5 h-4 w-4 flex-none" />
      <div className="min-w-0 flex-1">
        <div className="whitespace-pre-wrap break-words">{error}</div>
        {hint && <div className="mt-1 font-medium">{hint}</div>}
      </div>
      {onRetry && <Button variant="secondary" onClick={onRetry}>Retry</Button>}
    </div>
  );
}

export function Notice({ tone = "info", children }) {
  const t = {
    info: ["border-blue-200 bg-blue-50 text-blue-800", Info],
    warn: ["border-amber-200 bg-amber-50 text-amber-900", AlertTriangle],
    ok: ["border-emerald-200 bg-emerald-50 text-emerald-800", CheckCircle2],
  }[tone];
  const Icon = t[1];
  return <div className={cx("flex items-start gap-2 rounded-lg border p-3 text-sm", t[0])}><Icon className="mt-0.5 h-4 w-4 flex-none" /><div className="min-w-0">{children}</div></div>;
}

export function Empty({ children }) {
  return <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">{children}</div>;
}

export function Table({ columns, rows, empty = "No data", rowKey, onRowClick, selectedKey, maxHeight }) {
  if (!rows?.length) return <Empty>{empty}</Empty>;
  return (
    <div className="overflow-auto rounded-lg border border-slate-200" style={maxHeight ? { maxHeight } : undefined}>
      <table className="min-w-full divide-y divide-slate-200 text-sm">
        <thead className="sticky top-0 bg-slate-50">
          <tr>{columns.map((c) => <th key={c.key} className={cx("whitespace-nowrap px-3 py-2 text-left text-xs font-semibold text-slate-600", c.align === "right" && "text-right")}>{c.label}</th>)}</tr>
        </thead>
        <tbody className="divide-y divide-slate-100 bg-white">
          {rows.map((r, i) => {
            const k = rowKey ? rowKey(r) : i;
            return (
              <tr key={k} onClick={onRowClick ? () => onRowClick(r) : undefined}
                className={cx(onRowClick && "cursor-pointer hover:bg-indigo-50/50", selectedKey === k && "bg-indigo-50")}>
                {columns.map((c) => (
                  <td key={c.key} className={cx("whitespace-nowrap px-3 py-1.5 text-slate-700", c.align === "right" && "text-right tabular-nums")}>
                    {c.render ? c.render(r) : r[c.key] ?? "—"}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function ProgressBar({ value, label }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-slate-500"><span className="truncate">{label}</span><span className="tabular-nums">{pct}%</span></div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-indigo-500 transition-all" style={{ width: `${pct}%` }} /></div>
    </div>
  );
}

export function JobPanel({ job }) {
  if (!job) return null;
  return (
    <div className="space-y-2 rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-center justify-between text-xs"><span className="font-mono text-slate-500">job {job.id} · {job.type}</span><StatusBadge status={job.status} /></div>
      {!["completed", "failed"].includes(job.status) && <ProgressBar value={job.progress} label={job.message} />}
      {job.status === "failed" && <ErrorBox error={job.message} />}
    </div>
  );
}

export const fmt = {
  n: (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d })),
  int: (v) => (v === null || v === undefined ? "—" : Math.round(v).toLocaleString()),
  pct: (v, d = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`),
  money: (v, cur = "USD") => (v === null || v === undefined ? "—" : `${Math.round(v).toLocaleString()} ${cur}`),
  bytes: (b) => {
    if (b === null || b === undefined) return "—";
    const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; let x = b;
    while (x >= 1024 && i < u.length - 1) { x /= 1024; i += 1; }
    return `${x.toFixed(i ? 1 : 0)} ${u[i]}`;
  },
  time: (t) => (t ? new Date(t * 1000).toLocaleString() : "—"),
};
