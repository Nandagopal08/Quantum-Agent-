import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import {
  Activity, Atom, BarChart3, Bot, BrainCircuit, Cpu, Database, FlaskConical, GitCompare, HardDrive, LayoutDashboard,
  Lightbulb, Menu, Network, Settings, Waves, X,
} from "lucide-react";
import { api } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Badge, cx } from "../components/ui";

export const NAV = [
  { to: "/", label: "Dashboard Overview", icon: LayoutDashboard },
  { to: "/datasets", label: "Dataset Management", icon: Database },
  { to: "/analysis", label: "Dataset Analysis", icon: BarChart3 },
  { to: "/training", label: "Model Training", icon: BrainCircuit },
  { to: "/predictive", label: "Predictive Maintenance", icon: Activity },
  { to: "/quantum", label: "Quantum AI Laboratory", icon: Atom },
  { to: "/qseq", label: "Quantum Sequence Lab", icon: Waves },
  { to: "/twin", label: "Digital Twin", icon: Cpu },
  { to: "/agents", label: "Agentic AI Control Center", icon: Bot },
  { to: "/counterfactual", label: "Counterfactual Analysis", icon: GitCompare },
  { to: "/optimization", label: "Maintenance Optimization", icon: Network },
  { to: "/explainability", label: "Explainable AI", icon: Lightbulb },
  { to: "/research", label: "Research Evaluation", icon: FlaskConical },
  { to: "/storage", label: "Storage Management", icon: HardDrive },
  { to: "/settings", label: "System Settings", icon: Settings },
];

function StorageBanner({ health, error }) {
  if (error) {
    return (
      <div className="border-b border-rose-200 bg-rose-50 px-4 py-2 text-sm text-rose-800">
        Backend unreachable: {error}
      </div>
    );
  }
  const s = health?.storage;
  if (!s) return null;
  if (!s.mounted) {
    return (
      <div className="border-b border-rose-200 bg-rose-50 px-4 py-2 text-sm text-rose-800">
        <b>Google Drive is not mounted.</b> Downloads, training and saving are blocked until it is.
        <pre className="mt-1 whitespace-pre-wrap text-xs">{s.instructions}</pre>
      </div>
    );
  }
  if (!s.is_google_drive) {
    return (
      <div className="border-b border-amber-200 bg-amber-50 px-4 py-1.5 text-xs text-amber-900">
        <b>Development storage mode:</b> data is saved to <code>{s.project_root || "a local folder"}</code>, <b>not Google Drive</b>.
        Use Colab or Drive for desktop for persistent research storage.
      </div>
    );
  }
  return null;
}

export default function AppLayout() {
  const [open, setOpen] = useState(false);
  const { data: health, error } = useApi(api.health, []);
  const llm = health?.llm;
  return (
    <div className="flex min-h-screen bg-slate-50">
      <aside className={cx("fixed inset-y-0 left-0 z-30 w-64 transform bg-slate-900 text-slate-300 transition-transform lg:static lg:translate-x-0",
        open ? "translate-x-0" : "-translate-x-full")}>
        <div className="flex h-14 items-center justify-between border-b border-slate-800 px-4">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-500 text-white"><Atom className="h-5 w-5" /></div>
            <div className="leading-tight"><div className="text-sm font-semibold text-white">QA Digital Twin</div>
              <div className="text-[10px] uppercase tracking-wider text-slate-500">Predictive maintenance</div></div>
          </div>
          <button type="button" className="lg:hidden" onClick={() => setOpen(false)} aria-label="Close menu"><X className="h-5 w-5" /></button>
        </div>
        <nav className="h-[calc(100vh-3.5rem)] space-y-0.5 overflow-y-auto p-2">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === "/"} onClick={() => setOpen(false)}
              className={({ isActive }) => cx("flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm",
                isActive ? "bg-slate-800 font-medium text-white" : "hover:bg-slate-800/60 hover:text-white")}>
              <Icon className="h-4 w-4 flex-none" />{label}
            </NavLink>
          ))}
        </nav>
      </aside>
      {open && <div className="fixed inset-0 z-20 bg-slate-900/40 lg:hidden" onClick={() => setOpen(false)} />}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex h-14 items-center gap-3 border-b border-slate-200 bg-white/90 px-4 backdrop-blur">
          <button type="button" className="lg:hidden" onClick={() => setOpen(true)} aria-label="Open menu"><Menu className="h-5 w-5" /></button>
          <div className="truncate text-sm font-medium text-slate-700">Quantum-Enhanced Risk-Aware Counterfactual Agentic Digital Twin</div>
          <div className="ml-auto flex items-center gap-2">
            {health && <Badge tone={health.storage.mounted ? (health.storage.is_google_drive ? "green" : "amber") : "red"}>
              <HardDrive className="h-3 w-3" />{health.storage.is_google_drive ? "Google Drive" : health.storage.mode}</Badge>}
            {llm && <Badge tone={llm.available ? "green" : "slate"} title={llm.detail}>
              <Bot className="h-3 w-3" />{llm.available ? `${llm.provider}: ${llm.model}` : "LLM off (deterministic)"}</Badge>}
          </div>
        </header>
        <StorageBanner health={health} error={error} />
        <main className="mx-auto w-full max-w-[1500px] flex-1 p-4 lg:p-6"><Outlet /></main>
      </div>
    </div>
  );
}
