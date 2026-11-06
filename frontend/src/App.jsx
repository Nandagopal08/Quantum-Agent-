import { useEffect, useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { api } from "./services/api";
import AgentOnly from "./pages/AgentOnly";
import AppLayout from "./layouts/AppLayout";
import Dashboard from "./pages/Dashboard";
import DatasetManagement from "./pages/DatasetManagement";
import DatasetAnalysis from "./pages/DatasetAnalysis";
import ModelTraining from "./pages/ModelTraining";
import PredictiveMaintenance from "./pages/PredictiveMaintenance";
import QuantumLab from "./pages/QuantumLab";
import QuantumSequenceLab from "./pages/QuantumSequenceLab";
import DigitalTwin from "./pages/DigitalTwin";
import AgentControlCenter from "./pages/AgentControlCenter";
import CounterfactualAnalysis from "./pages/CounterfactualAnalysis";
import MaintenanceOptimization from "./pages/MaintenanceOptimization";
import ExplainableAI from "./pages/ExplainableAI";
import ResearchEvaluation from "./pages/ResearchEvaluation";
import StorageManagement from "./pages/StorageManagement";
import SystemSettings from "./pages/SystemSettings";

export const ROUTES = [
  ["/", Dashboard], ["/datasets", DatasetManagement], ["/analysis", DatasetAnalysis], ["/training", ModelTraining],
  ["/predictive", PredictiveMaintenance], ["/quantum", QuantumLab], ["/qseq", QuantumSequenceLab], ["/twin", DigitalTwin],
  ["/agents", AgentControlCenter], ["/counterfactual", CounterfactualAnalysis],
  ["/optimization", MaintenanceOptimization], ["/explainability", ExplainableAI], ["/research", ResearchEvaluation],
  ["/storage", StorageManagement], ["/settings", SystemSettings],
];

export function AppRoutes({ agentOnly = false }) {
  if (agentOnly) return <AgentOnly />;          // run_backend.py --agent-only: nothing but the agent
  return (
    <Routes>
      <Route path="/agent" element={<AgentOnly />} />
      <Route element={<AppLayout />}>
        {ROUTES.map(([path, Page]) => <Route key={path} path={path} element={<Page />} />)}
        <Route path="*" element={<div className="text-sm text-slate-500">Page not found.</div>} />
      </Route>
    </Routes>
  );
}

export default function App() {
  // The backend says whether it runs in agent-only mode; if it cannot be reached, show the full dashboard
  // (whose pages explain how to start the backend).
  const [mode, setMode] = useState(null);
  useEffect(() => { api.health().then((h) => setMode(h.ui_mode || "full")).catch(() => setMode("full")); }, []);
  if (mode === null) return <div className="p-6 text-sm text-slate-500">Loading…</div>;
  return <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><AppRoutes agentOnly={mode === "agent"} /></BrowserRouter>;
}
