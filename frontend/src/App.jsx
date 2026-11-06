import { BrowserRouter, Route, Routes } from "react-router-dom";
import AppLayout from "./layouts/AppLayout";
import Dashboard from "./pages/Dashboard";
import DatasetManagement from "./pages/DatasetManagement";
import DatasetAnalysis from "./pages/DatasetAnalysis";
import ModelTraining from "./pages/ModelTraining";
import PredictiveMaintenance from "./pages/PredictiveMaintenance";
import QuantumLab from "./pages/QuantumLab";
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
  ["/predictive", PredictiveMaintenance], ["/quantum", QuantumLab], ["/twin", DigitalTwin],
  ["/agents", AgentControlCenter], ["/counterfactual", CounterfactualAnalysis],
  ["/optimization", MaintenanceOptimization], ["/explainability", ExplainableAI], ["/research", ResearchEvaluation],
  ["/storage", StorageManagement], ["/settings", SystemSettings],
];

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        {ROUTES.map(([path, Page]) => <Route key={path} path={path} element={<Page />} />)}
        <Route path="*" element={<div className="text-sm text-slate-500">Page not found.</div>} />
      </Route>
    </Routes>
  );
}

export default function App() {
  return <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><AppRoutes /></BrowserRouter>;
}
