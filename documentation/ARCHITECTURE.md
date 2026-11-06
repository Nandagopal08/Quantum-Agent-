# System architecture

```mermaid
flowchart TD
    UI["React dashboard (Vite · Tailwind · Recharts · React Router · Axios)<br/>14 pages"]
    API["FastAPI backend<br/>REST + Server-Sent Events · token auth (optional)"]
    JOBS["Background jobs<br/>training · features · quantum · optimisation"]
    LG["LangGraph workflow<br/>7 agents + human-approval gate · shared state · MemorySaver"]
    LC["LangChain tools<br/>predict_rul · assess_risk · simulate_digital_twin · compare_maintenance_scenarios<br/>optimize_maintenance_schedule · explain_prediction · check_data_quality"]
    LLM["LLM (Ollama / OpenAI-compatible)<br/>narrative only, numbers validated"]
    ML["Classical ML<br/>RF · XGBoost · LightGBM · CatBoost · LSTM · GRU · CNN-LSTM · Transformer<br/>Isolation Forest · OC-SVM · AE · LSTM-AE"]
    Q["Quantum (PennyLane)<br/>VQC · quantum kernel · hybrid QNN · QAOA"]
    DT["Digital twin · risk engine · counterfactuals · QUBO solvers (MILP · greedy · SA) · XAI"]
    DS["Dataset pipelines<br/>C-MAPSS (RUL) · IMS (vibration features)"]
    ST[("Google Drive<br/>datasets · models · checkpoints · reports · SQLite DB")]

    UI -- "Axios / EventSource" --> API
    API --> JOBS
    API --> LG
    LG --> LC
    LG -. "narrative" .-> LLM
    LC --> ML & Q & DT
    JOBS --> ML & Q & DS
    ML & Q & DT & DS --> ST
    API --> ST
```

## LangGraph workflow

Generated from the compiled graph (`documentation/agent_graph.mmd`):

```mermaid
flowchart LR
    S([start]) --> DM[Data Monitoring]
    DM -- usable data --> PM[Predictive Maintenance]
    DM -- fatal error --> E([end: failed])
    PM -- predictions --> RA[Risk Assessment]
    PM -- fatal error --> E
    RA -- "≥1 asset medium+ risk" --> TW[Digital Twin]
    RA -- all low risk --> DC[Decision Coordination]
    TW --> CF[Counterfactual Reasoning]
    CF -- "≥2 assets need maintenance" --> QO[Quantum Optimization]
    CF -- otherwise --> DC
    QO --> DC
    DC -- "action ≥ approval gate" --> HA[Human Approval<br/>interrupt]
    DC -- no approval needed --> F([end: completed])
    HA --> F2([end: approved / rejected])
```

| Agent | Tools | Output (shared state) | On error |
|---|---|---|---|
| Data Monitoring | `check_data_quality` | `monitoring`, `valid_assets` | fatal → run ends `failed` |
| Predictive Maintenance | `predict_rul` | `predictions` | per asset; fatal if all fail |
| Risk Assessment | `assess_risk` | `risk`, routing | per asset |
| Digital Twin | `simulate_digital_twin` | `twin` (simulated) | non-fatal |
| Counterfactual Reasoning | `compare_maintenance_scenarios` | `counterfactual`, routing | non-fatal |
| Quantum Optimization | `optimize_maintenance_schedule` | `optimization` | non-fatal |
| Decision Coordination | LLM (narrative) | `decision` | non-fatal; deterministic narrative |
| Human Approval | — (`interrupt`) | `approval`, work orders | run stays paused |

## Data flow and storage

```
Google Drive  /content/drive/MyDrive/Quantum_Agentic_Digital_Twin/
├── datasets/raw|processed|features|dataset_reports      ← backend/storage/datasets.py, backend/datasets/*
├── models/classical_ai|quantum_ai|agentic_ai            ← model.pt + model_info.json, checkpoints/, agent run traces
├── digital_twin/simulation_results|equipment_states
├── experiments/training_logs|evaluation_metrics|comparison_reports, qadt.db (SQLite)
├── optimization/qaoa_results|maintenance_schedules
├── explainability/shap|counterfactuals
├── visualizations/  reports/  configuration/app_settings.json
```

All writes are atomic (temp file + `os.replace`); every storage call goes through `require_drive()`, which
raises with mounting instructions instead of falling back to local disk. `/content/qadt_scratch` is used only
for temporary serialisation.

## Code layout

| Path | Contents |
|---|---|
| `backend/main.py` | FastAPI app, middleware (token auth, access log), static dashboard |
| `backend/api/` | Routers: system, datasets, training, assets (prediction/twin/counterfactual/XAI), optimisation, agents |
| `backend/agents/` | `state.py`, `graph.py`, `nodes/`, `runner.py` (background execution, approval), `llm.py`, `chat.py` |
| `backend/langchain_tools/` | LangChain `StructuredTool`s over the real computations |
| `backend/datasets/` | C-MAPSS and IMS pipelines (separate) |
| `backend/classical_models/` | RUL models, anomaly detectors, metrics, PyTorch networks |
| `backend/quantum_models/` | PennyLane experiments |
| `backend/risk/`, `digital_twin/`, `counterfactual/`, `optimization/`, `explainability/` | Domain engines |
| `backend/services/fleet.py` | Asset IDs → data, models, predictions, twins (cached) |
| `backend/storage/` | Google Drive layer (mount, datasets, checkpoints, artefacts, storage page) |
| `backend/db.py`, `jobs.py`, `events.py`, `settings.py` | Database, background jobs, SSE bus, configurable assumptions |
| `frontend/src/` | `pages/` (14), `layouts/`, `components/`, `hooks/`, `services/api.js` |
| `notebooks/` | Colab: storage setup, run the app with a tunnel, train & evaluate everything |
| `scripts/` | Batch training, docs generation, Colab bundle, macOS OpenMP fix |
