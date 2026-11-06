# Quantum-Enhanced Risk-Aware Counterfactual Agentic Digital Twin

Explainable, cost-optimised predictive maintenance on NASA C-MAPSS (turbofan engines) and NASA IMS
(bearings). It combines:
- classical and quantum ML;
- a risk engine (risk = PoF × consequence);
- software digital twins;
- counterfactual maintenance scenarios;
- QAOA scheduling against classical optimisers;
- SHAP-based explanations;
- seven LangGraph agents with human approval.

It is delivered as a FastAPI backend and a React dashboard, with Google Drive as the permanent store.

| | |
|---|---|
| Frontend | React 18 · Vite · Tailwind CSS · Recharts · React Router · Axios · Lucide (14 pages) |
| Backend | FastAPI (REST + Server-Sent Events) · background jobs · SQLite (PostgreSQL optional) |
| Agents | LangGraph (state, routing, interrupt-based approval) · LangChain (tools, prompts, structured output) · Ollama / OpenAI-compatible |
| ML | scikit-learn · XGBoost · LightGBM · CatBoost · PyTorch · SHAP |
| Quantum | PennyLane (VQC, quantum kernel, hybrid QNN, QAOA) |
| Storage | Google Drive via Colab, Drive for desktop, or an explicitly flagged development folder |

## Start here

- **New here? Follow [HOW_TO_RUN.md](HOW_TO_RUN.md)** for step-by-step instructions (Colab or your own computer).
- **Full setup reference:** [documentation/INSTALLATION_AND_EXECUTION.md](documentation/INSTALLATION_AND_EXECUTION.md). Colab
  is three notebooks; local setup is a handful of commands.
- **Measured results:** [documentation/RESULTS.md](documentation/RESULTS.md), generated from saved run
  reports and never edited by hand.
- **How it works:** [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) ·
  [documentation/METHODOLOGY.md](documentation/METHODOLOGY.md) (with limitations and future work) ·
  [documentation/API.md](documentation/API.md).
- **Project status:** [documentation/AUDIT.md](documentation/AUDIT.md) ·
  [documentation/CHECKLIST.md](documentation/CHECKLIST.md).

## Quick start (local, development storage)

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/fix_macos_openmp.sh                                   # macOS only
(cd frontend && npm install && npx vite build)
export QADT_STORAGE_MODE=development QADT_DRIVE_ROOT=$PWD/.dev_storage
.venv/bin/python -m backend.storage download                  # NASA datasets (resumable, CRC-verified)
.venv/bin/python scripts/train_all_rul.py FD001 && .venv/bin/python scripts/train_all_anomaly.py
.venv/bin/uvicorn backend.main:app --port 8000                # → http://localhost:8000
```

Development mode stores data locally and the dashboard says so on every page. For research, use Colab or
Google Drive for desktop; see the guide.

## Ground rules the code enforces

- **Storage:** persistent data is written only through the Drive layer. If Drive is missing, the code raises
  with mounting instructions instead of silently using local disk.
- **Numbers:** every number shown or reported comes from a model, simulation or optimiser run. LLM text is
  checked against tool outputs and replaced if it cites an unsupported number.
- **Labels:** simulated values are tagged `simulated` in the API and UI; measured values are tagged `measured`.
- **Approval:** maintenance actions at or above the configured risk gate need an engineer's approval.
- **Claims:** quantum and QAOA results are reported against classical baselines as measured. No advantage is
  claimed.
