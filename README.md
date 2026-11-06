# Quantum-Enhanced Risk-Aware Counterfactual Agentic Digital Twin

Explainable, cost-optimised predictive maintenance on NASA C-MAPSS (turbofan engines) and NASA IMS
(bearings). It combines:
- classical and quantum ML;
- a risk engine (risk = PoF × consequence);
- software digital twins;
- counterfactual maintenance scenarios;
- QAOA scheduling against classical optimisers;
- SHAP-based explanations;
- seven LangGraph agents with human approval;
- quantum sequence models (Quantum LSTM, Quantum Transformer, Quantum CNN-LSTM) for RUL and for imputing
  missing sensor data, with a single LangGraph agent that imputes and forecasts.

Everything runs from standalone Python files (`backend/` modules, `scripts/` entry points); there are no notebooks.

It is delivered as a FastAPI backend and a React dashboard, with Google Drive as the permanent store.

| | |
|---|---|
| Frontend | React 18 · Vite · Tailwind CSS · Recharts · React Router · Axios · Lucide (15 pages) |
| Backend | FastAPI (REST + Server-Sent Events) · background jobs · SQLite (PostgreSQL optional) |
| Agents | LangGraph (state, routing, interrupt-based approval) · LangChain (tools, prompts, structured output) · Ollama / OpenAI-compatible |
| ML | scikit-learn · XGBoost · LightGBM · CatBoost · PyTorch · SHAP |
| Quantum | PennyLane (VQC, quantum kernel, hybrid QNN, Quantum LSTM / Transformer / CNN-LSTM, QAOA) · Qiskit (circuit execution and verification) |
| Storage | Google Drive via Colab, Drive for desktop, or an explicitly flagged development folder |

## Start here

- **New here? Follow [HOW_TO_RUN.md](HOW_TO_RUN.md)** for step-by-step instructions (Colab or your own computer).
- **Full setup reference:** [documentation/INSTALLATION_AND_EXECUTION.md](documentation/INSTALLATION_AND_EXECUTION.md). In Colab
  the notebook page is only a terminal for `!python scripts/...`; one entry point runs everything.
- **Quantum sequence models and missing data:** [documentation/QUANTUM_SEQUENCE_MODELS.md](documentation/QUANTUM_SEQUENCE_MODELS.md).
- **Measured results:** [documentation/RESULTS.md](documentation/RESULTS.md), generated from saved run
  reports and never edited by hand.
- **How it works:** [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) ·
  [documentation/METHODOLOGY.md](documentation/METHODOLOGY.md) (with limitations and future work) ·
  [documentation/API.md](documentation/API.md).
- **Project status:** [documentation/FINAL_AUDIT_REPORT.md](documentation/FINAL_AUDIT_REPORT.md) (latest audit: issues,
  fixes, tests, verified vs unverified results) · [documentation/AUDIT_REPORT.md](documentation/AUDIT_REPORT.md) ·
  [documentation/AUDIT.md](documentation/AUDIT.md) · [documentation/CHECKLIST.md](documentation/CHECKLIST.md).
- **Verify the numbers yourself:** `python scripts/verify_results.py` recomputes every quantum-sequence metric from
  the raw predictions; `python scripts/validate_colab_drive.py` checks Google Drive persistence (not yet run on real
  Colab).

## Quick start (local, development storage)

```bash
python3.12 -m venv .venv
.venv/bin/python scripts/setup_environment.py --storage-mode development --drive-root ./.dev_storage
(cd frontend && npm install && npx vite build)
export QADT_STORAGE_MODE=development QADT_DRIVE_ROOT=$PWD/.dev_storage
.venv/bin/python scripts/run_pipeline.py          # datasets → training → quantum → twin → QAOA → agents → evaluation
.venv/bin/python scripts/run_backend.py           # → http://localhost:8000
```

Colab: `!python scripts/setup_environment.py`, `!python scripts/run_pipeline.py`, `!python scripts/run_backend.py --tunnel`
(see [HOW_TO_RUN.md](HOW_TO_RUN.md)).

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
