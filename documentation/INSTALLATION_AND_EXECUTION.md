# Installation and execution guide

The backend, the training workflows and all experiments are standalone Python files: `backend/` contains the
modules and `scripts/` the entry points. The project contains no Jupyter or Colab notebooks. React (`frontend/`)
is the only non-Python layer. The beginner walk-through is [../HOW_TO_RUN.md](../HOW_TO_RUN.md).

## Entry points

| Script | Backend module(s) | Main options |
|---|---|---|
| `setup_environment.py` | `backend.storage.drive`, `backend.workflows.setup_storage_step` | `--skip-install`, `--check-only` |
| `download_datasets.py` | `backend.storage.datasets` | `--datasets`, `--force` |
| `preprocess_datasets.py` | `backend.datasets.cmapss`, `backend.datasets.ims` | `--subsets`, `--ims-tests`, `--skip-ims` |
| `train_all_models.py` | `backend.classical_models.rul`, `.anomaly` | `--subset`, `--rul-models`, `--epochs`, `--ims-tests`, `--skip-rul`, `--skip-anomaly` |
| `run_quantum_experiments.py` | `backend.quantum_models.qml`, `.qseq`, `.seq_experiments`, `.qiskit_backend` | `--suite qml\|sequence\|qiskit\|all`, `--seeds`, `--epochs`, `--rates`, `--n-qubits` |
| `run_quantum_seq.py` | same (sequence suite only) | `--seeds`, `--models`, `--rates`, `--quick` |
| `run_digital_twin.py` | `backend.digital_twin`, `backend.services.fleet` | `--assets`, `--horizon`, `--load`, `--maintenance-at` |
| `run_optimization.py` | `backend.optimization` | `--assets`, `--n-assets`, `--qaoa-layers` |
| `run_agent_workflow.py` | `backend.agents` (maintenance), `backend.agents.missing_data` | `maintenance --decision approve`, `missing-data --missing-rate --pattern --imputer` |
| `evaluate_all.py` | `backend.explainability`, `scripts/generate_docs.py` | `--explain-models`, `--skip-explain` |
| `run_tests.py` | `backend/tests`, `frontend/src/__tests__` | `--backend-only`, extra pytest args |
| `run_pipeline.py` | all of `backend.workflows` | `--steps`, `--skip`, `--profile full\|quick`, `--restart`, `--keep-going` |
| `run_backend.py` | `backend.main` (FastAPI + built dashboard) | `--port`, `--tunnel`, `--token`, `--ollama-model` |
| `make_colab_bundle.py` | — | `--skip-frontend-build`, `--output` |
| `fix_macos_openmp.py` | — | `--venv` |

Shared behaviour (`scripts/_cli.py`):

- Python `logging` to the console and to `<project>/reports/logs/<script>.log`.
- `--summary-json` writes the step's result.
- Exit codes: `0` success, `1` error, `2` storage unavailable, `3` missing prerequisite (dataset or model), `130` interrupted.
- Every stage writes its outputs through the storage layer (`backend/storage`), never to ad-hoc paths.

Checkpointing and resuming:

- **Pipeline:** `run_pipeline.py` records each step in `configuration/pipeline_state.json` and skips completed
  steps on the next run.
- **Training:** deep RUL models save a checkpoint every epoch (`CheckpointManager`) and resume on re-run.
- **Datasets:** downloads resume from the partial file, and IMS features are cached.

## A. Google Colab (data persists in Google Drive)

Colab is used only as a runtime terminal for the `.py` files.

```bash
python scripts/make_colab_bundle.py           # on your computer → dist/qadt_code.zip; upload it to My Drive
```

```python
from google.colab import drive; drive.mount('/content/drive')        # Colab can mount Drive only from the kernel
!unzip -oq /content/drive/MyDrive/qadt_code.zip -d /content/drive/MyDrive/Quantum_Agentic_Digital_Twin_code
%cd /content/drive/MyDrive/Quantum_Agentic_Digital_Twin_code
!python scripts/setup_environment.py           # or: %run scripts/setup_environment.py (also mounts Drive)
!python scripts/run_pipeline.py                # full workflow; re-run after a disconnect to resume
!python scripts/run_backend.py --tunnel        # private dashboard link
```

Code lives in `MyDrive/Quantum_Agentic_Digital_Twin_code/`. All datasets, models, checkpoints, logs and results
live in `MyDrive/Quantum_Agentic_Digital_Twin/`. `/content` is used only as temporary scratch space.

## B. Local machine

Requirements: Python 3.11–3.12, Node 18+ (only to rebuild the dashboard), and optionally Ollama.

```bash
python3.12 -m venv .venv
.venv/bin/python scripts/setup_environment.py --storage-mode development --drive-root ./.dev_storage
(cd frontend && npm install && npx vite build)
```

| Mode | Environment / flags | Use |
|---|---|---|
| `colab` (default) | Drive mounted at `/content/drive` | Colab |
| `desktop` | `--storage-mode desktop --drive-mount ~/Library/CloudStorage/GoogleDrive-<account>` | Google Drive for desktop; data in `My Drive/Quantum_Agentic_Digital_Twin` |
| `development` | `--storage-mode development --drive-root ./.dev_storage` | Local testing only; **not** Google Drive; the UI shows a warning |

```bash
export QADT_STORAGE_MODE=development QADT_DRIVE_ROOT="$PWD/.dev_storage"      # or pass the flags to each script
.venv/bin/python scripts/run_pipeline.py
.venv/bin/python scripts/run_backend.py                 # API + dashboard at http://localhost:8000
(cd frontend && npm run dev)                            # dashboard with hot reload at http://localhost:5173
```

## LLM (optional)

- **Ollama:** `ollama serve`, then `ollama pull llama3.1`, then set the model in System Settings (or
  `QADT_LLM_MODEL`). In Colab, use `run_backend.py --ollama-model llama3.2:3b`.
- **OpenAI-compatible API:** set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`).
- **No LLM:** everything works, and agent narratives are deterministic. Every LLM narrative is rejected if it
  cites a number that doesn't appear in the tool outputs.

## Tests

```bash
.venv/bin/python scripts/run_tests.py            # 86 backend + 19 frontend tests
```

Backend tests run on small synthetic files in the real dataset formats, in a temporary development-mode
project (no network, no Drive). They cover:

- storage, pipelines and models
- the quantum sequence models (including the Qiskit-vs-PennyLane equivalence check)
- missing-data masks and imputers
- both agent workflows (the seven-agent maintenance workflow and the single missing-data agent, including its retry loop)
- the API and scripts (`--help` works, and the project contains no `.ipynb` files)

## Configuration reference

| Variable | Purpose |
|---|---|
| `QADT_STORAGE_MODE` | `colab` · `desktop` · `development` |
| `QADT_DRIVE_MOUNT` / `QADT_DRIVE_ROOT` | Drive mount point / project root inside it |
| `QADT_LOCAL_SCRATCH` | Temporary local folder (never used for persistent data) |
| `QADT_DATABASE_URL` | Optional PostgreSQL URL (default: SQLite in Drive) |
| `QADT_API_TOKEN` | Required on every API call when set (`run_backend.py --tunnel` generates one) |
| `QADT_LLM_PROVIDER` / `QADT_LLM_MODEL` / `QADT_LLM_BASE_URL` | LLM defaults (can be overridden in System Settings) |
| `QADT_QUANTUM_DEVICE` | PennyLane device for the VQC/kernel/QNN experiments (default `default.qubit`) |
| `VITE_API_URL` / `VITE_API_PROXY` | Frontend: API base URL / dev-server proxy target |
