# Installation and execution guide

## A. Google Colab (primary research environment; data persists in Google Drive)

1. **Build the code bundle** on your computer (once, and again after code changes):
   ```bash
   scripts/make_colab_bundle.sh          # → dist/qadt_code.zip (backend, notebooks, built dashboard, docs)
   ```
2. **Storage and datasets.** Open `notebooks/00_storage_setup.ipynb` in Colab and run all cells.
   - It mounts Drive.
   - The first time, it asks you to upload `qadt_code.zip` and installs it into
     `MyDrive/Quantum_Agentic_Digital_Twin_code/`.
   - It creates the project folders.
   - It downloads and extracts both datasets into `MyDrive/Quantum_Agentic_Digital_Twin/`. C-MAPSS is 12 MB;
     IMS is 1.07 GB, and an interrupted download resumes.
3. **Train and evaluate.** Run `notebooks/02_train_and_evaluate_colab.ipynb`: all RUL models, IMS detectors,
   quantum experiments, explanations and a QAOA run. It's safe to re-run after a disconnect, because deep
   models resume from Drive checkpoints.
4. **Use the dashboard.** Run `notebooks/01_run_backend_colab.ipynb`. It starts FastAPI, which also serves the
   React app, and prints a private link (`https://….trycloudflare.com/?token=…`). Keep the link private; the
   token protects the API.

If the Drive mount drops (`Transport endpoint is not connected`), run
`from backend.storage import remount_drive; remount_drive()` and re-run the interrupted cell.

## B. Local machine

Requirements: Python 3.11–3.12, Node 18+, and optionally Ollama.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt        # exact tested versions: requirements-lock.txt
scripts/fix_macos_openmp.sh                       # macOS only: one OpenMP runtime for PyTorch + XGBoost/LightGBM
(cd frontend && npm install)
```

Choose a storage mode:

| Mode | Environment | Use |
|---|---|---|
| `colab` (default) | Drive mounted at `/content/drive` | Colab |
| `desktop` | `QADT_STORAGE_MODE=desktop QADT_DRIVE_MOUNT=~/Library/CloudStorage/GoogleDrive-<account>` | Google Drive for desktop; data in `My Drive/Quantum_Agentic_Digital_Twin` |
| `development` | `QADT_STORAGE_MODE=development QADT_DRIVE_ROOT=./.dev_storage` | Local testing only; **not** Google Drive, and the UI shows a warning |

Run:

```bash
export QADT_STORAGE_MODE=desktop QADT_DRIVE_MOUNT=~/Library/CloudStorage/GoogleDrive-<account>   # or development mode
.venv/bin/python -m backend.storage download            # datasets → storage (resumable, verified)
.venv/bin/python scripts/train_all_rul.py FD001         # 8 RUL models + comparison report
.venv/bin/python scripts/train_all_anomaly.py           # IMS features + detectors + comparison report
.venv/bin/uvicorn backend.main:app --port 8000          # API (+ built dashboard at http://localhost:8000)
(cd frontend && npm run dev)                            # dashboard with hot reload at http://localhost:5173
.venv/bin/python scripts/generate_docs.py               # refresh documentation/RESULTS.md and API.md
```

To serve the dashboard from FastAPI on one port, build it once with `(cd frontend && npx vite build)`.

## LLM (optional)

- **Ollama:** `ollama serve` and `ollama pull llama3.1`, then set the provider/model in System Settings (or
  `QADT_LLM_MODEL=llama3.1`).
- **OpenAI-compatible API:** set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) in the backend's
  environment and choose the `openai` provider in System Settings. Keys are never stored in settings files.
- **No LLM:** everything works; agent narratives are deterministic, and the chat panel says it needs an LLM.

## Tests

```bash
.venv/bin/python -m pytest backend/tests -q      # 40 tests: storage, pipelines, models, quantum, QUBO, agents, API
(cd frontend && npx vitest run)                  # 17 tests: all 14 pages render, error states, form state
```

Backend tests run on small synthetic files in the real dataset formats (no network). Research numbers come
only from the real datasets.

## Configuration reference

| Variable | Purpose |
|---|---|
| `QADT_STORAGE_MODE` | `colab` · `desktop` · `development` |
| `QADT_DRIVE_MOUNT` / `QADT_DRIVE_ROOT` | Drive mount point / project root inside it |
| `QADT_LOCAL_SCRATCH` | Temporary local folder (never persistent data) |
| `QADT_DATABASE_URL` | Optional PostgreSQL URL (default: SQLite in Drive) |
| `QADT_API_TOKEN` | Require this token on every API call (set automatically by the Colab notebook) |
| `QADT_LLM_PROVIDER` / `QADT_LLM_MODEL` / `QADT_LLM_BASE_URL` | LLM defaults (overridable in System Settings) |
| `QADT_QUANTUM_DEVICE` | PennyLane device for quantum experiments (default `default.qubit`) |
| `VITE_API_URL` / `VITE_API_PROXY` | Frontend: API base URL / dev-server proxy target |
