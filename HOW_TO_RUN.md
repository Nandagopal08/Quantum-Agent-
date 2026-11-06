# How to run the project, step by step

Everything runs from **Python scripts** in `scripts/`. There are no notebooks. In Google Colab, the notebook
page is only a terminal for running those scripts.

There are two ways to run it:

- **Option A: Google Colab + Google Drive** (recommended). Datasets, models and results are saved in Google Drive.
- **Option B: your own computer.** Data is saved in Google Drive for desktop, or in a local test folder.

| Script | What it does | Time (CPU) |
|---|---|---|
| `scripts/setup_environment.py` | Installs dependencies, mounts Drive, creates the folders, checks the stack | 2–5 min |
| `scripts/download_datasets.py` | Downloads, verifies and extracts C-MAPSS (12 MB) and IMS (1.07 GB); resumable | 1–10 min |
| `scripts/preprocess_datasets.py` | C-MAPSS preprocessing and EDA; IMS feature extraction | ~5 min |
| `scripts/train_all_models.py` | 8 RUL models and the IMS anomaly detectors; resumes from checkpoints | ~15 min |
| `scripts/run_quantum_experiments.py` | VQC, quantum kernel, QNN; Quantum LSTM / Transformer / CNN-LSTM with missing data; Qiskit check | 20–70 min |
| `scripts/run_digital_twin.py` | Simulates future operation of engines and bearings | < 1 min |
| `scripts/run_optimization.py` | QAOA vs MILP, greedy and annealing maintenance schedules | < 1 min |
| `scripts/run_agent_workflow.py` | LangGraph agents: the seven-agent maintenance workflow, or the single missing-data agent | < 1 min |
| `scripts/evaluate_all.py` | SHAP and other explanations; writes `RESULTS.md` to `<project>/reports/docs/` (`--publish-docs` → `documentation/`) | ~2 min |
| `scripts/verify_results.py` | Recomputes every quantum-sequence metric (incl. the accuracy table) from the raw predictions and `RUL_FD001.txt`, and re-runs every seed's saved models | ~10 min |
| `scripts/validate_colab_drive.py` | Checks Google Drive persistence for real (two phases around a runtime restart) | ~1 min |
| `scripts/run_tests.py` | Backend (pytest) and frontend (vitest) tests | ~1 min |
| **`scripts/run_pipeline.py`** | **All of the above in order (environment check first, result verification last); skips steps already done** | ~3–4 h (CPU) |
| `scripts/run_backend.py` | Starts the API and dashboard (`--tunnel` for Colab) | — |

Every script has `--help`, and they all accept `--storage-mode`, `--drive-root`, `--drive-mount`, `--log-level`,
`--log-file` and `--summary-json`. Logs are also written to `<project>/reports/logs/<script>.log`.

---

## Option A: Google Colab + Google Drive

### What you need

- A Google account with about **10 GB free** in Google Drive.
- This project folder on your computer.

### Step 1: package the code (on your computer)

```bash
cd "/Users/nandagopal/Desktop/ISI Project"
python scripts/make_colab_bundle.py          # → dist/qadt_code.zip
```

Upload `dist/qadt_code.zip` to **My Drive** (drag it into drive.google.com). Repeat this whenever you change
the code.

### Step 2: run the scripts in Colab

Open <https://colab.research.google.com>, create an empty notebook, and type these commands into cells.
The notebook is only the terminal; all the work happens in the `.py` files.

```python
# 1. mount Google Drive and unpack the code into Drive (first session, or after re-uploading the zip)
from google.colab import drive; drive.mount('/content/drive')
!unzip -oq /content/drive/MyDrive/qadt_code.zip -d /content/drive/MyDrive/Quantum_Agentic_Digital_Twin_code
%cd /content/drive/MyDrive/Quantum_Agentic_Digital_Twin_code
```

```python
# 2. install dependencies and create the project folders in Drive
!python scripts/setup_environment.py
```

```python
# 3. the whole workflow: datasets → training → quantum → twin → QAOA → agents → evaluation
!python scripts/run_pipeline.py
```

If Colab disconnects, run cells 1 and 3 again. The pipeline skips the steps it already finished, and training
continues from the last checkpoint saved in Drive. To run only some steps:

```python
!python scripts/run_pipeline.py --steps sequence,agents
!python scripts/run_quantum_experiments.py --suite sequence --seeds 42,43,44
```

### Step 3: open the dashboard

```python
!python scripts/run_backend.py --tunnel
```

It prints a link like `https://something.trycloudflare.com/?token=abc123…`. Open it in your browser.
**Keep the link private**, because the token protects the app: anyone with the full link has full access. With a
token, the API docs (`/docs`) are switched off, every `/api` call needs the token, and the dashboard removes the
token from the address bar after reading it. The dashboard works while this cell is running.

Optional AI narratives: `!python scripts/run_backend.py --tunnel --ollama-model llama3.2:3b`. Without an LLM,
everything still works and the agents write a standard summary.

---

## Option B: your own computer

### What you need

- Python 3.12, and Node.js 18 or newer (only to rebuild the dashboard)
- About 10 GB of free disk space
- macOS only: `brew install libomp`

### Step 1: install (one time)

```bash
cd "/Users/nandagopal/Desktop/ISI Project"
python3.12 -m venv .venv
.venv/bin/python scripts/setup_environment.py --storage-mode development --drive-root ./.dev_storage
cd frontend && npm install && npx vite build && cd ..
```

`setup_environment.py` installs `requirements.txt` and, on macOS, applies the OpenMP fix (`scripts/fix_macos_openmp.py`).

> If a `.venv` folder is moved after creation, its launchers (`pip`, `uvicorn`, …) keep the old path. The project's
> `.venv` was repaired on 2026-11-06 (launcher paths rewritten, nothing reinstalled). `.venv/bin/python -m pip …`
> always works regardless.

### Step 2: choose where data is stored

Run this in every new terminal, or copy `.env.example` to `.env` and export it:

```bash
export QADT_STORAGE_MODE=development QADT_DRIVE_ROOT="$PWD/.dev_storage"      # local test folder (not Drive)
# or, with Google Drive for desktop:
export QADT_STORAGE_MODE=desktop QADT_DRIVE_MOUNT=~/Library/CloudStorage/GoogleDrive-<your-email>
```

Development mode is clearly marked: the dashboard shows a yellow banner on every page.

### Step 3: run

```bash
.venv/bin/python scripts/run_pipeline.py                    # everything (or the individual scripts above)
.venv/bin/python scripts/run_backend.py                     # → http://localhost:8000
```

Some useful single runs:

```bash
.venv/bin/python scripts/run_quantum_seq.py FD001 --seeds 42,43,44        # quantum sequence models + missing data (~35–80 min)
.venv/bin/python scripts/verify_results.py                                # recompute and check every reported number
.venv/bin/python scripts/run_quantum_experiments.py --suite qiskit        # Qiskit vs PennyLane check of the saved models
.venv/bin/python scripts/run_agent_workflow.py missing-data --asset FD001-test-unit34 --missing-rate 0.4 --pattern block
.venv/bin/python scripts/run_agent_workflow.py maintenance --n-assets 5 --decision approve
.venv/bin/python scripts/generate_docs.py --publish                       # write the verified numbers into documentation/RESULTS.md
.venv/bin/python scripts/run_pipeline.py --profile quick --restart         # 10–15 min pipeline check
```

---

## Using the dashboard

1. **Dataset Management**: check that both datasets show **Ready**.
2. **Model Training**: trained models and their scores.
3. **Predictive Maintenance**: the risk for every engine and bearing.
4. **Quantum AI Laboratory**: VQC, quantum kernel and QNN against classical baselines.
5. **Quantum Sequence Lab**: Quantum LSTM / Transformer / CNN-LSTM against classical twins. It shows the RUL
   error table, the **accuracy table** ("fails within 30 cycles": accuracy, precision, recall, F1, ROC-AUC,
   share within ±10/±20 cycles), paired statistics, the Qiskit cross-check, imputation error against missing
   rate, downstream RUL, and the **missing-data agent**: choose an engine, a missing rate and a pattern, then
   click *Run agents* to see its steps live (data quality → plan → impute → check, retrying the next-best imputer
   when the check rejects one → forecast → report).
6. **Digital Twin**, **Counterfactual Analysis**, **Maintenance Optimization**.
7. **Agentic AI Control Center**: the seven-agent maintenance workflow with human approval.
8. **Research Evaluation** and **System Settings**.

---

## Common problems

| Problem | Fix |
|---|---|
| Colab: `No module named 'backend'` | Run the `%cd /content/drive/MyDrive/Quantum_Agentic_Digital_Twin_code` line (Step 2, cell 1). |
| `Google Drive is not mounted` | Run `from google.colab import drive; drive.mount('/content/drive')`, or `%run scripts/setup_environment.py`. |
| `Transport endpoint is not connected` | Drive disconnected. Run cell 1 again, then re-run the script; it resumes. |
| Exit code 3, "missing prerequisite" | An earlier step hasn't run yet; run `scripts/run_pipeline.py`, which runs the steps in order. |
| Dashboard link stopped working | The `run_backend.py` cell stopped. Run it again for a new link. |
| macOS: crash with `exit code 139` | `python scripts/fix_macos_openmp.py` |
| `Cannot reach the backend` | Start `scripts/run_backend.py`. |

## Validate Google Drive persistence (Colab)

This has **not** been run on real Colab/Drive by the developers; please run it once and keep the report.

```python
# session 1
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/Quantum_Agentic_Digital_Twin_code
!python scripts/validate_colab_drive.py --phase write
```

Then **Runtime ▸ Disconnect and delete runtime**, reconnect (a fresh VM), and:

```python
# session 2
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/Quantum_Agentic_Digital_Twin_code
!python scripts/validate_colab_drive.py --phase verify
```

Both must end with `PASSED`. The reports are in `MyDrive/Quantum_Agentic_Digital_Twin/reports/drive_validation_*.json`.
Also check: `!python scripts/run_pipeline.py` twice in a row — the second run must say "already completed,
skipping" for every finished step.

## What the results mean

The verified numbers, with their source, seed count and evaluation set, are in
[documentation/FINAL_VERIFIED_RESULTS.md](documentation/FINAL_VERIFIED_RESULTS.md), and the caveats are in
[documentation/FINAL_RESEARCH_LIMITATIONS.md](documentation/FINAL_RESEARCH_LIMITATIONS.md). In short, no quantum model
is significantly better than its classical twin, and no quantum advantage is claimed.

`scripts/evaluate_all.py` and `scripts/run_pipeline.py` write generated documents to `<project>/reports/docs/`.
Only `generate_docs.py --publish` (or `--publish-docs`) changes `documentation/RESULTS.md`, so a quick or test run can
never overwrite the published, verified numbers.

## Check that everything works

```bash
.venv/bin/python scripts/run_tests.py        # backend: 86 passed · frontend: 19 passed
```
