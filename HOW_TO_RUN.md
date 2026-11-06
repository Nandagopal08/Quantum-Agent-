# How to run the project — step by step

There are two ways to run the project:

- **Option A — Google Colab + Google Drive.** Recommended. Your datasets, models and results are saved in
  Google Drive.
- **Option B — your own computer.** Good for testing. Data is saved in a local folder, not Google Drive.

---

## Option A — Google Colab + Google Drive

### What you need

- A Google account with about **10 GB free** in Google Drive (the bearing dataset is large).
- This project folder on your computer.

### Step 1 — Build the code package (on your computer)

Open a terminal in VS Code (**Terminal → New Terminal**) and run:

```bash
cd "/Users/nandagopal/Desktop/untitled folder 5"
scripts/make_colab_bundle.sh
```

This creates **`dist/qadt_code.zip`**. Run it again whenever you change the code.

### Step 2 — Set up storage and download the datasets (Colab)

1. Go to <https://colab.research.google.com>.
2. Click **File → Upload notebook** and choose `notebooks/00_storage_setup.ipynb`.
3. Click **Runtime → Run all**.
4. When asked, **allow access to Google Drive**.
5. When a **Choose Files** button appears, select `dist/qadt_code.zip`. You only do this once; the code is
   saved to `MyDrive/Quantum_Agentic_Digital_Twin_code/`.
6. Wait for the downloads to finish. C-MAPSS is 12 MB and takes seconds. IMS bearings is 1.07 GB and takes a
   few minutes.

When it finishes, both datasets show `ready=True` and a storage summary appears. Everything is now in
`MyDrive/Quantum_Agentic_Digital_Twin/`.

> If a download is interrupted, run the cell again. It continues from where it stopped and never downloads a
> dataset that is already in Drive.

### Step 3 — Train the models (Colab)

1. **File → Upload notebook** → `notebooks/02_train_and_evaluate_colab.ipynb`.
2. **Runtime → Run all** and allow Drive access.

This trains all the models, runs the quantum experiments, the explanations and an optimisation run. It takes
roughly 30–60 minutes. Everything is saved to Drive. If Colab disconnects, run it again: training continues
from the last saved checkpoint.

### Step 4 — Open the dashboard (Colab)

1. **File → Upload notebook** → `notebooks/01_run_backend_colab.ipynb`.
2. **Runtime → Run all** and allow Drive access.
3. At the end, the notebook prints a link like:

   ```
   https://something.trycloudflare.com/?token=abc123...
   ```

4. Open that link in your browser. That is the full dashboard.

> **Keep the link private.** The token in it protects the app. The dashboard works only while this Colab
> notebook is running; if Colab stops, run the notebook again for a new link.

**Optional AI narratives:** in that notebook, set `USE_OLLAMA = True` in the Ollama cell before running.
Without it everything still works; the agents write a standard, non-AI summary.

---

## Option B — Run on your own computer

### What you need

- Python 3.12 (`python3.12 --version`)
- Node.js 18 or newer (`node --version`)
- About 10 GB of free disk space
- macOS only: Homebrew's OpenMP library (`brew install libomp`)

### Step 1 — Install (one time)

```bash
cd "/Users/nandagopal/Desktop/untitled folder 5"
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
scripts/fix_macos_openmp.sh            # macOS only — prevents crashes between PyTorch and XGBoost
cd frontend && npm install && npx vite build && cd ..
```

### Step 2 — Choose where data is stored

Run this in every new terminal window before the commands below:

```bash
export QADT_STORAGE_MODE=development QADT_DRIVE_ROOT="$PWD/.dev_storage"
```

This saves data in the `.dev_storage/` folder. It is **not** Google Drive, and the dashboard shows a yellow
warning banner to remind you.

If you have **Google Drive for desktop** installed, use this instead to save directly into your Drive:

```bash
export QADT_STORAGE_MODE=desktop QADT_DRIVE_MOUNT=~/Library/CloudStorage/GoogleDrive-<your-email>
```

### Step 3 — Download the datasets

```bash
.venv/bin/python -m backend.storage download
```

### Step 4 — Train the models

```bash
.venv/bin/python scripts/train_all_rul.py FD001      # engine models (about 5 minutes)
.venv/bin/python scripts/train_all_anomaly.py        # bearing models (about 5 minutes)
```

### Step 5 — Start the app

```bash
.venv/bin/uvicorn backend.main:app --port 8000
```

Open **<http://localhost:8000>** in your browser. Press `Ctrl+C` in the terminal to stop.

**Optional AI narratives:** in another terminal, run `ollama serve`, then `ollama pull llama3.1`. Then set the
model to `llama3.1` in **System Settings** in the dashboard.

---

## Using the dashboard

A suggested order for a first look:

1. **Dataset Management** — check both datasets show **Ready**.
2. **Model Training** — see the trained models and their scores, or train new ones.
3. **Predictive Maintenance** — the risk of every engine and bearing.
4. **Digital Twin** — replay an engine's history and simulate its future.
5. **Counterfactual Analysis** — compare maintenance options and their costs.
6. **Agentic AI Control Center** — click **Run agents**, watch the steps live, then approve or reject the
   maintenance plan.
7. **Research Evaluation** — all result tables in one place.
8. **System Settings** — change costs and risk thresholds. The defaults are examples only.

---

## Common problems

| Problem | Fix |
|---|---|
| Colab: `No module named 'backend'` | Run Step 2 of Option A first, so the code is installed in Drive. |
| `Google Drive is not mounted` | Run the first cell of the notebook (`drive.mount(...)`) and allow access. |
| `Transport endpoint is not connected` | Drive disconnected. Run `from backend.storage import remount_drive; remount_drive()` and re-run the cell. |
| Dashboard says a model has not been trained | Run Step 3 (Option A) or Step 4 (Option B). |
| Dashboard link stopped working | The Colab notebook stopped. Run `01_run_backend_colab.ipynb` again for a new link. |
| macOS: Python crashes with `exit code 139` | Run `scripts/fix_macos_openmp.sh`. |
| `Cannot reach the backend` in the dashboard | The server is not running. Start it (Option A Step 4 or Option B Step 5). |

---

## Check that everything works (optional)

```bash
.venv/bin/python -m pytest backend/tests -q     # expected: 40 passed
cd frontend && npx vitest run                    # expected: 17 passed
```

More detail: [documentation/INSTALLATION_AND_EXECUTION.md](documentation/INSTALLATION_AND_EXECUTION.md)
