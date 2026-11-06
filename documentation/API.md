# API reference

Generated from the FastAPI OpenAPI schema by `scripts/generate_docs.py`. Interactive docs: `http://localhost:8000/docs`. When `QADT_API_TOKEN` is set, send `Authorization: Bearer <token>` (or `?token=` for Server-Sent Events).

## system

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health |
| GET | `/api/overview` | Overview |
| GET | `/api/storage` | Storage |
| POST | `/api/storage/init` | Storage Init |
| GET | `/api/settings` | Settings Get |
| PUT | `/api/settings` | Settings Put |
| POST | `/api/settings/reset` | Settings Reset |
| GET | `/api/settings/llm-status` | Settings Llm |

## datasets

| Method | Path | Description |
|---|---|---|
| GET | `/api/datasets` | List Datasets |
| POST | `/api/datasets/{name}/download` | Download |
| POST | `/api/datasets/{name}/upload` | Upload the dataset ZIP from your computer instead of downloading it (same verification applies). |
| GET | `/api/analysis/cmapss/{subset}` | Cmapss Eda |
| GET | `/api/analysis/cmapss/{subset}/pipeline` | Preprocessing decisions (dropped sensors, features, unit split) — recomputed if not saved. |
| GET | `/api/analysis/ims/{test}` | Ims Eda |
| POST | `/api/analysis/ims/{test}/features` | Ims Features |
| DELETE | `/api/datasets/{name}/partial` | Remove a stalled partial download so the next download starts from scratch. |

## training

| Method | Path | Description |
|---|---|---|
| GET | `/api/models` | Models |
| POST | `/api/training/rul` | Train Rul |
| POST | `/api/training/anomaly` | Train Anomaly |
| POST | `/api/quantum/run` | Quantum Run |
| GET | `/api/quantum/experiments` | Quantum Experiments |
| GET | `/api/jobs` | List Jobs |
| GET | `/api/jobs/{job_id}` | Get Job |
| GET | `/api/events/jobs` | Job Events |
| GET | `/api/training/logs/{run_name}` | Training Log |

## assets

| Method | Path | Description |
|---|---|---|
| GET | `/api/fleet/turbofan` | Fleet Turbofan |
| GET | `/api/fleet/bearings` | Fleet Bearings |
| GET | `/api/assets/{asset_id}/prediction` | Prediction |
| GET | `/api/twin/{asset_id}/state` | Twin State |
| GET | `/api/twin/{asset_id}/playback` | Twin Playback |
| POST | `/api/twin/{asset_id}/simulate` | Twin Simulate |
| GET | `/api/maintenance/events` | Maintenance Events |
| POST | `/api/twin/{asset_id}/maintenance` | Record Maintenance |
| POST | `/api/counterfactual` | Counterfactual |
| GET | `/api/explain/asset/{asset_id}` | Explain Asset |
| POST | `/api/explain/model` | Explain Model |
| GET | `/api/explain/reports` | Explain Reports |
| GET | `/api/files/{area}/{name}` | Serve a generated image (e.g. a SHAP plot) from a project result area. |

## optimization

| Method | Path | Description |
|---|---|---|
| POST | `/api/optimization/schedule` | Schedule |
| GET | `/api/optimization/results` | Optimization Results |
| GET | `/api/experiments/comparisons` | Every comparison report produced by actual runs (RUL, IMS anomaly, quantum vs classical). |
| GET | `/api/experiments/comparisons/{name}` | Comparison |

## agents

| Method | Path | Description |
|---|---|---|
| GET | `/api/agents/specs` | Specs |
| POST | `/api/agents/runs` | Start |
| GET | `/api/agents/runs` | Runs |
| GET | `/api/agents/runs/{run_id}` | Run |
| GET | `/api/agents/runs/{run_id}/events` | Run Events |
| POST | `/api/agents/runs/{run_id}/approval` | Approve |
| POST | `/api/agents/ask` | Ask Question |
