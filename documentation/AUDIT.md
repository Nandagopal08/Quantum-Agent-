# Phase 1 — Project audit (2026-11-06)

Scope inspected: every file in the project folder, the local toolchain, network access to the two
NASA datasets, and the storage layer's behaviour against the real C-MAPSS archive.

## What existed before this build

| Component | Status | Problems identified | Required modifications | Priority | Testing status |
|---|---|---|---|---|---|
| Google Drive storage layer (`qadt/storage`) | Complete | Package name/location does not match the required `backend/` structure. Drive check only supports Colab (FUSE mount) — no Drive-for-desktop, no explicit development mode for testing the web app on a machine without Drive. | Move to `backend/storage`, add Drive-for-desktop and an explicit, visibly-flagged development mode. | High | 20 unit tests pass; real C-MAPSS download → verify → nested extract verified end-to-end |
| Dataset download (C-MAPSS) | Complete | — | — | — | Real 12.4 MB archive downloaded, CRC-verified, nested `CMAPSSData.zip` unpacked, 4 train/4 test/4 RUL files found |
| Dataset download (IMS) | Complete, unverified on real data | Real archive is 1.07 GB ZIP → `IMS.7z` → (expected) `.rar` per test. Download from this machine stalls on TLS handshakes to S3. | None in code; verify in Colab. | Medium | `.7z` path unit-tested with py7zr; missing-RAR-tool path tested |
| Colab setup notebook | Complete | Imports `qadt.*` | Update to `backend.*` | Low | Not executed (needs Colab) |
| Preprocessing / EDA / features | **Missing** | — | Build separate C-MAPSS and IMS pipelines | Critical | — |
| Classical models | **Missing** | — | RUL regressors (tree + deep), IMS anomaly detectors | Critical | — |
| Quantum models | **Missing** | — | VQC, quantum kernel, QNN on 4–8 qubits (PennyLane) | High | — |
| Risk engine | **Missing** | — | PoF, health, risk = PoF × consequence, configurable thresholds | Critical | — |
| Digital twin | **Missing** | — | Playback of measured data + clearly labelled simulated futures | High | — |
| Counterfactual analysis | **Missing** | — | Five scenarios with cost model and explanations | High | — |
| QAOA + classical optimisers | **Missing** | — | QUBO, QAOA, greedy, simulated annealing, MILP | High | — |
| Explainability | **Missing** | — | SHAP, permutation importance, counterfactual explanations, uncertainty | High | — |
| Agents (LangGraph) | **Missing** | — | 7 nodes, shared state, conditional routing, human approval, LLM + deterministic fallback | Critical | — |
| FastAPI backend | **Missing** | — | Routers per module, background jobs, SSE events, SQLite | Critical | — |
| React frontend | **Missing** | — | Vite + React + Tailwind + Recharts, 14 pages | Critical | — |

## Environment findings

| Item | Finding | Consequence |
|---|---|---|
| Python | 3.14 default; 3.11/3.12/3.13 available | Build uses a Python 3.12 venv (`.venv`) for library compatibility; Colab uses its own 3.x |
| Node | v25.9 / npm 11 | Fine for Vite + React |
| Google Drive | Not available on this Mac (no Drive for desktop) | Drive exists only in Colab here. The backend therefore supports three explicit storage modes: `colab`, `desktop` (Drive for desktop), and `development` (local folder, banner shown in UI, never silent). Local integration testing uses `development` mode. |
| Ollama | Installed, not running | LLM features use a deterministic fallback until Ollama (or an API key) is configured |
| Network to NASA S3 | Intermittent (TLS handshake timeouts) | C-MAPSS fetched successfully; IMS 1 GB download not reliably possible from this machine |
| LLM API keys | None set | Fallback mode by default |

## Decisions

1. **PennyLane** for all quantum work (VQC, quantum kernel, QNN, QAOA) — one framework, pure-Python
   simulator, no native build issues.
2. **SciPy `milp` (HiGHS)** as the MILP baseline — exact optimum to compare QAOA and heuristics against.
3. **SQLite** stores job/experiment/agent-run records; it lives in Drive under `experiments/` with
   rollback journaling (WAL is unsafe on Drive's FUSE mount). PostgreSQL via `QADT_DATABASE_URL` is
   optional.
4. The LLM never produces numbers. Every numeric field in an agent's output comes from a tool call;
   the LLM only writes the narrative and is validated against the tool results.

## Status after the build (2026-11-06)

| Component | Status | Problems found and fixed during the build | Testing status |
|---|---|---|---|
| Storage layer | Complete | Moved to `backend/storage`; added desktop and development modes; storage report crashed outside Colab (fixed) | 20 unit tests + real downloads |
| Colab notebook | Fixed | The original run failed with `No module named 'qadt'` because the code was never copied to Drive; the notebooks now detect this and install the uploaded bundle | Structure checked; not yet run in Colab |
| Datasets | Complete | Nested-archive copies wasted Drive space (now removed after unpacking) | Real C-MAPSS + IMS |
| Classical models | Complete | macOS segfault/deadlock from two OpenMP runtimes (fix script); PyTorch nets were not picklable (moved to module level) | Real-data runs + tests |
| Quantum | Complete | PennyLane 0.45 specs API change; deprecated `SVC(probability=True)` replaced by Platt scaling | Real-data runs + test |
| Counterfactuals / twin | Complete | Clipped normal put ~24% probability on "already failed now" (now truncated normal) | Tests |
| Optimisation | Complete | Over-large penalty weight drowned the objective (QAOA gap 264% → 0–11%); QUBO verified by exhaustive search | Tests |
| Agents | Complete | Interrupts were caught as errors; LLM timeout from oversized context; validator false positives on asset IDs; tool args unconstrained; critical asset silently dropped by capacity (now escalated) | Tests + live Ollama runs |
| Frontend | Complete | All forms submitted stale initial values (`useAction` bug); Settings page spun forever on errors; phone-width overflow | Headless Chrome on 14 pages + 17 vitest tests |
