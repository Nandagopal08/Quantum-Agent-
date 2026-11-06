# Project audit — 2026-11-06 (second audit)

Scope: every Python module in `backend/` and `scripts/`, the React app in `frontend/src/`, configuration,
requirements, the virtual environment, tests, saved models and experiment outputs in `.dev_storage/`, and the
documentation. Most of the quantum-sequence and agent code was written earlier in this same working session; it
was audited with the same suspicion as the rest.

This file is the *initial* audit (written before any change). The fixes, the tests and their outcomes are in
[FINAL_AUDIT_REPORT.md](FINAL_AUDIT_REPORT.md).

## Baseline before changes

| Check | Result |
|---|---|
| `python -m compileall backend scripts` | clean |
| `python -m pip check` | no broken requirements |
| Backend tests | 59 passed |
| Frontend tests | 18 passed |
| Notebooks (`*.ipynb`) in the project | none |
| Saved models | 23 classical (78 MB), 18 quantum, 2 agentic folders |
| Comparison reports | 10 (incl. `qseq_benchmark_FD001.json`, 3 seeds, 1,863 s) |
| Datasets | C-MAPSS (SHA-256 `c9c5dec1…`, 14 files), IMS (`21001ac2…`, 9,465 files), 7.2 GB |
| Key versions | Python 3.12.2 (Anaconda base), torch 2.15.0, PennyLane 0.45.1, Qiskit 2.6.2, FastAPI 0.162.0, langchain-core 1.7.6 |

## Independently verified (no issue)

- The official test set is exactly the 100 engines in `RUL_FD001.txt`; `rul_true` equals `min(RUL_FD001, 125)`.
- All three seeds (42, 43, 44) were executed for all six models; reported means and sample SDs recompute exactly
  from the per-seed runs.
- NASA (PHM08) score: `exp(−d/13)−1` for early, `exp(d/10)−1` for late predictions — correct.
- No leakage found: validation engines are split by unit; scalers, regime clusters and dropped sensors are fit on
  training units; imputers are trained on training-engine windows, selected on validation engines and evaluated on
  test engines; model selection never reads test data.
- ZIP / 7z extraction rejects members outside the destination; `/api/files/{area}/{name}` confines reads to the
  result area and to image/HTML suffixes.
- Qiskit forward outputs equal PennyLane (max |Δ| ≈ 4×10⁻¹⁶) on the saved models.

## Issues found

**Note on the request text:** the RUL table in the request is labelled *MAE*, but 17.50 / 18.31 / 19.65 / 17.47 /
23.75 / 22.51 are **RMSE** values (the MAEs are 11.85 / 12.20 / 14.47 / 13.30 / 16.93 / 16.26).

| ID | Severity | Area | Issue |
|---|---|---|---|
| C1 | Critical (fixed earlier, re-verify) | Security | Static-file catch-all route served any file via `/%2e%2e/…` without a token. Fixed earlier in this session; must be re-verified with positive and negative tests. |
| H1 | High | Security / robustness | `PUT /api/settings` accepts any keys, types and values: e.g. `optimization.max_qubits = 40` makes QAOA allocate a 2⁴⁰ statevector (memory DoS); string temperatures break the LLM; `llm.base_url` accepts any scheme/host (server-side request target). |
| H2 | High | Security | Dataset upload has no size limit and accepts any valid ZIP in place of the verified official archive (no SHA-256 check), silently replacing it. |
| H3 | High | Reproducibility | Per-engine RUL predictions are saved for the first seed only; seeds 43/44 cannot be recomputed from raw predictions. |
| H4 | High | Evaluation | Downstream-RUL rows use the "best quantum model on validation" chosen *per seed* (q_lstm in 2 seeds, q_transformer in 1), so rows mix models and have n = 1–2 seeds. |
| H5 | High | Evaluation | No paired statistics or confidence intervals; quantum-vs-twin statements rest on 3-seed means. |
| H6 | High | Data safety | Re-runs silently overwrite comparison reports and saved models, and append to the same training-log files (records of different runs mix). |
| H7 | High | Environment | 37 `.venv/bin` launchers (`pip`, `uvicorn`, `pytest`, …) and `activate` point at the old folder `untitled folder 5`; only `python -m …` works. |
| M1 | Medium | Agents | Hidden ground truth of simulated missingness travels through a traced tool result (published on the SSE event stream) and through the shared graph state; isolation is by convention only. |
| M2 | Medium | Agents / API | Uploaded windows (`values`) are not validated: wrong length crashes the Quantum Transformer (positional table is 24 steps), wrong channel count is silently accepted, size is unbounded. |
| M3 | Medium | Agents | Missing-data runs interrupted by a restart stay `running` forever; no recovery path. |
| M4 | Medium | Security | CORS `allow_origins=["*"]`; `/docs` and `/openapi.json` are public even when a token protects `/api` (public tunnel). |
| M5 | Medium | Security | Token check (`hmac.compare_digest` on `str`) raises → HTTP 500 for non-ASCII tokens. |
| M6 | Medium | Security | Error responses include absolute storage paths (Drive layout) in `detail`. |
| M7 | Medium | Evaluation | Test labels are capped at 125 (11/100 engines have a larger true RUL); the convention is not stated next to results and no uncapped metric is reported. |
| M8 | Medium | Quantum | No check that the circuits actually influence outputs (a residual path could let a model ignore them); no gradient-flow diagnostics. |
| M9 | Medium | Quantum | Qiskit check covers forward outputs only: no gradient comparison, parameter-mapping test, edge-case inputs, or mean-absolute-error reporting. |
| M10 | Medium | Evaluation | Summary lacks inference time, circuit depth/qubits per model, uncapped RMSE, CIs; no tool recomputes metrics from raw predictions. |
| M11 | Medium | Storage | Google Drive / Colab persistence never executed for real; no validation script exists. |
| M12 | Medium | Agents tests | No tests for planner fallback on a missing model, failed imputation, invalid/ungrounded LLM narrative, max-retry exhaustion, interruption recovery. |
| L1 | Low | Hygiene | CatBoost writes `catboost_info/` into the project root on every training run. |
| L2 | Low | Code | Unused `_lock` in `backend/agents/runner.py`. |
| L3 | Low | API | Every `ValueError` maps to HTTP 422, which can disguise server bugs as client errors. |
| L4 | Low | Frontend | Quantum Sequence Lab shows a hard-coded runtime hint ("≈15 min per seed"), no circuit metrics, no CIs, no Qiskit status. |
| L5 | Low | Archive safety | No limit on total extracted size (zip bomb) for uploaded archives. |
| L6 | Low | Logs | `--quick` benchmark runs (save=False) still append to the real training logs. |

## Hypotheses to test (not yet issues)

- Why are quantum imputers slightly worse than their twins? Candidates: optimisation (smaller gradients through
  the circuit), capacity (bounded ⟨Z⟩ read-outs), or the model routing around the circuit. To be measured with
  train-loss curves, gradient norms and circuit ablation.
- Are runs bit-for-bit reproducible per seed? To be checked by re-running the benchmark and comparing per-seed
  metrics with the saved report.
