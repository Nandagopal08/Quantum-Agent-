# Development checklist (status 2026-11-06)

✅ done and verified · ⚠️ done with a stated limitation · ⬜ not done

## Acceptance criteria

| Criterion | Status | Evidence |
|---|---|---|
| Both datasets download and are stored | ✅ | Real downloads: C-MAPSS 12.4 MB, IMS 1.07 GB (resumed after TLS timeouts); CRC-verified; IMS `.7z` + 3 `.rar` unpacked to 9 465 files |
| Dataset preprocessing works | ✅ | C-MAPSS pipeline drops the 6 known constant FD001 sensors, finds 6 regimes in FD004, engine-level split; IMS features on all 9 464 snapshots |
| ≥1 classical baseline per task | ✅ | 8 RUL models (FD001), 4 detectors × 3 IMS tests + cross-test supervised model |
| Quantum experiments execute and are recorded | ✅ | VQC, quantum kernel, QNN at 4 and 6 qubits with depth/gates/params/time vs same-input baselines |
| Predictions and risk assessment | ✅ | Fleet endpoints for 100 FD001 test engines and IMS bearings |
| Digital twin simulations | ✅ | Turbofan + bearing twins; measured playback and labelled simulations |
| Agents execute real tasks | ✅ | LangGraph run: 7 agents, 21 tool calls, routing, interrupt → UI approval → work orders |
| Counterfactual comparisons computed | ✅ | 5 scenarios, Monte Carlo, cost breakdown, explanation |
| QAOA + classical baselines execute | ✅ | MILP, greedy, SA, QAOA; QUBO verified against exhaustive search |
| Explainability outputs | ✅ | SHAP (trees), occlusion (sequence), permutation importance, counterfactual sensor explanations, uncertainty |
| Frontend ↔ backend integrated | ✅ | 14 pages load with zero console errors in headless Chrome; agent/twin/counterfactual flows driven through the UI; no horizontal scroll at 390 px |
| Critical tests pass | ✅ | 40 backend (pytest) + 17 frontend (vitest) |
| Documentation and execution instructions | ✅ | README, installation/execution, architecture, methodology, generated results and API docs |

## Phases

- [x] 1 Audit ([AUDIT.md](AUDIT.md))
- [x] 2 Google Drive storage (Colab / desktop / flagged development modes; atomic writes; resume; checkpoints)
- [x] 3 Dataset pipelines and EDA (separate C-MAPSS and IMS)
- [x] 4 Classical models and metrics (MAE, RMSE, R², NASA score, accuracy, precision, recall, F1, ROC-AUC, false-alarm rate, lead time)
- [x] 5 Quantum-enhanced models ⚠️ simulator only, single seed
- [x] 6 Risk engine (configurable thresholds, human-approval gate)
- [x] 7 LangGraph/LangChain agents ⚠️ paused runs expire on API restart (in-memory checkpointer)
- [x] 8 Digital twins
- [x] 9 Counterfactual analysis
- [x] 10 QAOA scheduling ⚠️ ≤ 16 qubits; relies on classical post-selection
- [x] 11 Explainability
- [x] 12 React dashboard (14 pages) + FastAPI (REST, SSE, background jobs, token auth)
- [x] 13 Integration and tests
- [x] 14 Documentation and verified results

## Not yet done

- [ ] Run the three Colab notebooks end-to-end in a real Colab runtime with Drive (built and validated here in development mode; Drive was unavailable on the build machine)
- [ ] Train and report FD002–FD004
- [ ] Multi-seed statistics for the quantum and QAOA comparisons
- [ ] Quantum hardware run (optional by design)
