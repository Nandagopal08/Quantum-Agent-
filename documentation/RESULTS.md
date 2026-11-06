# Verified results

Generated 2026-11-06 16:29 by `scripts/generate_docs.py` from reports written by actual runs (storage mode: `development`). Re-run the script after new experiments; do not edit numbers by hand.

## Datasets

| Dataset | Archive | SHA-256 | Extracted files | Downloaded |
|---|---|---|---|---|
| NASA C-MAPSS Turbofan Engine Degradation Simulation | 12.4 MB | `c9c5dec12a945a82…` | 14 | 2026-11-06T09:55:02+00:00 |
| NASA IMS Bearings (run-to-failure) | 1,075.6 MB | `21001ac266c465f5…` | 9465 | 2026-11-06T10:00:37+00:00 |

## RUL prediction — C-MAPSS FD001

Official test set: one prediction per engine at its last cycle vs `RUL_FD001.txt` (RUL capped at 125). PoF = P(RUL ≤ 30) from the calibrated Gaussian; coverage = share of test engines inside the 90 % conformal interval. Report: `rul_FD001_comparison.json` (2026-11-06 15:40).

| Model | RMSE | MAE | R² | NASA score | PoF ROC-AUC | 90 % interval coverage | Train time (s) |
|---|---|---|---|---|---|---|---|
| lstm | 13.37 | 10.34 | 0.889 | 307 | 0.991 | 94% | 44 |
| gru | 14.04 | 10.56 | 0.877 | 372 | 0.995 | 87% | 35 |
| transformer | 14.60 | 10.38 | 0.867 | 448 | 0.998 | 86% | 92 |
| cnn_lstm | 17.26 | 11.80 | 0.815 | 934 | 0.975 | 87% | 90 |
| catboost | 17.92 | 12.69 | 0.800 | 643 | 0.971 | 84% | 1 |
| xgboost | 18.32 | 12.81 | 0.791 | 701 | 0.972 | 86% | 1 |
| random_forest | 18.82 | 13.39 | 0.779 | 898 | 0.969 | 85% | 12 |
| lightgbm | 18.82 | 13.38 | 0.779 | 792 | 0.967 | 84% | 2 |

## Bearing anomaly detection — NASA IMS

Labels — pseudo-labels: first 25% healthy, last 10% of failed bearings anomalous (see backend/datasets/ims.py). Lead time = hours from the first sustained alarm (3 consecutive snapshots above the 99th-percentile baseline threshold) to the end of the test. Report: `ims_anomaly_comparison.json` (2026-11-06 15:48).

| Test | Model | ROC-AUC | Precision | Recall | F1 | False-alarm rate | Lead time (h) |
|---|---|---|---|---|---|---|---|
| 2nd_test | isolation_forest | 0.950 | 0.373 | 0.788 | 0.506 | 10.2% | b1: 46.83 |
| 2nd_test | one_class_svm | 0.961 | 0.249 | 1.000 | 0.398 | 23.3% | b1: 72.5 |
| 2nd_test | autoencoder | 0.966 | 0.273 | 1.000 | 0.430 | 20.5% | b1: 71.33 |
| 2nd_test | lstm_autoencoder | 0.974 | 0.283 | 1.000 | 0.441 | 19.6% | b1: 69.33 |
| 1st_test | isolation_forest | 0.998 | 0.831 | 1.000 | 0.908 | 3.4% | b3: 231.35, b4: 156.59 |
| 1st_test | one_class_svm | 0.996 | 0.799 | 0.995 | 0.887 | 4.2% | b3: 151.9, b4: 435.47 |
| 1st_test | autoencoder | 0.984 | 0.758 | 0.977 | 0.853 | 5.2% | b3: 81.72, b4: 438.3 |
| 1st_test | lstm_autoencoder | 0.998 | 0.801 | 1.000 | 0.890 | 4.1% | b3: 409.8, b4: 435.14 |
| 3rd_test | isolation_forest | 0.836 | 0.346 | 0.311 | 0.328 | 4.5% | b3: 42.17 |
| 3rd_test | one_class_svm | 0.846 | 0.359 | 0.585 | 0.445 | 8.0% | b3: 59.33 |
| 3rd_test | autoencoder | 0.826 | 0.331 | 0.622 | 0.432 | 9.7% | b3: 74.85 |
| 3rd_test | lstm_autoencoder | 0.890 | 0.349 | 0.570 | 0.433 | 8.2% | b3: 478.18 |
| 2nd_test | random_forest_supervised | 0.810 | 0.714 | 0.101 | 0.177 | 0.3% | b1: 2.33 |
| 1st_test | random_forest_supervised | 0.835 | 0.123 | 0.088 | 0.103 | 10.4% | b3: 385.23, b4: 91.88 |
| 3rd_test | random_forest_supervised | 0.687 | 0.361 | 0.250 | 0.295 | 3.4% | b3: 27.0 |

`random_forest_supervised` is trained on the other two tests and evaluated on the listed one.

## Quantum vs classical (PennyLane statevector simulator)

Classical baselines use the same PCA-reduced inputs. Single runs; **no quantum advantage is claimed.**

| Run | Task | Qubits | Depth | Gates | Params | Quantum | Best classical (same inputs) | Train (s) | Report |
|---|---|---|---|---|---|---|---|---|---|
| q_qnn_regressor_FD001_4q | rul_regression | 4 | 11 | 20 | 33 | rmse 24.990 | svr_rbf 18.619 | 0.7 | 2026-11-06 15:40 |
| q_qnn_regressor_FD001_6q | rul_regression | 6 | 15 | 30 | 61 | rmse 20.038 | svr_rbf 18.241 | 1.2 | 2026-11-06 16:26 |
| q_quantum_kernel_FD001_4q | failure_within_horizon | 4 | 13 | 28 | 66 | roc_auc 0.962 | svc_rbf 0.988 | 0.5 | 2026-11-06 15:39 |
| q_quantum_kernel_FD001_6q | failure_within_horizon | 6 | 19 | 54 | 160 | roc_auc 0.970 | svc_rbf 0.983 | 0.5 | 2026-11-06 16:26 |
| q_vqc_FD001_4q | failure_within_horizon | 4 | 9 | 20 | 26 | roc_auc 0.967 | logistic_regression 0.982 | 3.9 | 2026-11-06 16:05 |
| q_vqc_FD001_6q | failure_within_horizon | 6 | 12 | 30 | 38 | roc_auc 0.975 | svc_rbf 0.983 | 38.6 | 2026-11-06 16:26 |

## Maintenance scheduling: QAOA vs classical

Costs evaluated with the original cost model; gap relative to the exact MILP (HiGHS) optimum. QAOA's plan is the cheapest feasible bitstring among its most probable outcomes; `P(plan)` is the probability the circuit itself assigned to it.

| Run | Assets | QUBO vars | MILP cost | Greedy gap | Annealing gap | QAOA gap | QAOA p / depth | P(plan) | QAOA time (s) |
|---|---|---|---|---|---|---|---|---|---|
| schedule_20261106_160528 | 5 | 14 | 1,366,044 | 0.00% | 0.00% | 10.98% | 2 / 28 | 0.044% | 1.5 |

## Explainability (top drivers)

| Model | Top SHAP features | Top permutation-importance sensors |
|---|---|---|
| catboost | cycle_norm, s9_rm5, s15_rm5, s4_rm5 | cycle_norm (cycle_norm), s9 (Nc core speed), s4 (T50 LPT outlet temp) |
| lstm | n/a (sequence model) | s14 (NRc corrected core speed), s13 (NRf corrected fan speed), s11 (Ps30 HPC static pressure) |
| xgboost | cycle_norm, s4_rm5, s11_rm5, s15_rm5 | cycle_norm (cycle_norm), s11 (Ps30 HPC static pressure), s4 (T50 LPT outlet temp) |
