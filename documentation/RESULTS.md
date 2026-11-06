# Verified results

Generated 2026-11-06 18:26 by `scripts/generate_docs.py` from reports written by actual runs (storage mode: `development`). Re-run the script after new experiments; do not edit numbers by hand.

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

## Quantum sequence models — C-MAPSS FD001 (QLSTM, Quantum Transformer, Quantum CNN-LSTM)

4-qubit circuits × 2 variational layers (PennyLane `default.qubit`); 24-cycle windows of 15 sensors; 1200 training windows; seeds 42, 43, 44 (mean ± sd over 3). Each classical twin is the same architecture with the circuit replaced by a same-width classical layer. Report: `qseq_benchmark_FD001.json` (2026-11-06 18:26). **No quantum advantage is claimed.**

### RUL prediction (complete data, official test engines)

| Model | Test RMSE | MAE | R² | NASA score | Val RMSE | Params (circuit) | Train (s) |
|---|---|---|---|---|---|---|---|
| Transformer (classical twin) | 17.47 ± 0.25 | 13.30 | 0.810 | 667 | 17.76 | 745 | 1 |
| Quantum LSTM | 17.50 ± 0.16 | 11.85 | 0.809 | 1,036 | 17.61 | 649 (96) | 176 |
| LSTM (classical twin) | 18.31 ± 0.22 | 12.20 | 0.791 | 1,637 | 17.98 | 633 | 2 |
| Quantum Transformer | 19.65 ± 0.74 | 14.47 | 0.759 | 1,832 | 17.85 | 761 (96) | 15 |
| CNN-LSTM (classical twin) | 22.51 ± 0.54 | 16.26 | 0.684 | 3,656 | 19.82 | 741 | 1 |
| Quantum CNN-LSTM | 23.75 ± 1.78 | 16.93 | 0.647 | 14,961 | 21.08 | 761 (120) | 91 |

### Imputation RMSE on removed values — sensor outages (block gaps)

Standardised sensor units, held-out test engines, identical seeded masks for every method. Sorted by the middle rate.

| Method | 10% missing | 20% missing | 30% missing | 40% missing | 50% missing |
|---|---|---|---|---|---|
| Transformer (classical twin) | 0.6034 ± 0.0173 | 0.6215 ± 0.0003 | 0.6291 ± 0.0052 | 0.6334 ± 0.0064 | 0.6311 ± 0.0028 |
| Quantum Transformer | 0.6040 ± 0.0137 | 0.6226 ± 0.0047 | 0.6319 ± 0.0039 | 0.6346 ± 0.0065 | 0.6322 ± 0.0043 |
| LSTM (classical twin) | 0.6086 ± 0.0163 | 0.6266 ± 0.0044 | 0.6353 ± 0.0039 | 0.6379 ± 0.0045 | 0.6364 ± 0.0012 |
| Quantum LSTM | 0.6166 ± 0.0180 | 0.6360 ± 0.0013 | 0.6451 ± 0.0077 | 0.6499 ± 0.0072 | 0.6479 ± 0.0061 |
| CNN-LSTM (classical twin) | 0.6486 ± 0.0220 | 0.6699 ± 0.0067 | 0.6786 ± 0.0174 | 0.6839 ± 0.0152 | 0.6821 ± 0.0128 |
| Quantum CNN-LSTM | 0.6737 ± 0.0224 | 0.6964 ± 0.0110 | 0.7024 ± 0.0024 | 0.7086 ± 0.0082 | 0.7067 ± 0.0109 |
| linear interpolation | 0.6942 ± 0.0195 | 0.7197 ± 0.0038 | 0.7262 ± 0.0089 | 0.7304 ± 0.0089 | 0.7291 ± 0.0063 |
| LOCF | 0.7937 ± 0.0218 | 0.7916 ± 0.0158 | 0.8028 ± 0.0038 | 0.8037 ± 0.0271 | 0.8035 ± 0.0155 |
| mean fill | 0.8564 ± 0.0124 | 0.8658 ± 0.0016 | 0.8717 ± 0.0019 | 0.8715 ± 0.0054 | 0.8695 ± 0.0038 |

### Imputation RMSE on removed values — random missingness (MCAR)

Standardised sensor units, held-out test engines, identical seeded masks for every method. Sorted by the middle rate.

| Method | 10% missing | 20% missing | 30% missing | 40% missing | 50% missing |
|---|---|---|---|---|---|
| Transformer (classical twin) | 0.6150 ± 0.0089 | 0.6183 ± 0.0124 | 0.6225 ± 0.0051 | 0.6237 ± 0.0039 | 0.6259 ± 0.0056 |
| Quantum Transformer | 0.6173 ± 0.0132 | 0.6204 ± 0.0149 | 0.6245 ± 0.0073 | 0.6258 ± 0.0056 | 0.6283 ± 0.0037 |
| LSTM (classical twin) | 0.6214 ± 0.0094 | 0.6242 ± 0.0119 | 0.6295 ± 0.0053 | 0.6317 ± 0.0032 | 0.6347 ± 0.0044 |
| Quantum LSTM | 0.6299 ± 0.0087 | 0.6323 ± 0.0131 | 0.6400 ± 0.0060 | 0.6415 ± 0.0051 | 0.6453 ± 0.0094 |
| CNN-LSTM (classical twin) | 0.6591 ± 0.0040 | 0.6609 ± 0.0165 | 0.6668 ± 0.0105 | 0.6697 ± 0.0072 | 0.6753 ± 0.0143 |
| Quantum CNN-LSTM | 0.6783 ± 0.0075 | 0.6812 ± 0.0102 | 0.6897 ± 0.0045 | 0.6916 ± 0.0080 | 0.6991 ± 0.0084 |
| linear interpolation | 0.6946 ± 0.0040 | 0.6963 ± 0.0128 | 0.7083 ± 0.0085 | 0.7100 ± 0.0076 | 0.7190 ± 0.0052 |
| LOCF | 0.7890 ± 0.0047 | 0.7872 ± 0.0067 | 0.7994 ± 0.0087 | 0.8001 ± 0.0066 | 0.7973 ± 0.0107 |
| mean fill | 0.8624 ± 0.0071 | 0.8686 ± 0.0084 | 0.8678 ± 0.0054 | 0.8695 ± 0.0030 | 0.8676 ± 0.0027 |

### Downstream RUL RMSE after imputation — LSTM (classical twin), block

Complete data: 18.35 ± 0.29 cycles.

| Imputer | 10% missing | 20% missing | 30% missing | 40% missing | 50% missing |
|---|---|---|---|---|---|
| Transformer (classical twin) | 18.42 ± 0.83 | 18.58 ± 0.58 | 19.00 ± 0.36 | 19.25 ± 0.35 | 18.31 ± 0.12 |
| Quantum LSTM | 18.51 ± 0.80 | 18.66 ± 0.46 | 18.99 ± 0.42 | 19.33 ± 0.38 | 18.34 ± 0.07 |
| Quantum Transformer | 18.43 ± 0.81 | 18.60 ± 0.53 | 19.05 ± 0.29 | 19.12 ± 0.44 | 18.35 ± 0.01 |
| linear interpolation | 18.57 ± 0.82 | 18.77 ± 0.52 | 19.21 ± 0.45 | 19.69 ± 0.44 | 18.35 ± 0.17 |
| CNN-LSTM (classical twin) | 18.51 ± 0.90 | 18.76 ± 0.59 | 19.18 ± 0.48 | 19.52 ± 0.66 | 18.36 ± 0.18 |
| Quantum CNN-LSTM | 18.54 ± 0.78 | 18.76 ± 0.55 | 19.17 ± 0.45 | 19.63 ± 0.44 | 18.39 ± 0.20 |
| LSTM (classical twin) | 18.45 ± 0.88 | 18.65 ± 0.61 | 18.98 ± 0.46 | 19.18 ± 0.47 | 18.45 ± 0.05 |
| LOCF | 18.47 ± 0.70 | 18.74 ± 0.27 | 19.28 ± 0.31 | 19.93 ± 0.57 | 18.80 ± 0.25 |
| mean fill | 18.89 ± 1.38 | 19.13 ± 0.73 | 20.40 ± 0.55 | 21.54 ± 0.97 | 20.72 ± 0.08 |

### Downstream RUL RMSE after imputation — LSTM (classical twin), mcar

Complete data: 18.35 ± 0.29 cycles.

| Imputer | 10% missing | 20% missing | 30% missing | 40% missing | 50% missing |
|---|---|---|---|---|---|
| Transformer (classical twin) | 18.63 ± 0.48 | 18.46 ± 0.47 | 19.01 ± 0.37 | 18.70 ± 0.96 | 19.77 ± 0.74 |
| Quantum Transformer | 18.47 ± 0.50 | 18.59 ± 0.59 | 18.86 ± 0.41 | 18.67 ± 0.74 | 19.86 ± 0.84 |
| Quantum LSTM | 18.62 ± 0.54 | 18.46 ± 0.49 | 19.03 ± 0.07 | 18.95 ± 0.76 | 19.87 ± 0.94 |
| LSTM (classical twin) | 18.56 ± 0.61 | 18.54 ± 0.60 | 19.00 ± 0.26 | 18.73 ± 0.86 | 19.91 ± 0.84 |
| CNN-LSTM (classical twin) | 18.74 ± 0.61 | 18.48 ± 0.51 | 19.30 ± 0.17 | 18.90 ± 0.61 | 20.05 ± 0.93 |
| Quantum CNN-LSTM | 18.77 ± 0.52 | 18.51 ± 0.43 | 19.22 ± 0.08 | 18.98 ± 0.50 | 20.16 ± 1.02 |
| linear interpolation | 18.83 ± 0.50 | 18.57 ± 0.39 | 19.36 ± 0.02 | 19.02 ± 0.47 | 20.20 ± 1.13 |
| LOCF | 18.86 ± 0.44 | 18.76 ± 0.39 | 19.41 ± 0.09 | 19.17 ± 0.15 | 20.48 ± 0.94 |
| mean fill | 19.60 ± 0.80 | 20.44 ± 1.34 | 20.74 ± 0.62 | 24.55 ± 0.05 | 25.95 ± 0.13 |

### Downstream RUL RMSE after imputation — Transformer (classical twin), block

Complete data: 17.68 cycles.

| Imputer | 10% missing | 20% missing | 30% missing | 40% missing | 50% missing |
|---|---|---|---|---|---|
| Quantum LSTM | 18.25 | 17.52 | 17.95 | 18.38 | 17.15 |
| Transformer (classical twin) | 18.28 | 17.75 | 18.23 | 18.23 | 17.21 |
| Quantum Transformer | 18.28 | 17.50 | 18.15 | 18.38 | 17.28 |
| CNN-LSTM (classical twin) | 18.42 | 17.89 | 18.08 | 18.67 | 17.34 |
| LSTM (classical twin) | 18.28 | 17.42 | 18.21 | 18.32 | 17.48 |
| Quantum CNN-LSTM | 18.28 | 17.75 | 18.12 | 18.75 | 17.48 |
| linear interpolation | 18.36 | 17.74 | 18.09 | 18.75 | 17.78 |
| LOCF | 18.33 | 18.03 | 18.16 | 19.39 | 18.79 |
| mean fill | 17.95 | 20.10 | 22.47 | 25.30 | 28.35 |

### Downstream RUL RMSE after imputation — Transformer (classical twin), mcar

Complete data: 17.68 cycles.

| Imputer | 10% missing | 20% missing | 30% missing | 40% missing | 50% missing |
|---|---|---|---|---|---|
| LSTM (classical twin) | 17.71 | 18.55 | 17.50 | 17.51 | 18.82 |
| Quantum LSTM | 17.66 | 18.32 | 17.43 | 17.32 | 18.91 |
| Quantum Transformer | 17.80 | 18.10 | 17.34 | 17.14 | 19.04 |
| Transformer (classical twin) | 17.78 | 18.53 | 17.24 | 17.52 | 19.04 |
| Quantum CNN-LSTM | 17.54 | 18.66 | 17.46 | 17.53 | 19.07 |
| linear interpolation | 17.62 | 18.62 | 17.45 | 17.47 | 19.11 |
| CNN-LSTM (classical twin) | 17.59 | 18.49 | 17.55 | 17.58 | 19.13 |
| LOCF | 17.81 | 18.54 | 18.39 | 17.32 | 19.97 |
| mean fill | 18.64 | 20.01 | 21.96 | 23.35 | 26.31 |

### Downstream RUL RMSE after imputation — Quantum LSTM, block

Complete data: 17.59 ± 0.06 cycles.

| Imputer | 10% missing | 20% missing | 30% missing | 40% missing | 50% missing |
|---|---|---|---|---|---|
| Quantum CNN-LSTM | 17.83 ± 0.11 | 17.89 ± 0.18 | 18.13 ± 0.01 | 18.44 ± 0.23 | 17.79 ± 0.77 |
| CNN-LSTM (classical twin) | 17.85 ± 0.14 | 17.84 ± 0.20 | 18.13 ± 0.01 | 18.41 ± 0.05 | 17.80 ± 0.75 |
| linear interpolation | 17.87 ± 0.11 | 17.86 ± 0.26 | 18.16 ± 0.00 | 18.45 ± 0.15 | 17.84 ± 0.62 |
| Quantum Transformer | 17.79 ± 0.14 | 17.84 ± 0.13 | 18.11 ± 0.13 | 18.13 ± 0.07 | 17.88 ± 0.71 |
| Quantum LSTM | 17.83 ± 0.16 | 17.91 ± 0.15 | 17.99 ± 0.03 | 18.28 ± 0.05 | 18.00 ± 0.36 |
| Transformer (classical twin) | 17.77 ± 0.19 | 17.78 ± 0.09 | 18.13 ± 0.10 | 18.25 ± 0.17 | 18.04 ± 0.50 |
| LSTM (classical twin) | 17.76 ± 0.24 | 17.84 ± 0.10 | 18.00 ± 0.01 | 18.20 ± 0.01 | 18.17 ± 0.30 |
| LOCF | 17.76 ± 0.05 | 17.85 ± 0.39 | 18.34 ± 0.09 | 18.78 ± 0.18 | 18.26 ± 0.73 |
| mean fill | 18.19 ± 0.93 | 18.33 ± 0.25 | 19.52 ± 0.25 | 20.97 ± 1.32 | 21.21 ± 0.28 |

### Downstream RUL RMSE after imputation — Quantum LSTM, mcar

Complete data: 17.59 ± 0.06 cycles.

| Imputer | 10% missing | 20% missing | 30% missing | 40% missing | 50% missing |
|---|---|---|---|---|---|
| Transformer (classical twin) | 17.89 ± 0.55 | 17.83 ± 0.35 | 18.22 ± 0.22 | 18.07 ± 0.28 | 19.28 ± 0.41 |
| Quantum Transformer | 17.84 ± 0.48 | 17.89 ± 0.18 | 17.98 ± 0.24 | 17.98 ± 0.13 | 19.29 ± 0.73 |
| CNN-LSTM (classical twin) | 18.02 ± 0.59 | 17.83 ± 0.38 | 18.27 ± 0.32 | 18.27 ± 0.24 | 19.33 ± 0.52 |
| LSTM (classical twin) | 17.93 ± 0.64 | 17.88 ± 0.24 | 18.17 ± 0.12 | 18.11 ± 0.25 | 19.36 ± 0.29 |
| Quantum LSTM | 17.97 ± 0.50 | 17.93 ± 0.30 | 18.19 ± 0.35 | 18.27 ± 0.22 | 19.37 ± 0.56 |
| Quantum CNN-LSTM | 18.05 ± 0.56 | 17.86 ± 0.40 | 18.18 ± 0.52 | 18.29 ± 0.21 | 19.38 ± 0.49 |
| linear interpolation | 18.07 ± 0.54 | 17.86 ± 0.44 | 18.35 ± 0.39 | 18.33 ± 0.17 | 19.45 ± 0.68 |
| LOCF | 18.04 ± 0.58 | 17.90 ± 0.39 | 18.36 ± 0.28 | 18.47 ± 0.02 | 19.60 ± 0.58 |
| mean fill | 18.66 ± 1.24 | 20.34 ± 0.86 | 20.84 ± 0.64 | 24.40 ± 0.01 | 26.51 ± 0.64 |

### Downstream RUL RMSE after imputation — Quantum Transformer, block

Complete data: 18.80 cycles.

| Imputer | 10% missing | 20% missing | 30% missing | 40% missing | 50% missing |
|---|---|---|---|---|---|
| Quantum LSTM | 19.31 | 18.88 | 19.03 | 19.47 | 18.15 |
| linear interpolation | 19.29 | 19.05 | 19.03 | 19.80 | 18.44 |
| Quantum Transformer | 19.40 | 18.98 | 19.26 | 19.41 | 18.44 |
| Transformer (classical twin) | 19.34 | 19.02 | 19.32 | 19.40 | 18.56 |
| Quantum CNN-LSTM | 19.29 | 19.02 | 19.17 | 19.84 | 18.57 |
| LSTM (classical twin) | 19.37 | 19.05 | 19.23 | 19.49 | 18.58 |
| CNN-LSTM (classical twin) | 19.26 | 19.12 | 19.04 | 19.94 | 18.59 |
| LOCF | 19.22 | 19.00 | 19.12 | 20.68 | 19.11 |
| mean fill | 19.64 | 19.28 | 19.66 | 22.80 | 24.59 |

### Downstream RUL RMSE after imputation — Quantum Transformer, mcar

Complete data: 18.80 cycles.

| Imputer | 10% missing | 20% missing | 30% missing | 40% missing | 50% missing |
|---|---|---|---|---|---|
| LSTM (classical twin) | 19.71 | 19.31 | 18.88 | 19.07 | 19.54 |
| Quantum LSTM | 19.72 | 19.27 | 18.84 | 18.89 | 19.70 |
| linear interpolation | 19.90 | 19.29 | 19.12 | 19.43 | 19.85 |
| CNN-LSTM (classical twin) | 19.89 | 19.33 | 19.02 | 19.48 | 19.93 |
| Quantum Transformer | 19.66 | 19.07 | 18.95 | 19.16 | 19.96 |
| Quantum CNN-LSTM | 19.85 | 19.37 | 19.10 | 19.55 | 19.98 |
| Transformer (classical twin) | 19.59 | 19.38 | 19.19 | 19.33 | 20.03 |
| LOCF | 19.65 | 19.03 | 18.90 | 19.73 | 20.28 |
| mean fill | 19.95 | 20.18 | 20.74 | 24.25 | 25.63 |

### Quantum vs classical twin

| Architecture | RUL RMSE quantum / twin | Mean imputation RMSE quantum / twin |
|---|---|---|
| lstm | 17.50 / 18.31 | 0.6385 / 0.6286 |
| transformer | 19.65 / 17.47 | 0.6242 / 0.6224 |
| cnn_lstm | 23.75 / 22.51 | 0.6928 / 0.6695 |

### Qiskit cross-check of the trained quantum models

Saved models re-run with every circuit executed by Qiskit's `Statevector` instead of PennyLane.

| Model | Test windows | Qiskit circuit evaluations | Max output difference | Max RUL difference (cycles) |
|---|---|---|---|---|
| Quantum LSTM | 20 | 1,920 | 3.3e-16 | 4.2e-14 |
| Quantum Transformer | 20 | 1,920 | 3.3e-16 | 4.2e-14 |
| Quantum CNN-LSTM | 20 | 1,200 | 4.4e-16 | 5.6e-14 |

## Maintenance scheduling: QAOA vs classical

Costs evaluated with the original cost model; gap relative to the exact MILP (HiGHS) optimum. QAOA's plan is the cheapest feasible bitstring among its most probable outcomes; `P(plan)` is the probability the circuit itself assigned to it.

| Run | Assets | QUBO vars | MILP cost | Greedy gap | Annealing gap | QAOA gap | QAOA p / depth | P(plan) | QAOA time (s) |
|---|---|---|---|---|---|---|---|---|---|
| schedule_20261106_160528 | 5 | 14 | 1,366,044 | 0.00% | 0.00% | 10.98% | 2 / 28 | 0.044% | 1.5 |
| schedule_20261106_181509 | 5 | 14 | 1,366,044 | 0.00% | 0.00% | 10.98% | 2 / 28 | 0.044% | 1.3 |

## Explainability (top drivers)

| Model | Top SHAP features | Top permutation-importance sensors |
|---|---|---|
| catboost | cycle_norm, s9_rm5, s15_rm5, s4_rm5 | cycle_norm (cycle_norm), s9 (Nc core speed), s4 (T50 LPT outlet temp) |
| lstm | n/a (sequence model) | s14 (NRc corrected core speed), s13 (NRf corrected fan speed), s11 (Ps30 HPC static pressure) |
| xgboost | cycle_norm, s4_rm5, s11_rm5, s15_rm5 | cycle_norm (cycle_norm), s11 (Ps30 HPC static pressure), s4 (T50 LPT outlet temp) |
