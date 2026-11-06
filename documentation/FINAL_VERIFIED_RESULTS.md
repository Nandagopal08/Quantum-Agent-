# Final verified results — quantum sequence models on NASA C-MAPSS FD001

Status: **verified**. Every number below was recomputed independently from raw predictions and by re-running the saved
models of every seed (see [FINAL_REPRODUCIBILITY_REPORT.md](FINAL_REPRODUCIBILITY_REPORT.md)). The tables were generated
from the saved report, not typed by hand. **No quantum advantage is claimed.**

## Provenance (applies to every table unless stated)

| Field | Value |
|---|---|
| Source experiment | `scripts/run_quantum_seq.py FD001 --seeds 42,43,44`, run `20261106_185135` → `experiments/comparison_reports/qseq_benchmark_FD001.json` + `qseq_raw_FD001.json` (storage `.dev_storage`, development mode — not Google Drive) |
| Seeds | 3 (42, 43, 44); values are mean ± sample SD over seeds |
| Models | Quantum LSTM, Quantum Transformer, Quantum CNN-LSTM (4 qubits, 2 variational layers, PennyLane `default.qubit`) and same-size classical twins |
| Training data | 1200 windows (24 cycles × 15 sensors) from FD001 training engines; model selection on 400 windows from held-out validation engines |
| RUL evaluation set | the **100 official FD001 test engines** (last window each) vs `RUL_FD001.txt`; RUL capped at 125 (11 engines exceed it); "uncapped" uses the raw values |
| Imputation evaluation set | 400 windows from 93 test engines; values removed with seeded masks identical for every method; RMSE on removed values only, standardised units |
| Downstream evaluation set | the 100 official test-engine windows, masked, imputed by each method, forecast by each RUL model |
| Statistics | paired; 95 % bootstrap over engines (2000 resamples), seed-level paired t, Wilcoxon signed-rank with Holm adjustment; *significant* only if Holm p < 0.05 **and** both intervals exclude 0 |

## 1. RUL prediction on complete data (100 official test engines)

| Model | RMSE | MAE | R² | NASA score | RMSE uncapped | RMSE, circuits ablated | Params (circuit) | Circuit | Train s | Infer ms (100 engines) |
|---|---|---|---|---|---|---|---|---|---|---|
| Transformer twin | 17.47 ± 0.25 | 13.30 ± 0.12 | 0.810 ± 0.005 | 667 ± 40 | 18.74 ± 0.19 | 27.52 ± 3.16 | 745 | — | 1 | 1 |
| Quantum LSTM | 17.50 ± 0.16 | 11.85 ± 0.14 | 0.809 ± 0.003 | 1,036 ± 176 | 18.47 ± 0.24 | 42.09 ± 1.18 | 649 (96) | 4 qubits × 2 layers, depth 9, 20 gates, 4 circuits | 166 | 250 |
| LSTM twin | 18.31 ± 0.22 | 12.20 ± 0.17 | 0.791 ± 0.005 | 1,637 ± 166 | 19.27 ± 0.18 | 41.71 ± 0.49 | 633 | — | 3 | 3 |
| Quantum Transformer | 19.65 ± 0.74 | 14.47 ± 0.61 | 0.759 ± 0.018 | 1,832 ± 708 | 20.63 ± 0.71 | 24.92 ± 0.49 | 761 (96) | 4 qubits × 2 layers, depth 9, 20 gates, 4 circuits | 16 | 28 |
| CNN-LSTM twin | 22.51 ± 0.54 | 16.26 ± 0.73 | 0.684 ± 0.015 | 3,656 ± 489 | 23.39 ± 0.47 | 47.45 ± 1.49 | 741 | — | 2 | 3 |
| Quantum CNN-LSTM | 23.75 ± 1.78 | 16.93 ± 1.16 | 0.647 ± 0.054 | 14,961 ± 4,477 | 24.61 ± 1.65 | 41.25 ± 1.74 | 761 (120) | 4 qubits × 2 layers, depth 9, 20 gates, 5 circuits | 91 | 113 |

*Circuits ablated*: every circuit (or the twin's counterpart layer) replaced by its batch mean at inference. Training
and inference times were measured on a CPU shared with other jobs; treat them as upper bounds.

### Accuracy — "fails within 30 cycles" (same predictions, 100 official test engines, 25 positive)

RUL prediction is regression; this is its standard classification view. An engine is predicted to fail when
RUL_pred ≤ 30 cycles; ROC-AUC ranks engines by −RUL_pred; *within ±10 / ±20* is the share of engines whose predicted
RUL lies within 10 / 20 cycles of the true (capped) RUL. Mean ± SD over 3 seeds, computed from the
per-engine predictions in `qseq_raw_FD001.json` and re-checked with scikit-learn by `scripts/verify_results.py`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | False-alarm rate | Within ±10 cycles | Within ±20 cycles |
|---|---|---|---|---|---|---|---|---|
| Quantum LSTM | 0.927 ± 0.025 | 0.919 ± 0.031 | 0.773 ± 0.083 | 0.839 ± 0.061 | 0.975 ± 0.004 | 0.022 ± 0.008 | 0.643 ± 0.021 | 0.823 ± 0.021 |
| LSTM twin | 0.917 ± 0.012 | 0.933 ± 0.020 | 0.720 ± 0.069 | 0.811 ± 0.035 | 0.969 ± 0.001 | 0.018 ± 0.008 | 0.627 ± 0.012 | 0.830 ± 0.010 |
| Quantum Transformer | 0.910 ± 0.010 | 0.930 ± 0.022 | 0.693 ± 0.061 | 0.793 ± 0.032 | 0.964 ± 0.005 | 0.018 ± 0.008 | 0.477 ± 0.015 | 0.743 ± 0.006 |
| Transformer twin | 0.903 ± 0.023 | 0.912 ± 0.024 | 0.680 ± 0.106 | 0.776 ± 0.066 | 0.977 ± 0.004 | 0.022 ± 0.008 | 0.493 ± 0.012 | 0.770 ± 0.020 |
| Quantum CNN-LSTM | 0.897 ± 0.012 | 0.871 ± 0.066 | 0.693 ± 0.023 | 0.771 ± 0.018 | 0.923 ± 0.015 | 0.036 ± 0.020 | 0.417 ± 0.051 | 0.733 ± 0.029 |
| CNN-LSTM twin | 0.893 ± 0.006 | 0.882 ± 0.049 | 0.667 ± 0.061 | 0.757 ± 0.025 | 0.952 ± 0.001 | 0.031 ± 0.015 | 0.427 ± 0.055 | 0.767 ± 0.021 |

### Paired comparison, quantum vs classical twin (RMSE)

| Comparison | RMSE(quantum) − RMSE(twin) | 95% CI, engines | 95% CI, seeds | Wilcoxon p (Holm) | Engines where quantum lower | Verdict |
|---|---|---|---|---|---|---|
| Quantum LSTM vs LSTM twin | -0.81 | [-2.09, 0.24] | [-1.26, -0.35] | 0.901 | 48/100 | no significant difference |
| Quantum Transformer vs Transformer twin | 2.18 | [0.54, 3.81] | [-0.13, 4.50] | 0.122 | 42/100 | no significant difference |
| Quantum CNN-LSTM vs CNN-LSTM twin | 1.24 | [-0.70, 3.00] | [-1.86, 4.35] | 0.086 | 40/100 | no significant difference |

None of the three differences is significant under the pre-specified rule.

## 2. Imputation (RMSE on removed values, standardised sensor units)

**Random missingness (MCAR)**

| Method | 10% | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|
| *observed missing rate* | 10.1% | 20.1% | 30.0% | 40.0% | 50.0% |
| Transformer twin | 0.6150 ± 0.0089 | 0.6183 ± 0.0124 | 0.6225 ± 0.0051 | 0.6237 ± 0.0039 | 0.6259 ± 0.0056 |
| Quantum Transformer | 0.6173 ± 0.0132 | 0.6204 ± 0.0149 | 0.6245 ± 0.0073 | 0.6258 ± 0.0056 | 0.6283 ± 0.0037 |
| LSTM twin | 0.6214 ± 0.0094 | 0.6242 ± 0.0119 | 0.6295 ± 0.0053 | 0.6317 ± 0.0032 | 0.6347 ± 0.0044 |
| Quantum LSTM | 0.6299 ± 0.0087 | 0.6323 ± 0.0131 | 0.6400 ± 0.0060 | 0.6415 ± 0.0051 | 0.6453 ± 0.0094 |
| CNN-LSTM twin | 0.6591 ± 0.0040 | 0.6609 ± 0.0165 | 0.6668 ± 0.0105 | 0.6697 ± 0.0072 | 0.6753 ± 0.0143 |
| Quantum CNN-LSTM | 0.6783 ± 0.0075 | 0.6812 ± 0.0102 | 0.6897 ± 0.0045 | 0.6916 ± 0.0080 | 0.6991 ± 0.0084 |
| Linear interpolation | 0.6946 ± 0.0040 | 0.6963 ± 0.0128 | 0.7083 ± 0.0085 | 0.7100 ± 0.0076 | 0.7190 ± 0.0052 |
| LOCF | 0.7890 ± 0.0047 | 0.7872 ± 0.0067 | 0.7994 ± 0.0087 | 0.8001 ± 0.0066 | 0.7973 ± 0.0107 |
| Mean fill | 0.8624 ± 0.0071 | 0.8686 ± 0.0084 | 0.8678 ± 0.0054 | 0.8695 ± 0.0030 | 0.8676 ± 0.0027 |

**Sensor outages (block gaps)**

| Method | 10% | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|
| *observed missing rate* | 8.3% | 20.8% | 29.2% | 41.7% | 50.0% |
| Transformer twin | 0.6034 ± 0.0173 | 0.6215 ± 0.0003 | 0.6291 ± 0.0052 | 0.6334 ± 0.0064 | 0.6311 ± 0.0028 |
| Quantum Transformer | 0.6040 ± 0.0137 | 0.6226 ± 0.0047 | 0.6319 ± 0.0039 | 0.6346 ± 0.0065 | 0.6322 ± 0.0043 |
| LSTM twin | 0.6086 ± 0.0163 | 0.6266 ± 0.0044 | 0.6353 ± 0.0039 | 0.6379 ± 0.0045 | 0.6364 ± 0.0012 |
| Quantum LSTM | 0.6166 ± 0.0180 | 0.6360 ± 0.0013 | 0.6451 ± 0.0077 | 0.6499 ± 0.0072 | 0.6479 ± 0.0061 |
| CNN-LSTM twin | 0.6486 ± 0.0220 | 0.6699 ± 0.0067 | 0.6786 ± 0.0174 | 0.6839 ± 0.0152 | 0.6821 ± 0.0128 |
| Quantum CNN-LSTM | 0.6737 ± 0.0224 | 0.6964 ± 0.0110 | 0.7024 ± 0.0024 | 0.7086 ± 0.0082 | 0.7067 ± 0.0109 |
| Linear interpolation | 0.6942 ± 0.0195 | 0.7197 ± 0.0038 | 0.7262 ± 0.0089 | 0.7304 ± 0.0089 | 0.7291 ± 0.0063 |
| LOCF | 0.7937 ± 0.0218 | 0.7916 ± 0.0158 | 0.8028 ± 0.0038 | 0.8037 ± 0.0271 | 0.8035 ± 0.0155 |
| Mean fill | 0.8564 ± 0.0124 | 0.8658 ± 0.0016 | 0.8717 ± 0.0019 | 0.8715 ± 0.0054 | 0.8695 ± 0.0038 |

Block masks are built from whole missing steps, so the 10 % target yields 8.3 % (2 of 24 steps); observed rates are
shown.

**Paired comparisons at 30 % missing** (all 10 conditions are in the report)

| Pattern | Comparison | RMSE difference | 95% CI engines | 95% CI seeds | Holm p | Verdict |
|---|---|---|---|---|---|---|
| mcar | Quantum LSTM vs LSTM twin | 0.0105 | [0.0090, 0.0120] | [-0.0003, 0.0212] | 1.9e-14 | consistent on these engines, not across seeds |
| mcar | Quantum Transformer vs Transformer twin | 0.0020 | [0.0008, 0.0033] | [-0.0063, 0.0104] | 2.0e-03 | consistent on these engines, not across seeds |
| mcar | Quantum CNN-LSTM vs CNN-LSTM twin | 0.0230 | [0.0172, 0.0286] | [-0.0058, 0.0517] | 8.7e-12 | consistent on these engines, not across seeds |
| mcar | Quantum LSTM vs Linear interpolation | -0.0683 | [-0.0767, -0.0591] | [-0.0786, -0.0580] | 5.0e-16 | A lower error (significant) |
| mcar | Quantum Transformer vs Linear interpolation | -0.0838 | [-0.0935, -0.0732] | [-0.0881, -0.0795] | 5.0e-16 | A lower error (significant) |
| mcar | Quantum CNN-LSTM vs Linear interpolation | -0.0185 | [-0.0221, -0.0149] | [-0.0342, -0.0029] | 4.1e-15 | A lower error (significant) |
| mcar | LSTM twin vs Linear interpolation | -0.0788 | [-0.0876, -0.0691] | [-0.0882, -0.0694] | 5.0e-16 | A lower error (significant) |
| mcar | Transformer twin vs Linear interpolation | -0.0858 | [-0.0957, -0.0753] | [-0.0951, -0.0765] | 5.0e-16 | A lower error (significant) |
| mcar | CNN-LSTM twin vs Linear interpolation | -0.0415 | [-0.0496, -0.0324] | [-0.0603, -0.0228] | 1.7e-15 | A lower error (significant) |
| block | Quantum LSTM vs LSTM twin | 0.0098 | [0.0081, 0.0115] | [-0.0017, 0.0213] | 9.4e-14 | consistent on these engines, not across seeds |
| block | Quantum Transformer vs Transformer twin | 0.0028 | [0.0016, 0.0039] | [-0.0031, 0.0087] | 6.4e-04 | consistent on these engines, not across seeds |
| block | Quantum CNN-LSTM vs CNN-LSTM twin | 0.0238 | [0.0168, 0.0312] | [-0.0141, 0.0618] | 3.7e-10 | consistent on these engines, not across seeds |
| block | Quantum LSTM vs Linear interpolation | -0.0811 | [-0.0936, -0.0692] | [-0.0882, -0.0740] | 5.0e-16 | A lower error (significant) |
| block | Quantum Transformer vs Linear interpolation | -0.0943 | [-0.1090, -0.0809] | [-0.1079, -0.0807] | 5.0e-16 | A lower error (significant) |
| block | Quantum CNN-LSTM vs Linear interpolation | -0.0238 | [-0.0288, -0.0189] | [-0.0416, -0.0060] | 9.5e-16 | A lower error (significant) |
| block | LSTM twin vs Linear interpolation | -0.0909 | [-0.1043, -0.0785] | [-0.1034, -0.0783] | 5.0e-16 | A lower error (significant) |
| block | Transformer twin vs Linear interpolation | -0.0971 | [-0.1116, -0.0838] | [-0.1064, -0.0878] | 5.0e-16 | A lower error (significant) |
| block | CNN-LSTM twin vs Linear interpolation | -0.0476 | [-0.0594, -0.0365] | [-0.0702, -0.0251] | 5.0e-16 | A lower error (significant) |

Across all 10 conditions (5 rates × 2 patterns):

- learned imputer vs linear interpolation: **60 / 60** comparisons significantly favour the learned imputer;
- quantum vs twin (30 comparisons): **0** favour the quantum model; **4** favour the twin significantly
  (Quantum CNN-LSTM 1, Quantum LSTM 3); **21** favour the twin on these
  engines but not consistently across seeds; **5** show no difference.

## 3. Downstream RUL after imputation, 50 % missing (RMSE in cycles; rows: imputer, columns: RUL model)

**Random missingness (MCAR)**

| Imputer (50 % missing) | Quantum LSTM | LSTM twin | Quantum Transformer | Transformer twin | Quantum CNN-LSTM | CNN-LSTM twin |
|---|---|---|---|---|---|---|
| **complete data** | 17.50 ± 0.16 | 18.31 ± 0.22 | 19.65 ± 0.74 | 17.47 ± 0.25 | 23.75 ± 1.78 | 22.51 ± 0.54 |
| Mean fill | 26.50 ± 0.45 | 25.97 ± 0.10 | 26.14 ± 0.58 | 27.94 ± 1.47 | 28.45 ± 1.19 | 27.68 ± 2.06 |
| LOCF | 19.29 ± 0.68 | 20.28 ± 0.75 | 21.69 ± 1.23 | 20.41 ± 0.39 | 23.57 ± 2.17 | 23.16 ± 0.65 |
| Linear interpolation | 19.21 ± 0.63 | 20.13 ± 0.80 | 21.30 ± 1.27 | 19.98 ± 0.77 | 23.65 ± 1.73 | 22.99 ± 0.27 |
| Quantum LSTM | 19.34 ± 0.40 | 19.91 ± 0.67 | 20.92 ± 1.07 | 19.82 ± 0.91 | 23.25 ± 1.71 | 22.59 ± 0.31 |
| LSTM twin | 19.28 ± 0.25 | 19.91 ± 0.59 | 20.63 ± 0.95 | 19.79 ± 0.84 | 23.23 ± 1.47 | 22.60 ± 0.18 |
| Quantum Transformer | 19.33 ± 0.52 | 19.96 ± 0.62 | 20.73 ± 0.67 | 19.72 ± 0.65 | 23.01 ± 1.35 | 22.42 ± 0.26 |
| Transformer twin | 19.21 ± 0.31 | 19.78 ± 0.52 | 20.66 ± 0.54 | 19.81 ± 0.90 | 23.60 ± 0.97 | 22.66 ± 0.58 |
| Quantum CNN-LSTM | 19.18 ± 0.49 | 20.12 ± 0.73 | 21.22 ± 1.08 | 20.12 ± 0.93 | 23.66 ± 1.40 | 23.06 ± 0.34 |
| CNN-LSTM twin | 19.10 ± 0.55 | 20.05 ± 0.66 | 21.30 ± 1.19 | 20.16 ± 0.90 | 23.38 ± 1.81 | 22.99 ± 0.26 |

**Sensor outages (block)**

| Imputer (50 % missing) | Quantum LSTM | LSTM twin | Quantum Transformer | Transformer twin | Quantum CNN-LSTM | CNN-LSTM twin |
|---|---|---|---|---|---|---|
| **complete data** | 17.50 ± 0.16 | 18.31 ± 0.22 | 19.65 ± 0.74 | 17.47 ± 0.25 | 23.75 ± 1.78 | 22.51 ± 0.54 |
| Mean fill | 21.42 ± 0.41 | 21.52 ± 1.38 | 25.02 ± 1.34 | 29.42 ± 1.29 | 27.88 ± 1.76 | 26.19 ± 1.33 |
| LOCF | 18.09 ± 0.59 | 18.80 ± 0.18 | 21.47 ± 2.12 | 20.39 ± 1.52 | 25.73 ± 2.77 | 23.29 ± 1.65 |
| Linear interpolation | 17.80 ± 0.44 | 18.50 ± 0.28 | 20.14 ± 1.47 | 18.80 ± 0.98 | 24.23 ± 2.19 | 22.59 ± 1.43 |
| Quantum LSTM | 17.93 ± 0.28 | 18.52 ± 0.32 | 19.76 ± 1.40 | 18.27 ± 0.97 | 23.32 ± 2.21 | 22.01 ± 1.35 |
| LSTM twin | 18.02 ± 0.34 | 18.57 ± 0.21 | 19.85 ± 1.14 | 18.54 ± 0.92 | 23.40 ± 2.18 | 22.20 ± 1.36 |
| Quantum Transformer | 17.85 ± 0.50 | 18.51 ± 0.28 | 19.81 ± 1.19 | 18.20 ± 0.79 | 23.41 ± 1.71 | 22.03 ± 1.54 |
| Transformer twin | 17.98 ± 0.37 | 18.50 ± 0.34 | 19.73 ± 1.03 | 18.34 ± 0.98 | 23.46 ± 1.76 | 21.79 ± 1.25 |
| Quantum CNN-LSTM | 17.77 ± 0.54 | 18.56 ± 0.33 | 20.12 ± 1.35 | 18.71 ± 1.14 | 23.75 ± 2.13 | 22.38 ± 1.33 |
| CNN-LSTM twin | 17.78 ± 0.53 | 18.53 ± 0.31 | 20.09 ± 1.30 | 18.51 ± 1.10 | 23.97 ± 2.20 | 22.57 ± 1.38 |

All rates (10–50 %) for every imputer × RUL model are in the report (546 rows, 3 seeds each).
Mean filling degrades every RUL model strongly. The learned imputers' clearly better reconstruction carries through
only weakly: relative to linear interpolation they change downstream RUL RMSE by -0.91 to +0.21 cycles across
all rates, patterns, imputers and RUL models (negative = better).

## 4. Why the quantum models trail their twins — diagnostics

**RUL models (mean over seeds)**

| Model | Final train loss | Gradient norm at circuit/counterpart, epoch 1 → last | Mean change in prediction when ablated (cycles) |
|---|---|---|---|
| Quantum LSTM | 0.0203 | 0.0111 → 0.0059 | 33.5 |
| LSTM twin | 0.0218 | 0.0273 → 0.0057 | 33.2 |
| Quantum Transformer | 0.0183 | 0.0110 → 0.0063 | 11.8 |
| Transformer twin | 0.0174 | 0.0285 → 0.0086 | 17.8 |
| Quantum CNN-LSTM | 0.0197 | 0.0092 → 0.0073 | 32.7 |
| CNN-LSTM twin | 0.0222 | 0.0138 → 0.0076 | 30.9 |

**Imputers**

| Imputer | Val RMSE | Final train loss | Gradient norm at circuit/counterpart, epoch 1 → last | RMSE vs ablated (MCAR 30 %) |
|---|---|---|---|---|
| Quantum LSTM | 0.5969 ± 0.0081 | 0.3677 | 0.0016 → 0.0058 | 0.6400 vs 0.7100 |
| LSTM twin | 0.5868 ± 0.0074 | 0.3556 | 0.0030 → 0.0067 | 0.6295 vs 0.7106 |
| Quantum Transformer | 0.5849 ± 0.0057 | 0.3518 | 0.0036 → 0.0022 | 0.6245 vs 0.6252 |
| Transformer twin | 0.5826 ± 0.0074 | 0.3492 | 0.0065 → 0.0042 | 0.6225 vs 0.6244 |
| Quantum CNN-LSTM | 0.6403 ± 0.0042 | 0.4267 | 0.0008 → 0.0029 | 0.6897 vs 0.7089 |
| CNN-LSTM twin | 0.6283 ± 0.0131 | 0.4075 | 0.0021 → 0.0054 | 0.6668 vs 0.7088 |

1. **The circuits are genuinely used.** Ablating them raises RUL RMSE from ≈ 17.5 to ≈ 42 (Quantum LSTM) and to ≈ 41
   (Quantum CNN-LSTM); predictions move by ≈ 33 cycles on average. The models do not route around the circuits.
2. **Smaller gradients reach the circuit weights.** In the first epoch the gradient norm at the circuits is
   2.5–3.8× smaller than at the twin's counterpart layer. With the same learning rate and epoch budget the quantum
   models learn more slowly.
3. **Under-fitting, not over-fitting.** Every quantum imputer ends with a higher training loss than its twin, and
   validation error follows training error: an optimisation/capacity effect (bounded ⟨Z⟩ read-outs, small gradients)
   under an identical budget.
4. **Transformer imputers barely use their blocks** (ablation changes RMSE by < 0.002); they act mainly through the
   linear-interpolation residual, which is why quantum and classical Transformer imputers are almost equal.

These are observations for this budget (10 RUL / 8 imputation epochs, lr 0.01), not a general statement
about quantum models.

## 5. Qiskit / PennyLane equivalence (simulator vs simulator)

Source: `scripts/run_quantum_experiments.py --suite qiskit` → `qseq_qiskit_verification_FD001.json`.

| Check | Max abs error | Mean abs error |
|---|---|---|
| Circuit forward (16 cases incl. zeros, input ±π/2 bounds, large weights, weights = π) | 6.7e-16 | 1.8e-16 |
| ∂/∂inputs (Qiskit parameter shift vs PennyLane autograd) | 1.6e-15 | 2.5e-16 |
| ∂/∂weights | 1.4e-15 | 1.6e-16 |
| Parameter mapping (single-weight perturbations) | 7.2e-16 | 2.1e-16 |
| Saved rul model Quantum LSTM — 37 windows, 5 batch types, 3,552 Qiskit circuit runs | 3.9e-16 | 8.5e-17 |
| Saved rul model Quantum Transformer — 37 windows, 5 batch types, 3,552 Qiskit circuit runs | 3.3e-16 | 3.4e-17 |
| Saved rul model Quantum CNN-LSTM — 37 windows, 5 batch types, 2,220 Qiskit circuit runs | 6.7e-16 | 2.7e-16 |
| Saved impute model Quantum LSTM — 37 windows, 5 batch types, 3,552 Qiskit circuit runs | 2.2e-15 | 1.6e-16 |
| Saved impute model Quantum Transformer — 37 windows, 5 batch types, 3,552 Qiskit circuit runs | 1.3e-15 | 1.0e-16 |
| Saved impute model Quantum CNN-LSTM — 37 windows, 5 batch types, 4,440 Qiskit circuit runs | 2.1e-15 | 1.2e-16 |

Both are exact statevector simulations. **No quantum hardware was used**; this says nothing about noisy devices.

## 6. Not verified here

- Google Colab / Google Drive persistence (no Colab runtime was available; `scripts/validate_colab_drive.py` is ready).
- FD002–FD004, other datasets, shot-based or hardware execution, other training budgets.
- The classical full-size models, IMS detectors, VQC/kernel/QNN and QAOA results in `RESULTS.md` predate this audit
  and were not re-run in it.
