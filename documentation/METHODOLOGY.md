# Research methodology, limitations and future work

Measured numbers are in [RESULTS.md](RESULTS.md), which is generated from saved run reports.

## 1. Data

**NASA C-MAPSS** (PCoE data set 6, Saxena & Goebel 2008). Run-to-failure simulations of turbofan engines;
26 columns (unit, cycle, 3 operating settings, 21 sensors). Sub-sets FD001–FD004 (1 or 6 operating
conditions, 1 or 2 fault modes).

- RUL label: training RUL = last cycle − cycle, capped at 125 (piecewise-linear target). Test RUL = official
  `RUL_FDxxx.txt` value + (last cycle − cycle).
- Leakage control: validation split by engine (20 % of training engines), never by row; operating-condition
  clusters (k-means on settings, FD002/FD004), sensor-dropping decisions and normalisation statistics are
  fitted on training engines only.
- Features: informative sensors normalised per operating condition, 5-cycle rolling mean/std, normalised cycle.
  Sequence models use 30-cycle windows.
- Evaluation: one prediction per test engine at its last cycle (100 engines for FD001), the standard protocol.

**NASA IMS bearings** (PCoE data set 4, Lee et al. 2007). Three run-to-failure tests, 1-s vibration snapshots at
20 kHz every ~10 min (2 156 / 984 / 6 324 snapshots); the third test unpacks to `4th_test/`.

- Features per bearing and snapshot (averaged over its channels): time-domain (RMS, peak, peak-to-peak,
  crest, shape, impulse, clearance), statistical (kurtosis, skewness, entropy), frequency-domain (spectral
  centroid/spread/kurtosis, dominant frequency, band energies) and envelope-spectrum amplitudes at the
  defect frequencies of the Rexnord ZA-2115 bearing at 2 000 rpm (BPFO 236.4 Hz, BPFI 296.9 Hz, BSF 279.8 Hz,
  FTF 14.8 Hz).
- Labels: IMS has no per-snapshot labels. Evaluation uses an explicit assumption: the first 25 % of each test
  is healthy; the last 10 % of the bearings documented as failed is anomalous; the rest is unlabelled.
  These are pseudo-labels, and metrics depend on this choice.

The two datasets keep separate pipelines because they have different signals and labels.

## 2. Classical models

- RUL: Random Forest, XGBoost, LightGBM, CatBoost (early stopping on validation engines), LSTM, GRU, CNN-LSTM,
  Transformer encoder (Adam, ReduceLROnPlateau, early stopping, per-epoch Drive checkpoints).
- Uncertainty: σ = std of validation residuals; 90 % split-conformal interval from validation absolute errors.
  Probability of failure within h cycles = Φ((h − RUL̂)/σ). It is evaluated as a classifier against
  `RUL_true ≤ h` (ROC-AUC, precision, recall, F1, Brier score).
- Metrics: MAE, RMSE, R², and the PHM08 NASA score (asymmetric; late predictions are penalised more).
- IMS: Isolation Forest, One-Class SVM, dense autoencoder, LSTM autoencoder, all fitted on healthy-baseline
  snapshots. The threshold is the 99th percentile of baseline scores. Metrics: ROC-AUC, precision, recall,
  F1, accuracy, false-alarm rate, and detection lead time (first 3-snapshot sustained alarm → end of test). A
  supervised Random Forest is trained on two tests and evaluated on the third.

## 3. Quantum experiments

PennyLane, statevector simulator (`default.qubit`; `lightning.qubit` for QAOA). Inputs are reduced with PCA
(fitted on training engines) to n_qubits ∈ [2, 8] components and scaled to [0, π].

- VQC: angle embedding + StronglyEntanglingLayers, ⟨Z₀⟩ → logistic output.
- Quantum kernel: IQP feature map; fidelity kernel |⟨φ(x)|φ(x′)⟩|² computed from exact statevectors
  (equivalent to the overlap circuit on a simulator) → SVM with Platt scaling.
- Hybrid QNN: linear → tanh → angle-embedded BasicEntangler layer → linear head (RUL regression).
- Baselines: logistic regression, RBF-SVM and a small MLP (classification); ridge, RBF-SVR and a small MLP
  (regression). All baselines use the same PCA inputs.
- Recorded per run: qubits, layers, circuit depth and gate count after decomposition to device gates,
  trainable parameters, train and inference time, and metrics.

## 4. Risk, digital twin and counterfactuals

- Risk score = PoF × CoF, where CoF = (corrective cost + failure downtime × downtime cost) × criticality.
  Risk levels come from risk/CoF thresholds (default 0.10 / 0.30 / 0.60). Urgency comes from risk level and RUL.
- Bearings have no RUL labels. PoF comes from anomaly severity s = (score − baseline median) / (threshold −
  baseline median) via a logistic mapping (k = 3, s₀ = 1.5), a documented heuristic. Time-to-threshold is
  extrapolated from an exponential fit of recent scores.
- Digital twin: measured playback (dataset values and predictions on them), plus Monte Carlo futures:
  RUL₀ ~ N⁺(RUL̂, σ), wear rate = load^α (α = 2), noise 5 %, and restoration to 0.9 × RUL cap on maintenance.
  Sensors are projected through per-sensor linear maps fitted on training engines. Simulated values are
  tagged `source: simulated` throughout.
- Counterfactuals: five scenarios (continue, immediate PM, delayed PM, reduced load, optimal timing). Costs are
  split into maintenance, planned downtime, expected failure cost (P(fail) × CoF) and lost output, with no
  double counting. Optimal timing comes from a 25-point scan.

## 5. Maintenance scheduling (QAOA)

Binary x[i, t] = maintain asset i in slot t. Each assignment's expected cost is computed from the
truncated-normal failure model. QUBO penalties enforce at most one slot per asset and crew capacity per slot
(binary slack bits). The penalty weight is 1.5 × max |linear coefficient|, which enforces the constraints
without swamping the objective. The QUBO is verified by exhaustive search: its minimum equals the true
optimal plan cost.

Solvers: MILP (SciPy/HiGHS, exact), greedy, simulated annealing on the QUBO, and QAOA (p layers, COBYLA,
linear-ramp initialisation plus one random restart). QAOA reads out the 256 most probable bitstrings and keeps
the cheapest feasible one. The probability the circuit assigned to that plan is reported, because the
post-selection, not the circuit, often does most of the work.

## 6. Agents

LangGraph `StateGraph` over a shared `MaintenanceState`, with conditional routing on risk and maintenance
need. Agents obtain every number through LangChain tools. The LLM (Ollama or an OpenAI-compatible API, via
LangChain prompt templates and structured output) writes only the narrative. Any number in its output that
does not match a tool output (2 % tolerance) causes the narrative to be rejected and replaced by a
deterministic one, and the rejection is recorded. Consequential actions pause on a LangGraph `interrupt`
until an engineer approves or rejects them in the UI; approvals create work orders in the maintenance history.

## 7. Limitations

- **Simulator-only quantum results.** Small single-seed runs; no repeated-seed statistics, no noise models, no
  hardware runs. In every recorded run so far, classical models on the same inputs score higher.
- **QAOA scale.** At most 16 qubits on the simulator (≈ 5 assets × 2 slots plus slack). Plans depend on
  classical post-selection over the most probable outcomes.
- **IMS labels are assumptions.** Metrics change with `baseline_fraction`/`degraded_fraction`. Late in a test,
  neighbouring healthy bearings also show elevated vibration, which raises false alarms.
- **Bearing PoF and time-to-threshold are heuristics**, not trained RUL models.
- **Costs are illustrative assumptions** (System Settings); recommendations are only as good as these figures.
- **Uncertainty is marginal.** Conformal intervals and the Gaussian σ are not conditioned on the engine or the
  degradation stage.
- **Results so far are on FD001 only.** The pipeline supports FD002–FD004, but they have not been trained or
  reported yet.
- **Paused agent runs** live in an in-memory checkpointer and expire if the API restarts (marked `expired`).
- **SQLite on Drive** is safe for a single API process only. Use PostgreSQL (`QADT_DATABASE_URL`) for more.
- **Development mode** (`QADT_STORAGE_MODE=development`) stores data locally, for testing only. The UI shows
  a warning banner whenever it is active.

## 8. Future work

Multi-seed and cross-validated evaluation with confidence intervals; FD002–FD004 and joint training; noise-aware
and hardware quantum runs (IBM Quantum via `pennylane-qiskit`); warm-started and constraint-preserving
(XY-mixer) QAOA; condition-dependent uncertainty (quantile or deep-ensemble models); bearing RUL with
physics-informed degradation models; a persistent LangGraph checkpointer (Postgres); and user authentication
with roles for approvals.
