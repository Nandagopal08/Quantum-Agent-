# Quantum sequence models, missing data, and the missing-data agents

Code: [backend/quantum_models/qseq.py](../backend/quantum_models/qseq.py) (models),
[seq_experiments.py](../backend/quantum_models/seq_experiments.py) (experiments),
[qiskit_backend.py](../backend/quantum_models/qiskit_backend.py) (Qiskit execution),
[backend/missing_data/](../backend/missing_data/) (masks and baseline imputers), and
[backend/agents/missing_data/](../backend/agents/missing_data/) (single-agent workflow).
Run: `python scripts/run_quantum_experiments.py --suite sequence --seeds 42,43,44`.
Measured numbers are in [RESULTS.md](RESULTS.md); they are generated from the saved report and never typed by hand.

## 1. Models

Every quantum layer is the same **VQC block**:

```
x ∈ Rⁿⁱⁿ → Linear(nᵢₙ, q) → tanh·π/2 → AngleEmbedding(RY) → StronglyEntanglingLayers(L) → ⟨Z₀…Z_q⟩ → Linear(q, nₒᵤₜ)
```

The block uses q = 4 qubits and L = 2 layers, which gives 24 circuit parameters. It is trained end to end with
PyTorch through PennyLane `default.qubit` with backpropagation.

| Model | Where the circuits are | Circuits per model |
|---|---|---|
| **Quantum LSTM (QLSTM)**, after Chen, Yoo & Fang (2020) | The four gate transforms (forget, input, candidate, output) over `[x_t, h_{t−1}]`; the standard LSTM cell algebra | 4 (evaluated at every time step) |
| **Quantum Transformer**, after Di Sipio et al. (2022) | The query, key and value projections and the feed-forward sub-layer; attention weights are classical `softmax(QKᵀ/√d)`, with residual connections and layer norm | 4 |
| **Quantum CNN-LSTM** | A 1-D *quanvolution*: one VQC applied to every causal patch of k = 4 steps × all sensors (stride 2 for RUL, 1 for imputation), followed by a QLSTM | 5 |

**Classical twins.** `c_lstm`, `c_transformer` and `c_cnn_lstm` are the same architectures with the circuit
replaced by a classical `Linear(q, q) + tanh` layer of the same width. Parameter counts stay within 3% of the
quantum models. A quantum-vs-twin comparison therefore isolates the contribution of the circuit itself.

**Qiskit.** [qiskit_backend.py](../backend/quantum_models/qiskit_backend.py) rebuilds each trained circuit gate by
gate in Qiskit and evaluates it with `Statevector`. `qiskit_execution(model)` runs a whole trained model on Qiskit.
The pipeline re-runs every saved quantum model this way and records the largest difference from PennyLane in
`qseq_qiskit_verification_<subset>.json`. Circuits can also be exported to OpenQASM 3 for hardware runs.

## 2. Data

NASA C-MAPSS FD001, using the leakage-free pipeline in `backend/datasets/cmapss.py`. Validation engines are
split by unit, and scalers are fitted on training units only. Inputs are 24-cycle windows of the 15 informative
normalised sensors.

- **Training:** 1,200 random windows from training engines.
- **Validation:** 400 windows from validation engines, used for model selection and early stopping.
- **Test:** the official last window of each of the 100 test engines, scored against `RUL_FD001.txt` with RUL
  capped at 125.
- **Imputation benchmark:** 400 windows from the test engines.

## 3. Missing-data protocol

- **Patterns.** In `mcar`, every value is dropped independently with probability *r*. In `block`, each sensor
  loses contiguous gaps of 2 to T/3 steps until about *r* of its steps are missing, which simulates sensor
  outages.
- **Rates.** 10, 20, 30, 40 and 50%. The observed rate is reported, because block masks round to whole steps.
- **Ground truth.** C-MAPSS is complete, so the true value behind every removed entry is known. Errors are
  computed only on removed entries, in standardised sensor units.
- **Baselines.** Per-sensor training mean, last observation carried forward (LOCF), and linear interpolation.
- **Learned imputers.** These are the same six architectures in sequence mode. The input is
  `[linear-interpolated window, mask]` (2 × 15 channels), and the output is a residual correction to the
  interpolation; observed values are always kept. Each architecture is trained once on random rates from 5 to 60%
  and on both patterns, with fresh masks every epoch, so a single rate-agnostic model serves every condition.
- **Evaluation.** Every method sees identical seeded masks.
- **Downstream test.** The test-engine windows are masked, imputed by each method, and fed to the best quantum
  RUL model (chosen on validation engines) and to its classical twin.
- **Seeds.** Everything repeats per seed and is reported as mean ± sd.

## 4. The missing-data agent (single LangGraph agent)

One agent, `md_agent` (graph: `START → md_agent → END`), carries out six steps with its tools. Each step is
recorded in the run's `steps` log and streamed live as an `agent_step` event.

```
data quality ─(unusable)──────────────────────────────────────────────→ report
     │
     ▼
   plan → impute → check ─(rejected, attempts left)─→ plan (next-best method)
                     └─(accepted / out of attempts)─→ forecast → report
```

| Step | Tools | What it decides |
|---|---|---|
| Data quality | `observe_asset`, `profile_missingness` | Missing rate, gap lengths, likely pattern, weak sensors; stops the run above 90% missing |
| Plan | `rank_imputers` | Picks the untried method with the lowest **measured** benchmark RMSE at the nearest evaluated rate for the detected pattern |
| Impute | `run_imputation` | Runs the method, plus a label-free self-check: hides 15% of the *observed* values, re-imputes them, and compares the error with linear interpolation on the same values |
| Check | — | Rejects the imputation if the self-check exceeds 1.5× the benchmark expectation, if the method is more than 10% worse than linear on the self-check, or if any imputed value is beyond ±6σ; on rejection the agent goes back to *plan* (up to `max_attempts`, default 3 — a hard bound on the loop) and, when attempts run out, keeps the best attempt |
| Forecast | `forecast_rul` | RUL with the best quantum sequence model, its 90% interval, and its classical twin for comparison |
| Report | (LLM, optional) | Deterministic report; an optional LLM narrative is rejected if it cites any number missing from the tool outputs |

A failure in *data quality* ends the run (`failed`); a failed imputation counts as a rejected attempt; a failed
*check* stops the retry loop; a failed *forecast* gives `completed_without_forecast`. (Until 2026-11-06 these steps
were six separate LangGraph agents; they were merged into one agent on request, with identical tools, rules and
outputs.)

**Hidden truth.** The true values behind simulated missingness live in a per-run vault (`TRUTH_VAULT` in
`graph.py`), outside the LangGraph state and outside the traced tool results, so neither the agent's decisions nor any event stream
ever sees them. Only the report step takes them out, after all decisions are made, to score the run: the true imputation
RMSE against linear interpolation, the true RUL, and the forecast from complete data. A test asserts that neither
the final state nor the run's event history contains them.

**Structured, validated LLM output.** The report step asks the LLM for a `ReporterNarrative` (Pydantic: `summary`,
`caveats`). An unparseable or invalid response, an exception, or any number not present in the tool outputs causes
rejection; the deterministic narrative is used and the reason is recorded in the run.

**Uploaded windows.** A user can also upload a window with real gaps (`values`, with `null` for missing). The Data
Quality step checks it: the channel count must match the 15 trained sensors (standardised like the training
data), at least 24 steps are required (longer windows are trimmed to the last 24), and requests are capped at
500 × 64 values. There is no truth and no score for uploaded data.

**Recovery.** Runs left `running` by an API restart are marked `interrupted` at start-up; `POST
/api/qseq/agents/runs/{id}/rerun` repeats the request (deterministic for a given seed) and links the new run to the
old one (`rerun_of`).

**Interfaces.**

- API: `POST /api/qseq/agents/runs`
- Live events: `GET /api/qseq/agents/runs/{id}/events`
- Command line: `scripts/run_agent_workflow.py missing-data`
- Dashboard: the **Quantum Sequence Lab** page
- Storage: runs are saved to `models/agentic_ai/missing_data_runs/`

## 5. Evaluation protocol and verification

- **Label convention.** Targets and the primary test metrics use RUL capped at 125 (piecewise-linear convention, as
  for the project's classical models). 11 of the 100 FD001 test engines have a true RUL above 125; *uncapped*
  metrics compare the same predictions with the raw `RUL_FD001.txt` values and are reported alongside.
- **Seeds.** Seeds change weight initialisation, mini-batch order, training masks and evaluation masks; the window
  subsample and the engine split are fixed, so quantum and twin are compared on identical data (paired design).
- **Paired statistics** (`backend/quantum_models/stats.py`). For each architecture, RMSE(quantum) − RMSE(twin):
  95 % percentile bootstrap over **engines** (windows of one engine overlap, so engines are resampled, with the same
  resample for every seed), Wilcoxon signed-rank on per-engine MSE differences, a seed-level paired-t interval, and
  Holm adjustment within each family. A difference is called *significant* only if the Holm p < 0.05 **and** both
  intervals exclude 0. The engine bootstrap ignores training randomness and the 3-seed interval is wide, so
  this rule is deliberately conservative. The same analysis compares each learned imputer with linear
  interpolation and with its twin, per pattern and rate.
- **Circuit ablation.** At inference every circuit (or the twin's counterpart layer) is replaced by its batch mean;
  the rise in error measures how much the model depends on it. Gradient norms reaching the circuit weights (or the
  counterpart layer) are logged in the first and last epoch.
- **Provenance.** Every (model, seed) per-engine prediction and every per-window imputation error is saved in
  `qseq_raw_<subset>.json`; `scripts/verify_results.py` recomputes every reported metric, mean and SD from those raw
  values and the original `RUL_FD001.txt`. Reports and models are versioned on overwrite (`_history/`, last 5) and
  training logs carry the run tag, so re-runs never mix or silently replace results.
- **Qiskit.** Circuit-level checks (forward, ∂/∂inputs by parameter shift vs autograd, ∂/∂weights, parameter
  mapping, edge cases) and model-level checks (all quantum RUL and imputation models on five kinds of input batch).
  This is simulator-to-simulator equivalence; no quantum hardware was used.

## 6. Limitations

- The circuits run on a noiseless statevector simulator with 4 qubits. Shot noise and hardware noise are not
  modelled, and the Qiskit path executes the same exact simulation.
- The models are deliberately small, with 600–900 parameters, to keep circuit cost manageable. The full-size
  classical deep models in `RESULTS.md` (the LSTM has 64,065 parameters) are a different size class.
- C-MAPSS sensor noise is large compared with the degradation signal, so the error floor for imputation is high.
  Linear interpolation is already close to that floor, which limits how much any learned imputer can gain.
- The missingness is simulated (MCAR and block). Real field data may be *missing not at random*.
- Results cover FD001 only.
