# Final research limitations

Scope: the quantum sequence-model study (Quantum LSTM / Transformer / CNN-LSTM vs classical twins), the
missing-data experiments, the single missing-data agent, and the wider project. Read together with
[FINAL_VERIFIED_RESULTS.md](FINAL_VERIFIED_RESULTS.md).

## 1. What the results do *not* show

- **No quantum advantage.** No quantum model is significantly better than its same-size classical twin on any
  metric; in imputation the twin is better or equal in all 30 paired comparisons. The quantum models are
  *competitive* at this size (Quantum LSTM RMSE 17.50 vs twin 18.31, not significant), nothing more.
- **Not hardware results.** Every circuit was run on exact, noiseless statevector simulators (PennyLane
  `default.qubit`; Qiskit `Statevector` for the cross-check). Shot noise, gate errors, decoherence, connectivity
  and queueing are not modelled. The Qiskit check proves simulator-to-simulator equivalence only.
- **Not a statement about quantum models in general.** The gap to the twins is measured for one budget (4 qubits,
  2 layers, 10/8 epochs, Adam lr 0.01, batch 32). The diagnostics (2.5–3.8× smaller gradients at the circuits,
  higher training loss) suggest a longer or differently tuned schedule could narrow it; this was not tested,
  and tuning only the quantum models would break the fair-comparison design.

## 2. Model size and comparability

- The models are deliberately small (633–761 parameters, 96–120 of them in circuits) so the circuits stay cheap
  to simulate. The project's full-size classical LSTM (64 hidden units, 2 layers, 46 input features: 64,065
  parameters; RMSE 13.37) belongs to a different size class and are not a like-for-like comparison.
- The classical twins replace each circuit by a `Linear(4, 4) + tanh` layer of the same width. Other twin designs
  (deeper, wider, untied) could change the comparison.
- The Transformer imputers barely use their quantum/classical blocks (ablation changes RMSE by < 0.002); their
  quality comes mostly from the interpolation residual path.

## 3. Data and evaluation

- **One sub-dataset.** Only C-MAPSS FD001 (one operating condition, one fault mode). FD002–FD004 (six operating
  conditions and/or two fault modes) were not run.
- **Capped labels.** The primary RUL metrics use RUL capped at 125 (the training target). 11 of 100 test engines
  have a larger true RUL; uncapped metrics are reported alongside and are about 1 cycle worse.
- **Simulated missingness.** MCAR and block outages are synthetic, applied to complete data. Real field data may be
  missing *not* at random (e.g. sensors failing because the engine is degrading); that case is not covered.
- **Imputation error floor.** C-MAPSS sensor noise is large relative to the degradation trend. Linear
  interpolation already reaches 0.69–0.73 standardised RMSE, and the best learned imputer reaches about 0.60–0.63;
  part of the remaining error is irreducible noise.
- **Weak link to decisions.** Better imputation barely changes downstream RUL error: −0.91 to +0.21 cycles vs linear
  interpolation. Only mean filling hurts strongly.
- **Windows overlap.** The 400 imputation windows come from 97 test engines and overlap within an engine. The
  statistics resample engines, not windows, for this reason.

## 4. Statistics

- **Three seeds.** Seed-level intervals are wide (t with 2 degrees of freedom). The engine bootstrap treats each
  trained model as fixed, so it captures test-set variation but not training randomness. Requiring both
  intervals to exclude 0 is deliberately conservative and can miss small real effects.
- Seeds change initialisation, batch order and masks. The training subsample (1,200 windows) and the engine split
  are fixed, so variation from data selection is not represented.
- Multiple comparisons are Holm-adjusted *within* families (3 RUL comparisons; 9 per imputation condition), not
  across the whole study.

## 5. Accuracy table

- "Accuracy" refers to the classification view "fails within 30 cycles" (25 positives among 100 engines). With
  25 positives, one engine more or less changes recall by 0.04, so small differences between models are within
  noise (seed SDs 0.006–0.106; recall varies most).

## 6. Engineering and environment

- **Google Colab / Google Drive: not validated.** All runs used `development` storage on one macOS machine. Drive
  mounting, FUSE latency, quota and runtime restarts were never exercised. `scripts/validate_colab_drive.py`
  exists for this but was not run.
- **Timing.** Training and inference times were measured on a shared CPU while other jobs ran. They are upper bounds
  and are not comparable across machines.
- **Determinism.** Results were bit-identical across three runs on this machine and software stack (PyTorch 2.15,
  PennyLane 0.45.1, NumPy as in `requirements-lock.txt`). Other CPUs, BLAS builds or library versions may differ in
  the last digits.
- **Agent.** The missing-data agent is a deterministic tool-using LangGraph agent. The LLM, if enabled, writes only
  the narrative, and it was tested with a fake LLM rather than a real one in this audit.
- **Earlier results not re-run.** The classical RUL models, IMS detectors, VQC/kernel/QNN and QAOA results in
  `RESULTS.md` predate this audit. Their saved reports were inventoried and backed up but not re-executed.

## 7. Recommended future work

1. Run `scripts/validate_colab_drive.py` (both phases around a real runtime restart) and the full pipeline in Colab.
2. Repeat with more seeds (≥ 10) and with resampled training subsets, so the seed-level inference has power.
3. Matched-budget learning-rate/epoch sweeps applied identically to quantum models and twins, to test whether the
   gradient gap explains the performance gap.
4. FD002–FD004 and a real dataset with naturally missing values; a missing-not-at-random scenario.
5. Shot-based simulation, then hardware runs, through the existing Qiskit circuit export (OpenQASM 3).
