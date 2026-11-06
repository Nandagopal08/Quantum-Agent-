import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import QuantumSequenceLab from "../pages/QuantumSequenceLab";

// A minimal report in the backend's shape. The page must show these exact values (nothing hard-coded) and
// an explicit "unavailable" state where data is missing (here: no Qiskit verification).
const REPORT = {
  name: "qseq_benchmark_FD001", seeds: [1, 2], total_seconds: 600, sensors: Array.from({ length: 15 }, (_, i) => `s${i}`),
  config: { subset: "FD001" },
  data: { window: 24, n_test_engines: 100, n_eval_windows: 400, n_eval_engines: 97, rul_cap: 125, test_engines_above_cap: 11 },
  rul: {
    summary: [
      { model: "q_lstm", quantum: true, rmse: 17.123, rmse_std: 0.456, mae: 11.5, mae_std: 0.2, r2: 0.82, nasa_score: 512, val_rmse: 18,
        rmse_uncapped: 19.87, rmse_ablated: 31.42, trainable_params: 649, quantum_params: 96, n_qubits: 4, circuit_layers: 2,
        circuit: { depth: 9, gates: 20 }, circuits: 4, train_seconds: 120, inference_seconds: 0.2 },
      { model: "c_lstm", quantum: false, rmse: 18.31, rmse_std: 0.22, mae: 12.2, mae_std: 0.1, r2: 0.8, nasa_score: 600, val_rmse: 19,
        trainable_params: 633, quantum_params: 0, train_seconds: 5, inference_seconds: 0.003 },
    ],
    accuracy: [{ model: "q_lstm", accuracy: 0.927, accuracy_std: 0.025, precision: 0.919, precision_std: 0.031, recall: 0.773,
      recall_std: 0.083, f1: 0.839, f1_std: 0.061, roc_auc: 0.975, roc_auc_std: 0.004, false_alarm_rate: 0.02,
      false_alarm_rate_std: 0.01, within_10: 0.643, within_10_std: 0.021, within_20: 0.823, within_20_std: 0.021 }],
  },
  imputation: {
    evaluation: [
      { pattern: "mcar", rate: 0.3, method: "linear", rmse: 0.708 },
      { pattern: "mcar", rate: 0.3, method: "q_lstm", rmse: 0.64 },
    ],
    models: [{ model: "q_lstm", quantum: true, val_rmse: 0.6, val_rmse_std: null, final_train_loss: 0.38, first_epoch_grad_norm: 0.01,
      last_epoch_grad_norm: 0.02, ablation_mcar30: { rmse: 0.64, rmse_ablated: 0.70 }, trainable_params: 750, quantum_params: 96 }],
  },
  downstream_rul: [
    { pattern: "none", rate: 0, method: "complete_data", rul_model: "q_lstm", rul_rmse: 17.6, rul_rmse_std: 0.1 },
    { pattern: "mcar", rate: 0.5, method: "mean", rul_model: "q_lstm", rul_rmse: 26.51, rul_rmse_std: 0.64 },
  ],
  statistics: {
    method: "paired test", rul: [{ a: "q_lstm", b: "c_lstm", rmse_diff: -0.81, bootstrap_ci95: [-1.9, 0.3], seed_t_ci95: [-2.1, 0.5],
      wilcoxon_p_holm: 0.21, significant: false, verdict: "no significant difference" }],
    imputation: [],
  },
};

vi.mock("../services/api", async (orig) => {
  const real = await orig();
  const notFound = () => Promise.reject(Object.assign(new Error("404"), { response: { data: { detail: "no Qiskit verification yet" } } }));
  return {
    ...real,
    api: {
      qseqReport: () => Promise.resolve(REPORT), qseqQiskit: notFound, qseqRuns: () => Promise.resolve([]),
      qseqRunEventsUrl: () => null, jobEventsUrl: () => null,
    },
  };
});

test("Quantum Sequence Lab shows report values and marks missing data as unavailable", async () => {
  render(<MemoryRouter><QuantumSequenceLab /></MemoryRouter>);
  expect(await screen.findByText("17.12 ± 0.46")).toBeInTheDocument();          // RMSE from the report
  expect(screen.getByText("19.87")).toBeInTheDocument();                        // uncapped RMSE
  expect(screen.getByText("0.927 ± 0.025")).toBeInTheDocument();                // accuracy table
  expect(screen.getByText("0.839 ± 0.061")).toBeInTheDocument();
  expect(screen.getByText("31.42")).toBeInTheDocument();                        // ablated RMSE
  expect(screen.getByText(/4q × 2L · depth 9 · 20 gates · 4 circuits/)).toBeInTheDocument();
  expect(screen.getByText(/11 engines exceed it/)).toBeInTheDocument();
  expect(screen.getByText("Quantum vs classical twin — paired statistics (RUL)")).toBeInTheDocument();
  expect(screen.getByText("no significant difference")).toBeInTheDocument();
  expect(screen.getByText("[-1.900, 0.300]")).toBeInTheDocument();             // CI straight from the report
  expect(screen.getByText("26.51 ± 0.64")).toBeInTheDocument();                 // downstream cell
  expect(await screen.findByText("no Qiskit verification yet")).toBeInTheDocument();
  expect(screen.getAllByText("unavailable").length).toBeGreaterThan(0);
  expect(screen.getByText(/last run: 5 min per seed/)).toBeInTheDocument();     // derived, not hard-coded
  expect(screen.queryByText(/≈15 min/)).not.toBeInTheDocument();
});
