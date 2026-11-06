import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import QuantumSequenceLab from "../pages/QuantumSequenceLab";

const RUN = {
  run_id: "abc", status: "completed",
  state: {
    trace: [{ agent: "md_agent", status: "completed" }],
    steps: [
      { step: "data_quality", status: "completed", summary: "38.1% missing (mcar)" },
      { step: "plan", status: "completed", summary: "using requested imputer mean" },
      { step: "impute", status: "completed", summary: "mean imputed 137 values" },
      { step: "check", status: "completed", summary: "mean: rejected → re-plan" },
      { step: "plan", status: "completed", summary: "chose linear" },
      { step: "impute", status: "completed", summary: "linear imputed 137 values" },
      { step: "check", status: "completed", summary: "linear: accepted" },
      { step: "forecast", status: "completed", summary: "q_lstm: RUL 12.3 cycles" },
      { step: "report", status: "completed", summary: "done" },
    ],
    attempts: [{ method: "mean", holdout_rmse: 1.339, linear_holdout_rmse: 0.485 }, { method: "linear", holdout_rmse: 0.485, linear_holdout_rmse: 0.485 }],
    report: { text: "Imputed with linear after 2 attempt(s); rejected: mean.", evaluation: {} },
    forecast: { prediction: { model: "q_lstm", rul_pred: 12.3, interval_90: [0, 40] } },
  },
};

vi.mock("../services/api", async (orig) => {
  const real = await orig();
  const fail = () => Promise.reject(Object.assign(new Error("404"), { response: { data: { detail: "none" } } }));
  return {
    ...real,
    api: {
      qseqReport: fail, qseqQiskit: fail, qseqRunEventsUrl: () => null, jobEventsUrl: () => null,
      qseqRuns: () => Promise.resolve([{ run_id: "abc", status: "completed", request: { asset_id: "FD001-test-unit34", missing_rate: 0.4, pattern: "mcar" }, summary: { imputer: "linear", rul_pred: 12.3 } }]),
      qseqRun: () => Promise.resolve(RUN),
    },
  };
});

test("a saved run shows the single agent's step log, including the retry", async () => {
  render(<MemoryRouter><QuantumSequenceLab /></MemoryRouter>);
  fireEvent.click(await screen.findByText("FD001-test-unit34"));
  expect(await screen.findByText("Missing-Data Maintenance Agent")).toBeInTheDocument();
  expect(screen.getByText("mean: rejected → re-plan")).toBeInTheDocument();
  expect(screen.getByText("Plan ×2")).toBeInTheDocument();
  expect(screen.getByText("q_lstm: RUL 12.3 cycles")).toBeInTheDocument();
  expect(screen.getByText(/rejected: mean/)).toBeInTheDocument();
});
