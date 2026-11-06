"""Tests for the verification machinery: Qiskit gradients/mapping, paired statistics, circuit ablation, report and
model history on overwrite, the trimmed report API and the environment check."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_qiskit_matches_pennylane_including_gradients_and_edge_cases():
    from backend.quantum_models.qiskit_backend import verify_circuit_equivalence

    for q, layers in ((4, 2), (3, 1), (2, 3)):
        r = verify_circuit_equivalence(q, layers, n_cases=4, seed=q)
        for k in ("forward", "grad_inputs", "grad_weights", "parameter_mapping"):
            assert r[k]["max_abs"] < 1e-10, (q, layers, k, r[k])
        assert {"zeros", "large weights"} <= set(r["edge_cases"])


def test_qiskit_detects_a_wrong_circuit():
    """The comparison must be able to fail: a deliberately wrong entangler order is caught."""
    import pennylane as qml

    from backend.quantum_models import qiskit_backend as qb

    rng = np.random.default_rng(1)
    x, w = rng.uniform(-1, 1, 4), rng.normal(size=(2, 4, 3))
    wrong = qb.vqc_circuit(x, w).reverse_bits()          # little/big-endian mix-up
    from qiskit.quantum_info import Statevector

    p = Statevector(wrong).probabilities()
    idx = np.arange(len(p))
    bad = np.array([np.sum(p * (1 - 2 * ((idx >> i) & 1))) for i in range(4)])
    ref = torch.stack(list(qb._pennylane_block_circuit(4)(torch.tensor(x), torch.tensor(w)))).numpy()
    assert np.abs(bad - ref).max() > 1e-3
    assert np.abs(qb.expvals(x, w) - ref).max() < 1e-12
    assert qml is not None


def test_paired_statistics_are_conservative():
    from backend.quantum_models import stats

    rng = np.random.default_rng(0)
    units = np.repeat(np.arange(50), 4)                       # 50 engines × 4 windows (clusters)
    se = rng.gamma(2, 1, (3, len(units)))
    n = np.ones_like(se)
    same = stats.finalize([stats.paired_rmse_difference(se, n, se, n, units, 500)])[0]
    assert same["rmse_diff"] == 0 and not same["significant"] and same["bootstrap_ci95"][0] <= 0 <= same["bootstrap_ci95"][1]
    worse = stats.finalize([stats.paired_rmse_difference(se * 4, n, se, n, units, 500)])[0]
    assert worse["rmse_diff"] > 0 and worse["significant"] and worse["verdict"].startswith("B lower")
    # consistent on engines but noisy across seeds → not called significant
    se_b = se.copy()
    se_a = se * np.array([[1.6], [0.95], [1.5]])
    mixed = stats.finalize([stats.paired_rmse_difference(se_a, n, se_b, n, units, 500)])[0]
    assert not mixed["significant"]
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_circuit_ablation_changes_outputs_and_restores():
    from backend.quantum_models.qseq import SeqModelConfig, ablate_circuits, build_model

    for name in ("q_lstm", "c_lstm", "q_transformer"):
        m = build_model(SeqModelConfig(name=name, n_features=3, window=6), seed=0)
        m.eval()
        x = torch.randn(5, 6, 3, dtype=torch.float64)
        with torch.no_grad():
            y = m(x)
            with ablate_circuits(m):
                y_abl = m(x)
            y_again = m(x)
        assert not torch.allclose(y, y_abl), name
        assert torch.allclose(y, y_again), name
        assert not any(b.ablate for b in m.blocks())


def test_reports_and_models_keep_history_on_overwrite(project):
    from backend.storage import list_models, paths, save_model, save_result
    from backend.storage.fileio import read_json

    for v in range(1, 9):
        save_result("comparison_reports", "history_probe", {"v": v})
    hist = sorted((paths.area("comparison_reports") / "_history" / "history_probe").glob("*.json"))
    assert len(hist) == 5 and read_json(hist[-1])["v"] == 7           # last 5 previous versions kept
    assert read_json(paths.area("comparison_reports") / "history_probe.json")["v"] == 8
    save_model({"a": 1}, "quantum_ai", "history_model_probe")
    save_model({"a": 2}, "quantum_ai", "history_model_probe")
    assert list((paths.models("quantum_ai") / "_history" / "history_model_probe").iterdir())
    names = [m["name"] for m in list_models("quantum_ai")]
    assert names.count("history_model_probe") == 1, "history copies must not appear as models"


def test_report_api_summary_omits_bulk_fields(project):
    from fastapi.testclient import TestClient

    from backend.main import app
    from backend.storage import save_result

    save_result("comparison_reports", "qseq_benchmark_FD002", {
        "name": "qseq_benchmark_FD002", "rul": {"summary": [], "runs": [{"model": "q_lstm", "history": [1, 2]}],
                                                "predictions": {"q_lstm": [1]}},
        "imputation": {"runs": [{"model": "q_lstm", "history": [1]}], "evaluation": []}})
    with TestClient(app) as c:
        s = c.get("/api/qseq/report", params={"subset": "FD002"}).json()
        full = c.get("/api/qseq/report", params={"subset": "FD002", "detail": "full"}).json()
        assert "predictions" not in s["rul"] and "history" not in s["rul"]["runs"][0]
        assert "history" not in s["imputation"]["runs"][0]
        assert full["rul"]["predictions"] == {"q_lstm": [1]}
        assert c.get("/api/qseq/report", params={"subset": "../x"}).status_code == 422


def test_environment_check_enforces_python_only_rule():
    from backend.workflows import check_environment

    r = check_environment()
    assert r["notebooks"] == [] and not r["missing"] and r["pip_ok"]
    assert {"torch", "pennylane", "qiskit", "langgraph", "fastapi"} <= set(r["versions"])


def test_rul_accuracy_table_matches_hand_computed_values():
    from backend.quantum_models.seq_experiments import rul_accuracy_table

    # 4 engines: true RUL 10, 20, 80, 200(uncapped; capped 125); predictions 15, 50, 70, 120
    preds = [{"rul_true": t, "rul_true_uncapped": tu, "rul_pred": p}
             for t, tu, p in ((10, 10, 15), (20, 20, 50), (80, 80, 70), (125, 200, 120))]
    row = rul_accuracy_table({"m": {"1": preds, "2": preds}})[0]
    # positives: engines 1, 2; predicted positive (≤ 30): engine 1 → TP 1, FN 1, TN 2
    assert row["accuracy"] == 0.75 and row["precision"] == 1.0 and row["recall"] == 0.5
    assert row["within_10"] == 0.75 and row["within_20"] == 0.75 and row["roc_auc"] == 1.0
    assert row["accuracy_std"] == 0.0 and row["n_engines"] == 4 and row["positives"] == 2
