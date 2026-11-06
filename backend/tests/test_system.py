"""End-to-end backend tests on synthetic fixtures (see conftest.py).

Run:  .venv/bin/python -m pytest backend/tests -q
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest


# ---------------------------------------------------------------- datasets


def test_cmapss_pipeline_has_no_leakage(project):
    from backend.datasets import cmapss

    d = cmapss.prepare(cmapss.CmapssConfig("FD001"))
    assert not set(d.train["unit"]) & set(d.val["unit"]), "validation engines must not appear in training"
    assert set(d.dropped_sensors) >= {"s1", "s5", "s10", "s16", "s18", "s19"}
    assert d.test_last["unit"].nunique() == len(d.test_last) == 10
    assert (d.train["rul"] <= 125).all() and (d.train.groupby("unit")["rul_true"].min() == 0).all()
    X, y, _ = cmapss.make_sequences(d.val, d.feature_cols, 30)
    assert X.shape[1:] == (30, len(d.feature_cols)) and len(X) == len(y)
    eda = cmapss.eda("FD001")
    assert eda["train_units"] == 30 and eda["top_sensors"]


def test_ims_features_and_labels(project):
    from backend.datasets import ims

    assert ims.available_tests() == ["2nd_test"]
    data = ims.prepare_anomaly("2nd_test")
    f = data.features
    assert f["bearing"].nunique() == 4 and f["file_index"].nunique() == 80
    late_b1 = f[(f["bearing"] == 1) & (f["life_fraction"] > 0.9)]["env_bpfo"].mean()
    early_b1 = f[(f["bearing"] == 1) & f["is_baseline"]]["env_bpfo"].mean()
    assert late_b1 > 5 * early_b1, "envelope feature should pick up the 236 Hz outer-race fault"
    assert set(data.eval_df["label"].unique()) == {0, 1}


# ---------------------------------------------------------------- models


@pytest.fixture(scope="module")
def xgb(project):
    from backend.classical_models.rul import load_rul_model, train_rul_model

    result = train_rul_model("xgboost", "FD001")
    return result, load_rul_model("xgboost", "FD001")


def test_rul_training_saves_and_reloads(xgb):
    result, model = xgb
    assert result["test"]["rmse"] < 40 and np.isfinite(result["test"]["nasa_score"])
    assert model.sigma > 0 and model.q90 > 0
    assert 0 <= result["failure_classification"]["accuracy"] <= 1


def test_sequence_model_checkpoints_and_resumes(project):
    from backend.classical_models.rul import train_rul_model
    from backend.storage import CheckpointManager

    train_rul_model("gru", "FD001", epochs=2)
    ck = CheckpointManager("rul_gru_FD001", category="classical_ai")
    assert ck.latest()["step"] == 1
    r = train_rul_model("gru", "FD001", epochs=3)          # resumes at epoch 2 from the Drive checkpoint
    log = [l for l in __import__("backend.storage", fromlist=["x"]).ExperimentLogger("rul_gru_FD001").records()
           if "epoch" in l]
    assert [l["epoch"] for l in log][-1] == 2 and ck.latest()["step"] == 2
    assert np.isfinite(r["test"]["rmse"])


def test_anomaly_detector_detects_fault(project):
    from backend.classical_models.anomaly import train_detector

    r = train_detector("isolation_forest", "2nd_test")
    assert r["metrics"]["roc_auc"] > 0.9
    assert r["detection"]["1"]["lead_time_hours"] > 0


def test_quantum_kernel_runs_and_records_circuit(project):
    from backend.quantum_models.qml import QuantumConfig, run_experiment

    r = run_experiment(QuantumConfig(model="quantum_kernel", n_qubits=3, layers=1, n_train=60))
    q = r["quantum"]
    assert q["qubits"] == 3 and q["circuit_depth"] > 0 and q["gate_count"] > 0
    assert r["classical_baselines"] and "quantum advantage" in r["comparison"]["note"]


# ---------------------------------------------------------------- risk, twin, counterfactual


def test_risk_score_is_pof_times_cof(project):
    from backend.risk.engine import assess_turbofan, consequence_of_failure

    a = assess_turbofan("x", 8, sigma=10, q90=15)
    assert a.risk_score == pytest.approx(a.probability_of_failure * consequence_of_failure("turbofan"))
    assert a.risk_level == "critical" and a.requires_approval
    assert assess_turbofan("y", 120, sigma=10, q90=15).risk_level == "low"


def test_counterfactual_prefers_maintenance_for_failing_asset(project):
    from backend.counterfactual.scenarios import AssetState, compare_scenarios

    r = compare_scenarios(AssetState("x", "turbofan", 8, 10), horizon=60, samples=3000)
    by = {s["scenario"]: s for s in r["scenarios"]}
    assert r["best"] != "continue_operation"
    assert by["immediate_maintenance"]["failure_probability"] < 0.05, "a running asset cannot have failed at t=0"
    assert by["continue_operation"]["failure_probability"] > 0.9
    assert all(s["source"] == "simulated" for s in r["scenarios"])
    healthy = compare_scenarios(AssetState("y", "turbofan", 400, 10), horizon=60, samples=2000)
    assert healthy["best"] == "continue_operation"


def test_twin_separates_measured_and_simulated(project, xgb):
    from backend.digital_twin.twin import SimulationParams
    from backend.services import fleet

    tw = fleet.turbofan_twin("FD001-test-unit1")
    assert {p["source"] for p in tw.playback()["series"]} == {"measured"}
    sim = tw.simulate(SimulationParams(horizon=20, maintenance_at=5))
    assert {p["source"] for p in sim["series"]} == {"simulated"} and sim["series"][4]["maintenance"]


# ---------------------------------------------------------------- optimisation


def test_qubo_minimum_equals_true_optimum(project):
    from backend.optimization.problem import Asset, default_problem, qubo_energy
    from backend.optimization.solvers import solve_brute_force, solve_milp

    p = default_problem([Asset("a", "turbofan", 10, 12), Asset("b", "turbofan", 30, 12), Asset("c", "bearing", 15, 5)])
    Q, off, nv = p.to_qubo()
    e, bits = min((qubo_energy(Q, off, np.array(b)), b) for b in itertools.product((0, 1), repeat=nv))
    exact = solve_brute_force(p)
    assert e == pytest.approx(exact["total_expected_cost"], rel=1e-9)
    assert p.evaluate(np.array(bits))["feasible"]
    assert solve_milp(p)["total_expected_cost"] == pytest.approx(exact["total_expected_cost"], rel=1e-9)


def test_all_solvers_return_feasible_plans(project):
    from backend.optimization.problem import Asset, default_problem
    from backend.optimization.solvers import compare_solvers

    p = default_problem([Asset("a", "turbofan", 10, 12), Asset("b", "turbofan", 20, 12),
                         Asset("c", "turbofan", 90, 12)])
    r = compare_solvers(p, qaoa_layers=1, qaoa_steps=30)
    names = {x["solver"] for x in r["results"]}
    assert names == {"milp_highs", "greedy", "simulated_annealing", "qaoa"}
    for x in r["results"]:
        assert x["feasible"] and x["gap_percent"] >= -1e-6
    q = next(x for x in r["results"] if x["solver"] == "qaoa")
    assert q["qubits"] > 0 and q["circuit_depth"] > 0


# ---------------------------------------------------------------- LLM guard


def test_llm_number_validator():
    from backend.agents.llm import ungrounded_numbers

    ctx = {"pof": 0.939, "cost": 478953, "id": "FD001-test-unit81"}
    assert ungrounded_numbers("unit81 has PoF 93.9% and risk 478,953 USD", ctx) == []
    assert ungrounded_numbers("unit81 has PoF 37.5% and cost 999,999", ctx) == ["37.5%", "999,999"]


# ---------------------------------------------------------------- agents


def _wait(runner, run_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        row = runner.db.get_row(runner.db.agent_runs, run_id)
        if row["status"] not in ("running", "resuming"):
            return row
        time.sleep(0.2)
    raise TimeoutError


def test_agent_workflow_with_human_approval(project, xgb):
    from backend.agents import runner
    from backend.services import fleet

    fleet.clear_caches()
    ids = fleet.default_assets("FD001", 3)
    row = runner.run_sync({"asset_ids": ids, "horizon": 60, "use_llm": False})
    trace = [t["agent"] for t in row["state"]["trace"]]
    assert trace[:3] == ["data_monitoring", "predictive_maintenance", "risk_assessment"]
    assert row["state"]["decision"]["narrative_source"] == "deterministic"
    for a in row["state"]["decision"]["actions"]:
        assert a["risk_score"] == pytest.approx(row["state"]["risk"][a["asset_id"]]["risk_score"])
    if row["status"] == "awaiting_approval":
        runner.submit_approval(row["id"], True, "pytest")
        done = _wait(runner, row["id"])
        assert done["status"] == "approved" and done["approval"]["work_orders"]
        with pytest.raises(ValueError):
            runner.submit_approval(row["id"], True, "again")


def test_low_risk_assets_skip_simulation(project, xgb):
    from backend.agents import runner
    from backend.services import fleet

    healthy = fleet.fleet_overview("FD001")["assets"][-1]
    if healthy["risk_level"] != "low":
        pytest.skip("synthetic fleet has no low-risk engine")
    row = runner.run_sync({"asset_ids": [healthy["asset_id"]], "use_llm": False})
    agents = [t["agent"] for t in row["state"]["trace"]]
    assert "digital_twin" not in agents and agents[-1] == "decision_coordination"
    assert row["state"]["route_log"][0]["to"] == "decision_coordination"


def test_agent_fails_cleanly_on_unknown_asset(project):
    from backend.agents import runner

    row = runner.run_sync({"asset_ids": ["FD001-test-unit9999"], "use_llm": False})
    assert row["status"] == "failed" and row["state"]["errors"]


# ---------------------------------------------------------------- API


@pytest.fixture(scope="module")
def client(project, xgb):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        yield c


def test_api_read_endpoints(client):
    for path in ("/api/health", "/api/overview", "/api/storage", "/api/datasets", "/api/models",
                 "/api/fleet/turbofan", "/api/assets/FD001-test-unit1/prediction", "/api/twin/FD001-test-unit1/playback",
                 "/api/analysis/cmapss/FD001", "/api/agents/specs", "/api/settings", "/api/experiments/comparisons"):
        r = client.get(path)
        assert r.status_code == 200, (path, r.text[:300])
    assert client.get("/api/health").json()["storage"]["is_google_drive"] is False


def test_api_errors_are_mapped(client):
    assert client.get("/api/assets/nope/prediction").status_code == 404
    assert client.post("/api/training/rul", json={"model": "magic"}).status_code == 422
    assert client.get("/api/assets/FD001-test-unit1/prediction", params={"model": "lstm"}).status_code == 409


def test_api_counterfactual_and_simulation(client):
    r = client.post("/api/counterfactual", json={"asset_id": "FD001-test-unit1"}).json()
    assert len(r["scenarios"]) == 5 and r["explanation"]
    s = client.post("/api/twin/FD001-test-unit1/simulate", json={"horizon": 15, "load": 1.3}).json()
    assert len(s["series"]) == 15


def test_api_background_job_and_settings(client):
    job = client.post("/api/training/rul", json={"model": "random_forest"}).json()
    t0 = time.time()
    while time.time() - t0 < 120:
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("completed", "failed"):
            break
        time.sleep(0.3)
    assert j["status"] == "completed", j["error"]
    s = client.put("/api/settings", json={"risk": {"thresholds": {"critical": 0.7}}}).json()
    assert s["risk"]["thresholds"]["critical"] == 0.7 and s["risk"]["thresholds"]["high"] == 0.3
    client.post("/api/settings/reset")


def test_drive_required_when_not_mounted(project):
    from backend import config
    from backend.storage import DriveNotMountedError, require_drive

    saved = config.get_config()
    config.configure(storage_mode="colab", drive_mount="/nonexistent/drive",
                     project_root="/nonexistent/drive/MyDrive/x")
    try:
        with pytest.raises(DriveNotMountedError, match="drive.mount"):
            require_drive()
    finally:
        config.configure(storage_mode=saved.storage_mode, drive_mount=saved.drive_mount,
                         project_root=saved.project_root)
