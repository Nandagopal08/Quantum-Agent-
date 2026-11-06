"""Quantum sequence models, missing-data tools and the missing-data agent workflow (synthetic fixtures)."""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------- missing-data utilities


def test_masks_hit_requested_rates_and_patterns_are_detected():
    from backend.missing_data import PATTERNS, make_mask, missingness_profile

    rng = np.random.default_rng(0)
    for pattern in PATTERNS:
        for rate in (0.1, 0.3, 0.5):
            m = make_mask((200, 24, 6), rate, pattern, rng)
            p = missingness_profile(m)
            assert abs(p["missing_rate"] - rate) < 0.03, (pattern, rate, p["missing_rate"])
            assert p["likely_pattern"] == pattern
    assert make_mask((3, 10, 2), 0.0, "block", rng).all()
    with pytest.raises(ValueError):
        make_mask((1, 5, 1), 1.0, "mcar", rng)


def test_linear_imputer_matches_numpy_interp_and_never_reads_hidden_values():
    from backend.missing_data import impute_baseline, make_mask

    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 20, 4)).cumsum(1)
    m = make_mask(x.shape, 0.4, "mcar", rng)
    lin = impute_baseline("linear", x, m)
    for n in range(30):
        for c in range(4):
            o = m[n, :, c] > 0
            if o.any():
                ref = np.interp(np.arange(20), np.where(o)[0], x[n, o, c])
                assert np.allclose(lin[n, :, c], ref)
    poisoned = np.where(m > 0, x, 1e9)
    for method in ("mean", "locf", "linear"):
        out = impute_baseline(method, poisoned, m, means=np.zeros(4))
        assert np.abs(out).max() < 1e6, method
        assert np.allclose(out[m > 0], x[m > 0])


# ---------------------------------------------------------------- models


@pytest.mark.parametrize("name", ["q_lstm", "q_transformer", "q_cnn_lstm", "c_lstm", "c_transformer", "c_cnn_lstm"])
def test_sequence_models_shapes_and_gradients(name):
    from backend.quantum_models.qseq import SeqModelConfig, build_model

    for task, n_out in (("regression", 1), ("sequence", 5)):
        m = build_model(SeqModelConfig(name=name, n_features=5, n_out=n_out, task=task, window=12), seed=0)
        y = m(torch.randn(3, 12, 5, dtype=torch.float64))
        assert y.shape == ((3,) if task == "regression" else (3, 12, 5))
        y.sum().backward()
        grads = [p.grad for p in m.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        if name.startswith("q_"):
            qgrad = [p.grad for b in m.blocks() for p in b.q.parameters()]
            assert any(g.abs().sum() > 0 for g in qgrad), "circuit parameters must receive gradient"


def test_quantum_and_twin_have_comparable_size():
    from backend.quantum_models.qseq import ARCHS, SeqModelConfig, build_model

    for arch in ARCHS:
        q = build_model(SeqModelConfig(name=f"q_{arch}")).describe()
        c = build_model(SeqModelConfig(name=f"c_{arch}")).describe()
        assert q["quantum_params"] > 0 and c["quantum_params"] == 0
        assert abs(q["trainable_params"] - c["trainable_params"]) / c["trainable_params"] < 0.1


def test_vqc_block_equals_direct_pennylane_circuit():
    import pennylane as qml

    from backend.quantum_models.qseq import VQCBlock

    torch.manual_seed(0)
    b = VQCBlock(3, 2, n_qubits=3, layers=2).double()
    x = torch.randn(4, 3, dtype=torch.float64)
    w = next(iter(b.q.parameters())).detach()
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def ref(z):
        qml.AngleEmbedding(z, wires=range(3), rotation="Y")
        qml.StronglyEntanglingLayers(w, wires=range(3))
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    with torch.no_grad():
        z = torch.tanh(b.inp(x)) * (np.pi / 2)
        expected = b.out(torch.stack([torch.stack(list(ref(zi))) for zi in z]).to(torch.float64))
        assert torch.allclose(b(x), expected, atol=1e-8)


def test_saved_model_roundtrip():
    from backend.quantum_models.qseq import SeqModelConfig, build_model, model_from_state, state_to_numpy

    m = build_model(SeqModelConfig(name="q_transformer", n_features=4, window=8), seed=3)
    m.eval()
    x = torch.randn(2, 8, 4, dtype=torch.float64)
    m2 = model_from_state(m.cfg.to_dict(), state_to_numpy(m))
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x))


# ---------------------------------------------------------------- benchmark + the missing-data agent (synthetic C-MAPSS)


@pytest.fixture(scope="module")
def qseq_report(project):
    from backend.quantum_models.seq_experiments import SeqExperimentConfig, run_benchmark

    cfg = SeqExperimentConfig(window=12, n_train=96, n_val=48, n_eval=48, epochs=1, imputation_epochs=1,
                              models=("q_lstm", "q_transformer", "q_cnn_lstm", "c_lstm"), missing_rates=(0.2, 0.5))
    return run_benchmark(cfg, seeds=(1,))


def test_benchmark_report_structure(qseq_report):
    r = qseq_report
    assert {x["model"] for x in r["rul"]["summary"]} == {"q_lstm", "q_transformer", "q_cnn_lstm", "c_lstm"}
    assert all(np.isfinite(x["rmse"]) for x in r["rul"]["summary"])
    methods = {x["method"] for x in r["imputation"]["evaluation"]}
    assert methods == {"mean", "locf", "linear", "q_lstm", "q_transformer", "q_cnn_lstm", "c_lstm"}
    assert {(x["pattern"], x["rate"]) for x in r["imputation"]["evaluation"]} == \
        {(p, q) for p in ("mcar", "block") for q in (0.2, 0.5)}
    rank = r["imputation"]["ranking"]["mcar"]["0.2"]
    assert [x["rmse"] for x in rank] == sorted(x["rmse"] for x in rank)
    down = {(x["method"], x["rul_model"]) for x in r["downstream_rul"]}
    assert any(m == "complete_data" for m, _ in down)
    assert all(rm in ("q_lstm", "q_transformer", "q_cnn_lstm", "c_lstm") for _, rm in down)
    # learned imputers keep observed values and start from linear interpolation, so they never do much worse
    by = {(x["pattern"], x["rate"], x["method"]): x["rmse"] for x in r["imputation"]["evaluation"]}
    assert by[("mcar", 0.5, "linear")] < by[("mcar", 0.5, "mean")]


def test_missing_data_agents_end_to_end(qseq_report):
    from backend.agents.missing_data import tools
    from backend.agents.missing_data.runner import execute, get_run

    tools.clear_caches()
    r = execute("mdtest1", {"asset_id": "FD001-test-unit3", "missing_rate": 0.3, "pattern": "block", "seed": 1,
                            "use_llm": False})
    s = r["state"]
    assert r["status"] == "completed", r
    assert [t["agent"] for t in s["trace"]] == ["md_agent"]                  # a single agent
    steps = [x["step"] for x in s["steps"]]
    assert steps[0] == "data_quality" and steps[-1] == "report" and "forecast" in steps
    assert all(x["status"] == "completed" for x in s["steps"])
    assert s["quality"]["likely_pattern"] == "block"
    assert s["forecast"]["prediction"]["model"].startswith("q_")
    assert np.isfinite(s["report"]["evaluation"]["imputation_rmse_true"])
    # agents never see the truth: the observation has no hidden values
    assert "truth" not in s["observation"] and "rul_true" not in s["observation"]
    imputed = np.asarray(s["imputation"]["values"], float)
    obs = s["observation"]
    for t, row in enumerate(obs["values"]):
        for c, v in enumerate(row):
            if v is not None:
                assert imputed[t, c] == pytest.approx(v)
    assert get_run("mdtest1")["status"] == "completed"


def test_critic_rejects_bad_imputer_and_planner_replans(qseq_report):
    from backend.agents.missing_data.runner import execute

    r = execute("mdtest2", {"asset_id": "FD001-test-unit3", "missing_rate": 0.4, "pattern": "mcar", "seed": 2,
                            "imputer": "mean", "use_llm": False})
    s = r["state"]
    tried = [a["method"] for a in s["attempts"]]
    assert tried[0] == "mean" and len(tried) >= 2
    assert s["route_log"][1] == {**s["route_log"][1], "from": "check", "to": "plan"}
    assert s["imputation"]["method"] != "mean"
    assert s["report"]["summary"]["rejected"][0] == "mean"


def test_agents_stop_when_too_much_is_missing(qseq_report):
    from backend.agents.missing_data.runner import execute

    r = execute("mdtest3", {"asset_id": "FD001-test-unit3", "missing_rate": 0.97, "seed": 0, "use_llm": False})
    assert r["status"] == "insufficient_data"
    assert [x["step"] for x in r["state"]["steps"]] == ["data_quality", "report"]


def test_qseq_api(qseq_report):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        assert "q_lstm" in c.get("/api/qseq/models").json()["models"]
        assert c.get("/api/qseq/report").json()["name"] == "qseq_benchmark_FD001"
        specs = c.get("/api/qseq/agents/specs").json()["agents"]
        assert len(specs) == 1 and specs[0]["key"] == "md_agent"
        assert specs[0]["extra"]["steps"] == ["data_quality", "plan", "impute", "check", "forecast", "report"]
        assert c.post("/api/qseq/agents/runs", json={"missing_rate": 1.2}).status_code == 422
        assert c.post("/api/qseq/agents/runs", json={"imputer": "magic"}).status_code == 422
        assert c.get("/api/qseq/agents/runs/does-not-exist").status_code == 404


def test_qiskit_execution_reproduces_pennylane():
    from backend.quantum_models.qiskit_backend import compare_backends, to_qasm, vqc_circuit
    from backend.quantum_models.qseq import QUANTUM_SEQ_MODELS, SeqModelConfig, build_model

    rng = np.random.default_rng(0)
    for name in QUANTUM_SEQ_MODELS:
        m = build_model(SeqModelConfig(name=name, n_features=3, window=6), seed=2)
        m.eval()
        r = compare_backends(m, rng.normal(size=(3, 6, 3)))
        assert r["max_abs_diff"] < 1e-9 and r["qiskit_circuit_evaluations"] > 0, name
    w = rng.normal(size=(2, 4, 3))
    qc = vqc_circuit([0.1, 0.2, 0.3, 0.4], w)
    assert qc.num_qubits == 4 and qc.count_ops()["cx"] == 8
    assert to_qasm([0.1, 0.2, 0.3, 0.4], w).startswith("OPENQASM 3")


def test_scripts_have_no_notebooks_and_parse_help():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    found = [p for p in root.rglob("*.ipynb") if ".venv" not in p.parts and "node_modules" not in p.parts]
    assert not found, found
    for script in ("run_pipeline.py", "run_quantum_experiments.py", "run_agent_workflow.py", "train_all_models.py",
                   "download_datasets.py", "setup_environment.py", "run_backend.py", "evaluate_all.py"):
        r = subprocess.run([sys.executable, str(root / "scripts" / script), "--help"], capture_output=True, text=True,
                           timeout=120)
        assert r.returncode == 0 and "usage" in r.stdout.lower(), (script, r.stderr[-500:])


def test_static_route_cannot_escape_frontend_dist(project):
    from pathlib import Path

    from fastapi.testclient import TestClient

    from backend.main import app

    if not (Path(__file__).resolve().parents[2] / "frontend" / "dist" / "index.html").exists():
        pytest.skip("dashboard not built")
    with TestClient(app) as c:
        for p in ("/%2e%2e/%2e%2e/requirements.txt", "/..%2F..%2Frequirements.txt", "/%2e%2e/package.json"):
            r = c.get(p)
            assert b"fastapi" not in r.content and b"qadt-dashboard" not in r.content, p
            assert r.headers["content-type"].startswith("text/html"), p
        assert c.get("/api/does-not-exist").status_code == 404


# ---------------------------------------------------------------- agent edge cases (fixture trains q_lstm,
# q_transformer, q_cnn_lstm and c_lstm only, so c_transformer / c_cnn_lstm are "missing models")


def _md(run_id, **req):
    from backend.agents.missing_data.runner import execute

    base = {"asset_id": "FD001-test-unit3", "missing_rate": 0.3, "pattern": "mcar", "seed": 4, "use_llm": False}
    return execute(run_id, {**base, **req})


def test_planner_falls_back_when_requested_model_is_missing(qseq_report):
    r = _md("mdmiss1", imputer="c_transformer")
    s = r["state"]
    first = s["attempts"][0]
    assert first["method"] == "c_transformer" and "failed" in first
    assert s["route_log"][1]["to"] == "plan"                          # the check sent it back
    assert s["imputation"]["method"] != "c_transformer" and r["status"] == "completed"


def test_missing_rul_model_gives_report_without_forecast(qseq_report):
    r = _md("mdmiss2", rul_model="c_transformer")
    assert r["status"] == "completed_without_forecast"
    assert any(x["step"] == "forecast" and x["status"] == "failed" for x in r["state"]["steps"])
    assert r["state"]["report"]["summary"]["rul_pred"] is None


def test_retry_limit_is_respected_and_falls_back_to_best_attempt(qseq_report):
    r = _md("mdmax1", imputer="mean", max_attempts=1, missing_rate=0.4)
    s = r["state"]
    assert len(s["attempts"]) == 1
    rev = s["review"]
    assert rev["accepted"] is False and rev["retry"] is False and rev["confidence"] == "low"
    assert rev["fallback_method"] == "mean"
    assert [x["step"] for x in s["steps"]].count("plan") == 1


def test_reporter_rejects_invalid_or_ungrounded_llm_output(qseq_report, monkeypatch):
    from backend.agents.missing_data import graph

    class Fake:
        def __init__(self, result):
            self.result = result

        def with_structured_output(self, schema):
            outer = self

            class Runner:
                def invoke(self, _msgs):
                    if isinstance(outer.result, Exception):
                        raise outer.result
                    return outer.result
            return Runner()

    cases = {
        "ungrounded": (graph.ReporterNarrative(summary="The engine has 987.6 cycles left."), "ungrounded"),
        "error": (RuntimeError("model timed out"), "rejected"),
        "invalid": ({"caveats": ["no summary"]}, "rejected"),
        "grounded": (graph.ReporterNarrative(summary="Imputation was accepted and a forecast was made."), None),
    }
    for name, (result, reason) in cases.items():
        monkeypatch.setattr(graph, "get_chat_model", lambda *a, _r=result, **k: Fake(_r))
        rep = _md(f"mdllm{name}", use_llm=True)["state"]["report"]
        if reason is None:
            assert rep["llm"]["used"] is True and rep["text"].startswith("Imputation was accepted")
        else:
            assert rep["llm"]["used"] is False and reason in rep["llm"]["reason"], (name, rep["llm"])
            assert rep["text"] == rep["deterministic_text"]


def test_ground_truth_never_reaches_state_or_event_stream(qseq_report):
    import json

    from backend.agents.missing_data.graph import TRUTH_VAULT
    from backend.events import bus

    r = _md("mdtruth1", missing_rate=0.4)
    assert "evaluation_truth" not in r["state"] and "truth" not in json.dumps(r["state"]["observation"])
    assert r["state"]["report"]["evaluation"]["imputation_rmse_true"] >= 0      # scoring still happened
    events = json.dumps(bus.history("run:mdtruth1"), default=str)
    assert '"truth"' not in events and "rul_true" not in events
    assert "mdtruth1" not in TRUTH_VAULT


def test_uploaded_windows_are_validated(qseq_report):
    from backend.agents.missing_data.tools import expected_input

    exp = expected_input("FD001")
    W, C = exp["window"], len(exp["sensors"])
    good = [[0.1 if (t + c) % 4 else None for c in range(C)] for t in range(W + 5)]
    r = _md("mdup1", asset_id=None, values=good)
    assert r["status"] == "completed" and len(r["state"]["observation"]["values"]) == W
    assert r["state"]["report"]["evaluation"] == {}                   # no truth for uploaded data
    bad_channels = [[0.0] * (C - 1) for _ in range(W)]
    r = _md("mdup2", asset_id=None, values=bad_channels)
    assert r["status"] == "failed" and "channels" in r["state"]["trace"][0]["summary"]
    r = _md("mdup3", asset_id=None, values=[[0.0] * C for _ in range(W - 2)])
    assert r["status"] == "failed" and "steps" in r["state"]["trace"][0]["summary"]


def test_interrupted_runs_are_marked_and_can_be_rerun(qseq_report):
    import time

    from backend.agents.missing_data import runner
    from backend.storage.fileio import write_json

    write_json(runner._folder() / "mdstale1.json", {"run_id": "mdstale1", "status": "running", "created_at": 0,
                                                    "request": {"asset_id": "FD001-test-unit3", "missing_rate": 0.2,
                                                                "pattern": "mcar", "seed": 1, "use_llm": False}})
    assert runner.mark_interrupted() >= 1
    assert runner.get_run("mdstale1")["status"] == "interrupted"
    new_id = runner.rerun("mdstale1")
    t0 = time.time()
    while time.time() - t0 < 120 and (runner.get_run(new_id) or {}).get("status") == "running":
        time.sleep(0.25)
    rec = runner.get_run(new_id)
    assert rec["status"] == "completed" and rec["request"]["rerun_of"] == "mdstale1"


def test_single_agent_retry_loop_is_bounded_even_if_the_check_misbehaves(qseq_report, monkeypatch):
    from backend.agents.missing_data import graph

    monkeypatch.setattr(graph, "_critic_decision", lambda state: {"accepted": False, "retry": True, "reasons": ["forced"],
                                                                  "attempt": 0, "max_attempts": 99, "confidence": "low"})
    r = _md("mdbound1", max_attempts=2)
    steps = [x["step"] for x in r["state"]["steps"]]
    assert steps.count("plan") == 2 and steps.count("check") == 2 and steps[-1] == "report"

    def boom(state):
        raise RuntimeError("check crashed")

    monkeypatch.setattr(graph, "critic", boom)
    r = _md("mdbound2")
    s = r["state"]
    assert [x["step"] for x in s["steps"]].count("plan") == 1             # a crashing check stops the loop
    assert any(x["step"] == "check" and x["status"] == "failed" for x in s["steps"])
    assert r["status"] in ("completed", "completed_without_forecast")


def test_agent_status_reports_readiness(qseq_report):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        s = c.get("/api/qseq/agents/status").json()
        assert s["ready"] is True and s["benchmark_report"] is True and s["hint"] is None
        assert {"q_lstm", "q_transformer", "q_cnn_lstm", "c_lstm"} <= set(s["rul_models"])
        assert {"linear", "q_lstm"} <= set(s["imputers"])
        s2 = c.get("/api/qseq/agents/status", params={"subset": "FD003"}).json()
        assert s2["ready"] is False and "run_quantum_seq.py FD003" in s2["hint"]
