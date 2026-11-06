"""Single entry point: run the full research workflow end to end, resumably.

Steps (in order):  environment → download → preprocess → train → quantum → sequence (incl. Qiskit check) → twin →
                   optimization → agents → evaluate → verify

    python scripts/run_pipeline.py                          # everything; skips steps already completed
    python scripts/run_pipeline.py --steps train,sequence   # only these
    python scripts/run_pipeline.py --restart                # ignore the step log and redo everything
    python scripts/run_pipeline.py --profile quick          # small epochs/subsamples to check the pipeline

Progress is recorded in <project>/configuration/pipeline_state.json (Google Drive in colab/desktop mode).
A step that fails stops the run (or is recorded and skipped with --keep-going); re-running resumes at the
first incomplete step, and training inside a step resumes from its checkpoints in storage.
"""

from __future__ import annotations

import json
import sys
import time

from _cli import csv, log, parser, run

STEPS = ("environment", "download", "preprocess", "train", "quantum", "sequence", "twin", "optimization", "agents",
         "evaluate", "verify")
ALWAYS_RUN = {"environment", "verify"}     # cheap checks that are repeated on every invocation
PROFILES = {
    "full":  {"rul_epochs": 40, "anomaly_epochs": 60, "qml_epochs": 20, "qml_train": 200, "seq_epochs": 10,
              "seq_imp_epochs": 8, "seq_train": 1200, "seeds": [42, 43, 44], "ims_tests": ["2nd_test", "1st_test", "3rd_test"]},
    "quick": {"rul_epochs": 3, "anomaly_epochs": 5, "qml_epochs": 3, "qml_train": 80, "seq_epochs": 2,
              "seq_imp_epochs": 2, "seq_train": 256, "seeds": [42], "ims_tests": ["2nd_test"]},
}

ap = parser(__doc__)
ap.add_argument("--steps", default=",".join(STEPS))
ap.add_argument("--skip", default="", help="steps to leave out")
ap.add_argument("--profile", choices=tuple(PROFILES), default="full")
ap.add_argument("--subset", default="FD001")
ap.add_argument("--restart", action="store_true", help="ignore completed steps in the step log")
ap.add_argument("--keep-going", action="store_true", help="record a failed step and continue with the next")
ap.add_argument("--no-llm", action="store_true")
ap.add_argument("--publish-docs", action="store_true", help="evaluate step writes into documentation/ (repository)")
a = ap.parse_args()


def step_functions(p: dict) -> dict:
    from backend import workflows as w

    s = a.subset
    return {
        "environment": lambda: w.check_environment(),
        "verify": lambda: w.verify_results(s),
        "download": lambda: w.download_datasets(),
        "preprocess": lambda: w.preprocess_datasets([s], p["ims_tests"]),
        "train": lambda: {"rul": w.train_rul_suite(s, epochs=p["rul_epochs"]),
                          "anomaly": w.train_anomaly_suite(p["ims_tests"], p["anomaly_epochs"])},
        "quantum": lambda: w.quantum_qml_suite(s, n_train=p["qml_train"], epochs=p["qml_epochs"]),
        "sequence": lambda: w.quantum_sequence_suite(s, p["seeds"], epochs=p["seq_epochs"],
                                                     imputation_epochs=p["seq_imp_epochs"], n_train=p["seq_train"]),
        "twin": lambda: w.run_digital_twin(subset=s),
        "optimization": lambda: w.run_optimization(subset=s),
        "agents": lambda: {"maintenance": w.run_maintenance_agents(subset=s, decision="approve", reviewer="pipeline",
                                                                   use_llm=not a.no_llm),
                           "missing_data": [w.run_missing_data_agents(f"{s}-test-unit34", rate, pat, use_llm=not a.no_llm)
                                            for rate, pat in ((0.2, "mcar"), (0.5, "block"))]},
        "evaluate": lambda: {"explainability": w.run_explainability(s), "docs": w.evaluate_all(publish=a.publish_docs)},
    }


def main():
    from backend.langchain_tools.tools import to_jsonable
    from backend.storage import paths
    from backend.storage.fileio import read_json, write_json

    wanted = [x for x in csv(a.steps) or STEPS if x not in (csv(a.skip) or [])]
    unknown = [x for x in wanted if x not in STEPS]
    if unknown:
        raise ValueError(f"unknown steps {unknown}; choose from {STEPS}")
    state_file = paths.configuration / "pipeline_state.json"
    state = {} if a.restart else (read_json(state_file) or {})
    state.setdefault("steps", {})
    profile = PROFILES[a.profile]
    fns = step_functions(profile)
    failed = []
    for name in (x for x in STEPS if x in wanted):
        prev = state["steps"].get(name, {})
        if (name not in ALWAYS_RUN and prev.get("status") == "completed" and prev.get("profile") == a.profile
                and prev.get("subset") == a.subset):
            log.info("── %s: already completed %s, skipping (use --restart to redo)", name, prev.get("finished"))
            continue
        log.info("── %s: starting", name)
        t0 = time.time()
        state["steps"][name] = {"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "profile": a.profile, "subset": a.subset}
        write_json(state_file, state)
        try:
            result = fns[name]()
            state["steps"][name].update(status="completed", seconds=round(time.time() - t0, 1),
                                        finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                        summary=json.loads(json.dumps(to_jsonable(result), default=str))
                                        if len(json.dumps(to_jsonable(result), default=str)) < 20000 else "saved")
            log.info("── %s: completed in %.0f s", name, time.time() - t0)
        except Exception as exc:
            state["steps"][name].update(status="failed", error=f"{exc.__class__.__name__}: {exc}",
                                        seconds=round(time.time() - t0, 1))
            write_json(state_file, state)
            log.exception("── %s: failed", name)
            failed.append(name)
            if not a.keep_going:
                raise
        write_json(state_file, state)
    log.info("pipeline state: %s", {k: v["status"] for k, v in state["steps"].items()})
    if failed:
        raise RuntimeError(f"steps failed: {failed}")
    return state


sys.exit(run("run_pipeline", a, main))
