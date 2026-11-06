"""Quantum sequence models (QLSTM / Quantum Transformer / Quantum CNN-LSTM), missing-data benchmark and the
single missing-data agent."""

from __future__ import annotations

import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from .. import jobs
from ..agents.missing_data import runner as md_runner
from ..agents.missing_data import tools as md_tools
from ..agents.missing_data.graph import agent_specs, mermaid
from ..missing_data import BASELINE_IMPUTERS, PATTERNS
from ..quantum_models.qseq import SEQ_MODELS
from ..quantum_models.seq_experiments import SeqExperimentConfig, load_report, run_benchmark
from ..storage import require_drive
from .common import sse

router = APIRouter(prefix="/api/qseq", tags=["quantum sequence models"])
ASSET_RE = re.compile(r"FD00[1-4]-(test|train)-unit\d{1,3}")
SeqName = Literal["q_lstm", "q_transformer", "q_cnn_lstm", "c_lstm", "c_transformer", "c_cnn_lstm"]


class BenchmarkRequest(BaseModel):
    subset: Literal["FD001", "FD002", "FD003", "FD004"] = "FD001"
    models: list[SeqName] = Field(default_factory=lambda: list(SEQ_MODELS), min_length=1)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=5)
    epochs: int = Field(10, ge=1, le=100)
    imputation_epochs: int = Field(8, ge=1, le=100)
    n_train: int = Field(1200, ge=64, le=10_000)
    n_qubits: int = Field(4, ge=2, le=8)
    layers: int = Field(2, ge=1, le=6)
    missing_rates: list[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5], min_length=1)

    @model_validator(mode="after")
    def _rates(self):
        if any(not 0 < r < 0.95 for r in self.missing_rates):
            raise ValueError("missing rates must be in (0, 0.95)")
        return self


class AgentRunRequest(BaseModel):
    asset_id: str | None = Field("FD001-test-unit34", description="C-MAPSS engine, e.g. FD001-test-unit34")
    missing_rate: float = Field(0.3, ge=0, lt=1)
    pattern: Literal["mcar", "block"] = "mcar"
    seed: int = 0
    imputer: str | None = Field(None, description="Force the first imputer (the agent's check may still reject it)")
    rul_model: SeqName | None = None
    max_attempts: int = Field(3, ge=1, le=6)
    use_llm: bool = True
    values: list[list[float | None]] | None = Field(None, max_length=500,
                                                   description="Upload a window instead (T × C, null = missing)")
    sensors: list[str] | None = Field(None, max_length=64)
    subset: str = "FD001"

    @model_validator(mode="after")
    def _check(self):
        if self.values is None and not self.asset_id:
            raise ValueError("give asset_id or values")
        if self.values is not None and any(len(r) > 64 for r in self.values):
            raise ValueError("at most 64 channels")
        if self.values is None and not ASSET_RE.fullmatch(self.asset_id or ""):
            raise ValueError("asset_id must look like FD001-test-unit34")
        if self.imputer and self.imputer not in (*BASELINE_IMPUTERS, *SEQ_MODELS):
            raise ValueError(f"imputer must be one of {(*BASELINE_IMPUTERS, *SEQ_MODELS)}")
        return self


@router.get("/models")
def models():
    return {"models": list(SEQ_MODELS), "baseline_imputers": list(BASELINE_IMPUTERS), "patterns": list(PATTERNS)}


@router.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    require_drive()
    cfg = SeqExperimentConfig(subset=req.subset, models=tuple(req.models), epochs=req.epochs,
                              imputation_epochs=req.imputation_epochs, n_train=req.n_train, n_qubits=req.n_qubits,
                              layers=req.layers, missing_rates=tuple(sorted(req.missing_rates)))

    def work(progress):
        report = run_benchmark(cfg, seeds=tuple(req.seeds), progress=progress)
        md_tools.clear_caches()
        return {"name": report["name"], "total_seconds": report["total_seconds"]}

    return jobs.submit("qseq_benchmark", req.model_dump(), work)


@router.get("/report")
def report(subset: str = "FD001", detail: Literal["summary", "full"] = "summary"):
    """The benchmark report. ``summary`` (default) omits per-epoch training histories and per-engine predictions,
    which the dashboard does not display; ``full`` returns the saved file unchanged."""
    require_drive()
    if not re.fullmatch(r"FD00[1-4]", subset):
        raise HTTPException(422, "subset must be FD001…FD004")
    r = _cached_report(subset)
    if r is None:
        raise HTTPException(404, "no quantum sequence benchmark yet; run POST /api/qseq/benchmark "
                                 "or scripts/run_quantum_seq.py")
    r = {**r, "rul": {**r["rul"], "accuracy": _accuracy(subset, r)}}
    if detail == "full":
        return r
    out = {**r, "rul": {k: v for k, v in r["rul"].items() if k != "predictions"},
           "imputation": {**r["imputation"]}}
    out["rul"]["runs"] = [{k: v for k, v in x.items() if k != "history"} for x in r["rul"].get("runs", [])]
    out["imputation"]["runs"] = [{k: v for k, v in x.items() if k != "history"} for x in r["imputation"].get("runs", [])]
    return out


_REPORT_CACHE: dict[str, tuple[float, dict]] = {}
_ACC_CACHE: dict[str, tuple[str, list]] = {}


def _accuracy(subset: str, report: dict) -> list | None:
    """Failure-within-30-cycles accuracy table from the raw per-seed predictions (None if the raw file is missing
    or belongs to another run)."""
    from ..quantum_models.seq_experiments import rul_accuracy_table
    from ..storage import paths
    from ..storage.fileio import read_json

    tag = report.get("run_tag") or ""
    hit = _ACC_CACHE.get(subset)
    if hit and hit[0] == tag:
        return hit[1]
    raw = read_json(paths.area("comparison_reports") / f"qseq_raw_{subset}.json")
    table = rul_accuracy_table(raw["rul_predictions"]) if raw and raw.get("run_tag") == tag else None
    _ACC_CACHE[subset] = (tag, table)
    return table


def _cached_report(subset: str) -> dict | None:
    """Re-read the report only when the file changes (it is ~MBs and read on every page load)."""
    from ..storage import paths

    f = paths.area("comparison_reports") / f"qseq_benchmark_{subset}.json"
    try:
        mtime = f.stat().st_mtime
    except FileNotFoundError:
        return None
    hit = _REPORT_CACHE.get(subset)
    if hit and hit[0] == mtime:
        return hit[1]
    r = load_report(subset)
    if r is not None:
        _REPORT_CACHE[subset] = (mtime, r)
    return r


@router.get("/qiskit")
def qiskit_verification(subset: str = "FD001"):
    """Latest Qiskit-vs-PennyLane verification (circuit-level gradients and model-level outputs)."""
    require_drive()
    from ..storage import paths
    from ..storage.fileio import read_json

    if not re.fullmatch(r"FD00[1-4]", subset):
        raise HTTPException(422, "subset must be FD001…FD004")
    r = read_json(paths.area("comparison_reports") / f"qseq_qiskit_verification_{subset}.json")
    if r is None:
        raise HTTPException(404, "no Qiskit verification yet; run scripts/run_quantum_experiments.py --suite qiskit")
    return r


@router.get("/agents/status")
def agent_status(subset: str = "FD001"):
    """Is the agent ready? It needs the quantum-sequence benchmark report (imputer ranking) and trained models."""
    require_drive()
    if not re.fullmatch(r"FD00[1-4]", subset):
        raise HTTPException(422, "subset must be FD001…FD004")
    from ..storage import list_models

    names = {m["name"] for m in list_models("quantum_ai")}
    imputers = [m for m in SEQ_MODELS if f"qseq_impute_{m}_{subset}" in names]
    rul = [m for m in SEQ_MODELS if f"qseq_rul_{m}_{subset}" in names]
    report = _cached_report(subset) is not None
    ready = report and any(m.startswith("q_") for m in rul)
    return {"subset": subset, "ready": ready, "benchmark_report": report, "imputers": list(BASELINE_IMPUTERS) + imputers,
            "rul_models": rul,
            "hint": None if ready else f"train the models first: python scripts/run_quantum_seq.py {subset} --seeds 42"}


@router.get("/agents/specs")
def specs():
    return {"agents": agent_specs(), "graph_mermaid": mermaid()}


@router.post("/agents/runs")
def start_run(req: AgentRunRequest):
    require_drive()
    return {"run_id": md_runner.start_run(req.model_dump())}


@router.get("/agents/runs")
def runs(limit: int = 30):
    require_drive()
    return md_runner.list_runs(limit)


@router.get("/agents/runs/{run_id}")
def run(run_id: str):
    require_drive()
    r = md_runner.get_run(run_id)
    if r is None:
        raise HTTPException(404, "run not found")
    return r


@router.post("/agents/runs/{run_id}/rerun")
def rerun(run_id: str):
    require_drive()
    try:
        return {"run_id": md_runner.rerun(run_id), "rerun_of": run_id}
    except KeyError:
        raise HTTPException(404, "run not found") from None


@router.get("/agents/runs/{run_id}/events")
def run_events(run_id: str, request: Request):
    return sse(f"run:{run_id}", request, replay=True,
               stop_types=("completed", "failed", "insufficient_data", "completed_without_forecast"))
