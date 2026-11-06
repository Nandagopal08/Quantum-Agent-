"""Model training jobs, model catalogue, job status, training logs, quantum experiments."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import db, jobs
from ..classical_models.anomaly import DETECTORS, train_detector
from ..classical_models.rul import ALL_MODELS, SEQUENCE, train_rul_model
from ..quantum_models.qml import QUANTUM_MODELS, QuantumConfig, run_experiment
from ..services import fleet
from ..storage import list_models, paths, require_drive
from .common import saved_results, sse

router = APIRouter(prefix="/api", tags=["training"])


class RulTrainRequest(BaseModel):
    model: Literal["random_forest", "xgboost", "lightgbm", "catboost", "lstm", "gru", "cnn_lstm", "transformer"]
    subset: Literal["FD001", "FD002", "FD003", "FD004"] = "FD001"
    epochs: int = Field(40, ge=1, le=300)
    seed: int = 42


class AnomalyTrainRequest(BaseModel):
    model: Literal["isolation_forest", "one_class_svm", "autoencoder", "lstm_autoencoder"]
    test: Literal["1st_test", "2nd_test", "3rd_test"] = "2nd_test"
    threshold_quantile: float = Field(0.99, gt=0.5, lt=1)
    epochs: int = Field(60, ge=1, le=500)


class QuantumRequest(BaseModel):
    model: Literal["vqc", "quantum_kernel", "qnn_regressor"] = "vqc"
    subset: Literal["FD001", "FD002", "FD003", "FD004"] = "FD001"
    n_qubits: int = Field(4, ge=2, le=8)
    layers: int = Field(2, ge=1, le=6)
    n_train: int = Field(300, ge=40, le=1000)
    epochs: int = Field(30, ge=1, le=200)
    lr: float = Field(0.05, gt=0, le=1)
    seed: int = 42
    shots: int | None = Field(None, ge=10, le=100_000)


@router.get("/models")
def models():
    require_drive()
    return {"available": {"rul": list(ALL_MODELS), "sequence_models": list(SEQUENCE), "anomaly": list(DETECTORS),
                          "quantum": list(QUANTUM_MODELS)},
            "trained": list_models()}


@router.post("/training/rul")
def train_rul(req: RulTrainRequest):
    require_drive()

    def work(progress):
        r = train_rul_model(req.model, req.subset, req.epochs, req.seed, progress=progress)
        fleet.clear_caches()
        r.pop("predictions", None)
        return r

    return jobs.submit("train_rul", req.model_dump(), work)


@router.post("/training/anomaly")
def train_anomaly(req: AnomalyTrainRequest):
    require_drive()

    def work(progress):
        r = train_detector(req.model, req.test, threshold_quantile=req.threshold_quantile, epochs=req.epochs,
                           progress=progress)
        fleet.clear_caches()
        return r

    return jobs.submit("train_anomaly", req.model_dump(), work)


@router.post("/quantum/run")
def quantum_run(req: QuantumRequest):
    require_drive()
    return jobs.submit("quantum_experiment", req.model_dump(),
                       lambda progress: run_experiment(QuantumConfig(**req.model_dump()), progress=progress))


@router.get("/quantum/experiments")
def quantum_experiments():
    require_drive()
    return [r["data"] | {"saved_at": r["modified"]} for r in saved_results("comparison_reports", "quantum_vs_classical_")]


@router.get("/jobs")
def list_jobs(limit: int = 50, type: str | None = None, status: str | None = None):
    require_drive()
    return db.list_rows(db.jobs, limit=limit, type=type, status=status)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    require_drive()
    row = db.get_row(db.jobs, job_id)
    if row is None:
        raise HTTPException(404, "job not found")
    return row


@router.get("/events/jobs")
def job_events(request: Request):
    return sse("jobs", request, replay=False)


@router.get("/training/logs/{run_name}")
def training_log(run_name: str):
    require_drive()
    path = paths.area("training_logs") / f"{run_name}.jsonl"
    if not path.exists():
        raise HTTPException(404, f"no training log {run_name}")
    import json

    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
