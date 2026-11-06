"""Agentic AI: LangGraph workflow runs, live events (SSE), human approval, chat."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import db
from ..agents import runner
from ..agents.chat import ask
from ..agents.graph import agent_specs, mermaid
from ..agents.llm import llm_status
from ..services import fleet
from ..storage import require_drive
from .common import sse

router = APIRouter(prefix="/api/agents", tags=["agents"])


class RunRequest(BaseModel):
    asset_ids: list[str] | None = Field(None, max_length=12)
    subset: str = "FD001"
    n_assets: int = Field(5, ge=1, le=12)
    horizon: float = Field(60, gt=0, le=300)
    optimize: bool = True
    use_llm: bool = True
    force_full_analysis: bool = False
    model: str | None = None


class ApprovalRequest(BaseModel):
    approved: bool
    reviewer: str = Field(min_length=1, max_length=128)
    comment: str | None = Field(None, max_length=2000)
    asset_ids: list[str] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@router.get("/specs")
def specs():
    return {"agents": agent_specs(), "graph_mermaid": mermaid(), "llm": llm_status()}


@router.post("/runs")
def start(req: RunRequest):
    require_drive()
    ids = req.asset_ids or fleet.default_assets(req.subset, req.n_assets)
    for aid in ids:
        fleet.parse_asset(aid)
    payload = req.model_dump() | {"asset_ids": ids}
    return {"run_id": runner.start_run(payload), "asset_ids": ids}


@router.get("/runs")
def runs(limit: int = 30):
    require_drive()
    return [{k: r[k] for k in ("id", "status", "request", "created_at", "updated_at", "error", "approval")}
            for r in db.list_rows(db.agent_runs, limit=limit)]


@router.get("/runs/{run_id}")
def run(run_id: str):
    require_drive()
    row = db.get_row(db.agent_runs, run_id)
    if row is None:
        raise HTTPException(404, "run not found")
    return row


@router.get("/runs/{run_id}/events")
def run_events(run_id: str, request: Request):
    return sse(f"run:{run_id}", request, replay=True, stop_types=("completed", "failed", "approved", "rejected"))


@router.post("/runs/{run_id}/approval")
def approve(run_id: str, req: ApprovalRequest):
    require_drive()
    try:
        runner.submit_approval(run_id, req.approved, req.reviewer, req.comment, req.asset_ids)
    except KeyError:
        raise HTTPException(404, "run not found") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"run_id": run_id, "status": "resuming"}


@router.post("/ask")
def ask_question(req: AskRequest):
    require_drive()
    return ask(req.question)
