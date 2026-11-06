"""Maintenance-schedule optimisation (QAOA vs classical) and research comparison reports."""

from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import jobs
from ..langchain_tools.tools import optimize_maintenance_schedule
from ..services import fleet
from ..settings import get_settings, update_settings
from ..storage import require_drive, save_result
from .common import saved_result, saved_results

router = APIRouter(prefix="/api", tags=["optimization", "experiments"])


class ScheduleRequest(BaseModel):
    asset_ids: list[str] | None = Field(None, max_length=8)
    subset: str = "FD001"
    n_assets: int = Field(5, ge=2, le=8)
    qaoa_layers: int = Field(2, ge=1, le=6)
    qaoa_steps: int | None = Field(None, ge=10, le=1000)
    crew_capacity_per_slot: int | None = Field(None, ge=1, le=4)
    slots: int | None = Field(None, ge=1, le=3)


@router.post("/optimization/schedule")
def schedule(req: ScheduleRequest):
    require_drive()
    ids = req.asset_ids or fleet.default_assets(req.subset, req.n_assets)
    patch = {k: v for k, v in {"qaoa_steps": req.qaoa_steps, "crew_capacity_per_slot": req.crew_capacity_per_slot,
                               "slots": req.slots}.items() if v is not None}

    def work(progress):
        if patch:
            update_settings({"optimization": patch})
        progress(0.05, f"optimising {len(ids)} assets")
        result = optimize_maintenance_schedule(ids, req.qaoa_layers)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        result["generated_at"] = stamp
        result["settings"] = get_settings()["optimization"]
        save_result("qaoa_results", f"schedule_{stamp}", result)
        best = next(r for r in result["results"] if r.get("solver") == result["best_solver"])
        save_result("maintenance_schedules", f"schedule_{stamp}", {"solver": result["best_solver"], "plan": best["plan"],
                                                                     "total_expected_cost": best["total_expected_cost"]})
        return result

    return jobs.submit("optimization", {**req.model_dump(), "asset_ids": ids}, work)


@router.get("/optimization/results")
def optimization_results(limit: int = 20):
    require_drive()
    rows = saved_results("qaoa_results", "schedule_")
    return [r["data"] for r in sorted(rows, key=lambda r: -r["modified"])[:limit]]


@router.get("/experiments/comparisons")
def comparisons():
    """Every comparison report produced by actual runs (RUL, IMS anomaly, quantum vs classical)."""
    require_drive()
    return [{"name": r["name"], "saved_at": r["modified"], "data": r["data"]} for r in saved_results("comparison_reports")]


@router.get("/experiments/comparisons/{name}")
def comparison(name: str):
    require_drive()
    return saved_result("comparison_reports", name)
