"""Predictions, risk, digital twin, counterfactuals, explainability, maintenance history."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import db, jobs
from ..counterfactual.scenarios import AssetState, compare_scenarios
from ..digital_twin.twin import SimulationParams
from ..explainability import explain as xai
from ..services import fleet
from ..storage import paths, require_drive, save_result
from .common import saved_results

router = APIRouter(prefix="/api", tags=["assets"])


class SimulateRequest(BaseModel):
    horizon: int = Field(60, ge=1, le=500)
    load: float = Field(1.0, gt=0, le=2.0)
    maintenance_at: int | None = Field(None, ge=1)
    samples: int | None = Field(None, ge=100, le=20000)
    seed: int | None = None
    model: str | None = None
    as_of: int | None = None


class CounterfactualRequest(BaseModel):
    asset_id: str
    horizon: float | None = Field(None, gt=0, le=500)
    delay: float | None = Field(None, ge=0)
    reduced_load: float = Field(0.7, gt=0, lt=1)
    model: str | None = None
    as_of: int | None = None


class MaintenanceRecord(BaseModel):
    action: str = Field(min_length=2, max_length=64)
    scheduled_for: str = "now"
    status: str = "completed"
    performed_by: str = Field(min_length=1, max_length=128)
    notes: str | None = None


@router.get("/fleet/turbofan")
def fleet_turbofan(subset: str = "FD001", model: str | None = None):
    require_drive()
    return fleet.fleet_overview(subset, model)


@router.get("/fleet/bearings")
def fleet_bearings(test: str = "2nd_test", model: str | None = None, as_of: int | None = None):
    require_drive()
    return fleet.bearing_overview(test, model, as_of)


@router.get("/assets/{asset_id}/prediction")
def prediction(asset_id: str, model: str | None = None, as_of: int | None = None):
    require_drive()
    return fleet.asset_prediction(asset_id, model, as_of)


@router.get("/twin/{asset_id}/state")
def twin_state(asset_id: str, model: str | None = None, as_of: int | None = None):
    require_drive()
    a = fleet.parse_asset(asset_id)
    hist = db.list_rows(db.maintenance_events, asset_id=asset_id)
    if a.kind == "turbofan":
        tw = fleet.turbofan_twin(asset_id, model)
        tw.history = hist
        return tw.state()
    tw = fleet.bearing_twin(asset_id, model)
    tw.history = hist
    return tw.state(as_of)


@router.get("/twin/{asset_id}/playback")
def twin_playback(asset_id: str, model: str | None = None, as_of: int | None = None, stride: int = 1):
    require_drive()
    a = fleet.parse_asset(asset_id)
    if a.kind == "turbofan":
        return fleet.turbofan_twin(asset_id, model).playback()
    return fleet.bearing_twin(asset_id, model).playback(as_of, stride=max(1, stride))


@router.post("/twin/{asset_id}/simulate")
def twin_simulate(asset_id: str, req: SimulateRequest):
    require_drive()
    params = SimulationParams(horizon=req.horizon, load=req.load, maintenance_at=req.maintenance_at,
                              samples=req.samples, seed=req.seed)
    result = fleet.simulate(asset_id, params, req.model, req.as_of)
    save_result("simulation_results", f"{asset_id}_latest", result)
    return result


@router.get("/maintenance/events")
def maintenance_events(asset_id: str | None = None, limit: int = 200):
    require_drive()
    return db.list_rows(db.maintenance_events, limit=limit, asset_id=asset_id)


@router.post("/twin/{asset_id}/maintenance")
def record_maintenance(asset_id: str, rec: MaintenanceRecord):
    require_drive()
    fleet.parse_asset(asset_id)
    return db.insert_row(db.maintenance_events, id=db.new_id(), asset_id=asset_id, action=rec.action,
                         scheduled_for=rec.scheduled_for, status=rec.status, approved_by=rec.performed_by,
                         run_id=None, details={"notes": rec.notes, "source": "manual entry"}, created_at=time.time())


@router.post("/counterfactual")
def counterfactual(req: CounterfactualRequest):
    require_drive()
    p = fleet.asset_prediction(req.asset_id, req.model, req.as_of)
    if p["rul_pred"] is None:
        raise HTTPException(409, "this bearing shows no upward degradation trend yet, so there is no time-to-failure "
                                 "estimate to build scenarios on; pick a later snapshot (as_of)")
    state = AssetState(req.asset_id, p["equipment_type"], float(p["rul_pred"]), float(p["sigma"] or 1.0),
                       float(p.get("rul_cap", 125)))
    result = compare_scenarios(state, req.horizon, req.delay, req.reduced_load)
    result["prediction"] = {k: p[k] for k in ("model", "rul_pred", "sigma", "time_unit") if k in p}
    result["risk"] = p["risk"]
    save_result("counterfactuals", f"{req.asset_id}_scenarios", result)
    return result


@router.get("/explain/asset/{asset_id}")
def explain_asset(asset_id: str, model: str | None = None):
    require_drive()
    a = fleet.parse_asset(asset_id)
    if a.kind == "turbofan":
        tw = fleet.turbofan_twin(asset_id, model)
        return {"asset_id": asset_id, "model": tw.model.name, "kind": tw.model.kind,
                "local": xai.local_attribution(tw.model, tw.data, a.unit),
                "counterfactual": xai.counterfactual_explanation(tw.model, tw.data, a.unit),
                "uncertainty": xai.uncertainty(tw.model, tw.data, a.unit)}
    d = fleet.ims_features(a.test)
    return {"asset_id": asset_id, "deviation": xai.bearing_deviation(d.features, d.feature_cols, a.bearing)}


class ExplainModelRequest(BaseModel):
    model: str
    subset: str = "FD001"
    unit: int | None = None


@router.post("/explain/model")
def explain_model(req: ExplainModelRequest):
    require_drive()

    def work(progress):
        m = fleet.rul_model(req.model, req.subset)
        return xai.explain_rul_model(m, fleet.cmapss_data(req.subset), req.unit, progress=progress)

    return jobs.submit("explain_model", req.model_dump(), work, heavy=False)


@router.get("/explain/reports")
def explain_reports():
    require_drive()
    return [{"name": r["name"], "saved_at": r["modified"], **r["data"]} for r in saved_results("shap", "rul_")]


@router.get("/files/{area}/{name}")
def project_file(area: str, name: str):
    """Serve a generated image (e.g. a SHAP plot) from a project result area."""
    require_drive()
    folder = paths.area(area).resolve()
    target = (folder / name).resolve()
    if folder not in target.parents or target.suffix.lower() not in (".png", ".svg", ".html"):
        raise HTTPException(404, "not found")
    if not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)
