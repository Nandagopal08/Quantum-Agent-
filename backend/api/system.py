"""Health, dashboard overview, storage, settings."""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__, db
from ..agents.llm import llm_status
from ..config import get_config
from ..services import fleet
from ..settings import get_settings, reset_settings, update_settings
from ..storage import DATASETS, MOUNT_INSTRUCTIONS, is_drive_mounted, list_models, storage_report, verify_dataset
from ..storage.drive import ensure_project_structure, require_drive

router = APIRouter(prefix="/api", tags=["system"])


def _storage_info() -> dict:
    c = get_config()
    mounted = is_drive_mounted(c)
    return {"mode": c.storage_mode, "is_google_drive": c.is_google_drive, "mounted": mounted,
            "project_root": str(c.project_root), "drive_mount": str(c.drive_mount),
            "local_scratch": str(c.local_scratch), "instructions": None if mounted else MOUNT_INSTRUCTIONS}


@router.get("/health")
def health():
    import os

    return {"status": "ok", "version": __version__, "storage": _storage_info(), "llm": llm_status(),
            "auth_required": bool(os.environ.get("QADT_API_TOKEN"))}


@router.get("/overview")
def overview():
    require_drive()
    datasets = {n: verify_dataset(n) for n in DATASETS}
    models = list_models()
    out = {"storage": _storage_info(), "llm": llm_status(), "datasets": datasets,
           "models": {"total": len(models), "by_category": {}}, "fleet": None, "bearings": None,
           "recent_jobs": db.list_rows(db.jobs, limit=8), "recent_runs": [
               {k: r[k] for k in ("id", "status", "created_at", "updated_at", "request")}
               for r in db.list_rows(db.agent_runs, limit=6)],
           "maintenance_events": len(db.list_rows(db.maintenance_events, limit=1000))}
    for m in models:
        out["models"]["by_category"][m["category"]] = out["models"]["by_category"].get(m["category"], 0) + 1
    rul = [m for m in models if m["name"].startswith("rul_")]
    out["models"]["rul"] = sorted([{"name": m["name"], **{k: m["metrics"].get(k) for k in ("rmse", "mae", "r2", "nasa_score")}}
                                   for m in rul], key=lambda r: r["rmse"] or 1e9)
    try:
        out["fleet"] = fleet.fleet_overview("FD001", limit=10)
    except Exception as exc:
        out["fleet_error"] = str(exc)
    try:
        test = "2nd_test"
        out["bearings"] = fleet.bearing_overview(test)
    except Exception as exc:
        out["bearings_error"] = str(exc)
    return out


@router.get("/storage")
def storage():
    return {"info": _storage_info(), "report": storage_report()}


@router.post("/storage/init")
def storage_init():
    return {"project_root": str(ensure_project_structure()), "info": _storage_info()}


@router.get("/settings")
def settings_get():
    require_drive()
    return get_settings()


@router.put("/settings")
def settings_put(patch: dict):
    require_drive()
    s = update_settings(patch)
    fleet.clear_caches()
    return s


@router.post("/settings/reset")
def settings_reset():
    require_drive()
    return reset_settings()


@router.get("/settings/llm-status")
def settings_llm():
    return llm_status()
