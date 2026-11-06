"""Runs the missing-data agent workflow in a background thread, streams events, persists every run to Drive
(``models/agentic_ai/missing_data_runs/<run_id>.json``)."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from ... import db
from ...events import bus, current_run
from ...langchain_tools.tools import to_jsonable
from ...storage import paths
from ...storage.fileio import read_json, write_json
from .graph import TRUTH_VAULT, build_graph

log = logging.getLogger("qadt.agents.missing_data")
_graph = build_graph()
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="md-agent")


def _folder():
    return paths.root / "models" / "agentic_ai" / "missing_data_runs"


def _save(record: dict) -> None:
    write_json(_folder() / f"{record['run_id']}.json", record)


def execute(run_id: str, request: dict) -> dict:
    """Run synchronously (used by the background pool, tests and scripts)."""
    token = current_run.set(run_id)
    channel = f"run:{run_id}"
    t0 = time.time()
    record = {"run_id": run_id, "workflow": "missing_data", "request": request, "status": "running",
              "created_at": t0}
    try:
        state = _graph.invoke({"run_id": run_id, "request": request, "status": "running", "trace": [],
                               "route_log": [], "errors": [], "attempts": []}, {"recursion_limit": 40})
        state = to_jsonable(state)
        status = state.get("status") or "completed"
        if status == "running":
            status = "completed"
        record.update(status=status, state=state)
    except Exception as exc:
        log.exception("missing-data run %s failed", run_id)
        record.update(status="failed", error=f"{exc.__class__.__name__}: {exc}")
    record.update(seconds=round(time.time() - t0, 2), updated_at=time.time())
    TRUTH_VAULT.pop(run_id, None)
    try:
        _save(record)
    finally:
        bus.publish(channel, {"type": "run_status", "status": record["status"]})
        current_run.reset(token)
    return record


def mark_interrupted() -> int:
    """Called at API start-up: runs left 'running' by a previous process can no longer finish."""
    n = 0
    folder = _folder()
    if not folder.is_dir():
        return 0
    for f in folder.glob("*.json"):
        r = read_json(f) or {}
        if r.get("status") == "running":
            r.update(status="interrupted", error="the API stopped before the run finished; use rerun",
                     updated_at=time.time())
            write_json(f, r)
            n += 1
    return n


def rerun(run_id: str) -> str:
    """Start a new run with the same request (the workflow is deterministic for a given seed)."""
    old = get_run(run_id)
    if old is None:
        raise KeyError(run_id)
    return start_run({**old["request"], "rerun_of": run_id})


def start_run(request: dict) -> str:
    run_id = db.new_id()
    bus.publish(f"run:{run_id}", {"type": "run_status", "status": "running", "request": request})
    _save({"run_id": run_id, "workflow": "missing_data", "request": request, "status": "running",
           "created_at": time.time()})
    _pool.submit(execute, run_id, request)
    return run_id


def get_run(run_id: str) -> dict | None:
    if not run_id.isalnum():
        return None
    return read_json(_folder() / f"{run_id}.json")


def list_runs(limit: int = 30) -> list[dict]:
    folder = _folder()
    if not folder.is_dir():
        return []
    files = sorted(folder.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:limit]
    out = []
    for f in files:
        r = read_json(f) or {}
        s = (r.get("state") or {}).get("report", {}).get("summary", {})
        out.append({"run_id": r.get("run_id"), "status": r.get("status"), "request": r.get("request"),
                    "created_at": r.get("created_at"), "seconds": r.get("seconds"), "summary": s})
    return out
