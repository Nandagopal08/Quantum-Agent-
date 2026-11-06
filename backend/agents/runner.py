"""Runs the LangGraph workflow in background threads, streams events, and persists every run.

Run records live in the database (``agent_runs``) and a full JSON trace is written to Drive under
``models/agentic_ai/runs/<run_id>.json``. Paused runs (awaiting approval) are held by the in-process
checkpointer; if the API restarts, they are marked ``expired`` and must be re-run.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from langgraph.types import Command

from .. import db
from ..events import bus, current_run
from ..langchain_tools.tools import to_jsonable
from ..storage import paths
from ..storage.fileio import write_json
from .graph import build_graph
from .llm import llm_status

log = logging.getLogger("qadt.agents")
_graph = build_graph()
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent-run")
_lock = threading.Lock()


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": 40}


def _persist(run_id: str, status: str, error: str | None = None) -> dict:
    state = to_jsonable(_graph.get_state(_config(run_id)).values)
    db.update_row(db.agent_runs, run_id, status=status, state=state, approval=state.get("approval"),
                  llm=(state.get("decision") or {}).get("llm"), error=error)
    write_json(paths.root / "models" / "agentic_ai" / "runs" / f"{run_id}.json",
               {"run_id": run_id, "status": status, "error": error, "state": state})
    return state


def _execute(run_id: str, payload) -> None:
    token = current_run.set(run_id)
    channel = f"run:{run_id}"
    try:
        interrupted = False
        for chunk in _graph.stream(payload, _config(run_id), stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupted = True
                bus.publish(channel, {"type": "approval_required",
                                      "request": to_jsonable(chunk["__interrupt__"][0].value)})
        snapshot = _graph.get_state(_config(run_id))
        values = snapshot.values
        if interrupted or snapshot.next:
            status = "awaiting_approval"
        else:
            status = values.get("status") or "completed"
        _persist(run_id, status)
        bus.publish(channel, {"type": "run_status", "status": status})
    except Exception as exc:  # unexpected engine failure (node errors are handled inside nodes)
        log.exception("agent run %s failed", run_id)
        try:
            _persist(run_id, "failed", str(exc))
        except Exception:
            db.update_row(db.agent_runs, run_id, status="failed", error=str(exc))
        bus.publish(channel, {"type": "run_status", "status": "failed", "error": str(exc)})
    finally:
        current_run.reset(token)


def start_run(request: dict) -> str:
    run_id = db.new_id()
    now = time.time()
    db.insert_row(db.agent_runs, id=run_id, status="running", request=request, state={}, approval=None,
                  llm=llm_status(), error=None, created_at=now, updated_at=now)
    bus.publish(f"run:{run_id}", {"type": "run_status", "status": "running", "request": request})
    initial = {"run_id": run_id, "request": request, "status": "running", "trace": [], "route_log": [], "errors": []}
    _pool.submit(_execute, run_id, initial)
    return run_id


def submit_approval(run_id: str, approved: bool, reviewer: str, comment: str | None = None,
                    asset_ids: list[str] | None = None) -> None:
    row = db.get_row(db.agent_runs, run_id)
    if row is None:
        raise KeyError(run_id)
    if row["status"] != "awaiting_approval":
        raise ValueError(f"run {run_id} is {row['status']}, not awaiting approval")
    snapshot = _graph.get_state(_config(run_id))
    if not snapshot.next:
        db.update_row(db.agent_runs, run_id, status="expired",
                      error="paused state lost (API restarted); start a new run")
        raise ValueError("this run's paused state is no longer available (the API was restarted); start a new run")
    db.update_row(db.agent_runs, run_id, status="resuming")
    bus.publish(f"run:{run_id}", {"type": "approval_received", "approved": approved, "reviewer": reviewer})
    _pool.submit(_execute, run_id, Command(resume={"approved": approved, "reviewer": reviewer, "comment": comment,
                                                   "asset_ids": asset_ids}))


def expire_stale_runs() -> int:
    """Called at API start-up: runs paused in a previous process can no longer resume."""
    n = 0
    for r in db.list_rows(db.agent_runs, limit=500):
        if r["status"] in ("running", "resuming", "awaiting_approval"):
            db.update_row(db.agent_runs, r["id"], status="expired", error="API restarted before the run finished")
            n += 1
    return n


def run_sync(request: dict, timeout: float = 600) -> dict:
    """Start a run and wait for it to finish or pause (used by tests and scripts)."""
    run_id = start_run(request)
    t0 = time.time()
    while time.time() - t0 < timeout:
        row = db.get_row(db.agent_runs, run_id)
        if row["status"] not in ("running", "resuming"):
            return row
        time.sleep(0.25)
    raise TimeoutError(run_id)
