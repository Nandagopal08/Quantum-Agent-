"""Background jobs (training, feature extraction, quantum experiments, optimisation).

Heavy jobs run one at a time on a dedicated worker so CPU-bound training never competes with itself;
light jobs use a small separate pool. Progress is written to the ``jobs`` table and streamed on the
``jobs`` event channel. Jobs left running by a previous process are marked ``interrupted`` on start-up
(training resumes from its last Drive checkpoint when re-submitted).
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from . import db
from .events import bus
from .langchain_tools.tools import to_jsonable

log = logging.getLogger("qadt.jobs")
_heavy = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job-heavy")
_light = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job-light")
JobFn = Callable[[Callable[[float, str], None]], Any]


def _progress(job_id: str, job_type: str):
    last = [0.0]

    def report(fraction: float, message: str) -> None:
        now = time.time()
        fraction = float(max(0.0, min(1.0, fraction)))
        if now - last[0] < 0.5 and fraction < 1.0:   # throttle DB writes
            bus.publish("jobs", {"type": "job_progress", "job_id": job_id, "job_type": job_type,
                                 "progress": fraction, "message": message})
            return
        last[0] = now
        db.update_row(db.jobs, job_id, progress=fraction, message=message)
        bus.publish("jobs", {"type": "job_progress", "job_id": job_id, "job_type": job_type,
                             "progress": fraction, "message": message})
    return report


def _run(job_id: str, job_type: str, fn: JobFn) -> None:
    db.update_row(db.jobs, job_id, status="running", message="started")
    bus.publish("jobs", {"type": "job_status", "job_id": job_id, "job_type": job_type, "status": "running"})
    try:
        result = to_jsonable(fn(_progress(job_id, job_type)))
        db.update_row(db.jobs, job_id, status="completed", progress=1.0, message="completed", result=result)
        bus.publish("jobs", {"type": "job_status", "job_id": job_id, "job_type": job_type, "status": "completed"})
    except Exception as exc:
        log.exception("job %s (%s) failed", job_id, job_type)
        db.update_row(db.jobs, job_id, status="failed", message=str(exc)[:500],
                      error=traceback.format_exc(limit=6))
        bus.publish("jobs", {"type": "job_status", "job_id": job_id, "job_type": job_type, "status": "failed",
                             "error": str(exc)})


def submit(job_type: str, params: dict, fn: JobFn, heavy: bool = True) -> dict:
    job_id = db.new_id()
    now = time.time()
    row = db.insert_row(db.jobs, id=job_id, type=job_type, status="queued", progress=0.0, message="queued",
                        params=params, result=None, error=None, created_at=now, updated_at=now)
    bus.publish("jobs", {"type": "job_status", "job_id": job_id, "job_type": job_type, "status": "queued"})
    (_heavy if heavy else _light).submit(_run, job_id, job_type, fn)
    return row


def mark_interrupted() -> int:
    n = 0
    for r in db.list_rows(db.jobs, limit=1000):
        if r["status"] in ("queued", "running"):
            db.update_row(db.jobs, r["id"], status="interrupted",
                          message="API restarted; re-submit to resume from the last checkpoint")
            n += 1
    return n
