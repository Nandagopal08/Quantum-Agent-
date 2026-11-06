"""Shared API helpers: error mapping, SSE streaming, saved-result lookup."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..events import bus, sse_format
from ..services.fleet import AssetNotFound, ModelNotTrained
from ..storage import DatasetNotReadyError, DriveNotMountedError, paths
from ..storage.fileio import read_json

log = logging.getLogger("qadt.api")


def redact(message: str) -> str:
    """Replace absolute storage locations in client-facing messages with placeholders."""
    from ..config import get_config

    cfg = get_config()
    for path, label in ((cfg.project_root, "<project>"), (cfg.local_scratch, "<scratch>"), (cfg.drive_mount, "<drive>")):
        for variant in {str(path), str(path.expanduser().resolve()) if path else ""}:
            if variant and len(variant) > 1:
                message = message.replace(variant, label)
    return message


def install_error_handlers(app: FastAPI) -> None:
    def handler(status: int, code: str):
        async def h(_req: Request, exc: Exception):
            if status >= 500:
                log.error("%s: %s", code, exc)
            detail = str(exc) if code == "drive_not_mounted" else redact(str(exc))   # mount help names the mount point
            return JSONResponse(status_code=status, content={"error": code, "detail": detail})
        return h

    app.add_exception_handler(DriveNotMountedError, handler(503, "drive_not_mounted"))
    app.add_exception_handler(DatasetNotReadyError, handler(409, "dataset_not_ready"))
    app.add_exception_handler(ModelNotTrained, handler(409, "model_not_trained"))
    app.add_exception_handler(AssetNotFound, handler(404, "asset_not_found"))
    app.add_exception_handler(FileNotFoundError, handler(404, "not_found"))
    app.add_exception_handler(ValueError, handler(422, "invalid_request"))


def sse(channel: str, request: Request, replay: bool = True, stop_types: tuple[str, ...] = ()) -> StreamingResponse:
    async def gen():
        q = bus.subscribe(channel)
        try:
            if replay:
                for ev in bus.history(channel):
                    yield sse_format(ev)
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield sse_format(ev)
                if ev.get("type") == "run_status" and ev.get("status") in stop_types:
                    break
        finally:
            bus.unsubscribe(channel, q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def saved_results(area: str, prefix: str = "") -> list[dict]:
    folder = paths.area(area)
    if not folder.is_dir():
        return []
    out = []
    for f in sorted(folder.glob(f"{prefix}*.json")):
        data = read_json(f)
        if data is not None:
            out.append({"name": f.stem, "modified": f.stat().st_mtime, "data": data})
    return out


def saved_result(area: str, name: str) -> dict:
    data = read_json(paths.area(area) / f"{name}.json")
    if data is None:
        raise FileNotFoundError(f"no saved result {area}/{name}")
    return data


def jsonable(obj):
    return json.loads(json.dumps(obj, default=str))
