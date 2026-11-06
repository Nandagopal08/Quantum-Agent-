"""FastAPI application.

Run (from the project root):
    uvicorn backend.main:app --reload --port 8000
In Colab, see notebooks/01_run_backend_colab.ipynb (mounts Drive, starts the API, opens a tunnel).
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import agents, assets, datasets, optimization, system, training
from .api.common import install_error_handlers
from .storage import DriveNotMountedError
from .storage.drive import ensure_project_structure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("qadt.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        root = ensure_project_structure()
        from . import jobs
        from .agents import runner

        n_jobs, n_runs = jobs.mark_interrupted(), runner.expire_stale_runs()
        log.info("storage ready at %s (%d interrupted job(s), %d expired run(s))", root, n_jobs, n_runs)
    except DriveNotMountedError as exc:
        # Start anyway so the UI can show the mounting instructions; storage endpoints return 503.
        log.error("Google Drive is not available — storage endpoints will return 503.\n%s", exc)
    yield


app = FastAPI(title="Quantum Agentic Digital Twin API", version=__version__, lifespan=lifespan,
              description="Predictive maintenance with classical/quantum ML, digital twins, counterfactuals, "
                          "QAOA scheduling and LangGraph agents. Storage: Google Drive.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
install_error_handlers(app)


_TOKEN = os.environ.get("QADT_API_TOKEN")   # set when the API is exposed beyond localhost (e.g. a Colab tunnel)


@app.middleware("http")
async def require_token(request: Request, call_next):
    if _TOKEN and request.url.path.startswith("/api") and request.url.path != "/api/health":
        auth = request.headers.get("authorization", "")
        given = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else request.query_params.get("token", "")
        if not hmac.compare_digest(given, _TOKEN):
            return JSONResponse(status_code=401, content={"error": "unauthorized",
                                                          "detail": "missing or wrong API token; open the link printed by the Colab notebook"})
    return await call_next(request)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    if request.url.path.startswith("/api") and not request.url.path.startswith("/api/events"):
        log.info("%s %s → %d (%.0f ms)", request.method, request.url.path, response.status_code,
                 1000 * (time.time() - t0))
    return response


for r in (system, datasets, training, assets, optimization, agents):
    app.include_router(r.router)

# Serve the built React app (frontend/dist) from the same port when it exists.
_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = _dist / path
        return FileResponse(f if path and f.is_file() else _dist / "index.html")
