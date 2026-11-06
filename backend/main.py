"""FastAPI application.

Run (from the project root):
    uvicorn backend.main:app --reload --port 8000
Or: python scripts/run_backend.py [--tunnel]   (Colab: !python scripts/run_backend.py --tunnel prints a private link).
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
from .api import agents, assets, datasets, optimization, qseq, system, training
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

        from .agents.missing_data import runner as md_runner

        n_jobs, n_runs, n_md = jobs.mark_interrupted(), runner.expire_stale_runs(), md_runner.mark_interrupted()
        log.info("storage ready at %s (%d interrupted job(s), %d expired run(s), %d interrupted missing-data run(s))",
                 root, n_jobs, n_runs, n_md)
    except DriveNotMountedError as exc:
        # Start anyway so the UI can show the mounting instructions; storage endpoints return 503.
        log.error("Google Drive is not available — storage endpoints will return 503.\n%s", exc)
    yield


_TOKEN = os.environ.get("QADT_API_TOKEN")   # set when the API is exposed beyond localhost (e.g. a Colab tunnel)
# Browser origins allowed to call the API cross-origin (the dashboard itself is same-origin).
_CORS = [o.strip() for o in os.environ.get(
    "QADT_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000"
).split(",") if o.strip()]

app = FastAPI(title="Quantum Agentic Digital Twin API", version=__version__, lifespan=lifespan,
              description="Predictive maintenance with classical/quantum ML, digital twins, counterfactuals, "
                          "QAOA scheduling and LangGraph agents. Storage: Google Drive.",
              # with a token (public tunnel) the interactive docs and schema are not served
              docs_url=None if _TOKEN else "/docs", redoc_url=None if _TOKEN else "/redoc",
              openapi_url=None if _TOKEN else "/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=_CORS, allow_methods=["GET", "POST", "PUT", "DELETE"],
                   allow_headers=["Authorization", "Content-Type"])
install_error_handlers(app)


def _token_ok(given: str) -> bool:
    return hmac.compare_digest(given.encode("utf-8", "surrogateescape"), _TOKEN.encode("utf-8"))


@app.middleware("http")
async def require_token(request: Request, call_next):
    if _TOKEN and request.url.path.startswith("/api") and request.url.path != "/api/health":
        auth = request.headers.get("authorization", "")
        given = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else request.query_params.get("token", "")
        if not _token_ok(given):
            return JSONResponse(status_code=401, content={"error": "unauthorized",
                                                          "detail": "missing or wrong API token; open the link printed by scripts/run_backend.py --tunnel"})
    return await call_next(request)


# Agent-only mode (QADT_UI_MODE=agent, set by scripts/run_backend.py --agent-only): the dashboard shows only the
# missing-data agent and the API serves only what that page needs; everything else is 404.
_AGENT_ONLY = os.environ.get("QADT_UI_MODE", "").lower() == "agent"
_AGENT_API_PREFIXES = ("/api/qseq/agents/", "/api/qseq/models")    # plus /api/health


@app.middleware("http")
async def agent_only_mode(request: Request, call_next):
    p = request.url.path
    if _AGENT_ONLY and p.startswith("/api") and not (p == "/api/health" or p.startswith(_AGENT_API_PREFIXES)):
        return JSONResponse(status_code=404, content={"error": "not_found",
                                                      "detail": "not available in agent-only mode"})
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")     # the access token can travel in the URL
    response.headers.setdefault("X-Frame-Options", "DENY")
    return response


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    if request.url.path.startswith("/api") and not request.url.path.startswith("/api/events"):
        log.info("%s %s → %d (%.0f ms)", request.method, request.url.path, response.status_code,
                 1000 * (time.time() - t0))
    return response


for r in (system, datasets, training, assets, optimization, agents, qseq):
    app.include_router(r.router)

# Serve the built React app (frontend/dist) from the same port when it exists.
_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    _dist_root = _dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            return JSONResponse(status_code=404, content={"error": "not_found", "detail": f"no API route /{path}"})
        f = (_dist_root / path).resolve()
        # only files inside frontend/dist are served (blocks ../ traversal, e.g. /%2e%2e/%2e%2e/.env)
        if path and f.is_file() and f.is_relative_to(_dist_root):
            return FileResponse(f)
        return FileResponse(_dist_root / "index.html")
