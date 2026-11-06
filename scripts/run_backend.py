"""Start the FastAPI backend, which also serves the built React dashboard, optionally behind a public tunnel.

    python scripts/run_backend.py                         # http://localhost:8000
    python scripts/run_backend.py --tunnel                # Colab: Cloudflare quick tunnel + random access token
    python scripts/run_backend.py --ollama-model llama3.2:3b --tunnel

With --tunnel the API is public, so an access token is generated (or taken from --token / $QADT_API_TOKEN)
and the printed link carries it. Keep the link private. Ctrl+C stops the API (and the tunnel).
"""

from __future__ import annotations

import os
import platform
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from _cli import ROOT, log, parser, configure

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
# A quick-tunnel hostname is random hyphenated words (e.g. https://lazy-owl-tree-blue.trycloudflare.com).
# api.trycloudflare.com appears in cloudflared's *error* messages and must never be printed as the dashboard link.
TUNNEL_RE = re.compile(r"https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com")


def tunnel_url(line: str) -> str | None:
    m = TUNNEL_RE.search(line)
    return m.group(0) if m else None

ap = parser(__doc__)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--tunnel", action="store_true", help="expose through a Cloudflare quick tunnel")
ap.add_argument("--token", default=None, help="API token (generated automatically with --tunnel)")
ap.add_argument("--ollama-model", default=None, help="start a local Ollama server with this model for narratives")
ap.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
ap.add_argument("--agent-only", action="store_true",
                help="serve only the missing-data agent page, and only its API endpoints")
ap.add_argument("--tunnel-timeout", type=int, default=90, help="seconds to wait for the tunnel URL")
ap.add_argument("--cloudflared", default=None, help="path to the cloudflared binary (default: PATH / download)")
a = ap.parse_args()


def wait_for(url: str, seconds: int = 120) -> bool:
    for _ in range(seconds):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(1)
    return False


def cloudflared() -> str:
    if a.cloudflared:
        return a.cloudflared
    found = shutil.which("cloudflared")
    if found:
        return found
    if platform.system() != "Linux":
        raise SystemExit("cloudflared not found. Install it (e.g. brew install cloudflared) or omit --tunnel.")
    target = Path.home() / ".cache" / "qadt" / "cloudflared"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading cloudflared …")
        urllib.request.urlretrieve(CLOUDFLARED_URL, target)
        target.chmod(target.stat().st_mode | stat.S_IEXEC)
    return str(target)


def start_ollama(model: str, env: dict) -> subprocess.Popen | None:
    if not shutil.which("ollama"):
        if platform.system() == "Linux":
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True,
                           stdout=subprocess.DEVNULL)
        else:
            log.warning("ollama not installed; agents will use deterministic narratives")
            return None
    proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wait_for("http://localhost:11434/api/tags", 30)
    subprocess.run(["ollama", "pull", model], check=True)
    env.update(QADT_LLM_PROVIDER="ollama", QADT_LLM_MODEL=model)
    return proc


def _sigterm(*_):
    raise KeyboardInterrupt      # run the cleanup in main()'s finally block, so no child process is orphaned


def main() -> int:
    signal.signal(signal.SIGTERM, _sigterm)
    configure(a, "run_backend")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    procs: list[subprocess.Popen] = []
    if a.ollama_model:
        p = start_ollama(a.ollama_model, env)
        if p:
            procs.append(p)
    token = a.token or os.environ.get("QADT_API_TOKEN") or (secrets.token_urlsafe(24) if a.tunnel else None)
    if token:
        env["QADT_API_TOKEN"] = token
    host = a.host          # the tunnel connects through localhost, so no need to listen on other interfaces
    if a.agent_only:
        env["QADT_UI_MODE"] = "agent"
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        log.warning("dashboard not built (frontend/dist missing): the API runs, the UI will not be served")
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", host, "--port", str(a.port)]
    if a.reload:
        cmd.append("--reload")
    api = subprocess.Popen(cmd, cwd=ROOT, env=env)
    procs.append(api)
    try:
        if not wait_for(f"http://127.0.0.1:{a.port}/api/health"):
            log.error("API did not start; see the output above")
            return 1
        local = f"http://localhost:{a.port}/" + (f"?token={token}" if token else "")
        log.info("%s running at %s", "Agent-only page" if a.agent_only else "API and dashboard", local)
        if a.tunnel:
            tun = subprocess.Popen([cloudflared(), "tunnel", "--url", f"http://localhost:{a.port}", "--no-autoupdate"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            procs.append(tun)
            found: list[str] = []
            recent: list[str] = []

            def reader():
                for line in tun.stdout:
                    recent.append(line.rstrip())
                    del recent[:-20]
                    log.debug("cloudflared: %s", line.rstrip())
                    if not found and (u := tunnel_url(line)):
                        found.append(u)

            threading.Thread(target=reader, daemon=True).start()
            t0 = time.time()
            while not found and tun.poll() is None and time.time() - t0 < a.tunnel_timeout:
                time.sleep(0.2)
            if not found:
                log.error("the tunnel did not start (%s); stopping the API. Last cloudflared output:\n  %s",
                          "cloudflared exited" if tun.poll() is not None else f"no URL after {a.tunnel_timeout} s",
                          "\n  ".join(recent[-8:]) or "(none)")
                return 1
            print(f"\nOpen the dashboard (keep this link private):\n  {found[0]}/?token={token}\n", flush=True)
        while api.poll() is None:
            if a.tunnel and tun.poll() is not None:
                log.error("cloudflared stopped; stopping the API")
                return 1
            time.sleep(0.5)
        return api.returncode
    except KeyboardInterrupt:
        log.info("stopping …")
        return 0
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()


if __name__ == "__main__":
    sys.exit(main())
