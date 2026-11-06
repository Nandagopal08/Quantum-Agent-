"""Security regression tests: path traversal, token auth, headers, CORS, settings and upload validation, error
redaction, request-size limits. Positive and negative cases for every control."""

from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def client(project):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def token(monkeypatch):
    import backend.main as main

    monkeypatch.setattr(main, "_TOKEN", "s3cret-token")
    return "s3cret-token"


# ---------------------------------------------------------------- static files / path traversal


@pytest.mark.skipif(not (ROOT / "frontend" / "dist" / "index.html").exists(), reason="dashboard not built")
def test_static_route_serves_dashboard_but_never_files_outside_dist(client):
    assert client.get("/").status_code == 200 and b"<!doctype html" in client.get("/").content.lower()
    asset = next((ROOT / "frontend" / "dist" / "assets").glob("*.js"))
    r = client.get(f"/assets/{asset.name}")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]        # positive case
    secrets = [b"fastapi", b"qadt-dashboard", b"QADT_STORAGE_MODE", b"import "]
    for p in ("/%2e%2e/%2e%2e/requirements.txt", "/..%2F..%2Frequirements.txt", "/%2e%2e/package.json",
              "/%2e%2e%2f%2e%2e%2f.env.example", "/assets/..%2F..%2F..%2Fbackend%2Fmain.py",
              "/%2e%2e/%2e%2e/backend/config.py", "//etc/passwd", "/%2F%2Fetc%2Fpasswd"):
        r = client.get(p)
        assert not any(s in r.content for s in secrets), p
        assert r.status_code in (200, 404) and r.headers["content-type"].startswith(("text/html", "application/json")), p
    assert client.get("/api/not-a-route").status_code == 404


def test_result_file_endpoint_is_confined(client):
    for p in ("/api/files/shap/..%2F..%2F..%2Frequirements.txt", "/api/files/shap/x.json",
              "/api/files/not_an_area/x.png", "/api/files/shap/%2e%2e%2fconfiguration%2fapp_settings.json"):
        assert client.get(p).status_code in (404, 422), p


# ---------------------------------------------------------------- authentication


def test_token_required_when_configured(client, token):
    assert client.get("/api/health").status_code == 200                       # liveness stays open
    assert client.get("/api/qseq/models").status_code == 401                   # no token
    assert client.get("/api/qseq/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/qseq/models", params={"token": "s3cret-tokeñ"}).status_code == 401   # non-ASCII: 401 not 500
    assert client.get("/api/qseq/models", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert client.get("/api/qseq/models", params={"token": token}).status_code == 200            # SSE style


def test_no_token_means_open_local_api(client):
    assert client.get("/api/qseq/models").status_code == 200


def test_docs_and_schema_hidden_when_token_set(project):
    code = ("import os; os.environ['QADT_API_TOKEN']='t'; from fastapi.testclient import TestClient; "
            "from backend.main import app; c=TestClient(app); "
            "print(c.get('/docs').status_code, c.get('/openapi.json').status_code, "
            "c.get('/openapi.json').headers['content-type'])")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300,
                         env={**__import__("os").environ, "QADT_STORAGE_MODE": "development",
                              "QADT_DRIVE_ROOT": str(project)})
    docs, schema, ctype = out.stdout.strip().split()[-3:]
    assert "json" not in ctype, out.stdout                   # schema not served (SPA fallback or 404 instead)


# ---------------------------------------------------------------- headers and CORS


def test_security_headers_and_cors(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["x-frame-options"] == "DENY"
    evil = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers
    ok = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"


# ---------------------------------------------------------------- input validation


def test_settings_reject_unknown_keys_wrong_types_and_unsafe_values(client):
    bad = [{"nope": 1}, {"llm": {"temperature": "hot"}}, {"llm": {"base_url": "file:///etc/passwd"}},
           {"optimization": {"max_qubits": 40}}, {"risk": {"thresholds": {"high": 0.05}}},
           {"equipment": {"turbofan": {"preventive_maintenance_cost": -5}}}, {"llm": {"provider": "evil"}},
           {"simulation": {"monte_carlo_samples": 10**9}}, [1, 2]]
    for patch in bad:
        assert client.put("/api/settings", json=patch).status_code == 422, patch
    r = client.put("/api/settings", json={"optimization": {"qaoa_layers": 3}})
    assert r.status_code == 200 and r.json()["optimization"]["qaoa_layers"] == 3
    client.post("/api/settings/reset")


def _zip_bytes(name="x.txt", data=b"hello") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, data)
    return buf.getvalue()


def test_dataset_upload_limits_and_official_archive_check(client, monkeypatch):
    import backend.api.datasets as ds

    from backend.storage.datasets import archive_path

    before = archive_path("cmapss").read_bytes()
    r = client.post("/api/datasets/cmapss/upload", files={"file": ("a.zip", b"not a zip", "application/zip")})
    assert r.status_code == 422
    r = client.post("/api/datasets/cmapss/upload", files={"file": ("a.zip", _zip_bytes(), "application/zip")})
    assert r.status_code == 422 and "official" in r.json()["detail"]
    monkeypatch.setattr(ds, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/api/datasets/cmapss/upload", files={"file": ("a.zip", _zip_bytes(), "application/zip")})
    assert r.status_code == 413
    assert client.post("/api/datasets/nope/upload", files={"file": ("a.zip", b"x")}).status_code == 404
    assert archive_path("cmapss").read_bytes() == before, "a rejected upload must not touch the stored archive"
    assert not list(archive_path("cmapss").parent.glob("*.upload")), "partial upload left behind"


def test_zip_bomb_guard(tmp_path, monkeypatch):
    from backend.storage import datasets

    z = tmp_path / "big.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        f.writestr("zeros.bin", b"\0" * (2 << 20))
    monkeypatch.setattr(datasets, "MAX_EXTRACT_BYTES", 1 << 20)
    with pytest.raises(datasets.DatasetIntegrityError):
        datasets._extract_zip(z, tmp_path / "out")
    assert not (tmp_path / "out" / "zeros.bin").exists()


def test_zip_slip_is_rejected(tmp_path):
    from backend.storage import datasets

    z = tmp_path / "slip.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../../escaped.txt", b"x")
    with pytest.raises(datasets.DatasetIntegrityError):
        datasets._extract_zip(z, tmp_path / "out")
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_agent_request_limits(client):
    assert client.post("/api/qseq/agents/runs", json={"asset_id": "../../etc"}).status_code == 422
    assert client.post("/api/qseq/agents/runs", json={"values": [[0.0]] * 501}).status_code == 422
    assert client.post("/api/qseq/agents/runs", json={"values": [[0.0] * 65]}).status_code == 422
    assert client.post("/api/qseq/agents/runs", json={"missing_rate": -0.1}).status_code == 422


def test_error_messages_do_not_leak_storage_paths(client, project):
    r = client.get("/api/qseq/agents/runs/doesnotexist")
    assert r.status_code == 404
    r = client.get("/api/training/logs/no_such_run")
    assert str(project) not in r.text
    from backend.api.common import redact

    assert str(project) not in redact(f"No saved model in {project}/models/quantum_ai")
    assert "<project>" in redact(f"No saved model in {project}/models/quantum_ai")


def test_failed_tunnel_stops_the_api_and_never_prints_a_bogus_link(project, tmp_path):
    """cloudflared's error messages contain https://api.trycloudflare.com — that must not become the dashboard link,
    and a failed tunnel must stop the API (no server left running)."""
    import os
    import socket

    fake = tmp_path / "cloudflared"
    fake.write_text('#!/bin/sh\necho \'ERR failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel"\'\nexit 1\n')
    fake.chmod(0o755)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "QADT_STORAGE_MODE": "development", "QADT_DRIVE_ROOT": str(project)}
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_backend.py"), "--port", str(port), "--tunnel",
                          "--cloudflared", str(fake), "--log-file", str(tmp_path / "be.log")],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert out.returncode == 1, out.stdout[-2000:]
    assert "Open the dashboard" not in out.stdout and "tunnel did not start" in out.stdout
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", port)) != 0, "API still listening after the tunnel failed"


def _run_app(project, code: str, **env_extra) -> str:
    import os

    env = {**os.environ, "QADT_STORAGE_MODE": "development", "QADT_DRIVE_ROOT": str(project), **env_extra}
    out = subprocess.run([sys.executable, "-c", "from fastapi.testclient import TestClient; from backend.main import app; "
                          "c = TestClient(app); " + code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    return out.stdout.strip().splitlines()[-1]


def test_agent_only_mode_serves_only_the_agent_api(project):
    code = ("import json; r = {p: c.get(p).status_code for p in ['/api/health', '/api/qseq/models', '/api/qseq/agents/runs', "
            "'/api/qseq/agents/status', '/api/qseq/report', '/api/settings', '/api/fleet/turbofan', '/api/datasets', "
            "'/api/agents/specs']}; r['mode'] = c.get('/api/health').json()['ui_mode']; print(json.dumps(r))")
    import json

    r = json.loads(_run_app(project, code, QADT_UI_MODE="agent"))
    assert r["mode"] == "agent"
    assert r["/api/health"] == 200 and r["/api/qseq/models"] == 200
    assert r["/api/qseq/agents/runs"] == 200 and r["/api/qseq/agents/status"] == 200
    for p in ("/api/qseq/report", "/api/settings", "/api/fleet/turbofan", "/api/datasets", "/api/agents/specs"):
        assert r[p] == 404, p
    full = json.loads(_run_app(project, "import json; print(json.dumps({'mode': c.get('/api/health').json()['ui_mode'], "
                                        "'settings': c.get('/api/settings').status_code}))"))
    assert full == {"mode": "full", "settings": 200}


def test_health_hides_paths_when_token_is_set(project):
    import json

    r = json.loads(_run_app(project, "import json; print(json.dumps(c.get('/api/health').json()['storage']))",
                            QADT_API_TOKEN="t"))
    assert set(r) == {"mode", "is_google_drive", "mounted", "instructions"}
    assert str(project) not in json.dumps(r)
