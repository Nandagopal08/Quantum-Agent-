"""Experiment database (SQLAlchemy). SQLite in Drive by default; PostgreSQL via QADT_DATABASE_URL.

SQLite on Google Drive's FUSE mount uses rollback journaling (WAL needs shared memory the mount does
not provide). One API process should own the database at a time.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any

from sqlalchemy import JSON, Column, Float, MetaData, String, Table, Text, create_engine, event, insert, select, update
from sqlalchemy.engine import Engine

from .storage import paths, require_drive

metadata = MetaData()

jobs = Table(
    "jobs", metadata,
    Column("id", String(36), primary_key=True), Column("type", String(64)), Column("status", String(24)),
    Column("progress", Float), Column("message", Text), Column("params", JSON), Column("result", JSON),
    Column("error", Text), Column("created_at", Float), Column("updated_at", Float),
)
agent_runs = Table(
    "agent_runs", metadata,
    Column("id", String(36), primary_key=True), Column("status", String(32)), Column("request", JSON),
    Column("state", JSON), Column("approval", JSON), Column("llm", JSON), Column("error", Text),
    Column("created_at", Float), Column("updated_at", Float),
)
maintenance_events = Table(
    "maintenance_events", metadata,
    Column("id", String(36), primary_key=True), Column("asset_id", String(64), index=True),
    Column("action", String(64)), Column("scheduled_for", String(64)), Column("status", String(24)),
    Column("approved_by", String(128)), Column("run_id", String(36)), Column("details", JSON),
    Column("created_at", Float),
)

_engine: Engine | None = None
_engine_lock = threading.Lock()


def _url() -> str:
    url = os.environ.get("QADT_DATABASE_URL")
    if url:
        return url
    require_drive()
    db_path = paths.root / "experiments" / "qadt.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def engine() -> Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            url = _url()
            kw: dict[str, Any] = {"future": True}
            if url.startswith("sqlite"):
                kw["connect_args"] = {"check_same_thread": False, "timeout": 30}
            _engine = create_engine(url, **kw)
            if url.startswith("sqlite"):
                @event.listens_for(_engine, "connect")
                def _pragmas(conn, _record):
                    cur = conn.cursor()
                    cur.execute("PRAGMA journal_mode=DELETE")
                    cur.execute("PRAGMA synchronous=FULL")
                    cur.close()
            metadata.create_all(_engine)
        return _engine


def reset_engine() -> None:
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None


def _clean(v: Any) -> Any:
    return json.loads(json.dumps(v, default=str))


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def insert_row(table: Table, **values) -> dict:
    values = {k: _clean(v) if isinstance(v, (dict, list)) else v for k, v in values.items()}
    with engine().begin() as conn:
        conn.execute(insert(table).values(**values))
    return values


def update_row(table: Table, row_id: str, **values) -> None:
    values = {k: _clean(v) if isinstance(v, (dict, list)) else v for k, v in values.items()}
    if "updated_at" in table.c:
        values["updated_at"] = time.time()
    with engine().begin() as conn:
        conn.execute(update(table).where(table.c.id == row_id).values(**values))


def get_row(table: Table, row_id: str) -> dict | None:
    with engine().connect() as conn:
        r = conn.execute(select(table).where(table.c.id == row_id)).mappings().first()
        return dict(r) if r else None


def list_rows(table: Table, limit: int = 100, **filters) -> list[dict]:
    q = select(table)
    for k, v in filters.items():
        if v is not None:
            q = q.where(table.c[k] == v)
    q = q.order_by(table.c.created_at.desc()).limit(limit)
    with engine().connect() as conn:
        return [dict(r) for r in conn.execute(q).mappings()]
