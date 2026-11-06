"""Atomic file writes on Drive.

Writes go to a temporary sibling file and are then moved into place with
``os.replace``, so a runtime that dies mid-write never leaves a truncated file
under the final name.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .drive import drive_guard


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tmp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")


def atomic_write(path: Path, writer: Callable[[Path], None]) -> Path:
    """Call ``writer(tmp_path)`` then move ``tmp_path`` onto ``path``."""
    path = Path(path)
    with drive_guard(f"Writing {path}"):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_sibling(path)
        try:
            writer(tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
    return path


def write_bytes(path: Path, data: bytes) -> Path:
    return atomic_write(path, lambda tmp: tmp.write_bytes(data))


def write_text(path: Path, text: str) -> Path:
    return atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def write_json(path: Path, data: Any) -> Path:
    return write_text(path, json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    with drive_guard(f"Reading {path}"):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default


def copy_file(src: Path, dst: Path) -> Path:
    """Copy a (typically local scratch) file onto Drive atomically."""
    return atomic_write(dst, lambda tmp: shutil.copyfile(src, tmp))


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with drive_guard(f"Hashing {path}"), open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_stats(path: Path, include_hidden: bool = False) -> tuple[int, int]:
    """Return ``(file_count, total_bytes)`` for a directory tree.

    Hidden files (markers and in-progress temp files) are skipped unless ``include_hidden``.
    """
    count = size = 0
    with drive_guard(f"Scanning {path}"):
        for root, _dirs, files in os.walk(path):
            for name in files:
                if name.startswith(".") and not include_hidden:
                    continue
                try:
                    size += os.stat(os.path.join(root, name)).st_size
                    count += 1
                except FileNotFoundError:
                    pass
    return count, size


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"
