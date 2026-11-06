"""Training checkpoints in Drive, so an interrupted Colab session can resume.

Usage::

    ckpt = CheckpointManager("lstm_fd001", category="classical_ai", every_seconds=600)
    state, info = ckpt.load_latest()
    start = info["step"] + 1 if info else 0
    for epoch in range(start, epochs):
        ...train...
        ckpt.maybe_save(epoch, lambda: {"model": model.state_dict(), "optim": opt.state_dict()},
                        metrics={"val_rmse": rmse})
    ckpt.save(final_state, epoch, final=True)

Each checkpoint is serialised to local scratch, copied to Drive under a
temporary name and renamed into place; ``index.json`` is updated last, so the
index only ever points at complete files.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Callable

from .drive import drive_guard, local_scratch_dir, require_drive
from .fileio import copy_file, read_json, utc_now, write_json
from .paths import paths


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def serialize(obj: Any, path: Path, fmt: str) -> None:
    if fmt == "torch":
        _torch().save(obj, path)
    else:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize(path: Path, fmt: str, map_location: Any = None) -> Any:
    # These files are written by this project into the user's own Drive, so full unpickling is expected.
    if fmt == "torch":
        return _torch().load(path, map_location=map_location, weights_only=False)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def resolve_format(serializer: str) -> str:
    if serializer == "auto":
        return "torch" if _torch() is not None else "pickle"
    if serializer not in ("torch", "pickle"):
        raise ValueError("serializer must be 'auto', 'torch' or 'pickle'")
    if serializer == "torch" and _torch() is None:
        raise ImportError("serializer='torch' requested but PyTorch is not installed")
    return serializer


class CheckpointManager:
    def __init__(self, run_name: str, category: str = "classical_ai", keep_last: int = 3,
                 every_n_steps: int | None = None, every_seconds: float | None = 600,
                 serializer: str = "auto"):
        self.run_name = run_name
        self.category = category
        self.keep_last = keep_last
        self.every_n_steps = every_n_steps
        self.every_seconds = every_seconds
        self.format = resolve_format(serializer)
        self._last_save_time = time.monotonic()
        self._last_saved_step: int | None = None

    @property
    def directory(self) -> Path:
        return paths.checkpoints(self.category, self.run_name)

    @property
    def index_file(self) -> Path:
        return self.directory / "index.json"

    def list(self) -> list[dict]:
        require_drive()
        return read_json(self.index_file, default={}).get("checkpoints", [])

    def latest(self) -> dict | None:
        """Newest checkpoint entry whose file is actually present in Drive."""
        with drive_guard("Reading checkpoints"):
            for entry in reversed(self.list()):
                if (self.directory / entry["file"]).is_file():
                    return entry
        return None

    def save(self, state: Any, step: int, metrics: dict | None = None, final: bool = False) -> Path:
        require_drive()
        ext = "pt" if self.format == "torch" else "pkl"
        filename = f"{'final' if final else 'ckpt'}_step{step:08d}.{ext}"
        local = local_scratch_dir("checkpoints", self.category, self.run_name) / filename
        serialize(state, local, self.format)
        size = local.stat().st_size
        target = copy_file(local, self.directory / filename)
        local.unlink(missing_ok=True)

        entries = [e for e in self.list() if e["file"] != filename]
        entries.append({"step": step, "file": filename, "format": self.format, "final": final,
                        "size_bytes": size, "saved_at": utc_now(), "metrics": metrics or {}})
        entries.sort(key=lambda e: (e["step"], e.get("final", False)))
        entries = self._prune(entries)
        write_json(self.index_file, {"run_name": self.run_name, "category": self.category,
                                     "checkpoints": entries})
        self._last_save_time = time.monotonic()
        self._last_saved_step = step
        return target

    def _prune(self, entries: list[dict]) -> list[dict]:
        if not self.keep_last:
            return entries
        rolling = [e for e in entries if not e.get("final")]
        drop = rolling[:-self.keep_last]
        with drive_guard("Pruning checkpoints"):
            for entry in drop:
                (self.directory / entry["file"]).unlink(missing_ok=True)
        dropped = {e["file"] for e in drop}
        return [e for e in entries if e["file"] not in dropped]

    def should_save(self, step: int) -> bool:
        if self._last_saved_step == step:
            return False
        if self.every_n_steps and (step + 1) % self.every_n_steps == 0:
            return True
        return bool(self.every_seconds) and time.monotonic() - self._last_save_time >= self.every_seconds

    def maybe_save(self, step: int, state_fn: Callable[[], Any], metrics: dict | None = None) -> Path | None:
        """Save if the step/time interval has elapsed. ``state_fn`` is only called when saving."""
        if self.should_save(step):
            return self.save(state_fn(), step, metrics)
        return None

    def load(self, entry: dict, map_location: Any = None) -> Any:
        require_drive()
        with drive_guard(f"Loading checkpoint {entry['file']}"):
            return deserialize(self.directory / entry["file"], entry.get("format", self.format), map_location)

    def load_latest(self, map_location: Any = None) -> tuple[Any, dict | None]:
        """Return ``(state, entry)`` for the newest checkpoint, or ``(None, None)`` when starting fresh."""
        entry = self.latest()
        if entry is None:
            return None, None
        print(f"[qadt.checkpoints] resuming {self.run_name} from step {entry['step']} ({entry['file']})")
        return self.load(entry, map_location), entry
