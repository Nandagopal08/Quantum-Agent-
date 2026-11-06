"""Shared command-line plumbing for the scripts: logging, storage options, error handling, JSON summaries.

Every script accepts:
    --storage-mode {colab,desktop,development}   (default: $QADT_STORAGE_MODE or colab)
    --drive-root PATH       project folder (development: any local folder; colab/desktop: inside Drive)
    --drive-mount PATH      Drive mount point (desktop mode: ~/Library/CloudStorage/GoogleDrive-<account>)
    --log-level LEVEL       DEBUG | INFO | WARNING
    --log-file PATH         also write the log to a file (default: <project>/reports/logs/<script>.log)
    --summary-json PATH     write the step's result summary as JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("qadt.cli")


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("storage and logging")
    g.add_argument("--storage-mode", choices=("colab", "desktop", "development"), default=None)
    g.add_argument("--drive-root", default=None)
    g.add_argument("--drive-mount", default=None)
    g.add_argument("--log-level", default="INFO")
    g.add_argument("--log-file", default=None)
    g.add_argument("--summary-json", default=None)
    return ap


def csv(value: str | None, cast: Callable = str) -> list | None:
    return [cast(v.strip()) for v in value.split(",") if v.strip()] if value else None


def configure(args: argparse.Namespace, script: str) -> None:
    """Apply storage options (before backend modules read the config) and set up logging."""
    for flag, env in (("storage_mode", "QADT_STORAGE_MODE"), ("drive_root", "QADT_DRIVE_ROOT"),
                      ("drive_mount", "QADT_DRIVE_MOUNT")):
        if getattr(args, flag):
            os.environ[env] = str(Path(getattr(args, flag)).expanduser()) if flag != "storage_mode" else getattr(args, flag)
    from backend import config

    config.reset_config()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_file = args.log_file
    if log_file is None:
        try:
            from backend.storage import paths
            from backend.storage.drive import is_drive_mounted

            if config.get_config().storage_mode == "development" or is_drive_mounted():
                log_file = str(paths.root / "reports" / "logs" / f"{script}.log")
        except Exception:
            log_file = None
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log.info("%s · storage mode %s · root %s%s", script, config.get_config().storage_mode,
             config.get_config().project_root, f" · log {log_file}" if log_file else "")


def run(script: str, args: argparse.Namespace, fn: Callable[[], Any], mount: bool = True) -> int:
    """Configure, prepare storage, run ``fn`` and translate failures into exit codes.

    Exit codes: 0 success · 2 storage unavailable · 3 missing prerequisite (dataset/model) · 1 other error.
    """
    configure(args, script)
    from backend.services.fleet import ModelNotTrained
    from backend.storage import DatasetNotReadyError, DriveNotMountedError
    from backend.workflows import setup_storage_step

    t0 = time.time()
    try:
        setup_storage_step(mount=mount)
        result = fn()
    except DriveNotMountedError as exc:
        log.error("storage unavailable:\n%s", exc)
        return 2
    except (DatasetNotReadyError, ModelNotTrained, FileNotFoundError) as exc:
        log.error("missing prerequisite: %s", exc)
        log.error("run the earlier steps first (download_datasets.py → preprocess_datasets.py → train_all_models.py)")
        return 3
    except KeyboardInterrupt:
        log.warning("interrupted; completed work is saved and the next run resumes from checkpoints")
        return 130
    except Exception as exc:
        log.error("%s failed: %s\n%s", script, exc, traceback.format_exc())
        return 1
    log.info("%s finished in %.0f s", script, time.time() - t0)
    if args.summary_json:
        from backend.langchain_tools.tools import to_jsonable

        Path(args.summary_json).write_text(json.dumps(to_jsonable(result), indent=1, default=str))
    return 0
