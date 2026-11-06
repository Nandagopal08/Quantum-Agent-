"""Google Drive mounting, verification and disconnect handling.

There is deliberately no local-only fallback: every persistent read or write
calls :func:`require_drive` first, which raises :class:`DriveNotMountedError`
with mounting instructions when Drive is unavailable.
"""

from __future__ import annotations

import contextlib
import errno
import functools
import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, TypeVar

from ..config import PROJECT_DIRS, StorageConfig, get_config

if TYPE_CHECKING:
    from .paths import ProjectPaths

T = TypeVar("T")

MOUNT_INSTRUCTIONS = """\
Google Drive is not mounted, so nothing can be downloaded, trained or saved.

To mount it, run this in a Colab cell (not with `!python`, it must run in the notebook kernel):

    from google.colab import drive
    drive.mount('/content/drive')

or, using this project:

    from backend.storage import setup_storage
    paths = setup_storage()

Approve the Google account access prompt, then re-run the cell that failed."""

RECONNECT_INSTRUCTIONS = """\
The connection to Google Drive was lost (the Drive FUSE mount stopped responding).
Data already written to Drive is safe. To reconnect, run in a Colab cell:

    from backend.storage import remount_drive
    remount_drive()

or manually:

    from google.colab import drive
    drive.flush_and_unmount()
    drive.mount('/content/drive', force_remount=True)

Then re-run the interrupted step. Downloads resume from the partial file and
training resumes from the latest checkpoint in Drive."""

# errno values the Drive FUSE mount produces when it drops out underneath us.
_DISCONNECT_ERRNOS = {errno.ENOTCONN, errno.EIO, errno.ESTALE}


class DriveNotMountedError(RuntimeError):
    def __init__(self, detail: str = "", instructions: str = MOUNT_INSTRUCTIONS):
        message = f"{detail}\n\n{instructions}" if detail else instructions
        super().__init__(message)


class DriveDisconnectedError(DriveNotMountedError):
    def __init__(self, detail: str = ""):
        super().__init__(detail, RECONNECT_INSTRUCTIONS)


def in_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:  # "google" itself is not installed
        return False


def _is_mount_point(path: Path) -> bool:
    return os.path.ismount(path)


def is_drive_mounted(config: StorageConfig | None = None) -> bool:
    config = config or get_config()
    try:
        if config.storage_mode == "development":
            return True
        if config.storage_mode == "desktop":
            # Drive for desktop is a File Provider folder, not a mount point; it always contains "My Drive".
            return (config.drive_mount / "My Drive").is_dir() or (config.drive_mount / "MyDrive").is_dir()
        return _is_mount_point(config.drive_mount) and any(config.drive_mount.iterdir())
    except OSError:
        return False


_dev_warning_shown = False


def _warn_development_mode(config: StorageConfig) -> None:
    global _dev_warning_shown
    if not _dev_warning_shown:
        import logging

        logging.getLogger("qadt.storage").warning(
            "QADT_STORAGE_MODE=development: data is stored in %s, NOT in Google Drive.", config.project_root)
        _dev_warning_shown = True


def _check_root_inside_mount(config: StorageConfig) -> None:
    if config.storage_mode == "development":
        return
    mount = config.drive_mount.resolve()
    root = config.project_root.resolve()
    if root != mount and mount not in root.parents:
        raise DriveNotMountedError(
            f"The project root {config.project_root} is not inside the Google Drive mount "
            f"{config.drive_mount}. Persistent data must live in Drive; set QADT_DRIVE_ROOT "
            f"(or backend.config.set_project_root) to a folder under {config.drive_mount}."
        )


def require_drive(config: StorageConfig | None = None) -> StorageConfig:
    """Raise unless Drive is mounted, reachable and contains the project root's parent."""
    config = config or get_config()
    _check_root_inside_mount(config)
    if config.storage_mode == "development":
        _warn_development_mode(config)
        config.project_root.mkdir(parents=True, exist_ok=True)
        return config
    if not is_drive_mounted(config):
        raise DriveNotMountedError(f"No Google Drive found at {config.drive_mount} "
                                   f"(QADT_STORAGE_MODE={config.storage_mode}).")
    try:
        # Touch the filesystem so a stale FUSE mount is detected here, not mid-write.
        os.stat(config.project_root.parent)
    except FileNotFoundError:
        raise DriveNotMountedError(
            f"{config.project_root.parent} does not exist in the mounted Drive. "
            "Check that QADT_DRIVE_ROOT points into MyDrive (or a shared drive you can access)."
        ) from None
    except OSError as exc:
        if exc.errno in _DISCONNECT_ERRNOS:
            raise DriveDisconnectedError(str(exc)) from exc
        raise
    return config


def mount_drive(force_remount: bool = False, config: StorageConfig | None = None) -> StorageConfig:
    """Mount Google Drive in Colab (no-op if it is already mounted)."""
    config = config or get_config()
    if is_drive_mounted(config) and not force_remount:
        return require_drive(config)
    if config.storage_mode == "desktop":
        raise DriveNotMountedError(
            f"Google Drive for desktop folder not found at {config.drive_mount}. Start Google Drive for desktop "
            "and set QADT_DRIVE_MOUNT to its folder (macOS: ~/Library/CloudStorage/GoogleDrive-<account>).")
    if not in_colab():
        raise DriveNotMountedError(
            "Automatic mounting needs Google Colab (google.colab is not importable here)."
        )
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(str(config.drive_mount), force_remount=force_remount)
    return require_drive(config)


def remount_drive(config: StorageConfig | None = None) -> StorageConfig:
    """Flush, unmount and remount Drive after a dropped connection."""
    config = config or get_config()
    if not in_colab():
        raise DriveNotMountedError("Remounting needs Google Colab.")
    from google.colab import drive  # type: ignore[import-not-found]

    with contextlib.suppress(Exception):
        drive.flush_and_unmount()
    return mount_drive(force_remount=True, config=config)


@contextlib.contextmanager
def drive_guard(action: str = "Drive operation") -> Iterator[None]:
    """Turn FUSE disconnect errors into :class:`DriveDisconnectedError`."""
    try:
        yield
    except DriveNotMountedError:
        raise
    except OSError as exc:
        if exc.errno in _DISCONNECT_ERRNOS:
            raise DriveDisconnectedError(f"{action} failed: {exc}") from exc
        raise


def guarded(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator form of :func:`drive_guard` that also calls :func:`require_drive`."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        require_drive()
        with drive_guard(func.__name__):
            return func(*args, **kwargs)

    return wrapper


def ensure_project_structure(config: StorageConfig | None = None) -> Path:
    """Create the full project folder tree in Drive. Returns the project root."""
    config = require_drive(config)
    with drive_guard("Creating project folders"):
        for rel in PROJECT_DIRS:
            (config.project_root / rel).mkdir(parents=True, exist_ok=True)
    return config.project_root


def setup_storage(force_remount: bool = False) -> "ProjectPaths":
    """Mount Drive, create the folder tree and return the project paths.

    This is the single entry point notebooks should call first.
    """
    from .paths import ProjectPaths

    config = mount_drive(force_remount=force_remount)
    ensure_project_structure(config)
    return ProjectPaths(config)


def local_scratch_dir(*parts: str) -> Path:
    """Temporary local directory (``/content/...``). Never use it for anything that must persist."""
    path = get_config().local_scratch.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
