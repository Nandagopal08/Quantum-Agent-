"""Project-wide storage configuration.

Google Drive is the only persistent storage location. The project root is
configurable through the ``QADT_DRIVE_ROOT`` environment variable or
:func:`set_project_root`, but it must always live inside the mounted Drive
(``QADT_DRIVE_MOUNT``, default ``/content/drive``). Local Colab storage under
``/content`` is used only as scratch space for temporary files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

ENV_DRIVE_MOUNT = "QADT_DRIVE_MOUNT"
ENV_PROJECT_ROOT = "QADT_DRIVE_ROOT"
ENV_LOCAL_SCRATCH = "QADT_LOCAL_SCRATCH"
ENV_STORAGE_MODE = "QADT_STORAGE_MODE"

DEFAULT_DRIVE_MOUNT = "/content/drive"
DEFAULT_PROJECT_ROOT = "/content/drive/MyDrive/Quantum_Agentic_Digital_Twin"
DEFAULT_LOCAL_SCRATCH = "/content/qadt_scratch"

# colab:       Google Drive mounted in Colab with drive.mount() (default).
# desktop:     Google Drive for desktop; QADT_DRIVE_MOUNT is e.g. ~/Library/CloudStorage/GoogleDrive-<account>.
# development: a plain local folder (QADT_DRIVE_ROOT must be set). Data is NOT in Google Drive; the API
#              reports this and the web UI shows a warning banner. Use only for local testing.
STORAGE_MODES = ("colab", "desktop", "development")

# Every directory below is created automatically under the project root.
PROJECT_DIRS: tuple[str, ...] = (
    "datasets/raw/cmapss",
    "datasets/raw/ims_bearings",
    "datasets/processed/cmapss",
    "datasets/processed/ims_bearings",
    "datasets/features",
    "datasets/dataset_reports",
    "models/classical_ai",
    "models/quantum_ai",
    "models/agentic_ai",
    "digital_twin/simulation_results",
    "digital_twin/equipment_states",
    "experiments/training_logs",
    "experiments/evaluation_metrics",
    "experiments/comparison_reports",
    "optimization/qaoa_results",
    "optimization/maintenance_schedules",
    "explainability/shap",
    "explainability/counterfactuals",
    "visualizations",
    "reports",
    "configuration",
)


@dataclass(frozen=True)
class StorageConfig:
    drive_mount: Path
    project_root: Path
    local_scratch: Path
    storage_mode: str = "colab"

    @classmethod
    def from_env(cls) -> "StorageConfig":
        mode = os.environ.get(ENV_STORAGE_MODE, "colab").strip().lower()
        if mode not in STORAGE_MODES:
            raise ValueError(f"{ENV_STORAGE_MODE}={mode!r}; expected one of {STORAGE_MODES}")
        mount = Path(os.environ.get(ENV_DRIVE_MOUNT, DEFAULT_DRIVE_MOUNT)).expanduser()
        if mode == "desktop":
            default_root = mount / "My Drive" / "Quantum_Agentic_Digital_Twin"
        else:
            default_root = Path(DEFAULT_PROJECT_ROOT)
        root = os.environ.get(ENV_PROJECT_ROOT)
        if mode == "development" and not root:
            raise ValueError(f"{ENV_STORAGE_MODE}=development requires {ENV_PROJECT_ROOT} to name a local folder")
        scratch_default = DEFAULT_LOCAL_SCRATCH if mode == "colab" else str(Path.home() / ".cache" / "qadt_scratch")
        return cls(
            drive_mount=mount,
            project_root=Path(root).expanduser() if root else default_root,
            local_scratch=Path(os.environ.get(ENV_LOCAL_SCRATCH, scratch_default)).expanduser(),
            storage_mode=mode,
        )

    @property
    def is_google_drive(self) -> bool:
        return self.storage_mode != "development"


_config: StorageConfig = StorageConfig.from_env()


def get_config() -> StorageConfig:
    return _config


def configure(
    project_root: str | os.PathLike | None = None,
    drive_mount: str | os.PathLike | None = None,
    local_scratch: str | os.PathLike | None = None,
    storage_mode: str | None = None,
) -> StorageConfig:
    """Override storage locations for this process and return the new config."""
    global _config
    changes = {}
    if storage_mode is not None:
        if storage_mode not in STORAGE_MODES:
            raise ValueError(f"storage_mode must be one of {STORAGE_MODES}")
        changes["storage_mode"] = storage_mode
    if project_root is not None:
        changes["project_root"] = Path(project_root)
    if drive_mount is not None:
        changes["drive_mount"] = Path(drive_mount)
    if local_scratch is not None:
        changes["local_scratch"] = Path(local_scratch)
    _config = replace(_config, **changes)
    return _config


def set_project_root(project_root: str | os.PathLike) -> StorageConfig:
    return configure(project_root=project_root)


def reset_config() -> StorageConfig:
    """Reload configuration from environment variables."""
    global _config
    _config = StorageConfig.from_env()
    return _config
