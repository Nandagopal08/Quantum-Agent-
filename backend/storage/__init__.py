"""Google Drive-backed persistent storage for the whole project.

Start every Colab session with::

    from backend.storage import setup_storage, prepare_datasets, show_storage_page
    paths = setup_storage()          # mounts Drive, creates the folder tree
    prepare_datasets()               # downloads/extracts only what is missing
    show_storage_page()
"""

from .artifacts import (
    ExperimentLogger,
    list_models,
    load_config,
    load_dataframe,
    load_model,
    load_splits,
    save_cleaned,
    save_config,
    save_dataframe,
    save_features,
    save_figure,
    save_metrics,
    save_model,
    save_report,
    save_result,
    save_splits,
)
from .checkpoints import CheckpointManager
from .datasets import (
    DATASETS,
    DatasetDownloadError,
    DatasetIntegrityError,
    DatasetNotReadyError,
    download_dataset,
    extract_dataset,
    find_files,
    load_metadata,
    prepare_datasets,
    require_dataset,
    verify_dataset,
    verify_zip,
)
from .drive import (
    MOUNT_INSTRUCTIONS,
    RECONNECT_INSTRUCTIONS,
    DriveDisconnectedError,
    DriveNotMountedError,
    drive_guard,
    ensure_project_structure,
    is_drive_mounted,
    mount_drive,
    remount_drive,
    require_drive,
    setup_storage,
)
from .paths import ProjectPaths, paths
from .status import show_storage_page, storage_report

__all__ = [
    "CheckpointManager",
    "DatasetDownloadError",
    "DatasetIntegrityError",
    "DatasetNotReadyError",
    "DATASETS",
    "download_dataset",
    "drive_guard",
    "DriveDisconnectedError",
    "DriveNotMountedError",
    "ensure_project_structure",
    "ExperimentLogger",
    "extract_dataset",
    "find_files",
    "is_drive_mounted",
    "list_models",
    "load_config",
    "load_dataframe",
    "load_metadata",
    "load_model",
    "load_splits",
    "mount_drive",
    "MOUNT_INSTRUCTIONS",
    "paths",
    "prepare_datasets",
    "ProjectPaths",
    "RECONNECT_INSTRUCTIONS",
    "remount_drive",
    "require_dataset",
    "require_drive",
    "save_cleaned",
    "save_config",
    "save_dataframe",
    "save_features",
    "save_figure",
    "save_metrics",
    "save_model",
    "save_report",
    "save_result",
    "save_splits",
    "setup_storage",
    "show_storage_page",
    "storage_report",
    "verify_dataset",
    "verify_zip",
]
