"""Saving and loading everything the project produces, always into Drive.

Backend modules should use these helpers instead of writing files themselves,
so locations and atomic-write behaviour stay consistent.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .checkpoints import deserialize, resolve_format, serialize
from .drive import drive_guard, local_scratch_dir, require_drive
from .fileio import atomic_write, copy_file, read_json, utc_now, write_json, write_text
from .paths import MODEL_CATEGORIES, paths

# ---------------------------------------------------------------- models


def model_dir(category: str, name: str) -> Path:
    return paths.models(category) / name


def save_model(model: Any, category: str, name: str, config: Mapping | None = None,
               metrics: Mapping | None = None, serializer: str = "auto") -> Path:
    """Save a trained model (classical, quantum or agentic) plus its config and metrics."""
    require_drive()
    fmt = resolve_format(serializer)
    filename = "model.pt" if fmt == "torch" else "model.pkl"
    local = local_scratch_dir("models", category, name) / filename
    serialize(model, local, fmt)
    target = copy_file(local, model_dir(category, name) / filename)
    local.unlink(missing_ok=True)
    write_json(model_dir(category, name) / "model_info.json", {
        "name": name, "category": category, "file": filename, "format": fmt,
        "saved_at": utc_now(), "config": dict(config or {}), "metrics": dict(metrics or {}),
    })
    return target


def load_model(category: str, name: str, map_location: Any = None) -> Any:
    """Load a model directly from Drive."""
    require_drive()
    info = read_json(model_dir(category, name) / "model_info.json")
    if info is None:
        raise FileNotFoundError(f"No saved model {category}/{name} in {paths.models(category)}")
    with drive_guard(f"Loading model {category}/{name}"):
        return deserialize(model_dir(category, name) / info["file"], info["format"], map_location)


def list_models(category: str | None = None) -> list[dict]:
    require_drive()
    found = []
    with drive_guard("Listing models"):
        for cat in [category] if category else MODEL_CATEGORIES:
            base = paths.models(cat)
            if base.is_dir():
                for info_file in sorted(base.glob("*/model_info.json")):
                    found.append(read_json(info_file))
    return found


# ---------------------------------------------------------------- configuration


def save_config(name: str, config: Mapping) -> Path:
    require_drive()
    return write_json(paths.configuration / f"{name}.json", dict(config))


def load_config(name: str, default: Any = None) -> Any:
    require_drive()
    return read_json(paths.configuration / f"{name}.json", default=default)


# ---------------------------------------------------------------- tabular data


def save_dataframe(df, path: Path, fmt: str = "auto") -> Path:
    """Write a pandas DataFrame as Parquet (when pyarrow is installed) or CSV."""
    require_drive()
    path = Path(path)
    if fmt == "auto":
        fmt = "parquet" if importlib.util.find_spec("pyarrow") else "csv"
    path = path.with_suffix(".parquet" if fmt == "parquet" else ".csv")
    if fmt == "parquet":
        return atomic_write(path, lambda tmp: df.to_parquet(tmp, index=False))
    return atomic_write(path, lambda tmp: df.to_csv(tmp, index=False))


def load_dataframe(path: Path):
    import pandas as pd

    require_drive()
    path = Path(path)
    with drive_guard(f"Reading {path}"):
        if not path.exists():
            for suffix in (".parquet", ".csv"):
                if path.with_suffix(suffix).exists():
                    path = path.with_suffix(suffix)
                    break
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def save_cleaned(dataset: str, name: str, df) -> Path:
    return save_dataframe(df, paths.cleaned(dataset) / name)


def save_features(dataset: str, name: str, df) -> Path:
    return save_dataframe(df, paths.features(dataset) / name)


def save_splits(dataset: str, splits: Mapping[str, Any], tag: str = "default") -> dict[str, Path]:
    """Save e.g. ``{"train": df, "test": df}`` under ``datasets/features/<dataset>/splits/<tag>/``."""
    out = {name: save_dataframe(df, paths.splits(dataset) / tag / name) for name, df in splits.items()}
    write_json(paths.splits(dataset) / tag / "splits.json", {
        "dataset": dataset, "tag": tag, "saved_at": utc_now(),
        "splits": {name: {"file": p.name, "rows": len(splits[name])} for name, p in out.items()},
    })
    return out


def load_splits(dataset: str, tag: str = "default") -> dict[str, Any]:
    info = read_json(paths.splits(dataset) / tag / "splits.json")
    if info is None:
        raise FileNotFoundError(f"No saved splits {dataset}/{tag} in {paths.splits(dataset)}")
    return {name: load_dataframe(paths.splits(dataset) / tag / meta["file"]) for name, meta in info["splits"].items()}


# ---------------------------------------------------------------- results, logs, figures


def save_result(area: str, name: str, data: Any) -> Path:
    """Save a JSON result into a named area, e.g. ``save_result("qaoa_results", "run_03", {...})``.

    Areas: simulation_results, equipment_states, evaluation_metrics, comparison_reports,
    qaoa_results, maintenance_schedules, shap, counterfactuals, reports, configuration, ...
    """
    require_drive()
    if hasattr(data, "to_csv"):
        return save_dataframe(data, paths.area(area) / name)
    return write_json(paths.area(area) / f"{name}.json", data)


def save_metrics(run_name: str, metrics: Mapping) -> Path:
    return save_result("evaluation_metrics", run_name, {"run_name": run_name, "saved_at": utc_now(), **metrics})


def save_figure(fig, name: str, area: str = "visualizations", dpi: int = 150) -> Path:
    """Save a matplotlib figure as PNG."""
    require_drive()
    path = paths.area(area) / f"{name}.png"
    return atomic_write(path, lambda tmp: fig.savefig(tmp, dpi=dpi, bbox_inches="tight", format="png"))


def save_report(name: str, text: str, suffix: str = ".md") -> Path:
    require_drive()
    return write_text(paths.reports / f"{name}{suffix}", text)


class ExperimentLogger:
    """Append-only JSON-lines training log in ``experiments/training_logs/<run>.jsonl``.

    Each line is flushed and fsynced, so the log survives a runtime disconnect up to the last record.
    """

    def __init__(self, run_name: str, config: Mapping | None = None):
        require_drive()
        self.run_name = run_name
        self.path = paths.area("training_logs") / f"{run_name}.jsonl"
        with drive_guard("Creating training log"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if config is not None:
            save_result("training_logs", f"{run_name}_config", dict(config))

    def log(self, **record: Any) -> None:
        line = json.dumps({"time": utc_now(), **record}, default=str)
        with drive_guard(f"Writing {self.path}"), open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def records(self) -> list[dict]:
        with drive_guard(f"Reading {self.path}"):
            if not self.path.exists():
                return []
            return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
