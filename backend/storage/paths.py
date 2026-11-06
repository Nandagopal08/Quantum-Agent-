"""Named Drive paths, so every backend module resolves locations the same way."""

from __future__ import annotations

from pathlib import Path

from ..config import StorageConfig, get_config

MODEL_CATEGORIES = ("classical_ai", "quantum_ai", "agentic_ai")

# Short names for result areas, used by backend.storage.artifacts.save_result.
RESULT_AREAS = {
    "features": "datasets/features",
    "dataset_reports": "datasets/dataset_reports",
    "simulation_results": "digital_twin/simulation_results",
    "equipment_states": "digital_twin/equipment_states",
    "training_logs": "experiments/training_logs",
    "evaluation_metrics": "experiments/evaluation_metrics",
    "comparison_reports": "experiments/comparison_reports",
    "qaoa_results": "optimization/qaoa_results",
    "maintenance_schedules": "optimization/maintenance_schedules",
    "shap": "explainability/shap",
    "counterfactuals": "explainability/counterfactuals",
    "visualizations": "visualizations",
    "reports": "reports",
    "configuration": "configuration",
}


class ProjectPaths:
    """Resolves project locations from the *current* config each time.

    Paths are computed lazily so a later :func:`backend.config.configure` call is
    picked up by modules that already hold a ``ProjectPaths`` instance.
    """

    def __init__(self, config: StorageConfig | None = None):
        self._config = config

    @property
    def root(self) -> Path:
        return (self._config or get_config()).project_root

    def __truediv__(self, rel: str) -> Path:
        return self.root / rel

    # datasets
    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    def raw(self, dataset: str) -> Path:
        return self.datasets / "raw" / dataset

    def processed(self, dataset: str) -> Path:
        return self.datasets / "processed" / dataset

    def cleaned(self, dataset: str) -> Path:
        return self.processed(dataset) / "cleaned"

    def features(self, dataset: str | None = None) -> Path:
        base = self.datasets / "features"
        return base / dataset if dataset else base

    def splits(self, dataset: str) -> Path:
        return self.features(dataset) / "splits"

    @property
    def dataset_reports(self) -> Path:
        return self.datasets / "dataset_reports"

    @property
    def dataset_metadata_file(self) -> Path:
        return self.dataset_reports / "dataset_metadata.json"

    # models
    def models(self, category: str) -> Path:
        if category not in MODEL_CATEGORIES:
            raise ValueError(f"Unknown model category {category!r}; expected one of {MODEL_CATEGORIES}")
        return self.root / "models" / category

    def checkpoints(self, category: str, run_name: str) -> Path:
        return self.models(category) / run_name / "checkpoints"

    # everything else
    def area(self, name: str) -> Path:
        try:
            return self.root / RESULT_AREAS[name]
        except KeyError:
            raise ValueError(f"Unknown result area {name!r}; expected one of {sorted(RESULT_AREAS)}") from None

    @property
    def configuration(self) -> Path:
        return self.root / "configuration"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


paths = ProjectPaths()
