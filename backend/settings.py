"""Application settings: cost model, risk thresholds, simulation and LLM configuration.

Stored in Drive at ``configuration/app_settings.json`` and editable from the System Settings page.
All monetary values are *assumptions* for demonstration and should be replaced with real figures.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from .storage import load_config, save_config

SETTINGS_NAME = "app_settings"

DEFAULTS: dict[str, Any] = {
    "currency": "USD",
    "equipment": {
        "turbofan": {
            "label": "Turbofan engine (C-MAPSS)", "time_unit": "cycle",
            "preventive_maintenance_cost": 60_000, "corrective_maintenance_cost": 450_000,
            "downtime_cost_per_unit": 1_500,       # per cycle out of service
            "preventive_downtime": 8, "failure_downtime": 40,   # cycles
            "lost_output_per_unit": 900,            # revenue lost per cycle per 100 % load reduction
            "criticality": 1.0, "restoration_fraction": 0.9,
        },
        "bearing": {
            "label": "Rolling-element bearing (IMS)", "time_unit": "hour",
            "preventive_maintenance_cost": 2_500, "corrective_maintenance_cost": 30_000,
            "downtime_cost_per_unit": 800,          # per hour
            "preventive_downtime": 4, "failure_downtime": 36,
            "lost_output_per_unit": 300,
            "criticality": 0.7, "restoration_fraction": 1.0,
        },
    },
    "risk": {
        # risk level from risk score = PoF × CoF, expressed as a fraction of the asset's CoF
        "thresholds": {"medium": 0.10, "high": 0.30, "critical": 0.60},
        "failure_horizon": 30,                     # PoF horizon in the asset's time unit
        "urgency_rul": {"immediate": 10, "soon": 30, "planned": 60},
        "human_approval_level": "high",            # actions at or above this level need approval
    },
    "simulation": {"monte_carlo_samples": 2000, "load_exponent": 2.0, "planning_horizon": 60, "seed": 7},
    "optimization": {"crew_capacity_per_slot": 2, "slots": 2, "qaoa_layers": 2, "qaoa_steps": 80,
                     "max_qubits": 16},
    "llm": {
        "provider": os.environ.get("QADT_LLM_PROVIDER", "ollama"),   # ollama | openai | none
        "model": os.environ.get("QADT_LLM_MODEL", "llama3.1:8b"),
        "base_url": os.environ.get("QADT_LLM_BASE_URL", "http://localhost:11434"),
        "temperature": 0.1, "timeout_seconds": 120,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def get_settings() -> dict:
    return _merge(DEFAULTS, load_config(SETTINGS_NAME, default={}) or {})


def update_settings(patch: dict) -> dict:
    current = load_config(SETTINGS_NAME, default={}) or {}
    merged = _merge(current, patch)
    _merge(DEFAULTS, merged)  # raises on structural problems
    save_config(SETTINGS_NAME, merged)
    return get_settings()


def reset_settings() -> dict:
    save_config(SETTINGS_NAME, {})
    return get_settings()


def equipment_costs(kind: str, settings: dict | None = None) -> dict:
    s = settings or get_settings()
    try:
        return s["equipment"][kind]
    except KeyError:
        raise ValueError(f"unknown equipment type {kind!r}") from None
