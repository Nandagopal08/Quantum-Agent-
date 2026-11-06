"""Shared LangGraph state passed between the seven maintenance agents."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class MaintenanceState(TypedDict, total=False):
    run_id: str
    request: dict[str, Any]                 # asset_ids, horizon, optimize, model
    stage: str                              # last completed agent
    monitoring: dict[str, dict]             # Data Monitoring → per-asset data-quality report
    valid_assets: list[str]
    predictions: dict[str, dict]            # Predictive Maintenance → RUL / time-to-alarm per asset
    risk: dict[str, dict]                   # Risk Assessment → PoF, CoF, risk score, level, urgency
    twin: dict[str, dict]                   # Digital Twin → simulated futures (simulated values)
    counterfactual: dict[str, dict]         # Counterfactual Reasoning → scenario comparison per asset
    optimization: dict | None               # Quantum Optimization → QAOA vs classical schedule
    decision: dict                          # Decision Coordination → actions, narrative, approval need
    approval: dict                          # Human approval outcome
    status: str
    trace: Annotated[list[dict], operator.add]       # one record per agent execution
    route_log: Annotated[list[dict], operator.add]   # conditional-routing decisions
    errors: Annotated[list[dict], operator.add]
