"""Common wrapper for agent nodes: timing, event publication, structured error handling."""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Callable

from langgraph.errors import GraphBubbleUp

from ...events import bus
from ..state import MaintenanceState


@dataclass
class AgentSpec:
    name: str
    key: str
    responsibilities: list[str]
    inputs: list[str]
    outputs: list[str]
    tools: list[str]
    runs_when: str
    on_error: str
    uses_llm: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


REGISTRY: dict[str, AgentSpec] = {}


def agent(spec: AgentSpec, fatal: bool = False):
    """Register and wrap a node. Non-fatal errors are recorded and the workflow continues degraded;
    fatal errors set status=failed so routing ends the run."""
    REGISTRY[spec.key] = spec

    def decorate(fn: Callable[[MaintenanceState], dict]):
        def node(state: MaintenanceState) -> dict:
            run_id = state.get("run_id", "")
            channel = f"run:{run_id}"
            bus.publish(channel, {"type": "agent_start", "agent": spec.key, "name": spec.name})
            t0 = time.time()
            try:
                update = fn(state) or {}
                seconds = round(time.time() - t0, 3)
                summary = update.pop("_summary", None)
                update.setdefault("trace", []).append({"agent": spec.key, "name": spec.name, "status": "completed",
                                                       "seconds": seconds, "summary": summary})
                update["stage"] = spec.key
                bus.publish(channel, {"type": "agent_end", "agent": spec.key, "status": "completed",
                                      "seconds": seconds, "summary": summary})
                for r in update.get("route_log", []):
                    bus.publish(channel, {"type": "route", **r})
                return update
            except GraphBubbleUp:  # interrupt() for human approval is control flow, not an error
                raise
            except Exception as exc:  # recorded in state; routing decides whether to continue
                seconds = round(time.time() - t0, 3)
                err = {"agent": spec.key, "error": f"{exc.__class__.__name__}: {exc}",
                       "traceback": traceback.format_exc(limit=4)}
                bus.publish(channel, {"type": "agent_end", "agent": spec.key, "status": "failed",
                                      "seconds": seconds, "error": err["error"]})
                out = {"errors": [err], "stage": spec.key,
                       "trace": [{"agent": spec.key, "name": spec.name, "status": "failed", "seconds": seconds,
                                  "summary": err["error"]}]}
                if fatal:
                    out["status"] = "failed"
                return out

        node.__name__ = spec.key
        return node

    return decorate

