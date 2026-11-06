"""LangGraph workflow of the seven maintenance agents.

    Data Monitoring → Predictive Maintenance → Risk Assessment ─┬─(elevated risk)→ Digital Twin
                                                               └─(all low)──────→ Decision Coordination
    Digital Twin → Counterfactual Reasoning ─┬─(≥2 assets need maintenance)→ Quantum Optimization
                                             └─(otherwise)─────────────────→ Decision Coordination
    Quantum Optimization → Decision Coordination ─┬─(approval required)→ Human Approval → END
                                                  └─(no approval needed)→ END
A fatal error in Data Monitoring or Predictive Maintenance ends the run with status=failed.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes.base import REGISTRY
from .nodes.counterfactual import counterfactual_reasoning
from .nodes.data_monitoring import data_monitoring
from .nodes.decision_coordination import decision_coordination
from .nodes.digital_twin import digital_twin
from .nodes.human_approval import human_approval
from .nodes.predictive_maintenance import predictive_maintenance
from .nodes.quantum_optimization import quantum_optimization
from .nodes.risk_assessment import risk_assessment
from .state import MaintenanceState

PIPELINE = ["data_monitoring", "predictive_maintenance", "risk_assessment", "digital_twin",
            "counterfactual_reasoning", "quantum_optimization", "decision_coordination", "human_approval"]


def _continue_or_end(next_node: str):
    def edge(state: MaintenanceState) -> str:
        return END if state.get("status") == "failed" else next_node
    return edge


def _last_route(frm: str, default: str):
    def edge(state: MaintenanceState) -> str:
        for r in reversed(state.get("route_log") or []):
            if r["from"] == frm:
                return r["to"]
        return default   # the agent failed before routing: continue on the safe path
    return edge


def _after_decision(state: MaintenanceState) -> str:
    return "human_approval" if (state.get("decision") or {}).get("requires_approval") else END


def build_graph(checkpointer=None):
    g = StateGraph(MaintenanceState)
    for name, fn in (("data_monitoring", data_monitoring), ("predictive_maintenance", predictive_maintenance),
                     ("risk_assessment", risk_assessment), ("digital_twin", digital_twin),
                     ("counterfactual_reasoning", counterfactual_reasoning),
                     ("quantum_optimization", quantum_optimization),
                     ("decision_coordination", decision_coordination), ("human_approval", human_approval)):
        g.add_node(name, fn)
    g.add_edge(START, "data_monitoring")
    g.add_conditional_edges("data_monitoring", _continue_or_end("predictive_maintenance"),
                            ["predictive_maintenance", END])
    g.add_conditional_edges("predictive_maintenance", _continue_or_end("risk_assessment"), ["risk_assessment", END])
    g.add_conditional_edges("risk_assessment", _last_route("risk_assessment", "decision_coordination"),
                            ["digital_twin", "decision_coordination"])
    g.add_edge("digital_twin", "counterfactual_reasoning")
    g.add_conditional_edges("counterfactual_reasoning", _last_route("counterfactual_reasoning", "decision_coordination"),
                            ["quantum_optimization", "decision_coordination"])
    g.add_edge("quantum_optimization", "decision_coordination")
    g.add_conditional_edges("decision_coordination", _after_decision, ["human_approval", END])
    g.add_edge("human_approval", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def agent_specs() -> list[dict]:
    return [REGISTRY[k].to_dict() for k in PIPELINE]


def mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()
