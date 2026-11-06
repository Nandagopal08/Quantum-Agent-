from ...langchain_tools.tools import call
from .base import AgentSpec, agent

SPEC = AgentSpec(
    name="Counterfactual Reasoning Agent", key="counterfactual_reasoning",
    responsibilities=["Compare five maintenance scenarios per asset with Monte Carlo simulation",
                      "Identify the lowest expected-cost scenario and explain why",
                      "Route: ≥2 assets needing maintenance and optimisation enabled → Quantum Optimization"],
    inputs=["twin", "risk", "request.horizon"], outputs=["counterfactual", "route_log"],
    tools=["compare_maintenance_scenarios"],
    runs_when="After the Digital Twin agent",
    on_error="Non-fatal per asset; Decision Coordination falls back to urgency-based actions for those assets.")


@agent(SPEC)
def counterfactual_reasoning(state):
    horizon = state["request"].get("horizon")
    targets = list(state.get("twin") or state["risk"])
    cf = {}
    for aid in targets:
        r = call("compare_maintenance_scenarios", asset_id=aid, horizon=horizon)
        if "error" not in r:
            r.pop("timing_curve", None)
            r.pop("assumptions", None)
        cf[aid] = r
    needing = [aid for aid, r in cf.items() if r.get("best") not in (None, "continue_operation")]
    if len(needing) >= 2 and state["request"].get("optimize", True):
        nxt, why = "quantum_optimization", f"{len(needing)} assets need maintenance; schedule jointly under capacity"
    else:
        nxt, why = "decision_coordination", (f"{len(needing)} asset(s) need maintenance; joint scheduling not needed"
                                             if state["request"].get("optimize", True) else "optimisation disabled")
    return {"counterfactual": cf, "route_log": [{"from": SPEC.key, "to": nxt, "reason": why}],
            "_summary": f"{len(needing)}/{len(cf)} asset(s) better off with maintenance; → {nxt}"}
