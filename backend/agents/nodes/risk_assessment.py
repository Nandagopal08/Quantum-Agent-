from ...langchain_tools.tools import call
from ...risk.engine import LEVELS
from .base import AgentSpec, agent

SPEC = AgentSpec(
    name="Risk Assessment Agent", key="risk_assessment",
    responsibilities=["Compute probability of failure, consequence of failure and risk = PoF × CoF",
                      "Classify risk level and maintenance urgency with the configured thresholds",
                      "Route: low risk everywhere → straight to Decision Coordination; otherwise → Digital Twin"],
    inputs=["predictions"], outputs=["risk", "route_log"], tools=["assess_risk"],
    runs_when="After predictions exist",
    on_error="Non-fatal per asset; assets without a risk assessment are excluded from later agents.")


@agent(SPEC)
def risk_assessment(state):
    risk = {aid: call("assess_risk", asset_id=aid, model=state["request"].get("model"))
            for aid in state["predictions"]}
    elevated = [aid for aid, r in risk.items() if LEVELS.index(r["risk_level"]) >= LEVELS.index("medium")]
    if elevated or state["request"].get("force_full_analysis"):
        nxt, why = "digital_twin", f"{len(elevated)} asset(s) at medium risk or above"
    else:
        nxt, why = "decision_coordination", "all assets low risk; simulation and optimisation not needed"
    worst = max(risk.values(), key=lambda r: r["risk_score"])
    return {"risk": risk, "route_log": [{"from": SPEC.key, "to": nxt, "reason": why}],
            "_summary": f"highest risk {worst['asset_id']}: {worst['risk_level']} "
                        f"(PoF {worst['probability_of_failure']:.1%}); → {nxt}"}
