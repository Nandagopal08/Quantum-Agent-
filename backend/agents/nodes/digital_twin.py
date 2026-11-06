from ...langchain_tools.tools import call
from ...risk.engine import LEVELS
from .base import AgentSpec, agent

SPEC = AgentSpec(
    name="Digital Twin Agent", key="digital_twin",
    responsibilities=["Simulate future operation of elevated-risk assets at nominal load (simulated values)",
                      "Report projected cumulative failure probability over the planning horizon"],
    inputs=["risk", "request.horizon"], outputs=["twin"], tools=["simulate_digital_twin"],
    runs_when="Risk Assessment found at least one asset at medium risk or above",
    on_error="Non-fatal: counterfactual reasoning can still run from the predictions.")


@agent(SPEC)
def digital_twin(state):
    horizon = int(state["request"].get("horizon") or 60)
    targets = [aid for aid, r in state["risk"].items()
               if LEVELS.index(r["risk_level"]) >= LEVELS.index("medium") or state["request"].get("force_full_analysis")]
    twin = {}
    for aid in targets:
        sim = call("simulate_digital_twin", asset_id=aid, horizon=horizon)
        sim["series"] = sim["series"][:: max(1, len(sim["series"]) // 30)]   # keep the trace compact
        twin[aid] = sim
    return {"twin": twin, "_summary": f"simulated {len(twin)} asset(s) over {horizon} steps"}
