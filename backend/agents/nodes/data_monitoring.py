from ...langchain_tools.tools import call
from .base import AgentSpec, agent

SPEC = AgentSpec(
    name="Data Monitoring Agent", key="data_monitoring",
    responsibilities=["Validate that each requested asset has usable recent sensor data",
                      "Flag missing values, flat-lined sensors and out-of-distribution readings"],
    inputs=["request.asset_ids"], outputs=["monitoring", "valid_assets"], tools=["check_data_quality"],
    runs_when="Always (entry point)",
    on_error="Fatal: without validated data no downstream agent can run; the workflow ends with status=failed.")


@agent(SPEC, fatal=True)
def data_monitoring(state):
    reports = {aid: call("check_data_quality", asset_id=aid) for aid in state["request"]["asset_ids"]}
    valid = [aid for aid, r in reports.items() if r.get("ok")]
    if not valid:
        raise RuntimeError("no requested asset has usable data")
    issues = sum(len(r.get("issues", [])) for r in reports.values())
    return {"monitoring": reports, "valid_assets": valid,
            "_summary": f"{len(valid)}/{len(reports)} assets usable, {issues} data issue(s) flagged"}
