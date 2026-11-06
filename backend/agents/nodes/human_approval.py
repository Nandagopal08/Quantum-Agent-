import time

from langgraph.types import interrupt

from ... import db
from .base import AgentSpec, agent

SPEC = AgentSpec(
    name="Human Approval Gate", key="human_approval",
    responsibilities=["Pause the workflow until an engineer approves or rejects consequential actions",
                      "On approval, create maintenance work orders in the maintenance history"],
    inputs=["decision"], outputs=["approval", "status"], tools=[],
    runs_when="Decision Coordination flagged an action at or above the approval risk gate",
    on_error="The run stays in awaiting_approval; no maintenance is recorded until a valid decision arrives.")


@agent(SPEC)
def human_approval(state):
    pending = [a for a in state["decision"]["actions"] if a["requires_approval"]]
    reply = interrupt({"type": "approval_request", "run_id": state["run_id"], "actions": pending})
    approved = bool(reply.get("approved"))
    selected = set(reply.get("asset_ids") or [a["asset_id"] for a in pending]) if approved else set()
    created = []
    for a in state["decision"]["actions"]:
        extra = a.get("capacity_limited") and a["asset_id"] in selected
        if extra or (a["is_maintenance"] and (a["asset_id"] in selected or not a["requires_approval"])):
            action = "maintenance_extra_capacity" if extra else a["action"]
            row = db.insert_row(db.maintenance_events, id=db.new_id(), asset_id=a["asset_id"], action=action,
                                scheduled_for=f"t+{a.get('scheduled_in') or 0:g}", status="scheduled",
                                approved_by=reply.get("reviewer") or "unknown", run_id=state["run_id"],
                                details={"risk_level": a["risk_level"], "basis": a["basis"],
                                         "comment": reply.get("comment")}, created_at=time.time())
            created.append(row["id"])
    approval = {"approved": approved, "reviewer": reply.get("reviewer"), "comment": reply.get("comment"),
                "approved_assets": sorted(selected), "work_orders": created, "decided_at": time.time()}
    return {"approval": approval, "status": "approved" if approved else "rejected",
            "_summary": f"{'approved' if approved else 'rejected'} by {reply.get('reviewer') or 'unknown'}; "
                        f"{len(created)} work order(s)"}
