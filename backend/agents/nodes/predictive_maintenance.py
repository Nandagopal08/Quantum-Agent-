from ...langchain_tools.tools import call
from .base import AgentSpec, agent

SPEC = AgentSpec(
    name="Predictive Maintenance Agent", key="predictive_maintenance",
    responsibilities=["Predict remaining useful life (turbofan) or time-to-alarm (bearing) with the trained models",
                      "Attach calibrated uncertainty (σ, 90% interval)"],
    inputs=["valid_assets", "request.model"], outputs=["predictions"], tools=["predict_rul"],
    runs_when="After Data Monitoring finds at least one usable asset",
    on_error="Per-asset failures are recorded and the asset is dropped; fatal only if every prediction fails.")


@agent(SPEC, fatal=True)
def predictive_maintenance(state):
    preds, failed = {}, []
    for aid in state["valid_assets"]:
        try:
            preds[aid] = call("predict_rul", asset_id=aid, model=state["request"].get("model"))
        except Exception as exc:
            failed.append({"agent": SPEC.key, "asset_id": aid, "error": str(exc)})
    if not preds:
        raise RuntimeError("; ".join(f["error"] for f in failed) or "no predictions")
    lowest = min(preds.values(), key=lambda p: p["rul_pred"] if p["rul_pred"] is not None else 1e9)
    return {"predictions": preds, "errors": failed,
            "_summary": f"{len(preds)} prediction(s); lowest RUL {lowest['asset_id']} = "
                        f"{lowest['rul_pred']:.1f} {lowest['time_unit']}s" if lowest["rul_pred"] is not None
            else f"{len(preds)} prediction(s)"}
