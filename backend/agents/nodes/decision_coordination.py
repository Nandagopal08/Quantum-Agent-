from ...risk.engine import LEVELS, needs_approval
from ...settings import get_settings
from ..llm import narrate
from .base import AgentSpec, agent

SPEC = AgentSpec(
    name="Decision Coordination Agent", key="decision_coordination",
    responsibilities=["Combine risk, counterfactual and optimisation results into one action per asset",
                      "Decide whether human approval is required (configured risk gate)",
                      "Produce a narrative with the LLM (numbers validated against tool outputs) or a deterministic one"],
    inputs=["risk", "counterfactual", "optimization", "predictions"], outputs=["decision"], tools=[],
    runs_when="Always reached (directly after Risk Assessment when all assets are low risk)",
    on_error="Non-fatal: a minimal decision listing urgency-based actions is produced.", uses_llm=True)

_ACTION_FOR_URGENCY = {"immediate": "immediate_maintenance", "soon": "schedule_maintenance",
                       "planned": "plan_maintenance", "monitor": "continue_monitoring"}


def _actions(state) -> list[dict]:
    cf = state.get("counterfactual") or {}
    opt = state.get("optimization") or {}
    plan = {}
    if opt.get("results"):
        best = next(r for r in opt["results"] if r.get("solver") == opt.get("best_solver"))
        plan = {p["asset_id"]: p for p in best["plan"]}
    actions = []
    for aid, r in sorted(state["risk"].items(), key=lambda kv: -kv[1]["risk_score"]):
        a = {"asset_id": aid, "risk_level": r["risk_level"], "urgency": r["urgency"],
             "probability_of_failure": r["probability_of_failure"], "risk_score": r["risk_score"]}
        if aid in plan:
            p = plan[aid]
            a.update(action="maintenance" if p["maintain"] else "continue_monitoring", basis="optimised_schedule",
                     scheduled_in=p["slot_time"], solver=opt["best_solver"])
            wanted = cf.get(aid, {}).get("best") not in (None, "continue_operation")
            if not p["maintain"] and wanted and LEVELS.index(r["risk_level"]) >= LEVELS.index("high"):
                # The schedule left this asset out because of crew capacity, not because it is safe.
                a.update(action="escalate_capacity_shortfall", capacity_limited=True,
                         note="needs maintenance per its counterfactual analysis but no crew slot was available")
        elif aid in cf and cf[aid].get("best"):
            best = next(s for s in cf[aid]["scenarios"] if s["scenario"] == cf[aid]["best"])
            a.update(action=cf[aid]["best"], basis="counterfactual", scheduled_in=best.get("maintenance_at"),
                     expected_cost=best["total_expected_cost"], failure_probability_after=best["failure_probability"],
                     saving_vs_continue=next(s for s in cf[aid]["scenarios"] if s["scenario"] == "continue_operation")
                     ["total_expected_cost"] - best["total_expected_cost"])
        else:
            a.update(action=_ACTION_FOR_URGENCY[r["urgency"]], basis="risk_urgency")
        a["is_maintenance"] = a["action"] not in ("continue_operation", "continue_monitoring", "reduce_load",
                                                  "escalate_capacity_shortfall")
        a["requires_approval"] = (a["is_maintenance"] or a.get("capacity_limited", False)) and needs_approval(r["risk_level"])
        actions.append(a)
    return actions


def _deterministic(actions: list[dict], state, currency: str) -> dict:
    maint = [a for a in actions if a["is_maintenance"]]
    lines = []
    for a in actions[:6]:
        txt = (f"{a['asset_id']}: {a['risk_level']} risk (PoF {a['probability_of_failure']:.1%}, "
               f"risk score {a['risk_score']:,.0f} {currency}) → {a['action'].replace('_', ' ')}")
        if a.get("scheduled_in") is not None:
            txt += f" at t+{a['scheduled_in']:g}"
        if a.get("saving_vs_continue"):
            txt += f", expected saving {a['saving_vs_continue']:,.0f} {currency} vs continuing"
        lines.append(txt)
    opt = state.get("optimization") or {}
    caveats = ["Costs and consequences are configurable assumptions (System Settings).",
               "Scenario and twin values are simulated from model predictions."]
    short = [a["asset_id"] for a in actions if a.get("capacity_limited")]
    if short:
        caveats.insert(0, f"Capacity shortfall: {', '.join(short)} need maintenance but exceed crew capacity "
                          "(System Settings → Optimisation). Add capacity or approve an extra slot.")
    if opt.get("results"):
        q = next((r for r in opt["results"] if r["solver"] == "qaoa"), None)
        if q and "gap_percent" in q:
            caveats.append(f"QAOA plan gap to the MILP optimum: {q['gap_percent']:.1f}% (measured on a simulator).")
    return {"summary": (f"{len(maint)} of {len(actions)} assets should receive maintenance; "
                        f"{sum(a['requires_approval'] for a in actions)} action(s) need human approval.")
            if maint else "No maintenance is needed now; continue monitoring all assets.",
            "rationale": lines, "caveats": caveats}


def _r(v):
    return round(v, 3) if isinstance(v, float) and abs(v) < 10 else (round(v) if isinstance(v, float) else v)


def _llm_context(actions: list[dict], state) -> dict:
    """Compact, rounded view of the computed results — everything the narrative may cite, nothing more."""
    cf = state.get("counterfactual") or {}
    opt = state.get("optimization") or {}
    return {
        "actions": [{k: _r(v) for k, v in a.items() if k not in ("requires_approval", "is_maintenance")}
                    for a in actions],
        "counterfactual_explanations": {aid: v.get("explanation", [])[:3] for aid, v in cf.items()},
        "optimization": None if not opt.get("results") else {
            "best_solver": opt.get("best_solver"),
            "solvers": [{"solver": r["solver"], "total_expected_cost": _r(r.get("total_expected_cost")),
                         "gap_percent": _r(r.get("gap_percent")), "feasible": r.get("feasible")}
                        for r in opt["results"] if "total_expected_cost" in r]},
        "approval_required_for": [a["asset_id"] for a in actions if a["requires_approval"]],
    }


@agent(SPEC)
def decision_coordination(state):
    s = get_settings()
    actions = _actions(state)
    deterministic = _deterministic(actions, state, s["currency"])
    context = _llm_context(actions, state)
    narrative, meta = narrate(context, s) if state["request"].get("use_llm", True) else (None, {"used": False,
                                                                                         "reason": "disabled for run"})
    text = narrative.model_dump() if narrative else deterministic
    requires = any(a["requires_approval"] for a in actions)
    decision = {"actions": actions, "narrative": text, "narrative_source": "llm" if narrative else "deterministic",
                "llm": meta, "deterministic_narrative": deterministic, "requires_approval": requires,
                "approval_gate": s["risk"]["human_approval_level"],
                "highest_risk_level": max((a["risk_level"] for a in actions), key=LEVELS.index, default="low")}
    return {"decision": decision, "status": "awaiting_approval" if requires else "completed",
            "_summary": f"{sum(a['is_maintenance'] for a in actions)} maintenance action(s); narrative from "
                        f"{decision['narrative_source']}; approval {'required' if requires else 'not required'}"}
