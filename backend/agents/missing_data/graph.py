"""Single-agent LangGraph workflow: quantum-model RUL forecasting from sensor windows with missing data.

One agent (``md_agent``) carries out the whole procedure with its tools, in this order:

    1. data quality   – obtain the window, profile its missingness (rate, gaps, likely pattern, weak sensors);
                        stop early when too little data is left to impute            [observe_asset, profile_missingness]
    2. plan           – choose the untried imputer with the lowest *measured* benchmark RMSE at the detected
                        rate and pattern                                              [rank_imputers]
    3. impute         – run it, with a label-free self-check (hide some observed values, re-impute, compare
                        with linear interpolation on the same values)                 [run_imputation]
    4. check          – accept, or reject and go back to 2 with the next-best method (bounded by max_attempts;
                        when attempts run out, keep the best attempt)
    5. forecast       – RUL with the best quantum sequence model and its classical twin   [forecast_rul]
    6. report         – deterministic report; an optional LLM narrative (structured output) is rejected if it
                        cites any number absent from the tool outputs

The true values behind injected missingness are kept out of the agent's inputs (per-run vault); they are used
only in step 6, after every decision is made, to score the run.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from pydantic import BaseModel, Field

from ..llm import get_chat_model, llm_status, ungrounded_numbers
import time

from ...events import bus
from ..nodes.base import AgentSpec, agent
from .tools import call, expected_input, window_and_truth

# Hidden truth of simulated missingness, per run. Deliberately outside the graph state and the event stream so
# no agent can read it; only the Reporter takes it out (after every decision is made) to score the run.
TRUTH_VAULT: dict[str, dict] = {}

MAX_MISSING = 0.9            # beyond this the window is not imputed
CRITIC_TOLERANCE = 1.5       # self-check RMSE may exceed the benchmark expectation by this factor
LINEAR_MARGIN = 1.10         # a learned imputer must not be >10 % worse than linear on the self-check
PLAUSIBLE_Z = 6.0            # imputed values beyond ±6σ of training data are implausible


class MissingDataState(TypedDict, total=False):
    run_id: str
    request: dict[str, Any]
    observation: dict           # values (None = missing), mask, sensors — what the agents see
    quality: dict
    plan: dict
    imputation: dict
    review: dict
    attempts: Annotated[list[dict], operator.add]
    forecast: dict
    report: dict
    status: str
    stage: str
    trace: Annotated[list[dict], operator.add]
    route_log: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]
    steps: list[dict]          # the agent's step log (data_quality, plan, impute, check, forecast, report)


SPEC = AgentSpec(
    name="Missing-Data Maintenance Agent", key="md_agent",
    responsibilities=["Profile missingness and stop when too little data remains",
                      "Choose the imputer with the lowest measured benchmark error for the detected rate and pattern",
                      "Impute with a label-free self-check; reject and retry with the next-best method when it fails",
                      "Forecast RUL with the best quantum sequence model and its classical twin",
                      "Report, optionally with an LLM narrative validated against the tool outputs"],
    inputs=["request"], outputs=["observation", "quality", "attempts", "imputation", "review", "forecast", "report"],
    tools=["observe_asset", "profile_missingness", "rank_imputers", "run_imputation", "forecast_rul"],
    runs_when="Always (single agent)",
    on_error="Data-quality failure is fatal (status failed); a failed imputation is retried with the next method; a "
             "failed forecast is reported (completed_without_forecast); LLM problems fall back to the deterministic report.",
    uses_llm=True, extra={"steps": ["data_quality", "plan", "impute", "check", "forecast", "report"]})

STEP_LABEL = {"data_quality": "Data quality", "plan": "Plan", "impute": "Impute", "check": "Check",
              "forecast": "Forecast", "report": "Report"}


# ---------------------------------------------------------------- steps of the agent


def data_quality(state):
    req = state["request"]
    if req.get("values") is not None:
        subset = req.get("subset", "FD001")
        exp = expected_input(subset)
        values = req["values"]
        W, C = exp["window"], len(exp["sensors"])
        if not values or any(len(r) != len(values[0]) for r in values):
            raise ValueError("uploaded window must be a non-empty rectangular T × C table")
        if len(values[0]) != C:
            raise ValueError(f"uploaded window has {len(values[0])} channels; the trained models expect {C} "
                             f"({', '.join(exp['sensors'])}), standardised like the training data")
        if len(values) < W:
            raise ValueError(f"uploaded window has {len(values)} steps; at least {W} are needed")
        values = values[-W:]                        # the models were trained on W-step windows
        obs = {"asset_id": req.get("asset_id") or "uploaded-window", "subset": subset, "values": values,
               "sensors": req.get("sensors") or exp["sensors"], "window": W,
               "mask": [[0.0 if v is None else 1.0 for v in row] for row in values], "source": "uploaded"}
    else:
        obs = call("observe_asset", asset_id=req["asset_id"], missing_rate=req.get("missing_rate", 0.3),
                   pattern=req.get("pattern", "mcar"), seed=req.get("seed", 0))
        _, truth = window_and_truth(req["asset_id"], req.get("missing_rate", 0.3), req.get("pattern", "mcar"),
                                    req.get("seed", 0))
        TRUTH_VAULT[state["run_id"]] = truth
    prof = call("profile_missingness", mask=obs["mask"], sensors=obs["sensors"])
    usable = prof["missing_rate"] < MAX_MISSING and len(prof["fully_missing_channels"]) < len(obs["sensors"])
    weak = [s for s, r in prof["per_channel"].items() if r > 0.8]
    quality = {**prof, "usable": usable, "weak_sensors": weak}
    route = {"from": "data_quality", "to": "plan" if usable else "report",
             "reason": (f"{prof['missing_rate']:.0%} missing, pattern {prof['likely_pattern']}" if usable else
                        f"{prof['missing_rate']:.0%} missing exceeds the {MAX_MISSING:.0%} limit")}
    return {"observation": obs, "quality": quality, "route_log": [route],
            "_summary": f"{prof['missing_rate']:.1%} missing ({prof['likely_pattern']}, mean gap "
                        f"{prof['mean_gap_length']:.1f} steps), {len(weak)} weak sensor(s)"}


def imputation_planner(state):
    q = state["quality"]
    tried = [a["method"] for a in state.get("attempts", [])]
    forced = state["request"].get("imputer")
    if forced and forced not in tried:
        return {"plan": {"method": forced, "expected_rmse": None, "reason": "requested by the user",
                         "alternatives": []}, "_summary": f"using requested imputer {forced}"}
    pattern = q["likely_pattern"] if q["likely_pattern"] != "none" else "mcar"
    r = call("rank_imputers", missing_rate=q["missing_rate"], pattern=pattern,
             subset=state["observation"]["subset"], exclude=tried)
    if not r["ranking"]:
        plan = {"method": "linear", "expected_rmse": None, "reason": "no untried method left; linear fallback",
                "alternatives": [], "evidence": r["evidence"]}
    else:
        top = r["ranking"][0]
        plan = {"method": top["method"], "expected_rmse": top["rmse"], "benchmark_rate": r["benchmark_rate"],
                "pattern": r["pattern"], "alternatives": r["ranking"][1:4], "evidence": r["evidence"],
                "reason": "lowest measured benchmark RMSE among untried methods"}
    return {"plan": plan, "_summary": f"chose {plan['method']} "
                                      + (f"(benchmark RMSE {plan['expected_rmse']:.3f})" if plan["expected_rmse"] else "")}


def imputer(state):
    plan = state["plan"]
    try:
        res = call("run_imputation", method=plan["method"], values=state["observation"]["values"],
                   subset=state["observation"]["subset"], seed=state["request"].get("seed", 0))
    except Exception as exc:
        att = {"method": plan["method"], "failed": f"{exc.__class__.__name__}: {exc}"}
        return {"imputation": {"method": plan["method"], "error": att["failed"]}, "attempts": [att],
                "_summary": f"{plan['method']} failed: {att['failed']}"}
    chk = res["self_check"]
    att = {"method": plan["method"], "expected_rmse": plan.get("expected_rmse"), **chk}
    return {"imputation": res, "attempts": [att],
            "_summary": f"{plan['method']} imputed {res['imputed_count']} values; self-check RMSE "
                        f"{chk.get('holdout_rmse', float('nan')):.3f} (linear {chk.get('linear_holdout_rmse', float('nan')):.3f})"}


def _critic_decision(state) -> dict:
    imp, plan = state["imputation"], state["plan"]
    attempts = state.get("attempts", [])
    max_attempts = int(state["request"].get("max_attempts", 3))
    reasons = []
    if "error" in imp:
        reasons.append("imputation failed")
    else:
        chk = imp["self_check"]
        h, lin, exp = chk.get("holdout_rmse"), chk.get("linear_holdout_rmse"), plan.get("expected_rmse")
        if h is not None and exp is not None and h > CRITIC_TOLERANCE * exp:
            reasons.append(f"self-check RMSE {h:.3f} > {CRITIC_TOLERANCE}× benchmark {exp:.3f}")
        if h is not None and lin is not None and plan["method"] not in ("linear",) and h > LINEAR_MARGIN * lin:
            reasons.append(f"self-check RMSE {h:.3f} worse than linear interpolation {lin:.3f}")
        if imp["max_abs_z"] > PLAUSIBLE_Z:
            reasons.append(f"implausible imputed value ({imp['max_abs_z']:.1f}σ)")
    accepted = not reasons
    retry = not accepted and len(attempts) < max_attempts
    return {"accepted": accepted, "retry": retry, "reasons": reasons, "attempt": len(attempts),
            "max_attempts": max_attempts,
            "confidence": "high" if accepted and len(attempts) == 1 else "medium" if accepted else "low"}


def critic(state):
    review = _critic_decision(state)
    to = "plan" if review["retry"] else "forecast"
    reason = "; ".join(review["reasons"]) or "self-check within tolerance"
    out = {"review": review, "route_log": [{"from": "check", "to": to, "reason": reason}]}
    if not review["accepted"] and not review["retry"]:
        # out of attempts: fall back to the best-scoring successful attempt
        ok = [a for a in state.get("attempts", []) if a.get("holdout_rmse") is not None]
        if ok:
            best = min(ok, key=lambda a: a["holdout_rmse"])
            review["fallback_method"] = best["method"]
            if best["method"] != state["imputation"].get("method"):
                res = call("run_imputation", method=best["method"], values=state["observation"]["values"],
                           subset=state["observation"]["subset"], seed=state["request"].get("seed", 0))
                out["imputation"] = res
    verdict = "accepted" if review["accepted"] else ("rejected → re-plan" if review["retry"] else "rejected, no attempts left")
    out["_summary"] = f"{state['imputation'].get('method')}: {verdict} ({reason})"
    return out


def forecaster(state):
    imp = state["imputation"]
    if "values" not in imp:
        raise RuntimeError("no successful imputation to forecast from")
    f = call("forecast_rul", values=imp["values"], subset=state["observation"]["subset"],
             model=state["request"].get("rul_model"))
    p = f["prediction"]
    tw = f.get("classical_twin")
    return {"forecast": f, "_summary": f"{p['model']}: RUL {p['rul_pred']:.1f} cycles"
                                       + (f" (twin {tw['model']}: {tw['rul_pred']:.1f})" if tw else "")}


class ReporterNarrative(BaseModel):
    """Structured output requested from the LLM; every number in it is checked against the tool outputs."""
    summary: str = Field(description="3-5 sentences: missing data, imputer choice and rejections, RUL forecast")
    caveats: list[str] = Field(default_factory=list, description="1-3 limitations the reader must know")


NARRATIVE_PROMPT = """You are the Reporter agent of a predictive-maintenance system that forecasts the remaining
useful life of turbofan engines from sensor windows with missing values, using quantum sequence models.
Using ONLY numbers present in the JSON, write 3-5 sentences: how much data was missing, which imputer
was used and why (including any rejected attempts), the RUL forecast with its interval, and the main caveat.
Never invent numbers."""


def _narrative(context: dict, use_llm: bool) -> tuple[str | None, dict]:
    meta: dict[str, Any] = {"status": llm_status()}
    llm = get_chat_model() if use_llm else None
    if llm is None:
        meta.update(used=False, reason="LLM disabled for this run" if not use_llm else meta["status"]["detail"])
        return None, meta
    try:
        out = llm.with_structured_output(ReporterNarrative).invoke(
            [("system", NARRATIVE_PROMPT), ("human", json.dumps(context, default=str))])
        if not isinstance(out, ReporterNarrative):
            out = ReporterNarrative.model_validate(out)
        text = " ".join([out.summary.strip(), *(c.strip() for c in out.caveats)]).strip()
        if not text:
            raise ValueError("empty narrative")
    except Exception as exc:          # invalid / unparseable / failed response → deterministic text
        meta.update(used=False, reason=f"LLM response rejected: {exc.__class__.__name__}: {str(exc)[:160]}")
        return None, meta
    bad = ungrounded_numbers(text, context)
    if bad:
        meta.update(used=False, reason="rejected: ungrounded numbers", ungrounded_numbers=bad[:10])
        return None, meta
    meta["used"] = True
    return text, meta


def reporter(state):
    q, f, rev = state.get("quality", {}), state.get("forecast"), state.get("review", {})
    imp = state.get("imputation", {})
    attempts = state.get("attempts", [])
    pred = (f or {}).get("prediction")
    summary = {
        "asset_id": state.get("observation", {}).get("asset_id"),
        "missing_rate": q.get("missing_rate"), "pattern": q.get("likely_pattern"),
        "imputer": imp.get("method"), "attempts": [a["method"] for a in attempts],
        "rejected": [a["method"] for a in attempts if a["method"] != imp.get("method")],
        "confidence": rev.get("confidence", "none") if q.get("usable", False) else "none",
        "rul_pred": pred["rul_pred"] if pred else None, "rul_interval_90": pred["interval_90"] if pred else None,
        "rul_model": pred["model"] if pred else None,
        "twin_rul_pred": (f.get("classical_twin") or {}).get("rul_pred") if f else None,
    }
    lines = [f"{summary['asset_id']}: {q.get('missing_rate', 0):.1%} of sensor values missing "
             f"(pattern: {q.get('likely_pattern')})."]
    if not q.get("usable", True):
        lines.append("Too little data to impute; no forecast was made.")
    else:
        lines.append(f"Imputed with {summary['imputer']} after {len(attempts)} attempt(s)"
                     + (f"; rejected: {', '.join(summary['rejected'])}" if summary["rejected"] else "") + ".")
        if pred:
            lo, hi = pred["interval_90"]
            lines.append(f"{pred['model']} forecasts RUL {pred['rul_pred']:.1f} cycles (90% interval {lo:.1f}–{hi:.1f}).")
    deterministic = " ".join(lines)
    text, meta = _narrative({"summary": summary, "quality": {k: q.get(k) for k in ("missing_rate", "likely_pattern",
                                                                                   "mean_gap_length", "weak_sensors")},
                             "attempts": attempts}, state["request"].get("use_llm", True))

    evaluation = {}
    truth = TRUTH_VAULT.pop(state.get("run_id", ""), None) or {}
    if truth.get("values") is not None and "values" in imp:
        x_true = np.asarray(truth["values"], float)
        mask = np.asarray(state["observation"]["mask"], float)
        x_imp = np.asarray(imp["values"], float)
        m = mask == 0
        evaluation["imputation_rmse_true"] = float(np.sqrt(np.mean((x_imp[m] - x_true[m]) ** 2))) if m.any() else 0.0
        from ...missing_data import impute_baseline

        evaluation["linear_rmse_true"] = (float(np.sqrt(np.mean((impute_baseline("linear", x_true[None], mask[None])[0][m]
                                                                 - x_true[m]) ** 2))) if m.any() else 0.0)
        if truth.get("rul_true") is not None:
            evaluation["rul_true"] = truth["rul_true"]
            if pred:
                evaluation["rul_abs_error"] = abs(pred["rul_pred"] - truth["rul_true"])
                full = call("forecast_rul", values=truth["values"], subset=state["observation"]["subset"],
                            model=pred["model"])
                evaluation["rul_pred_complete_data"] = full["prediction"]["rul_pred"]
        evaluation["note"] = "Available only because the missingness was simulated; not used by any agent decision."
    report = {"summary": summary, "text": text or deterministic, "deterministic_text": deterministic, "llm": meta,
              "evaluation": evaluation}
    status = "completed" if pred else ("insufficient_data" if not q.get("usable", True) else "completed_without_forecast")
    return {"report": report, "status": status, "_summary": deterministic}


# ---------------------------------------------------------------- the agent


_LISTS = ("attempts", "route_log", "errors")


@agent(SPEC, fatal=True)
def missing_data_agent(state):
    """Run the six steps with a working copy of the state; every step's result is recorded in ``steps``."""
    st = {**state, **{k: list(state.get(k) or []) for k in _LISTS}}
    channel = f"run:{state.get('run_id', '')}"
    steps: list[dict] = []

    def do(name, fn, fatal=False) -> bool:
        bus.publish(channel, {"type": "agent_step", "step": name, "label": STEP_LABEL[name], "status": "running"})
        t0 = time.time()
        try:
            upd = dict(fn(st) or {})
        except Exception as exc:
            if fatal:
                raise
            err = f"{exc.__class__.__name__}: {exc}"
            st["errors"].append({"step": name, "error": err})
            steps.append({"step": name, "status": "failed", "summary": err, "seconds": round(time.time() - t0, 3)})
            bus.publish(channel, {"type": "agent_step", "step": name, "status": "failed", "summary": err})
            return False
        summary = upd.pop("_summary", None)
        for k in _LISTS:
            st[k].extend(upd.pop(k, []))
        st.update(upd)
        steps.append({"step": name, "status": "completed", "summary": summary, "seconds": round(time.time() - t0, 3)})
        bus.publish(channel, {"type": "agent_step", "step": name, "status": "completed", "summary": summary})
        return True

    do("data_quality", data_quality, fatal=True)
    if st["quality"].get("usable"):
        max_attempts = int(st["request"].get("max_attempts", 3))
        for _ in range(max_attempts):              # hard bound, whatever happens inside the steps
            do("plan", imputation_planner)
            do("impute", imputer)
            if not do("check", critic) or not (st.get("review") or {}).get("retry"):
                break
        do("forecast", forecaster)
    do("report", reporter)
    keys = ("observation", "quality", "plan", "imputation", "review", "forecast", "report", "status", *_LISTS)
    return {**{k: st[k] for k in keys if k in st}, "steps": steps,
            "_summary": (st.get("report") or {}).get("deterministic_text")}


def build_graph():
    g = StateGraph(MissingDataState)
    # plain wrapper: the shared @agent decorator is annotated with MaintenanceState, and LangGraph would otherwise
    # filter this workflow's state down to that schema's keys
    g.add_node("md_agent", lambda state: missing_data_agent(state))
    g.add_edge(START, "md_agent")
    g.add_edge("md_agent", END)
    return g.compile()


def agent_specs() -> list[dict]:
    return [SPEC.to_dict()]


def mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()
