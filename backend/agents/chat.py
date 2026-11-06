"""Free-form questions answered by a LangChain tool-calling agent over the read-only tools.

The LLM chooses which tools to call; numbers in the final answer are checked against the tool
results and flagged when unsupported. Without an LLM the endpoint says so instead of guessing.
"""

from __future__ import annotations

import json
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..langchain_tools.tools import CHAT_TOOLS, to_jsonable
from .llm import get_chat_model, llm_status, ungrounded_numbers

SYSTEM = """You answer questions about a predictive-maintenance fleet (C-MAPSS turbofan engines with ids like
FD001-test-unit34 and IMS bearings with ids like IMS-2nd_test-bearing1). Always call tools to obtain numbers;
never estimate or invent numbers. Quote numbers exactly as the tools return them and name the tool used.
Say when values are simulated."""


def verified_facts(calls: list[dict], results: list[dict]) -> list[str]:
    """Key conclusions read directly from tool results, shown beside the LLM's prose so that a wrong
    claim in the prose (e.g. naming the wrong cheapest scenario) is visible to the reader."""
    facts = []
    for c, r in zip(calls, results):
        if "error" in r:
            facts.append(f"{c['tool']} failed: {r['error']}")
        elif c["tool"] == "predict_rul" and r.get("rul_pred") is not None:
            facts.append(f"{r['asset_id']}: predicted RUL {r['rul_pred']:.1f} {r.get('time_unit', '')}s "
                         f"(model {r['model']})")
        elif c["tool"] == "assess_risk":
            facts.append(f"{r['asset_id']}: risk {r['risk_level']}, PoF {r['probability_of_failure']:.1%}, "
                         f"urgency {r['urgency']}")
        elif c["tool"] == "compare_maintenance_scenarios" and r.get("scenarios"):
            best = next(x for x in r["scenarios"] if x["scenario"] == r["best"])
            facts.append(f"{r['asset_id']}: cheapest scenario over {r['horizon']:g} {r['time_unit']}s is "
                         f"'{best['label']}' ({best['total_expected_cost']:,.0f} expected cost)")
    return facts


def ask(question: str, max_steps: int = 4) -> dict:
    llm = get_chat_model()
    status = llm_status()
    if llm is None:
        return {"answer": None, "available": False, "llm": status,
                "message": "No LLM is available, so free-form questions cannot be answered. Configure Ollama or an "
                           "API provider in System Settings; all agent workflows still run with deterministic reports."}
    bound = llm.bind_tools(CHAT_TOOLS)
    tools = {t.name: t for t in CHAT_TOOLS}
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    calls, results = [], []
    t0 = time.time()
    for _ in range(max_steps):
        reply: AIMessage = bound.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            break
        for tc in reply.tool_calls:
            try:
                out = to_jsonable(tools[tc["name"]].invoke(tc["args"]))
            except Exception as exc:
                out = {"error": f"{exc.__class__.__name__}: {exc}"}
            calls.append({"tool": tc["name"], "args": tc["args"]})
            results.append(out)
            messages.append(ToolMessage(json.dumps(out, default=str)[:12000], tool_call_id=tc["id"]))
    answer = messages[-1].content if isinstance(messages[-1], AIMessage) else ""
    return {"answer": answer, "available": True, "llm": status, "tool_calls": calls,
            "verified_facts": verified_facts(calls, results),
            "ungrounded_numbers": ungrounded_numbers(answer, results) if results else
            ungrounded_numbers(answer, {}), "seconds": round(time.time() - t0, 2),
            "note": "Numbers listed under ungrounded_numbers did not appear in any tool result."}
