"""LLM access through LangChain (Ollama or an OpenAI-compatible API) with a deterministic fallback.

The LLM only writes narrative. Every number in its output is checked against the numbers produced
by tools; a narrative containing an unsupported number is rejected and the deterministic narrative
is used instead (the rejection is recorded in the run trace).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from typing import Any

from pydantic import BaseModel, Field

from ..settings import get_settings

log = logging.getLogger("qadt.llm")
_health_cache: dict[str, tuple[float, bool, str]] = {}


class DecisionNarrative(BaseModel):
    """Structured output the Decision Coordination agent asks the LLM for."""
    summary: str = Field(description="2-4 sentence plain-language summary of the recommendation")
    rationale: list[str] = Field(description="3-6 short bullet reasons, each grounded in the provided numbers")
    caveats: list[str] = Field(description="1-3 limitations or assumptions the reader must know")


def llm_status(settings: dict | None = None) -> dict:
    cfg = (settings or get_settings())["llm"]
    provider = cfg["provider"]
    if provider == "none":
        return {"provider": "none", "available": False, "model": None, "detail": "LLM disabled in settings"}
    if provider == "openai":
        ok = bool(os.environ.get("OPENAI_API_KEY"))
        return {"provider": "openai", "available": ok, "model": cfg["model"],
                "detail": "OPENAI_API_KEY set" if ok else "OPENAI_API_KEY not set"}
    key = f"{cfg['base_url']}|{cfg['model']}"
    cached = _health_cache.get(key)
    if cached and time.time() - cached[0] < 30:
        return {"provider": "ollama", "available": cached[1], "model": cfg["model"], "detail": cached[2]}
    try:
        with urllib.request.urlopen(f"{cfg['base_url'].rstrip('/')}/api/tags", timeout=2) as r:
            tags = [m["name"] for m in json.load(r).get("models", [])]
        want = cfg["model"]
        ok = want in tags or f"{want}:latest" in tags or want.removesuffix(":latest") in tags
        detail = "model available" if ok else f"model {want!r} not pulled (have: {', '.join(tags[:6])})"
    except Exception as exc:
        ok, detail = False, f"Ollama not reachable at {cfg['base_url']} ({exc.__class__.__name__})"
    _health_cache[key] = (time.time(), ok, detail)
    return {"provider": "ollama", "available": ok, "model": cfg["model"], "detail": detail}


def get_chat_model(settings: dict | None = None):
    s = settings or get_settings()
    status = llm_status(s)
    if not status["available"]:
        return None
    cfg = s["llm"]
    if cfg["provider"] == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"], timeout=cfg["timeout_seconds"],
                          base_url=os.environ.get("OPENAI_BASE_URL") or None)
    from langchain_ollama import ChatOllama

    return ChatOllama(model=cfg["model"], base_url=cfg["base_url"], temperature=cfg["temperature"],
                      client_kwargs={"timeout": cfg["timeout_seconds"]})


# ---------------------------------------------------------------- number grounding


_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")


def _numbers_in(obj: Any) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(_parse(m) for m in _NUM.findall(obj))
        out.extend(float(d) for d in re.findall(r"\d+", obj))   # digits inside ids, e.g. "unit81" → 81
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_numbers_in(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_numbers_in(v))
    return out


def _parse(token: str) -> float:
    t = token.replace(",", "")
    return float(t[:-1]) / 100 if t.endswith("%") else float(t)


def ungrounded_numbers(text: str, context: Any, rel_tol: float = 0.02) -> list[str]:
    """Numbers in ``text`` that do not match (within tolerance or rounding) any number in ``context``."""
    known = _numbers_in(context)
    bad = []
    for token in _NUM.findall(text):
        v = _parse(token)
        if abs(v) <= 10 and float(v).is_integer():   # counts and small ordinals ("3 assets", "step 2")
            continue
        if not any(abs(v - k) <= max(rel_tol * abs(k), 0.51 if abs(k) >= 1 else 0.006) for k in known):
            bad.append(token)
    return bad


SYSTEM_PROMPT = """You are the Decision Coordination agent of a predictive-maintenance digital twin.
You receive JSON produced by real models and optimisers: RUL predictions, failure probabilities, risk
scores, counterfactual scenario costs and an optimised maintenance schedule.

Rules:
- Use ONLY numbers that appear in the JSON. Never estimate, invent or recompute numbers.
- Do not change the recommended actions; explain them.
- Say clearly which values are simulated.
- Be concise and specific to the assets named."""


def narrate(context: dict, settings: dict | None = None) -> tuple[DecisionNarrative | None, dict]:
    """Ask the LLM for a structured narrative. Returns (narrative or None, metadata for the trace)."""
    from langchain_core.prompts import ChatPromptTemplate

    meta: dict[str, Any] = {"status": llm_status(settings)}
    llm = get_chat_model(settings)
    if llm is None:
        meta["used"] = False
        meta["reason"] = meta["status"]["detail"]
        return None, meta
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Maintenance analysis results:\n```json\n{context}\n```\nWrite the decision narrative."),
    ])
    t0 = time.time()
    try:
        chain = prompt | llm.with_structured_output(DecisionNarrative)
        out: DecisionNarrative = chain.invoke({"context": json.dumps(context, default=str, indent=1)})
    except Exception as exc:
        meta.update(used=False, reason=f"LLM call failed: {exc.__class__.__name__}: {exc}"[:300])
        return None, meta
    meta["latency_seconds"] = round(time.time() - t0, 2)
    text = " ".join([out.summary, *out.rationale, *out.caveats])
    bad = ungrounded_numbers(text, context)
    if bad:
        meta.update(used=False, reason="rejected: narrative contained numbers not present in tool outputs",
                    ungrounded_numbers=bad[:10], rejected_text=text[:1500])
        return None, meta
    meta["used"] = True
    return out, meta
