from ...langchain_tools.tools import call
from .base import AgentSpec, agent

SPEC = AgentSpec(
    name="Quantum Optimization Agent", key="quantum_optimization",
    responsibilities=["Formulate joint maintenance scheduling as a QUBO under crew capacity",
                      "Solve with QAOA and with MILP, greedy and simulated annealing baselines",
                      "Select the cheapest feasible plan and report the measured QAOA gap"],
    inputs=["counterfactual", "risk"], outputs=["optimization"], tools=["optimize_maintenance_schedule"],
    runs_when="At least two assets are better off with maintenance and optimisation is enabled",
    on_error="Non-fatal: Decision Coordination uses per-asset counterfactual recommendations instead.")


@agent(SPEC)
def quantum_optimization(state):
    cf = state["counterfactual"]
    ids = [aid for aid, r in cf.items() if r.get("best") not in (None, "continue_operation")]
    ids = sorted(ids, key=lambda a: -state["risk"][a]["risk_score"])[:6]
    result = call("optimize_maintenance_schedule", asset_ids=ids)
    q = next((r for r in result.get("results", []) if r["solver"] == "qaoa"), {})
    return {"optimization": result,
            "_summary": f"best solver {result.get('best_solver')}; QAOA gap "
                        f"{q.get('gap_percent', float('nan')):.1f}% on {q.get('qubits', '?')} qubits"}
