"""Run a LangGraph agent workflow from the command line.

  maintenance   7 agents: data monitoring → prediction → risk → digital twin → counterfactuals → QAOA
                optimisation → decision (+ human approval; answer it with --decision approve|reject)
  missing-data  1 agent:  data quality → plan → impute → check (retries the next-best imputer on rejection) →
                forecast (quantum sequence model) → report

    python scripts/run_agent_workflow.py maintenance --n-assets 5 --decision approve
    python scripts/run_agent_workflow.py missing-data --asset FD001-test-unit34 --missing-rate 0.4 --pattern block
"""
import sys

from _cli import csv, parser, run

ap = parser(__doc__)
ap.add_argument("workflow", choices=("maintenance", "missing-data"))
ap.add_argument("--no-llm", action="store_true", help="deterministic narratives only")
g = ap.add_argument_group("maintenance")
g.add_argument("--assets", default=None)
g.add_argument("--subset", default="FD001")
g.add_argument("--n-assets", type=int, default=5)
g.add_argument("--horizon", type=float, default=60)
g.add_argument("--decision", choices=("approve", "reject"), default=None,
               help="answer the human-approval gate (otherwise the run stays awaiting approval)")
g.add_argument("--reviewer", default="cli")
g = ap.add_argument_group("missing-data")
g.add_argument("--asset", default="FD001-test-unit34")
g.add_argument("--missing-rate", type=float, default=0.3)
g.add_argument("--pattern", choices=("mcar", "block"), default="mcar")
g.add_argument("--seed", type=int, default=0)
g.add_argument("--imputer", default=None, help="force the first imputer (the check may reject it)")
g.add_argument("--max-attempts", type=int, default=3)
a = ap.parse_args()


def main():
    from backend import workflows as w
    if a.workflow == "maintenance":
        return w.run_maintenance_agents(csv(a.assets), a.subset, a.n_assets, a.horizon, a.decision, a.reviewer,
                                        use_llm=not a.no_llm)
    return w.run_missing_data_agents(a.asset, a.missing_rate, a.pattern, a.seed, a.imputer, not a.no_llm, a.max_attempts)


sys.exit(run("run_agent_workflow", a, main))
