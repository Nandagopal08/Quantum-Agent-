"""Explainability reports (SHAP, permutation importance) and regeneration of documentation/RESULTS.md and API.md
from the saved run reports.

    python scripts/evaluate_all.py [--subset FD001] [--explain-models xgboost,catboost,lstm] [--skip-explain]
"""
import sys

from _cli import csv, parser, run

ap = parser(__doc__)
ap.add_argument("--subset", default="FD001")
ap.add_argument("--explain-models", default="xgboost,catboost,lstm")
ap.add_argument("--skip-explain", action="store_true")
ap.add_argument("--publish-docs", action="store_true", help="write RESULTS.md/API.md into documentation/ (repository)")
a = ap.parse_args()


def main():
    from backend import workflows as w
    out = {}
    if not a.skip_explain:
        out["explainability"] = w.run_explainability(a.subset, csv(a.explain_models))
    out["evaluation"] = w.evaluate_all(publish=a.publish_docs)
    return out


sys.exit(run("evaluate_all", a, main))
