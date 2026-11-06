"""Preprocess C-MAPSS (labels, condition normalisation, features, leakage-free splits, EDA) and extract IMS
bearing features (cached in storage).

    python scripts/preprocess_datasets.py [--subsets FD001,FD002] [--ims-tests 1st_test,2nd_test,3rd_test | --skip-ims]
"""
import sys

from _cli import csv, parser, run

ap = parser(__doc__)
ap.add_argument("--subsets", default="FD001")
ap.add_argument("--ims-tests", default=None, help="default: every extracted IMS test")
ap.add_argument("--skip-ims", action="store_true")
a = ap.parse_args()


def main():
    from backend.workflows import preprocess_datasets
    return preprocess_datasets(csv(a.subsets), [] if a.skip_ims else csv(a.ims_tests))


sys.exit(run("preprocess_datasets", a, main))
