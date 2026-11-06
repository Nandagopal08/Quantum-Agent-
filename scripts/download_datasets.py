"""Download, verify (CRC/SHA-256) and extract the NASA datasets into project storage. Resumable.

    python scripts/download_datasets.py [--datasets cmapss,ims_bearings] [--force]
"""
import sys

from _cli import csv, parser, run

ap = parser(__doc__)
ap.add_argument("--datasets", default=None, help="comma-separated: cmapss, ims_bearings (default: both)")
ap.add_argument("--force", action="store_true", help="re-download even if a verified copy exists")
a = ap.parse_args()


def main():
    from backend.workflows import download_datasets
    return download_datasets(csv(a.datasets), force=a.force)


sys.exit(run("download_datasets", a, main))
