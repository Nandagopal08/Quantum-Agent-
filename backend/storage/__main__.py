"""Command line: ``python -m backend.storage {init,download,verify,status}``.

Mounting needs the notebook kernel, so mount Drive in a cell first
(``setup_storage()``); these commands then work from ``!python -m ...`` cells.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..config import set_project_root
from .datasets import DATASETS, prepare_datasets, verify_dataset
from .drive import DriveNotMountedError, ensure_project_structure
from .status import print_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.storage")
    parser.add_argument("--root", help="project root inside the mounted Drive (overrides QADT_DRIVE_ROOT)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create the project folder tree in Drive")
    dl = sub.add_parser("download", help="download, verify and extract datasets")
    dl.add_argument("datasets", nargs="*", choices=list(DATASETS), help="default: all")
    dl.add_argument("--force", action="store_true", help="download and extract again even if present")
    vf = sub.add_parser("verify", help="check datasets are ready for training")
    vf.add_argument("datasets", nargs="*", help="default: all")
    sub.add_parser("status", help="print the storage report")
    args = parser.parse_args(argv)

    if args.root:
        set_project_root(args.root)
    try:
        if args.command == "init":
            print(f"Project folders ready in {ensure_project_structure()}")
        elif args.command == "download":
            ensure_project_structure()
            reports = prepare_datasets(args.datasets or None, force=args.force)
            print(json.dumps(reports, indent=2))
            return 0 if all(r["ready"] for r in reports.values()) else 2
        elif args.command == "verify":
            reports = [verify_dataset(n) for n in args.datasets or DATASETS]
            print(json.dumps(reports, indent=2))
            return 0 if all(r["ready"] for r in reports) else 2
        else:
            return print_status()
    except DriveNotMountedError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
