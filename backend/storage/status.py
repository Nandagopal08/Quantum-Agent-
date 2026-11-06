"""Storage management page: dataset sizes, files, model checkpoints and paths.

In Colab: ``show_storage_page()`` renders it in the notebook and saves a copy to
``reports/storage_status.html``. From a terminal: ``python -m backend.storage status``.
"""

from __future__ import annotations

import html
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..config import PROJECT_DIRS
from .datasets import DATASETS, load_metadata, verify_dataset
from .drive import DriveNotMountedError, drive_guard, require_drive
from .fileio import human_size, read_json, tree_stats, utc_now, write_text
from .paths import MODEL_CATEGORIES, paths

MAX_FILES_PER_FOLDER = 12


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _folder_listing(folder: Path, recursive: bool) -> list[dict]:
    if not folder.is_dir():
        return []
    files = folder.rglob("*") if recursive else folder.iterdir()
    entries = [p for p in files if p.is_file() and not p.name.startswith(".")]
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": str(p.relative_to(folder)), "size_bytes": p.stat().st_size, "modified": _mtime(p)}
            for p in entries[:MAX_FILES_PER_FOLDER]]


def _checkpoint_runs() -> list[dict]:
    runs = []
    for category in MODEL_CATEGORIES:
        base = paths.models(category)
        if not base.is_dir():
            continue
        for index_file in sorted(base.glob("*/checkpoints/index.json")):
            index = read_json(index_file, default={}) or {}
            entries = index.get("checkpoints", [])
            present = [e for e in entries if (index_file.parent / e["file"]).is_file()]
            latest = present[-1] if present else None
            runs.append({
                "category": category,
                "run_name": index_file.parent.parent.name,
                "path": str(index_file.parent),
                "count": len(present),
                "size_bytes": sum(e.get("size_bytes", 0) for e in present),
                "latest_step": latest["step"] if latest else None,
                "latest_saved_at": latest["saved_at"] if latest else None,
                "latest_metrics": latest.get("metrics", {}) if latest else {},
            })
    return runs


def storage_report() -> dict:
    """Collect everything the storage page shows. Raises if Drive is not mounted."""
    config = require_drive()
    with drive_guard("Building storage report"):
        usage = shutil.disk_usage(config.drive_mount if config.is_google_drive else config.project_root)
        folders = []
        for rel in PROJECT_DIRS:
            folder = config.project_root / rel
            count, size = tree_stats(folder) if folder.is_dir() else (0, 0)
            is_dataset_tree = rel.startswith("datasets/processed/")
            folders.append({
                "path": rel, "exists": folder.is_dir(), "file_count": count, "size_bytes": size,
                # Extracted dataset trees hold thousands of files; list only their top level.
                "files": _folder_listing(folder, recursive=not is_dataset_tree),
            })
        metadata = load_metadata()
        datasets = []
        for name in DATASETS:
            record = metadata.get(name, {})
            datasets.append({**verify_dataset(name), **{k: record.get(k) for k in (
                "source_url", "download_date", "file_size_bytes", "sha256", "dataset_version",
                "remote_last_modified", "nested_archives")}})
        return {
            "generated_at": utc_now(),
            "project_root": str(config.project_root),
            "storage_mode": config.storage_mode,
            "is_google_drive": config.is_google_drive,
            "drive_mount": str(config.drive_mount),
            "local_scratch": str(config.local_scratch),
            "drive_total_bytes": usage.total,
            "drive_used_bytes": usage.used,
            "drive_free_bytes": usage.free,
            "project_size_bytes": sum(f["size_bytes"] for f in folders),
            "datasets": datasets,
            "checkpoints": _checkpoint_runs(),
            "folders": folders,
        }


# ---------------------------------------------------------------- text


def format_text(report: dict) -> str:
    lines = [
        f"Project root : {report['project_root']}",
        f"Drive        : {human_size(report['drive_used_bytes'])} used of {human_size(report['drive_total_bytes'])}"
        f" ({human_size(report['drive_free_bytes'])} free); project uses {human_size(report['project_size_bytes'])}",
        "", "Datasets",
    ]
    for d in report["datasets"]:
        state = "ready" if d["ready"] else "NOT READY"
        lines.append(f"  {d['name']:<14} {state:<10} zip {human_size(d['file_size_bytes'] or 0):>9}  "
                     f"extracted {d['file_count']} files / {human_size(d['size_bytes'])}")
        lines += [f"      - {p}" for p in d["problems"]]
    lines += ["", "Checkpoints"]
    if not report["checkpoints"]:
        lines.append("  (none yet)")
    for c in report["checkpoints"]:
        lines.append(f"  {c['category']}/{c['run_name']:<24} {c['count']} saved, latest step {c['latest_step']} "
                     f"at {c['latest_saved_at']}  ({human_size(c['size_bytes'])})")
    lines += ["", "Folders"]
    for f in report["folders"]:
        lines.append(f"  {f['path']:<38} {f['file_count']:>6} files  {human_size(f['size_bytes']):>9}")
    return "\n".join(lines)


# ---------------------------------------------------------------- HTML

_CSS = """
.qadt{--bg:#fff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--card:#f6f8fa;--ok:#1a7f37;--bad:#cf222e;--warn:#9a6700;
 font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg);
 padding:16px;max-width:1100px}
@media (prefers-color-scheme:dark){.qadt{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#3d444d;--card:#151b23;
 --ok:#3fb950;--bad:#f85149;--warn:#d29922}}
html[theme=dark] .qadt,body.dark .qadt{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#3d444d;--card:#151b23;
 --ok:#3fb950;--bad:#f85149;--warn:#d29922}
.qadt h2{font-size:20px;margin:0 0 4px}.qadt h3{font-size:15px;margin:22px 0 8px}
.qadt .muted{color:var(--muted)}.qadt code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}
.qadt .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:12px 0}
.qadt .tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.qadt .tile b{display:block;font-size:18px}.qadt .bar{height:6px;background:var(--line);border-radius:3px;margin-top:6px}
.qadt .bar i{display:block;height:100%;background:var(--fg);border-radius:3px;opacity:.6}
.qadt table{border-collapse:collapse;width:100%;font-size:13px}.qadt .scroll{overflow-x:auto}
.qadt th,.qadt td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
.qadt th{color:var(--muted);font-weight:600}.qadt td.n{text-align:right;white-space:nowrap}
.qadt .ok{color:var(--ok);font-weight:600}.qadt .bad{color:var(--bad);font-weight:600}.qadt .warn{color:var(--warn)}
.qadt details{border-bottom:1px solid var(--line);padding:4px 0}.qadt summary{cursor:pointer;padding:4px 0}
.qadt .err{border:1px solid var(--bad);border-radius:8px;padding:12px 14px;background:var(--card)}
.qadt pre{white-space:pre-wrap;font:12px ui-monospace,Menlo,monospace;margin:8px 0 0}
"""


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def render_html(report: dict) -> str:
    used_pct = 100 * report["drive_used_bytes"] / max(report["drive_total_bytes"], 1)
    ready = sum(d["ready"] for d in report["datasets"])
    out = [f"<style>{_CSS}</style><div class='qadt'>",
           "<h2>Project storage</h2>",
           f"<div class='muted'>Google Drive · <code>{_e(report['project_root'])}</code> · "
           f"generated {_e(report['generated_at'])}</div>",
           "<div class='tiles'>",
           f"<div class='tile'><span class='muted'>Drive used</span><b>{human_size(report['drive_used_bytes'])}</b>"
           f"<span class='muted'>of {human_size(report['drive_total_bytes'])}, "
           f"{human_size(report['drive_free_bytes'])} free</span>"
           f"<div class='bar'><i style='width:{used_pct:.1f}%'></i></div></div>",
           f"<div class='tile'><span class='muted'>This project</span><b>{human_size(report['project_size_bytes'])}</b></div>",
           f"<div class='tile'><span class='muted'>Datasets ready</span><b>{ready} / {len(report['datasets'])}</b></div>",
           f"<div class='tile'><span class='muted'>Checkpointed runs</span><b>{len(report['checkpoints'])}</b></div>",
           "</div>"]

    out.append("<h3>Datasets</h3><div class='scroll'><table><tr><th>Dataset</th><th>Status</th>"
               "<th class='n'>ZIP</th><th class='n'>Extracted</th><th>Downloaded</th><th>Paths</th></tr>")
    for d in report["datasets"]:
        status = "<span class='ok'>Ready</span>" if d["ready"] else \
            "<span class='bad'>Not ready</span><br>" + "<br>".join(f"<span class='warn'>{_e(p)}</span>" for p in d["problems"])
        out.append(
            f"<tr><td><b>{_e(d['title'])}</b><br><span class='muted'>{_e(d['dataset_version'])}</span></td>"
            f"<td>{status}</td><td class='n'>{human_size(d['file_size_bytes'] or 0)}</td>"
            f"<td class='n'>{d['file_count']} files<br>{human_size(d['size_bytes'])}</td>"
            f"<td>{_e(d['download_date'] or '—')}</td>"
            f"<td><code>{_e(d['zip_path'])}</code><br><code>{_e(d['processed_path'])}</code></td></tr>")
    out.append("</table></div>")

    out.append("<h3>Model checkpoints</h3>")
    if report["checkpoints"]:
        out.append("<div class='scroll'><table><tr><th>Run</th><th class='n'>Saved</th><th class='n'>Latest step</th>"
                   "<th>Saved at</th><th>Metrics</th><th class='n'>Size</th></tr>")
        for c in report["checkpoints"]:
            metrics = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in c["latest_metrics"].items())
            out.append(f"<tr><td><b>{_e(c['run_name'])}</b><br><span class='muted'>{_e(c['category'])}</span></td>"
                       f"<td class='n'>{c['count']}</td><td class='n'>{_e(c['latest_step'])}</td>"
                       f"<td>{_e(c['latest_saved_at'])}</td><td>{_e(metrics) or '—'}</td>"
                       f"<td class='n'>{human_size(c['size_bytes'])}</td></tr>")
        out.append("</table></div>")
    else:
        out.append("<div class='muted'>No checkpoints yet. Training runs that use CheckpointManager appear here.</div>")

    out.append("<h3>Folders and files</h3>")
    for f in report["folders"]:
        label = (f"<code>{_e(f['path'])}</code> <span class='muted'>· {f['file_count']} files · "
                 f"{human_size(f['size_bytes'])}</span>")
        if not f["files"]:
            out.append(f"<details><summary>{label}</summary><div class='muted'>Empty</div></details>")
            continue
        rows = "".join(f"<tr><td><code>{_e(x['name'])}</code></td><td class='n'>{human_size(x['size_bytes'])}</td>"
                       f"<td>{_e(x['modified'])}</td></tr>" for x in f["files"])
        more = f["file_count"] - len(f["files"])
        tail = f"<div class='muted'>…and {more} more</div>" if more > 0 else ""
        out.append(f"<details><summary>{label}</summary><div class='scroll'><table>{rows}</table></div>{tail}</details>")

    out.append(f"<h3>Paths</h3><table>"
               f"<tr><td>Project root (Drive)</td><td><code>{_e(report['project_root'])}</code></td></tr>"
               f"<tr><td>Drive mount</td><td><code>{_e(report['drive_mount'])}</code></td></tr>"
               f"<tr><td>Local scratch (temporary, not persisted)</td><td><code>{_e(report['local_scratch'])}</code></td></tr>"
               f"</table></div>")
    return "".join(out)


def render_error_html(message: str) -> str:
    return (f"<style>{_CSS}</style><div class='qadt'><h2>Project storage</h2><div class='err'>"
            f"<span class='bad'>Google Drive is not available</span><pre>{_e(message)}</pre></div></div>")


def show_storage_page(save: bool = True) -> dict | None:
    """Render the storage page in the notebook. Shows mounting instructions if Drive is missing."""
    try:
        from IPython.display import HTML, display
    except ImportError:
        display = None

    try:
        report = storage_report()
    except DriveNotMountedError as exc:
        if display:
            display(HTML(render_error_html(str(exc))))
        else:
            print(exc)
        return None

    page = render_html(report)
    if save:
        write_text(paths.reports / "storage_status.html",
                   f"<!doctype html><meta charset='utf-8'><title>Project storage</title>{page}")
    if display:
        display(HTML(page))
    else:
        print(format_text(report))
    return report


def print_status() -> int:
    try:
        print(format_text(storage_report()))
    except DriveNotMountedError as exc:
        print(exc)
        return 1
    return 0
