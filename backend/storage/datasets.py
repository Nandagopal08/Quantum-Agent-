"""Dataset download, integrity verification and extraction, all on Google Drive.

Lifecycle per dataset::

    datasets/raw/<name>/<name>.zip.part   resumable download in progress (kept in Drive)
    datasets/raw/<name>/<name>.zip        complete, CRC-verified archive
    datasets/processed/<name>/...         extracted contents (nested archives unpacked too)
    datasets/dataset_reports/dataset_metadata.json   provenance + status for every dataset

A dataset is never downloaded again while a verified archive exists, and
extraction is skipped once its completion marker is present.
"""

from __future__ import annotations

import http.client
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .drive import _DISCONNECT_ERRNOS, DriveDisconnectedError, drive_guard, require_drive
from .fileio import read_json, sha256_file, tree_stats, utc_now, write_json
from .paths import paths

USER_AGENT = "Quantum-Agentic-Digital-Twin/0.1 (+research; Google Colab)"
EXTRACTION_MARKER = ".extraction_complete.json"
NESTED_ARCHIVE_SUFFIXES = (".zip", ".7z", ".rar")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    url: str
    archive_name: str
    version: str
    citation: str
    sha256: str | None = None          # SHA-256 of the official archive (uploads must match unless allowed)


DATASETS: dict[str, DatasetSpec] = {
    "cmapss": DatasetSpec(
        name="cmapss",
        title="NASA C-MAPSS Turbofan Engine Degradation Simulation",
        url="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
        archive_name="cmapss.zip",
        version="NASA PCoE Data Set 6 (2008)",
        citation="A. Saxena and K. Goebel (2008). Turbofan Engine Degradation Simulation Data Set, "
        "NASA Prognostics Data Repository, NASA Ames Research Center.",
        sha256="c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2",
    ),
    "ims_bearings": DatasetSpec(
        name="ims_bearings",
        title="NASA IMS Bearings (run-to-failure)",
        url="https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip",
        archive_name="ims_bearings.zip",
        version="NASA PCoE Data Set 4 (2007)",
        citation="J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, "
        "University of Cincinnati. Bearing Data Set, NASA Prognostics Data Repository.",
        sha256="21001ac266c465f5d345ec42d7b508c6a6328487fd9d4d7774422dd5ea10ad83",
    ),
}


class DatasetError(RuntimeError):
    pass


class DatasetDownloadError(DatasetError):
    pass


class DatasetIntegrityError(DatasetError):
    pass


class DatasetNotReadyError(DatasetError):
    pass


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}; available: {sorted(DATASETS)}") from None


def archive_path(name: str) -> Path:
    return paths.raw(name) / get_spec(name).archive_name


def _part_path(zip_path: Path) -> Path:
    return zip_path.with_name(zip_path.name + ".part")


def _log(msg: str) -> None:
    print(f"[qadt.datasets] {msg}", flush=True)


# ---------------------------------------------------------------- metadata


def load_metadata() -> dict[str, dict]:
    require_drive()
    return read_json(paths.dataset_metadata_file, default={}) or {}


def update_metadata(name: str, **fields) -> dict:
    """Merge ``fields`` into the record for ``name`` and save the metadata file."""
    metadata = load_metadata()
    spec = get_spec(name)
    record = metadata.get(name) or {
        "name": spec.name,
        "title": spec.title,
        "source_url": spec.url,
        "dataset_version": spec.version,
        "citation": spec.citation,
        "zip_path": str(archive_path(name)),
        "processed_path": str(paths.processed(name)),
        "extraction_status": "not_extracted",
    }
    record.update(fields)
    record["updated_at"] = utc_now()
    metadata[name] = record
    write_json(paths.dataset_metadata_file, metadata)
    return record


# ---------------------------------------------------------------- integrity


def verify_zip(path: Path, deep: bool = True) -> tuple[bool, str]:
    """Check a ZIP archive. ``deep`` reads every member and checks its CRC."""
    with drive_guard(f"Verifying {path}"):
        if not path.is_file():
            return False, "file does not exist"
        if not zipfile.is_zipfile(path):
            return False, "not a valid ZIP archive (missing or damaged central directory)"
        try:
            with zipfile.ZipFile(path) as zf:
                if not zf.infolist():
                    return False, "archive is empty"
                if deep:
                    bad = zf.testzip()
                    if bad is not None:
                        return False, f"CRC check failed for member {bad!r}"
        except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, ValueError) as exc:
            return False, f"archive is corrupt: {exc}"
    return True, "ok"


def _existing_archive_is_valid(name: str, zip_path: Path) -> bool:
    if not zip_path.is_file():
        return False
    record = load_metadata().get(name, {})
    size = zip_path.stat().st_size
    if record.get("zip_verified") and record.get("file_size_bytes") == size:
        ok, _ = verify_zip(zip_path, deep=False)
        if ok:
            return True
    # Unknown or mismatched copy (e.g. copied in by hand): check it fully before trusting it.
    ok, reason = verify_zip(zip_path, deep=True)
    if ok:
        _log(f"{name}: adopting existing archive {zip_path} after full integrity check")
        update_metadata(
            name,
            file_size_bytes=size,
            sha256=sha256_file(zip_path),
            zip_verified=True,
            download_date=record.get("download_date") or utc_now(),
        )
        return True
    quarantine = zip_path.with_name(zip_path.name + f".corrupt-{int(time.time())}")
    _log(f"{name}: existing archive failed verification ({reason}); moved to {quarantine.name}")
    os.replace(zip_path, quarantine)
    update_metadata(name, zip_verified=False, extraction_status="not_extracted")
    return False


# ---------------------------------------------------------------- download


def _remote_info(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            return {
                "size": int(length) if length else None,
                "etag": (resp.headers.get("ETag") or "").strip('"') or None,
                "last_modified": resp.headers.get("Last-Modified"),
                "accepts_ranges": resp.headers.get("Accept-Ranges", "").lower() == "bytes",
            }
    except (urllib.error.URLError, OSError) as exc:
        _log(f"HEAD request failed ({exc}); continuing without remote size/ETag")
        return {"size": None, "etag": None, "last_modified": None, "accepts_ranges": True}


class _Progress:
    def __init__(self, total: int | None, initial: int, label: str):
        self.total, self.done, self.label = total, initial, label
        self._bar = None
        self._next_pct = 0
        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(total=total, initial=initial, unit="B", unit_scale=True, desc=label)
        except ImportError:
            pass

    def update(self, n: int) -> None:
        self.done += n
        if self._bar is not None:
            self._bar.update(n)
        elif self.total:
            pct = int(self.done * 100 / self.total)
            if pct >= self._next_pct:
                _log(f"{self.label}: {pct}% ({self.done}/{self.total} bytes)")
                self._next_pct = pct + 5

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _download_resumable(url: str, part: Path, remote: dict, label: str, retries: int,
                        chunk_size: int, timeout: float) -> int:
    """Download ``url`` into ``part``, resuming from whatever is already there. Returns final size."""
    sidecar = part.with_name(part.name + ".json")
    total = remote["size"]
    attempt = 0
    while True:
        with drive_guard(f"Preparing {part}"):
            have = part.stat().st_size if part.exists() else 0
            previous = read_json(sidecar, default={}) or {}
        if have and remote["etag"] and previous.get("etag") not in (None, remote["etag"]):
            _log(f"{label}: remote file changed since the partial download began; restarting")
            have = 0
        if total is not None and have > total:
            have = 0
        if total is not None and have == total:
            return have

        headers = {"User-Agent": USER_AGENT}
        if have and remote["accepts_ranges"]:
            headers["Range"] = f"bytes={have}-"
            _log(f"{label}: resuming at byte {have}" + (f" of {total}" if total else ""))
        progress = None
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
                if resp.status != 206:
                    have = 0  # server ignored the range request; start over
                write_json(sidecar, {"url": url, "etag": remote["etag"], "size": total, "started_at": utc_now()})
                progress = _Progress(total, have, label)
                with drive_guard(f"Writing {part}"), open(part, "ab" if have else "wb") as fh:
                    while chunk := resp.read(chunk_size):
                        fh.write(chunk)
                        have += len(chunk)
                        progress.update(len(chunk))
                    fh.flush()
                    os.fsync(fh.fileno())
            if total is None or have >= total:
                return have
            raise DatasetDownloadError(f"connection closed early at {have}/{total} bytes")
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and total is not None and have == total:
                return have
            if exc.code == 416:
                part.unlink(missing_ok=True)
            elif 400 <= exc.code < 500:
                raise DatasetDownloadError(f"{label}: server returned HTTP {exc.code} for {url}") from exc
            error: Exception = exc
        except DriveDisconnectedError:
            raise
        except (urllib.error.URLError, http.client.HTTPException, DatasetDownloadError, TimeoutError, OSError) as exc:
            if isinstance(exc, OSError) and not isinstance(exc, urllib.error.URLError) \
                    and exc.errno in _DISCONNECT_ERRNOS:
                raise DriveDisconnectedError(f"{label}: {exc}") from exc
            error = exc
        finally:
            if progress is not None:
                progress.close()

        attempt += 1
        if attempt > retries:
            raise DatasetDownloadError(
                f"{label}: download failed after {retries} retries ({error}). The partial file is kept at "
                f"{part}; run the download again to resume from byte {have}."
            )
        delay = min(2 ** attempt, 60)
        _log(f"{label}: {error}; retry {attempt}/{retries} in {delay}s")
        time.sleep(delay)


def download_dataset(name: str, force: bool = False, retries: int = 5,
                     chunk_size: int = 8 << 20, timeout: float = 60) -> Path:
    """Download a dataset ZIP into Drive (skipped if a verified copy already exists)."""
    spec = get_spec(name)
    require_drive()
    zip_path = archive_path(name)
    with drive_guard("Creating dataset folders"):
        zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and _existing_archive_is_valid(name, zip_path):
        _log(f"{name}: verified archive already in Drive ({zip_path}); not downloading again")
        return zip_path

    part = _part_path(zip_path)
    remote = _remote_info(spec.url, timeout)
    update_metadata(name, download_status="in_progress", remote_etag=remote["etag"],
                    remote_last_modified=remote["last_modified"], remote_size_bytes=remote["size"])
    _log(f"{name}: downloading {spec.url} -> {zip_path}")
    size = _download_resumable(spec.url, part, remote, name, retries, chunk_size, timeout)

    ok, reason = verify_zip(part, deep=True)
    if not ok:
        part.unlink(missing_ok=True)
        update_metadata(name, download_status="failed", zip_verified=False, error=f"integrity check: {reason}")
        raise DatasetIntegrityError(f"{name}: downloaded archive failed verification ({reason}). "
                                    "The corrupt file was removed; run the download again.")
    with drive_guard(f"Finalising {zip_path}"):
        os.replace(part, zip_path)
        part.with_name(part.name + ".json").unlink(missing_ok=True)

    update_metadata(
        name,
        download_status="complete",
        download_date=utc_now(),
        file_size_bytes=size,
        sha256=sha256_file(zip_path),
        zip_verified=True,
        extraction_status="not_extracted",
        error=None,
    )
    _log(f"{name}: download complete and verified ({size} bytes)")
    return zip_path


# ---------------------------------------------------------------- extraction


def _safe_target(dest: Path, member: str) -> Path:
    dest_resolved = dest.resolve()
    target = (dest / member).resolve()
    if target != dest_resolved and dest_resolved not in target.parents:
        raise DatasetIntegrityError(f"archive member {member!r} would extract outside {dest}")
    return target


MAX_EXTRACT_BYTES = int(os.environ.get("QADT_MAX_EXTRACT_BYTES", 20 << 30))   # zip-bomb guard (default 20 GiB)


def _extract_zip(zip_path: Path, dest: Path) -> int:
    """Extract with path-traversal protection; members already present at full size are skipped,
    so an interrupted extraction picks up where it stopped."""
    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        total = sum(i.file_size for i in zf.infolist())
        if total > MAX_EXTRACT_BYTES:
            raise DatasetIntegrityError(f"{zip_path.name} would extract {total / 1e9:.1f} GB, above the "
                                        f"{MAX_EXTRACT_BYTES / 1e9:.1f} GB limit (QADT_MAX_EXTRACT_BYTES)")
        for info in zf.infolist():
            target = _safe_target(dest, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            if target.with_name(f".{target.name}.extracted").exists():
                continue  # nested archive already unpacked (and possibly removed); don't write it again
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.part")
            with zf.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            os.replace(tmp, target)
            written += 1
    return written


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DatasetError(f"{' '.join(cmd[:2])} failed: {(result.stderr or result.stdout).strip()[-500:]}")


def _extract_7z(path: Path, dest: Path) -> None:
    try:
        import py7zr
    except ImportError:
        py7zr = None
    if py7zr is not None:
        with py7zr.SevenZipFile(path, mode="r") as archive:
            for member in archive.getnames():
                _safe_target(dest, member)
            archive.extractall(path=dest)
        return
    tool = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
    if tool:
        return _run([tool, "x", "-y", f"-o{dest}", str(path)])
    raise DatasetError("no 7z extractor available. In Colab run: !pip install -q py7zr")


def _extract_rar(path: Path, dest: Path) -> None:
    if tool := shutil.which("unrar"):
        return _run([tool, "x", "-o+", "-idq", str(path), f"{dest}{os.sep}"])
    if tool := (shutil.which("7z") or shutil.which("7zz")):
        return _run([tool, "x", "-y", f"-o{dest}", str(path)])
    if tool := shutil.which("bsdtar"):
        return _run([tool, "-xf", str(path), "-C", str(dest)])
    raise DatasetError("no RAR extractor available. In Colab run: !apt-get -qq install -y unrar")


def _extract_nested(root: Path, max_depth: int = 3, remove_archives: bool = True) -> list[dict]:
    """Unpack archives found inside the extracted tree into sibling folders named after them.

    With ``remove_archives`` each nested archive is deleted once it has unpacked, so Drive does not
    hold a second copy of data that the verified raw ZIP can always reproduce.
    """
    results: list[dict] = []
    for _depth in range(max_depth):
        found = False
        for archive in sorted(p for p in root.rglob("*") if p.suffix.lower() in NESTED_ARCHIVE_SUFFIXES
                              and p.is_file() and not p.name.startswith(".")):
            done_marker = archive.with_name(f".{archive.name}.extracted")
            if done_marker.exists():
                if remove_archives:
                    archive.unlink(missing_ok=True)
                continue
            found = True
            out = archive.with_suffix("")
            out.mkdir(parents=True, exist_ok=True)
            entry = {"archive": str(archive.relative_to(root)), "output": str(out.relative_to(root))}
            try:
                suffix = archive.suffix.lower()
                if suffix == ".zip":
                    ok, reason = verify_zip(archive, deep=True)
                    if not ok:
                        raise DatasetIntegrityError(reason)
                    _extract_zip(archive, out)
                elif suffix == ".7z":
                    _extract_7z(archive, out)
                else:
                    _extract_rar(archive, out)
                done_marker.write_text(utc_now())
                entry["status"] = "extracted"
                if remove_archives:
                    archive.unlink()
                    entry["archive_removed"] = True
                _log(f"unpacked nested archive {entry['archive']}")
            except DriveDisconnectedError:
                raise
            except Exception as exc:  # recorded in metadata; the rest of the dataset stays usable
                entry.update(status="failed", error=str(exc))
                _log(f"could not unpack nested archive {entry['archive']}: {exc}")
            results.append(entry)
        if not found:
            break
    return results


def extract_dataset(name: str, force: bool = False, keep_nested_archives: bool = False) -> Path:
    """Extract a verified dataset archive into ``datasets/processed/<name>/``."""
    require_drive()
    zip_path = archive_path(name)
    out = paths.processed(name)
    marker = out / EXTRACTION_MARKER
    record = load_metadata().get(name, {})

    if not force and marker.exists() and record.get("extraction_status") == "extracted":
        _log(f"{name}: already extracted to {out}")
        return out
    if not record.get("zip_verified") or not zip_path.is_file():
        raise DatasetNotReadyError(f"{name}: no verified archive at {zip_path}. Run download_dataset({name!r}) first.")

    update_metadata(name, extraction_status="in_progress", extraction_started_at=utc_now())
    _log(f"{name}: extracting {zip_path.name} -> {out}")
    try:
        with drive_guard(f"Extracting {name}"):
            out.mkdir(parents=True, exist_ok=True)
            marker.unlink(missing_ok=True)
            _extract_zip(zip_path, out)
            nested = _extract_nested(out, remove_archives=not keep_nested_archives)
            file_count, total_bytes = tree_stats(out)
    except Exception as exc:
        update_metadata(name, extraction_status="failed", error=f"extraction: {exc}")
        raise

    status = "extracted" if all(n["status"] == "extracted" for n in nested) else "partial"
    summary = {"extracted_at": utc_now(), "file_count": file_count, "size_bytes": total_bytes,
               "nested_archives": nested}
    write_json(marker, summary)
    update_metadata(
        name,
        extraction_status=status,
        extracted_at=summary["extracted_at"],
        extracted_file_count=file_count,
        extracted_size_bytes=total_bytes,
        nested_archives=nested,
        error=None if status == "extracted" else "some nested archives could not be unpacked; see nested_archives",
    )
    _log(f"{name}: extraction {status} ({file_count} files)")
    return out


# ---------------------------------------------------------------- readiness


def verify_dataset(name: str) -> dict:
    """Report whether ``name`` is downloaded, verified and extracted in Drive."""
    require_drive()
    spec = get_spec(name)
    record = load_metadata().get(name, {})
    zip_path = archive_path(name)
    out = paths.processed(name)
    problems = []

    with drive_guard(f"Checking {name}"):
        zip_present = zip_path.is_file()
        if not zip_present:
            problems.append(f"archive missing: {zip_path}")
        elif record.get("file_size_bytes") not in (None, zip_path.stat().st_size):
            problems.append("archive size differs from the recorded download")
        if not record.get("zip_verified"):
            problems.append("archive has not passed integrity verification")
        marker = read_json(out / EXTRACTION_MARKER)
        if marker is None:
            problems.append(f"not extracted into {out}")
        file_count, size = tree_stats(out) if out.exists() else (0, 0)
        if marker and file_count < marker.get("file_count", 0):
            problems.append(f"{marker['file_count'] - file_count} extracted files are missing; re-run extraction")
    if record.get("extraction_status") == "partial":
        problems.append("some nested archives failed to unpack (see dataset_metadata.json)")

    return {
        "name": name,
        "title": spec.title,
        "ready": not problems,
        "zip_path": str(zip_path),
        "processed_path": str(out),
        "file_count": file_count,
        "size_bytes": size,
        "extraction_status": record.get("extraction_status", "not_extracted"),
        "problems": problems,
    }


def require_dataset(name: str) -> Path:
    """Call before training: returns the processed folder or raises with what to run."""
    report = verify_dataset(name)
    if not report["ready"]:
        raise DatasetNotReadyError(
            f"Dataset {name!r} is not ready in Google Drive:\n  - " + "\n  - ".join(report["problems"])
            + f"\n\nRun:  from backend.storage import prepare_datasets; prepare_datasets([{name!r}])"
        )
    return Path(report["processed_path"])


def prepare_datasets(names: Iterable[str] | None = None, force: bool = False) -> dict[str, dict]:
    """Download (if needed), verify and extract each dataset; returns ``verify_dataset`` reports."""
    results = {}
    for name in names or DATASETS:
        download_dataset(name, force=force)
        extract_dataset(name, force=force)
        results[name] = verify_dataset(name)
    return results


def find_files(name: str, pattern: str) -> list[Path]:
    """Glob inside a ready dataset, e.g. ``find_files('cmapss', 'train_FD00*.txt')``."""
    root = require_dataset(name)
    with drive_guard(f"Searching {name}"):
        return sorted(p for p in root.rglob(pattern) if p.is_file())


def dataset_specs() -> list[dict]:
    return [asdict(spec) for spec in DATASETS.values()]


if __name__ == "__main__":  # pragma: no cover
    for n in sys.argv[1:] or DATASETS:
        print(verify_dataset(n))
