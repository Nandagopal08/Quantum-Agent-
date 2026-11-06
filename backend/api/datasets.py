"""Dataset management and analysis (EDA, feature extraction)."""

from __future__ import annotations

import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import jobs
from ..datasets import cmapss, ims
from ..services import fleet
from ..storage import (DATASETS, download_dataset, extract_dataset, load_metadata, paths, require_drive,
                       verify_dataset, verify_zip)
from ..storage.datasets import archive_path, dataset_specs, update_metadata
from ..storage.fileio import read_json, sha256_file, utc_now
from .common import saved_result

router = APIRouter(prefix="/api", tags=["datasets"])


@router.get("/datasets")
def list_datasets():
    require_drive()
    meta = load_metadata()
    return [{"spec": s, "status": verify_dataset(s["name"]), "metadata": meta.get(s["name"], {}),
             "parts": {"cmapss": cmapss.available_subsets, "ims_bearings": ims.available_tests}[s["name"]]()}
            for s in dataset_specs()]


@router.post("/datasets/{name}/download")
def download(name: str, force: bool = False):
    if name not in DATASETS:
        raise HTTPException(404, f"unknown dataset {name}")
    require_drive()

    def work(progress):
        progress(0.05, "downloading (resumes automatically if interrupted)")
        download_dataset(name, force=force)
        progress(0.7, "extracting")
        extract_dataset(name, force=force)
        fleet.clear_caches()
        return verify_dataset(name)

    return jobs.submit("dataset_download", {"dataset": name, "force": force}, work, heavy=False)


@router.post("/datasets/{name}/upload")
async def upload(name: str, file: UploadFile = File(...)):
    """Upload the dataset ZIP from your computer instead of downloading it (same verification applies)."""
    if name not in DATASETS:
        raise HTTPException(404, f"unknown dataset {name}")
    require_drive()
    target = archive_path(name)
    part = target.with_name(target.name + ".upload")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(part, "wb") as fh:
        while chunk := await file.read(8 << 20):
            fh.write(chunk)
    ok, reason = verify_zip(part, deep=True)
    if not ok:
        part.unlink(missing_ok=True)
        raise HTTPException(422, f"uploaded file failed ZIP verification: {reason}")
    os.replace(part, target)
    update_metadata(name, download_status="complete", download_date=utc_now(), file_size_bytes=target.stat().st_size,
                    sha256=sha256_file(target), zip_verified=True, extraction_status="not_extracted",
                    upload_filename=file.filename, error=None)

    def work(progress):
        progress(0.1, "extracting uploaded archive")
        extract_dataset(name, force=True)
        fleet.clear_caches()
        return verify_dataset(name)

    return jobs.submit("dataset_extract", {"dataset": name, "uploaded": file.filename}, work, heavy=False)


@router.get("/analysis/cmapss/{subset}")
def cmapss_eda(subset: str, refresh: bool = False):
    require_drive()
    if not refresh:
        try:
            return saved_result("dataset_reports", f"cmapss_{subset}_eda")
        except FileNotFoundError:
            pass
    return cmapss.eda(subset)


@router.get("/analysis/cmapss/{subset}/pipeline")
def cmapss_pipeline(subset: str):
    """Preprocessing decisions (dropped sensors, features, unit split) — recomputed if not saved."""
    require_drive()
    info = read_json(paths.splits("cmapss") / subset / "pipeline.json")
    if info is None:
        cmapss.prepare(cmapss.CmapssConfig(subset=subset))
        info = read_json(paths.splits("cmapss") / subset / "pipeline.json")
    return {k: v for k, v in info.items() if k != "scaler"}


@router.get("/analysis/ims/{test}")
def ims_eda(test: str):
    require_drive()
    try:
        return saved_result("dataset_reports", f"ims_{test}_eda")
    except FileNotFoundError:
        pass
    cache = [p for p in paths.features("ims_bearings").glob(f"{test}_features.*")] if paths.features("ims_bearings").is_dir() else []
    if not cache:
        raise HTTPException(409, f"features for {test} not extracted yet; POST /api/analysis/ims/{test}/features")
    return ims.eda(test)


@router.post("/analysis/ims/{test}/features")
def ims_features(test: str, force: bool = False):
    if test not in ims.TESTS:
        raise HTTPException(404, f"unknown IMS test {test}")
    require_drive()

    def work(progress):
        df = ims.extract_features(test, force=force, progress=lambda f: progress(0.05 + 0.85 * f, "extracting"))
        progress(0.95, "computing EDA")
        rep = ims.eda(test)
        fleet.clear_caches()
        return {"rows": len(df), "snapshots": rep["snapshots"], "duration_hours": rep["duration_hours"]}

    return jobs.submit("ims_features", {"test": test, "force": force}, work)


@router.delete("/datasets/{name}/partial")
def delete_partial(name: str):
    """Remove a stalled partial download so the next download starts from scratch."""
    require_drive()
    part = archive_path(name).with_name(archive_path(name).name + ".part")
    removed = part.exists()
    part.unlink(missing_ok=True)
    part.with_name(part.name + ".json").unlink(missing_ok=True)
    return {"removed": removed}
