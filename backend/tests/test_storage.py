"""Storage layer tests. A temp directory stands in for the Drive mount and a local
HTTP server (with Range support) stands in for the NASA S3 bucket.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import errno
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import zipfile
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import config  # noqa: E402
from backend.storage import datasets, drive  # noqa: E402
from backend.storage import (  # noqa: E402
    CheckpointManager, DriveDisconnectedError, DriveNotMountedError, DatasetIntegrityError,
    DatasetNotReadyError, ExperimentLogger, download_dataset, extract_dataset, load_metadata,
    load_model, prepare_datasets, require_dataset, save_model, setup_storage, storage_report,
    verify_dataset,
)
from backend.storage.status import render_html  # noqa: E402


def make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


class RangeServer:
    """Serves ``self.files[path]`` with HEAD, Range and an optional mid-body drop."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.drop_after: int | None = None
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _send(self, head: bool):
                data = server.files.get(self.path)
                server.requests.append({"method": self.command, "range": self.headers.get("Range")})
                if data is None:
                    self.send_error(404)
                    return
                start, status = 0, 200
                if rng := self.headers.get("Range"):
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.end_headers()
                        return
                    status = 206
                body = data[start:]
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", '"v1"')
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.end_headers()
                if head:
                    return
                if server.drop_after is not None:
                    self.wfile.write(body[:server.drop_after])
                    server.drop_after = None
                    self.close_connection = True
                    return
                self.wfile.write(body)

            def do_HEAD(self):
                self._send(head=True)

            def do_GET(self):
                self._send(head=False)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class StorageTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = RangeServer()

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mount = self.tmp / "drive"
        (self.mount / "MyDrive").mkdir(parents=True)
        config.configure(project_root=self.mount / "MyDrive" / "QADT", drive_mount=self.mount,
                         local_scratch=self.tmp / "scratch")
        self.mount_patch = mock.patch.object(drive, "_is_mount_point", lambda p: Path(p) == self.mount)
        self.mount_patch.start()
        self.sleep_patch = mock.patch.object(datasets.time, "sleep", lambda _seconds: None)
        self.sleep_patch.start()
        self.server.drop_after = None
        self.server.requests.clear()
        self.orig_specs = dict(datasets.DATASETS)

    def tearDown(self):
        self.mount_patch.stop()
        self.sleep_patch.stop()
        datasets.DATASETS.clear()
        datasets.DATASETS.update(self.orig_specs)
        config.reset_config()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def serve(self, name: str, payload: bytes) -> None:
        self.server.files[f"/{name}.zip"] = payload
        datasets.DATASETS[name] = replace(datasets.DATASETS[name], url=f"{self.server.url}/{name}.zip")

    @property
    def gets(self):
        return [r for r in self.server.requests if r["method"] == "GET"]


class DriveTests(StorageTestCase):
    def test_setup_creates_full_structure(self):
        paths = setup_storage()
        for rel in config.PROJECT_DIRS:
            self.assertTrue((paths.root / rel).is_dir(), rel)

    def test_not_mounted_raises_with_instructions(self):
        self.mount_patch.stop()
        try:
            with self.assertRaises(DriveNotMountedError) as ctx:
                download_dataset("cmapss")
            self.assertIn("drive.mount('/content/drive')", str(ctx.exception))
            self.assertFalse((self.mount / "MyDrive" / "QADT").exists(), "must not create local fallback data")
        finally:
            self.mount_patch.start()

    def test_root_outside_drive_is_rejected(self):
        config.configure(project_root=self.tmp / "local_only")
        with self.assertRaises(DriveNotMountedError) as ctx:
            setup_storage()
        self.assertIn("not inside the Google Drive mount", str(ctx.exception))

    def test_disconnect_errno_becomes_reconnect_error(self):
        with self.assertRaises(DriveDisconnectedError) as ctx:
            with drive.drive_guard("test"):
                raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
        self.assertIn("force_remount=True", str(ctx.exception))


class DatasetTests(StorageTestCase):
    def setUp(self):
        super().setUp()
        setup_storage()
        inner = make_zip({"CMAPSSData/train_FD001.txt": os.urandom(20000), "CMAPSSData/readme.txt": b"hi"})
        self.payload = make_zip({"Turbofan/CMAPSSData.zip": inner, "Turbofan/Damage.pdf": b"%PDF" * 2000})
        self.serve("cmapss", self.payload)

    def test_download_extract_verify_and_metadata(self):
        result = prepare_datasets(["cmapss"])["cmapss"]
        self.assertTrue(result["ready"], result["problems"])
        files = list(Path(result["processed_path"]).rglob("train_FD001.txt"))
        self.assertEqual(len(files), 1, "nested ZIP should be unpacked")
        record = load_metadata()["cmapss"]
        for key in ("source_url", "download_date", "file_size_bytes", "sha256", "dataset_version"):
            self.assertTrue(record.get(key), key)
        self.assertEqual(record["file_size_bytes"], len(self.payload))
        self.assertEqual(record["extraction_status"], "extracted")
        self.assertEqual(require_dataset("cmapss"), Path(result["processed_path"]))

    def test_existing_valid_copy_is_not_downloaded_again(self):
        download_dataset("cmapss")
        self.server.requests.clear()
        download_dataset("cmapss")
        self.assertEqual(self.server.requests, [], "no network access expected for a verified copy")

    def test_interrupted_download_resumes_with_range(self):
        self.server.drop_after = len(self.payload) // 3
        download_dataset("cmapss")
        ranges = [r["range"] for r in self.gets]
        self.assertEqual(ranges[0], None)
        self.assertTrue(ranges[1].startswith(f"bytes={len(self.payload) // 3}-"), ranges)
        self.assertEqual(datasets.archive_path("cmapss").read_bytes(), self.payload)

    def test_resume_across_sessions_from_partial_file(self):
        zip_path = datasets.archive_path("cmapss")
        part = datasets._part_path(zip_path)
        part.write_bytes(self.payload[:1000])
        part.with_name(part.name + ".json").write_text(json.dumps({"etag": "v1"}))
        download_dataset("cmapss")
        self.assertEqual(self.gets[0]["range"], "bytes=1000-")
        self.assertEqual(zip_path.read_bytes(), self.payload)
        self.assertFalse(part.exists())

    def test_corrupt_download_is_rejected(self):
        bad = bytearray(self.payload)
        bad[40:60] = b"\x00" * 20  # damage compressed member data, CRC must catch it
        self.serve("cmapss", bytes(bad))
        with self.assertRaises(DatasetIntegrityError):
            download_dataset("cmapss")
        self.assertFalse(datasets.archive_path("cmapss").exists())
        self.assertFalse(load_metadata()["cmapss"]["zip_verified"])

    def test_corrupt_existing_copy_is_quarantined_and_redownloaded(self):
        zip_path = datasets.archive_path("cmapss")
        zip_path.write_bytes(b"not a zip")
        download_dataset("cmapss")
        self.assertEqual(zip_path.read_bytes(), self.payload)
        self.assertTrue(list(zip_path.parent.glob("cmapss.zip.corrupt-*")))

    def test_training_blocked_until_dataset_ready(self):
        with self.assertRaises(DatasetNotReadyError) as ctx:
            require_dataset("cmapss")
        self.assertIn("prepare_datasets(['cmapss'])", str(ctx.exception))

    def test_missing_extracted_files_are_detected(self):
        prepare_datasets(["cmapss"])
        next(datasets.paths.processed("cmapss").rglob("train_FD001.txt")).unlink()
        self.assertFalse(verify_dataset("cmapss")["ready"])

    def test_zip_slip_is_blocked(self):
        self.serve("cmapss", make_zip({"../../escape.txt": b"x"}))
        download_dataset("cmapss")
        with self.assertRaises(DatasetIntegrityError):
            extract_dataset("cmapss")
        self.assertFalse((self.mount / "MyDrive" / "escape.txt").exists())
        self.assertEqual(load_metadata()["cmapss"]["extraction_status"], "failed")

    def test_nested_archive_removed_after_unpacking_and_not_rewritten(self):
        out = prepare_datasets(["cmapss"])["cmapss"]["processed_path"]
        nested = Path(out) / "Turbofan" / "CMAPSSData.zip"
        self.assertFalse(nested.exists(), "unpacked nested archive should not take Drive space")
        self.assertTrue(load_metadata()["cmapss"]["nested_archives"][0]["archive_removed"])
        extract_dataset("cmapss", force=True)  # e.g. re-run after an interrupted extraction
        self.assertFalse(nested.exists())
        self.assertTrue(verify_dataset("cmapss")["ready"])

    def test_keep_nested_archives_option(self):
        download_dataset("cmapss")
        out = extract_dataset("cmapss", keep_nested_archives=True)
        self.assertTrue((out / "Turbofan" / "CMAPSSData.zip").is_file())

    @unittest.skipUnless(importlib.util.find_spec("py7zr"), "py7zr not installed")
    def test_nested_7z_is_unpacked(self):
        import py7zr

        buf = io.BytesIO()
        with py7zr.SevenZipFile(buf, "w") as archive:
            archive.writestr(os.urandom(5000), "1st_test/2003.10.22.12.06.24")
        self.serve("ims_bearings", make_zip({"4. Bearings/IMS.7z": buf.getvalue()}))
        result = prepare_datasets(["ims_bearings"])["ims_bearings"]
        self.assertTrue(result["ready"], result["problems"])
        self.assertTrue(list(Path(result["processed_path"]).rglob("2003.10.22.12.06.24")))

    def test_missing_rar_tool_marks_dataset_partial(self):
        self.serve("ims_bearings", make_zip({"IMS/1st_test.rar": b"Rar!\x1a\x07\x00fake"}))
        with mock.patch.object(datasets.shutil, "which", lambda _tool: None):
            result = prepare_datasets(["ims_bearings"])["ims_bearings"]
        self.assertFalse(result["ready"])
        self.assertEqual(result["extraction_status"], "partial")
        entry = load_metadata()["ims_bearings"]["nested_archives"][0]
        self.assertIn("apt-get -qq install -y unrar", entry["error"])
        self.assertTrue((Path(result["processed_path"]) / "IMS" / "1st_test.rar").exists(),
                        "a failed nested archive must be kept so it can be retried")


class CheckpointAndArtifactTests(StorageTestCase):
    def setUp(self):
        super().setUp()
        setup_storage()

    def test_checkpoint_save_resume_and_prune(self):
        ckpt = CheckpointManager("lstm_fd001", keep_last=2, every_n_steps=1, serializer="pickle")
        self.assertEqual(ckpt.load_latest(), (None, None))
        for step in range(4):
            ckpt.maybe_save(step, lambda: {"weights": [step]}, metrics={"loss": 1.0 / (step + 1)})
        resumed = CheckpointManager("lstm_fd001", serializer="pickle")
        state, info = resumed.load_latest()
        self.assertEqual((state, info["step"]), ({"weights": [3]}, 3))
        self.assertEqual(len(list(ckpt.directory.glob("ckpt_*.pkl"))), 2)
        self.assertFalse(list(ckpt.directory.glob(".*.tmp")))

    def test_model_roundtrip_and_log(self):
        save_model({"theta": [0.1, 0.2]}, "quantum_ai", "vqc_demo", config={"layers": 2}, serializer="pickle")
        self.assertEqual(load_model("quantum_ai", "vqc_demo"), {"theta": [0.1, 0.2]})
        log = ExperimentLogger("vqc_demo")
        log.log(epoch=0, loss=0.5)
        log.log(epoch=1, loss=0.4)
        self.assertEqual([r["epoch"] for r in log.records()], [0, 1])

    def test_storage_report_and_page(self):
        CheckpointManager("run_a", serializer="pickle").save({"x": 1}, 7, metrics={"rmse": 12.5})
        report = storage_report()
        self.assertEqual(report["checkpoints"][0]["latest_step"], 7)
        self.assertEqual(len(report["datasets"]), 2)
        page = render_html(report)
        self.assertIn("run_a", page)
        self.assertIn("Not ready", page)


if __name__ == "__main__":
    unittest.main()
