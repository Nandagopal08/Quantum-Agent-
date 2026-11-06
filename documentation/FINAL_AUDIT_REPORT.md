# Final audit report — 2026-11-06

Companion to the initial audit [AUDIT_REPORT.md](AUDIT_REPORT.md) (written before any change). Everything below was
executed on the build machine (macOS, Python 3.12.2, CPU only) in `development` storage mode unless stated otherwise.

## 1. Protection of existing work

| Item | Action |
|---|---|
| Source code | `backups/source_20261106_183403.tar.gz` (0.6 MB) before any change |
| Results, models, logs, checkpoints | `backups/results_models_20261106_183403.tar.gz` (27 MB; everything in `.dev_storage` except the re-downloadable 7.2 GB datasets; 41 saved models) |
| Environment | `backups/pip_freeze_20261106_183403.txt`, `backups/requirements-lock_20261106_183403.txt`, `backups/venv_bin_before_repair/` |
| Overwrites from now on | Reports in `comparison_reports/` and every saved model keep their previous 5 versions in `_history/` |

No model architecture, training procedure or hyper-parameter was changed. The only model-code addition is an
`ablate` flag (default off) used for diagnostics.

## 2. Issues, root causes, fixes and tests

| ID | Sev. | Issue | Root cause | Fix | Files | Test → outcome |
|---|---|---|---|---|---|---|
| C1 | Critical | Unauthenticated file read via `/%2e%2e/…` | SPA catch-all joined the raw path to `frontend/dist` | Resolve and require `is_relative_to(dist)`; unknown `/api/*` → 404 | `backend/main.py` | `test_static_route_*` (8 encodings + positive cases) → pass; live `curl` → dashboard HTML only |
| H1 | High | Settings accepted any key/type/value (e.g. 40-qubit QAOA, `file://` LLM URL) | `_merge` copied the patch verbatim | `validate_patch`: known keys, types, bounds, enums, http(s) URL, ordered thresholds → 422 | `backend/settings.py` | `test_settings_reject_*` (9 bad, 1 good) → pass |
| H2 | High | Upload: no size limit, any ZIP replaced the verified archive | No checks after `verify_zip` | 2 GiB cap (413), official SHA-256 required unless `allow_unofficial`, previous archive kept | `backend/api/datasets.py`, `storage/datasets.py` | `test_dataset_upload_limits_*` → pass; stored archive unchanged after rejections |
| H3 | High | Per-engine predictions only for seed 1 | Report kept `seeds[0]` only | All seeds' predictions (with uncapped truth) and per-window imputation errors in `qseq_raw_<subset>.json` | `seq_experiments.py` | `scripts/verify_results.py` → see §3 |
| H4 | High | Downstream rows mixed RUL models across seeds | "Best quantum model" chosen per seed | Every trained RUL model evaluated for every seed | `seq_experiments.py` | verifier "downstream rows cover every seed" |
| H5 | High | No paired statistics / CIs | Not implemented | `stats.py`: engine-cluster bootstrap, Wilcoxon, seed-level t, Holm; conservative verdict | `quantum_models/stats.py` | `test_paired_statistics_are_conservative` → pass |
| H6 | High | Re-runs silently overwrote reports/models; logs mixed runs | Plain overwrite; fixed log names | `_history/` versioning (last 5), run-tagged logs, `--quick` writes nothing | `storage/artifacts.py`, `seq_experiments.py` | `test_reports_and_models_keep_history_on_overwrite` → pass (it also caught a version-ordering bug, fixed) |
| H7 | High | 37 venv launchers pointed at `untitled folder 5` | Project folder moved after `python -m venv` | Rewrote the paths in 38 files (launchers, activate scripts, `pyvenv.cfg`); nothing reinstalled | `.venv/bin/*` | `pip/pytest/uvicorn --version`, `pip check` → ok |
| M1 | Medium | Hidden truth visible in event stream and graph state | Truth returned by a traced tool, stored in state | Per-run `TRUTH_VAULT`; tool returns observation only | `agents/missing_data/*` | `test_ground_truth_never_reaches_state_or_event_stream` → pass |
| M2 | Medium | Uploaded windows unvalidated | No checks | Channel count, ≥ 24 steps (trimmed), 500 × 64 cap, asset-id pattern | `graph.py`, `api/qseq.py` | `test_uploaded_windows_are_validated`, `test_agent_request_limits` → pass |
| M3 | Medium | Interrupted agent runs stuck in `running` | No start-up recovery | Marked `interrupted` at start-up; `POST …/rerun` | `runner.py`, `main.py`, `api/qseq.py` | `test_interrupted_runs_are_marked_and_can_be_rerun` → pass |
| M4 | Medium | CORS `*`; docs public behind tunnel | Defaults | Origin allow-list; docs/schema off with a token | `main.py` | `test_security_headers_and_cors`, `test_docs_and_schema_hidden_when_token_set` → pass |
| M5 | Medium | Non-ASCII token → HTTP 500 | `compare_digest` on `str` | Byte comparison | `main.py` | `test_token_required_when_configured` → pass |
| M6 | Medium | Errors leaked absolute storage paths | `str(exc)` returned | `redact()` → `<project>`/`<scratch>`/`<drive>` | `api/common.py` | `test_error_messages_do_not_leak_storage_paths` → pass |
| M7 | Medium | Capped-label convention unstated; no uncapped metric | — | Convention in report/docs; `*_uncapped` metrics | `seq_experiments.py`, docs | verifier "uncapped truth = RUL file" |
| M8 | Medium | No proof circuits matter | — | Circuit ablation + gradient-norm logging | `qseq.py`, `seq_experiments.py` | `test_circuit_ablation_changes_outputs_and_restores` → pass; §4 |
| M9 | Medium | Qiskit check forward-only | — | Gradients (parameter shift vs autograd), mapping, edge cases, mean error; all models, 5 batch types | `qiskit_backend.py`, `workflows.py` | `test_qiskit_matches_pennylane_*`, `test_qiskit_detects_a_wrong_circuit` → pass; §5 |
| M10 | Medium | Missing metrics/time/circuit info; no recomputation tool | — | Summary fields + `scripts/verify_results.py` | `seq_experiments.py`, `scripts/` | §3 |
| M11 | Medium | Drive never validated | No Colab available | `scripts/validate_colab_drive.py` (two phases around a real restart) | `scripts/` | Code path 13/13 in development mode; **real Drive: not run (no Colab here)** |
| M12 | Medium | Agent edge cases untested | — | 7 new tests | `tests/test_qseq.py` | all pass |
| L1 | Low | `catboost_info/` in project root | CatBoost default | `allow_writing_files=False` | `classical_models/rul.py` | — |
| L2 | Low | Unused `_lock` | Leftover | Removed | `agents/runner.py` | suite passes |
| L4 | Low | Hard-coded runtime hint; missing metrics in UI | — | Values derived from report; "unavailable" states; circuit/CI/Qiskit panels | `QuantumSequenceLab.jsx` | `quantumSequenceLab.test.jsx` → pass |
| L5 | Low | No zip-bomb limit | — | 20 GiB extraction cap | `storage/datasets.py` | `test_zip_bomb_guard` → pass |
| L6 | Low | `--quick` appended to real logs | Logger always on | Null logger when not saving | `seq_experiments.py` | — |
| N1 | High (found during testing) | `run_backend.py --tunnel`: a *failed* tunnel printed `https://api.trycloudflare.com/?token=…` as the dashboard link, the API kept running on 0.0.0.0, and SIGTERM orphaned uvicorn | URL regex matched cloudflared's error text; no failure detection; bind-all default; no signal handling | Strict quick-tunnel regex, wait-for-URL with timeout, stop on tunnel failure/exit, bind 127.0.0.1, SIGTERM cleanup | `scripts/run_backend.py` | Fake-binary success/failure runs + `test_failed_tunnel_stops_the_api_*` → pass |
| N2 | Medium (found) | `evaluate` step overwrote the repository's `RESULTS.md` from any storage (tests, quick runs) | Unconditional write | Writes to `<project>/reports/docs/` unless `--publish-docs` | `generate_docs.py`, `workflows.py`, scripts | Quick pipeline left `documentation/RESULTS.md` untouched |
| L3 | Low | Every `ValueError` → 422 | Broad handler | **Not changed** (changing it risks breaking clients); recorded as accepted risk | — | — |

**Incident during testing (disclosed).** While testing `--tunnel` I assumed `cloudflared` was not installed; it is
(`/opt/homebrew/bin/cloudflared`), so two test runs launched a real Cloudflare quick-tunnel request. The only
tunnel-related line captured was cloudflared's error referencing `api.trycloudflare.com`, which indicates the tunnel
request failed; all processes were stopped within minutes, and the random one-time tokens died with them. No further
real tunnels were started; the fixed logic was tested with fake binaries.

<!-- RESULTS -->
