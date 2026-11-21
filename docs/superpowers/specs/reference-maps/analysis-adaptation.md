> Extracted from the reference code under `src/courtshot` on 2026-11-20 and checked against the source by a second reader. Line numbers refer to the reference files at commit c2b1767. Nothing in the pipeline was run to produce this.

# Analysis and output layer: adapting it for `inference/`

Paths are under `src/courtshot/`. Nothing in the pipeline was run.

## 1. Device and network

- Torch is used only in `appearance._Embedder` (appearance.py:103-131). The device is `device or ("cuda" if available else "cpu")` (:111), so it **never picks MPS**, and no caller passes one (:163, :253): on a Mac, SigLIP runs on the CPU. `tracking/device.py:20-58` `detect_device()` already returns `"mps"`. In the copy, build one `_Embedder(device=detect_device())` at worker start and pass it through `extract(..., embedder=)` (:147); today the model is reloaded on every call. Whether SigLIP runs fully on MPS is not settled by the code.
- Precision is fp16 only on `cuda` (:112-114, :129), fp32 elsewhere; `COURTSHOT_FP16` is not read here. Vectors are stored as float16 (:172). Batch 64 (:104); crops are flushed at 256 (:188).
- `google/siglip-base-patch16-224` (`from_pretrained`, :113) downloads from the Hugging Face Hub on first use. Only `ImportError` is caught (identities.py:614); a failed download propagates out of `analyse_cache`. Fetch it at worker start.
- `extract` decodes from frame 0 and assumes `frame_id` is the decode index (:158-192). The rest of the analysis is numpy, scipy and cv2 on the CPU.

## 2. The process pool in `ball_flight.reconstruct`

- `ProcessPoolExecutor(max_workers, initializer=_init_worker, initargs=(solver,))` with no `mp_context` (ball_flight.py:631-635), so **spawn** on macOS. One pool per call, joined when the `with` ends. `pool.map` keeps job order.
- Gate (:630): a pool starts only if `workers > 1` and the process name is `"MainProcess"`.
  - Long-running worker, **any thread**: the pool starts (the name belongs to the process).
  - `multiprocessing` child (`Process-N`): the fit runs **serially, with no log line**.
  - `subprocess` child (a fresh interpreter): it is `MainProcess`, so the pool starts.
- Spawn re-imports the parent's main module in each fit process as `__mp_main__`, so unguarded module-level code (a poll loop) runs again. CPython 3.13 skips a main named `pkg.__main__` (`multiprocessing/spawn.py`, `_fixup_main_from_name`). The chain `ball_flight → ball_events → identities → tracking.pipeline` imports torch lazily today (tracking/ball.py:32); keep that, or every fit process loads torch.
- `_importable_main` (:552-571) deletes `__main__.__file__` only for a piped script (`__file__` set, no `__spec__`, file missing), and then changes process-global state. Otherwise it does nothing.
- Count: the `workers` argument, else `COURTSHOT_WORKERS` (`max(1, int(env))`; a non-integer raises `ValueError`), else `cpu_count // 2` (:574-580), capped at the job count (:627). `ball3d_for` passes no `workers` (match.py:157). `1` means no pool. Apple Silicon has no hyperthreading; tune it.
- Risks: no timeout, cancel or progress hook. Each fit process gets a pickled copy of the whole-match `_FlightSolver` (:466-483; size not measured). One flight's exception is re-raised by `pool.map` and all flights are lost (:635). The docstring reports an endless wait for piped scripts (:557-560). Whether fit processes outlive a SIGKILLed parent is not settled by this code.
- Recommendation: analyse in the worker's own process, or in one `subprocess` per job (cancel by kill), and call `ball_flight.run_on_tracking(..., workers=N)` (:681).

## 3. Caches and sidecar files

| File (next to the video) | Code | Key | Stale when | Per-job worker |
|---|---|---|---|---|
| `<stem>_tracking_cache.pkl`, `.key.json` | tracking/cache.py:27-33, 73-78 | video `name:size`, `calibration_sha1`, ball model `name:size`, `use_tracknet_v3`, `enable_pose`, `tracker`, `max_players`, `pose_closeups_only` (:98-107) | key differs or is unreadable (:122-131); it is then moved to `.old.pkl` (:140). No key file: reused with a warning (:113-121) | Keep in the job folder for retries, delete after |
| `<stem>_tracking_cache_appearance.npz` | appearance.py:208-210, 254 | inside the npz: `tracking_sha1`, `model`, `samples_per_second`, `crop` (:213-223) | key differs (:238). Remade only if the video exists, else `None` (:241-247). A corrupt file raises (:237) | Do not write; call `extract()` |
| `<stem>_tracking_cache_ball3d.pkl`, `.key.json` | match.py:92-97, 160-162 | `tracking_sha1`, `fps`, reprs of the **default** `BallSettings`, `FitSettings`, `StableIdConfig`, plus the swaps (:100-113, :145) | key differs or is unreadable; no key file: mtime (:116-131); unpickling error (:152); `refit=True` | Do not write. It is the slowest CPU step (match.py:9): pickle it in the job folder only if retries must skip it |

No key includes device or precision. `run()` also keeps an existing `<name>_tracked.mp4` by mtime (pipeline.py:269-282). `report.html`, the three CSVs, `summary.json` and `scene3d.html` (match_report.py:172-313, scene3d.py:278-316) become JSON files in the job folder. The Hugging Face cache is kept across jobs.

Both sidecar keys hash the tracking pickle on disk (appearance.py:215, match.py:105), so `analyse_cache` needs that file, and the in-memory path in `CourtShot.analyse` (pipeline.py:152-161) skips appearance and the 3D fit. The copy needs an in-memory `analyse`: `extract` → `Appearance` → `stabilize_tracking` → `ball_events.run_on_tracking` → `ball_flight.run_on_tracking` → `MatchAnalysis`.

## 4. Outputs as data

- **Report payload:** `build_report_data` (match_report.py:98-164) is pure and returns a `clean()`ed dict. Its inputs are made only inside `write_match_report` (:271-290): `player_arrays` (:274), ball or movement rallies (:277-278), `movement.analyse` (:279), `s.rally_source` (which **mutates** the Settings, :280), `shots.match_stats` (:284), the title from the cache stem (:289), `has_3d` (:290). The payload is not returned (:314).
- **Summary:** built inline (:293-312), with `ball_stats()` (:63-95) called only there. The returned dict holds NaN; only the file is cleaned (:313).
- **CSV rows:** formatted inside `write_csv` (:179-204), `write_ball_csv` (:207-222) and `write_shots_csv` (:225-260).
- **Scene payload:** `build_scene` (scene3d.py:216-275) is pure; `write_scene` adds the guard (:294-295) and `scene["video"]` (:299-303). `pose` is a flat int array n×13×3 and `ball` n×4 (x, y, z, flight index), in centimetres on viewer axes (:45-48), with −32000 for missing. Poor fits are included, marked `ok: false`.
- **Minimal refactor:** make :271-312 a `report_payload(analysis, settings)` returning the payload, the cleaned summary and the CSV inputs. Add `scene_payload(analysis)` = guard + `build_scene` + `video: None`. Keep the CSV writers. Drop `write_report`, `write_scene`, `print_summary`, `TEMPLATE`. Dump the scene with `allow_nan=False`, which :304 lacks.
- **Size:** about 17 M ints per hour at 30 fps, around 100 MB of JSON (estimate). Store it as a gzip file, not in SQLite rows.
- `export_video` computes `build_shots` again (pipeline.py:192).

## 5. Failure modes and what the caller gets today

| Input | Where | Today |
|---|---|---|
| No positions, or never 4 positioned detections in one frame | identities.py:143-150 | Raw IDs (`P_007`), `id_report == {}`, a warning. Then: no hits (ball_events.py:545), zero flights, `player_arrays` raises a **misleading** `ValueError` "... Is the court calibrated?" (movement.py:94-96), and `build_scene` would emit one player per raw ID (scene3d.py:105) |
| 4 seen, never 2+2 clear of the net | identities.py:457-462 | Seeds from the first frame with 4; teams can be wrong; no warning |
| Video missing; torch or transformers missing | appearance.py:241-247, identities.py:614 | `None`, a warning, identities from motion only |
| Download fails; video will not open; corrupt `.npz` | appearance.py:113, 160, 237 | Exception; the analysis fails |
| Too few crops near the seed; labelling not confirmed | identities.py:264-267, 340-342 | Motion only, a warning; `appearance_used` is still True (:190) |
| No ball detections, or no calibration | ball_events.py:538 | `ball` and `ball3d` are None; no shots; `write_scene` raises (scene3d.py:294) |
| Detections but no hits | ball_flight.py:627, 648 | `Ball3DResult` with no flights; `has_3d` is True |
| Flight with fewer than 6 points or no start candidates; poor fit | ball_flight.py:384, 397; 451 | Dropped silently; kept with `ok=False`, and shots use 2D values |
| Exception in one fit; singular homography (`LinAlgError`) | ball_flight.py:635, 111 | Uncaught; the whole 3D step fails |
| No player frames; fewer than 2 timestamped frames | scene3d.py:217; identities.py:597 | `ValueError` from `max()`; fps is 30, silently |
| Clip shorter than 5 s (`rally_window`), movement rallies in use | movement.py:79, 309 | `IndexError`: `np.convolve(mode="same")` returns `max(n, window)` values. Confirmed with numpy on a copy of these lines |
| Variable frame rate | ball_events.py:412, movement.py:112 | Not handled; times are `frame / fps` |

The worker should check `id_report` straight after `stabilize_tracking` and fail there with "four players were never on court together". Otherwise it returns a partial result with quality flags, and sets `has_3d` from `any(fl.ok)`. Build the calibration with `court.calibration_from_keypoints` (court.py:91-120), in source-video pixels: it fills the keypoints the camera refinement needs (ball_flight.py:128).

## 6. Tests

| Test file | Covers | For the copy |
|---|---|---|
| test_appearance_ids.py | `Appearance`, the sidecar, `_fix_swaps` (synthetic) | Unchanged; drop `TestSavedAppearance` (:91) |
| test_match_stats.py | `shots`, `build_report_data`, `clean` | Unchanged; drop the HTML test (:307) |
| test_shot_pose.py | `swing`, `analyse_ball`, `build_shots` | Unchanged |
| test_ball_speed.py | `frame_speeds`, `viz.frame` | Unchanged if `viz` is copied |
| test_reuse_and_candidates.py | cache keys, old caches, `drop_static` | It imports `match` and `pipeline.CourtShot` (:20-21), so nothing in it runs if those change. Move `TestDropStatic` (:262) out; rewrite `TestTrackingCacheKey` (:40); drop `TestOlderCaches` (:112) and `TestBall3dCacheKey` (:189) |
| test_court, test_schemas, test_player_detection, test_ball_detection, test_frame_annotator, test_mini_court | Layer 1, `viz` | Carry over (checked by imports only) |
| test_video_writer.py | `VideoWriter` | No edit needed for the codec: no test checks it, and the default is `mp4v` (io/video.py:104), not H.264. Add a codec check |

No test calls `reconstruct`, `fit_flight`, the camera, the pool, `extract`, `player_arrays`, `frame_rate`, `analyse_cache`, `write_match_report`, the CSV writers, `build_scene` or `export_video`. None covers a failed seed, a short clip or determinism.

"Unchanged" assumes the package keeps the name `courtshot`, which clashes with the original in one environment: rename it or use a separate venv.

## 7. Determinism

- **Claims:** CLAUDE.md says the report and replay are deterministic, and that `report` on an existing cache gives byte-identical `summary.json` and the three CSVs. ball_flight.py:602 says the result is the same for any worker count. No test checks either.
- **What supports them:** `pool.map` keeps order (:635). There is no randomness or clock. Sets are only counted, tested or sorted (ball_flight.py:607, scene3d.py:105). JSON key order is dict insertion order. `clean()` rounds to 4 decimals and the CSVs to 1-3, except the unrounded `time_s` in `shots.csv` (shots.py:355).
- **Scope:** the claim is for one machine, from an existing cache, where the SigLIP vectors and the 3D fit usually come from sidecars. The worker recomputes both. Across machines these can differ (not measured):
  - SigLIP in CUDA fp16 against MPS or CPU fp32, and video decoding. The swap threshold (identities.py:298) can flip, which changes identities and everything after.
  - `least_squares` and `polyfit` on another BLAS. `ok` (ball_flight.py:451) and the splits can flip. An unstable `np.argsort` picks the starts (:405); its tie order is not guaranteed.
- **Recommendation:** pin library versions. Test "same input twice on one machine gives identical JSON". Do not promise equality with the Windows/CUDA output.
