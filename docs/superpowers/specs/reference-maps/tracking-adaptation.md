> Extracted from the reference code under `src/courtshot` on 2026-11-20 and checked against the source by a second reader. Line numbers refer to the reference files at commit c2b1767. Nothing in the pipeline was run to produce this.

# Inference copy: plan from reading `src/courtshot`

## 0. What to do with each file

| File | Action |
|---|---|
| `pipeline.py` | **Copy and modify.** `CourtCalibration` objects only: drop `load_calibration` (`:44-57,92`). Configured absolute ball model path (`:37,89-91`). `cache_dir` = job directory (`:95`). Pass a device and a progress callback down (`:126-132,162-164`). Always write the video (drop `:271-280`). JSON instead of HTML (`:260-261,283-287`): `build_report_data` (`report/match_report.py:98`) and `build_scene` (`report/scene3d.py:216`) already return JSON-ready dicts. Needs `analysis/*` too. |
| `cli.py` | **Drop.** Only it configures logging (`:21-26`); the worker must. |
| `court.py` | **Copy and modify** `fit_homography` (§8). |
| `calibration.py` | **Drop**: OpenCV GUI (`:75-82,138-146,264-268`). `__init__.py:13` imports it; remove that line in the copy. Click order: `KEYPOINT_ORDER` (`:26-39`). |
| `weights.py` | **Copy and modify.** Configurable `WEIGHTS_DIR` (`:17`); drop the `~/.padex` fallback (`:46-49`); give `urlopen` a timeout (`:71`). |
| `schemas/tracking.py` | **Copy as-is.** |
| `io/video.py` | **Copy and modify**: H.264 writer (§4). |
| `tracking/__init__.py`, both yaml | **Copy as-is**, yaml next to `player.py` (`player.py:93`). |
| `tracking/ball.py` | **Copy and modify**: autocast (`:283`, §1). |
| `tracking/cache.py` | **Copy as-is.** Keep `load_pickle` (`:62-65`): `analysis/match.py:149` uses it. |
| `tracking/device.py` | **Copy as-is.** |
| `tracking/pipeline.py` | **Copy and modify**: callback at `:122`; close the pose pool in the `finally` (`:130-132`). |
| `tracking/player.py` | **Copy and modify**: `close()` for `_pose_pool` (`:352,368-369`). |
| `viz/*` | **Copy as-is.** |
| `pyproject.toml` | **Copy and modify** (§7). |
| `run_match.sh` | **Drop.** Keep its `caffeinate -dims -w <pid>` (`:17`). |

## 1. Device handling

All CUDA-specific code in `src/courtshot`:

- `tracking/device.py:21-59`: cuda, then mps, then cpu, cached in `_cached_device` (`:18`). MPS is picked when CUDA is absent.
- `device.py:62-71`: half precision only for `cuda*`. On MPS everything is FP32: `half=False` to ultralytics (`player.py:87,112,183,207,286`), `_half=False` in TrackNet (`ball.py:229`).
- `ball.py:283`: `torch.autocast("cuda", enabled=self._half)`. Disabled on MPS, so results are unaffected. Delete it in the copy.
- **The one real CPU fallback:** `analysis/appearance.py:111` picks `"cuda"` or `"cpu"`, and `extract()` builds `_Embedder()` without a device (`:163`), so SigLIP runs on the CPU on a Mac. Pass `detect_device()`.

Ultralytics:

- The device string goes into every call (`player.py:111,206,285`), resolved in `tracking/pipeline.py:55-63`. `CourtShot` cannot override it (`pipeline.py:126-132`); add a `device` argument so a job can be forced to CPU.
- Detectors built without a device pass `device=None` (`player.py:77,172,348`); ultralytics then chooses. Always pass it.

Not settled by the code:

- Whether every YOLO26 op exists on MPS. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing torch.
- Whether three models may use MPS at once: ball thread (`tracking/pipeline.py:96-105`), pose thread (`player.py:366-370`), main thread. `COURTSHOT_THREADS=0` (`device.py:82-88`) runs them on one thread; keep it.
- Speed on MPS. Results will differ numerically from CUDA.

## 2. Filesystem assumptions

- **Current directory:** ball model `checkpoints/padel_tracknetv3.pt` (`pipeline.py:37`; outside the wheel, `pyproject.toml:63-64`, so the inference package must ship it); output `output/<stem>/` (`pipeline.py:263-264`); a tracker yaml there wins over the shipped one (`player.py:92-94`).
- **Next to the video, or in `cache_dir`, named from the stem alone** (`tracking/cache.py:27-33`): `<stem>_tracking_cache.pkl`, `.key.json`, `.old.pkl` (`:140-141`), `_tracking_cache_appearance.npz` (`analysis/appearance.py:208-210`), `_tracking_cache_ball3d.pkl` and its key (`analysis/match.py:92-97`).
- **Collisions:**
  - Output names are fixed (`report.html`, `summary.json`, three CSVs, `scene3d.html`, `<stem>_tracked.mp4`): two videos with one stem overwrite each other.
  - The cache key identifies the video by `name:size` (`cache.py:91-95`). Same name, size and calibration: the wrong tracking is reused. Same stem, other size: the cache is set aside, replacing the earlier `.old.pkl`.
- **Retry traps:**
  - A cache without a key file is trusted (`cache.py:113-121`); pickle and `.npz` are written in place (`cache.py:76-77`, `appearance.py:254`). A crash mid-write fails every retry.
  - The tracked video is skipped when tracking came from the cache and the file exists and is not older than the 3D fit (`pipeline.py:271-277`): a truncated file is kept.
- **Rule:** one directory per job for input, cache and output, emptied before each attempt (or atomic cache writes, to resume). Keep caching on: appearance and the 3D fit need the cache file (`pipeline.py:152-161`).

## 3. Progress

1. **Models.** YOLO weights are resolved (downloaded if missing) and the person model loaded in `TrackingPipeline.__init__` (`tracking/pipeline.py:58-62`, `player.py:84,95,179`). Pose and TrackNet load on the first frame (`player.py:192`, `ball.py:222-248`).
2. **Median background:** 30 seeks (`tracking/pipeline.py:143-155`).
3. **Tracking loop, the slowest:** `tracking/pipeline.py:100-129`; report `processed / total_frames` at `:122`. The total is `CAP_PROP_FRAME_COUNT` (`io/video.py:75`); the loop cannot pass it (`io/video.py:48-50`) but can stop short (`:52-53`), so finish with done = total.
4. `flush()` (`tracking/pipeline.py:134`), cache saved (`pipeline.py:137`).
5. **Appearance:** one sequential pass over the video, `fid / last` in `analysis/appearance.py:175-192`.
6. Identities and 2D ball events (`analysis/match.py:78-80`): no loop exposed.
7. **3D fit, slow** (`match.py:156`): `analysis/ball_flight.py:635` is `list(pool.map(...))`. Iterate the map and report `i / len(jobs)`; order is kept. Serial path: `:637`.
8. **Tracked video, slow** (decode, draw, encode): `pipeline.py:217-238`, `frame_id / reader.frame_count`.
9. JSON outputs.

Callback: `on_progress(stage, done, total)`, throttled by the caller. Raising from it aborts the tracking loop cleanly (`tracking/pipeline.py:130-132`).

## 4. Tracked video

- Today: `cv2.VideoWriter`, fourcc `mp4v` (`io/video.py:104,117`): MPEG-4 Part 2, which browsers generally do not play. Written in place to `<out>/<stem>_tracked.mp4` (`pipeline.py:270`), source size, `CAP_PROP_FPS` (`:199,216`), no audio.
- (a) fourcc `avc1`: whether the installed opencv-python wheel encodes H.264 on macOS **cannot be known from the code**. `isOpened()` (`io/video.py:124`) may pass and `write()` reports nothing (`:129-132`). No `+faststart`.
- (b) `mp4v`, then an ffmpeg transcode.
- (c) **Safe option:** pipe BGR frames to ffmpeg: `-f rawvideo -pix_fmt bgr24 -s WxH -r fps -i - -c:v libx264 -pix_fmt yuv420p -movflags +faststart`. Even dimensions; redirect stderr; check the exit code; write to a temporary name, then rename.
- ffmpeg from Homebrew or the `imageio-ffmpeg` wheel (verify libx264 in its arm64 binary); check at worker start.
- Not settled: variable-frame-rate input; whether H.264 yuv420p with faststart is all the dashboard needs.

## 5. Headless and long-running concerns

- **GUI:** every window call is in `calibration.py`; none on the processing path.
- **Pools:**
  - Ball thread: per run, shut down in `finally` (`tracking/pipeline.py:96-98,130-132`).
  - Pose thread: never shut down by this code, not even by `reset()` (`player.py:352,368-369,380-385`). Add `close()`.
  - 3D fit: a `ProcessPoolExecutor` per run in a `with` (`analysis/ball_flight.py:630-635`); `cpu_count // 2` processes or `COURTSHOT_WORKERS` (`:574-580`). macOS spawns, so each fit process re-imports the worker `__main__`: keep the job loop under `if __name__ == "__main__"`, or every fit process pulls jobs. The pool starts only in a process named `MainProcess` (`:630`); in a `multiprocessing` child the fit silently runs serially (`:637`). `subprocess` is fine.
- **State between jobs:**
  - None survives today: `track()` builds a throwaway `TrackingPipeline` (`pipeline.py:126-132`), `extract()` a throwaway SigLIP (`appearance.py:163`). Four models reload per job (cost unknown).
  - If objects are reused, `TrackingPipeline.run()` never calls `reset()`. ByteTrack state sits in the YOLO object (`persist=True`, `player.py:108`); the only reset reloads the model (`:130-132`). `_pose_buffer` (`player.py:353,451-468`) is cleared by `PlayerDetector.reset()` (`:385`). `BallDetector.flush()` leaves `_frame_buffer` and the background tensor (`ball.py:173-185`); `reset()` clears them but keeps `_bg_source` (`:187-192`).
  - `_cached_device` is harmless.
  - Not settled: whether ultralytics restarts track IDs for a new YOLO object in the same process; whether MPS memory returns when models are dropped (call `torch.mps.empty_cache()`).
- **Memory:** every `PlayerFrame` (up to 8 per frame, 17 keypoints each) stays in RAM and is pickled at once (`tracking/pipeline.py:75,115`). Full-match footprint unknown; measure.
- **Simplest safe rule:** a new `TrackingPipeline` per job, as now; one `subprocess` per job removes every item above.

## 6. Network needs at first run

- `yolo26m.pt`, `yolo26m-pose.pt` from `https://github.com/rondo-labs/Padex/releases/download/v0.1.0/` to `~/.courtshot/weights/` (`weights.py:17-24,51-62`) at pipeline construction, via a fixed `<name>.tmp` (`:55`; fine for one worker per machine). No timeout, no checksum.
- SigLIP `google/siglip-base-patch16-224` via `from_pretrained` into the Hugging Face cache (`analysis/appearance.py:39,113`), **after tracking**. Only `ImportError` is caught (`analysis/identities.py:610-616`), so an offline first run fails the job there.
- No download for the ball model, nor for BoT-SORT (`model: auto`, `courtshot_botsort.yaml:17`; not the default).
- **Not settled:** what ultralytics and transformers contact by themselves. Pre-fetch at install; test once offline.

## 7. Dependencies

- Inference: all of `pyproject.toml:18-28` (numpy, opencv-python, ultralytics, lap, pydantic, scipy, torch, torchvision, transformers), Python 3.11+ (`:7`). `uv.lock` pins ultralytics 8.4.19 and torch 2.10.0 off Windows.
- Drop the Windows CUDA index (`:48-57`) and the console script (`:42-43`).
- opencv-python-headless is out: ultralytics 8.4.19 requires `opencv-python` (`uv.lock:2332`).
- Add ffmpeg. HTTP: `requests` already arrives with ultralytics (`uv.lock:2337`); declare it.
- GPLv3 (`pyproject.toml:6,12`) carries over to the copy.
- The server needs none of torch, torchvision, ultralytics, lap, opencv, scipy, transformers, numpy; pydantic arrives with FastAPI. So the server cannot fit or check a homography: the worker does (or browser JavaScript).

## 8. Calibration from clicked points

- **Signature:** `calibration_from_keypoints(keypoints_px: dict[str, (x, y)], frame_width: int, frame_height: int) -> CourtCalibration` (`court.py:91-121`).
- **Names:** the 12 in `COURT_MODEL.KEYPOINTS`, metres, origin near-left, "bottom" = near (`court.py:4,34-49`). Unknown names are skipped with a warning (`:104-106`).
- **Errors:** `ValueError` for fewer than 4 known points (`:110-111`) or a failed fit (`:82-83`).
- **Returns:** `homography_matrix` (3×3, pixels to metres), the frame size, pixel and metre point lists **without names** (`:118-119`), `reprojection_error` (RMS, metres). Keep the name-to-pixel dict too, as `calibration.py:330` does.
- **Validation today:** pydantic types only (`schemas/tracking.py:72-87`); no error limit.
- **Frame size is never compared with the video but is used:** 3D camera centre and pixel tolerances (`analysis/ball_flight.py:113,691-695`), 2D ball step (`analysis/ball_events.py:564`). `H` is applied as given (`tracking/pipeline.py:77`). Clicks must be in decoded-frame pixels (`calibration.py:304,316-320`); compare with `VideoReader.frame_size` (`io/video.py:78-82`). Rotation metadata handling is not settled by the code.
- **A bad calibration fails late:** off-court people are dropped (`player.py:470-477`); with no stable player IDs the report raises `ValueError` (`analysis/movement.py:92-96`) after every slow stage.
- **RANSAC weakness:** `cv2.findHomography(src, dst, cv2.RANSAC, 3.0)` (`court.py:81`). Per the OpenCV documentation (not this repo), the threshold is in destination units, here metres: only a click more than 3 m off is an outlier.
  - A rejected point leaves the RMS (`:85-87`) but stays in the stored lists (`:118-119`), which the 3D camera refinement uses (`ball_flight.py:128-130,689-691`).
  - With 4 points the fit is exact; the error says nothing.
- **Fixes:**
  - `method=0` on all points plus per-point and leave-one-out errors; `tools/check_calibration.py:36-44,58` has this and flags a point above 0.2 m and twice the median.
  - Reject above a configurable RMS. The hand-clicked `videos/match_calibration.json` (6 points, 720p) has 0.031 m.
  - Require in-frame points, preferably 6+, and a finite invertible `H` (`ball_flight.py:111` inverts it unguarded).
  - Mirrored clicks fit with low error: check near points are lower in the image than far ones, and left ones left of right ones.
