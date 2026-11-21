# Handoff: CourtShot server + inference build

Last updated: 2026-11-20

## Goal

A web app around the CourtShot pipeline: sign in, upload a padel match video, mark the court in the browser, queue it, have Apple Silicon Macs process it, and view the match stats, the tracked video and the 3D replay on a dashboard.

## Status

**Built and deployable locally; first version on branch `dev`.** (Kerem: "Start implementing", 2026-11-20, after asking to stop writing plan documents). The design counts as approved as written, including the 10 s to 15 min clip limit and `CourtShot-DesignDoc.html` as the dashboard brand.

- Design: `docs/superpowers/specs/2026-11-20-server-inference-design.md`; exact HTTP contract: `server/API.md`; reference maps: `docs/superpowers/specs/reference-maps/`.
- Build runs as one workflow of parallel subagents, each owning distinct paths:
  - backend: `server/` except `server/web/`
  - dashboard: scaffold, then three in parallel: app and pages (`server/web/` except below), report port (`server/web/src/report/`), 3D replay port (`server/web/src/replay/`)
  - CV copy: `inference/` except the worker paths; ends with a real run on `videos/final_f.mp4` (timings, fixtures)
  - worker: `inference/src/courtshot/worker/`, `inference/tests/worker/`, `inference/deploy/`, `inference/README.md`
  - then one reviewer and one fixer per part, then one cross-part contract audit
- Contract files written by the coordinator: `server/API.md`, `inference/src/courtshot/__init__.py`, `inference/src/courtshot/errors.py` (`JobFailure`).
- `uv` installed with Homebrew (0.13.9). A first reference run of the old pipeline failed on DNS errors while downloading wheels; not retried (the CV copy's real run replaces it).
- Git: the work is committed on branch `dev` and pushed to `origin/dev` (2026-11-21, at Kerem's request). Left out of the commits because they are Kerem's own untracked files and he has not said they belong in the repo: `CourtShot-DesignDoc.html` (the logo sheet) and `videos/final_f_cut.mp4`. Further commits only on request.
- 2026-11-21: builders, reviewers and fixers are done; the cross-part contract audit is running.
- First real run of the adapted pipeline (`run_job` on `videos/final_f.mp4`, 90 s, 1080p, on this M1 8 GB while other agents were also working): 1866 s in total, about 20 times real time. Tracking 1609 s, appearance 62 s, analysis 1 s, 3D fit 157 s (64 of 71 flights fit well), render 32 s. Peak memory footprint 4.9 GB. Outputs in `/tmp/courtshot-e2e/fixtures/` (report.json 44 KB, scene.json 1.4 MB, tracked.mp4 13.7 MB, plain.mp4 8.7 MB).
- Build workflow finished (16 agents): built, reviewed, fixed, cross-checked against `server/API.md`. Test counts: server 110, dashboard 207 (plus a clean `npm run build`), inference 231, worker 32. Reviews found 3 minor (backend), 1 major + 1 minor (CV), 2 major + 5 minor (worker), 1 major + 4 minor (dashboard); all fixed. The audit fixed a title-length mismatch and a minimum-points setting, and smoke-tested the live server with curl and with the real worker code.
- Real-run fixtures copied into the dashboard: `server/web/src/report/final_f.report.json`, `server/web/src/replay/final_f.scene.json`.
- Local deployment (Kerem asked for it on 2026-11-21): `scripts/local.sh` (setup, up, server, worker, seed, status, test, reset, help); data in `.courtshot-data/` (gitignored). Setup done on this Mac: user `admin` with a generated password in `.courtshot-data/admin.password`, worker `local-mac` (token in `.courtshot-data/worker.token`), models present. `up` runs server + worker; dashboard at http://127.0.0.1:8000. Fixed in the script: `seed` failed on videos larger than one chunk (SIGPIPE under pipefail); now it splits into chunk files.
- 2026-11-21 ~01:25: local server and worker stopped at Kerem's request (data kept in `.courtshot-data/`; restart with `scripts/local.sh up`). The worker now logs a status line per stage or percent change (`runner.py` Heartbeat); not yet seen live. Kerem asked about speed: options and an RTX 3070 estimate (about 4 to 7 min per 90 s clip, unmeasured) were given; running a worker on NVIDIA needs `cuda` allowed in the worker's device setting and check (Linux), plus process-group, lock and caffeinate replacements on Windows. Asked which OS the 3070 machine runs; no answer yet.
- Checked in headless Chrome: sign-in, matches list, match page while processing and the new-match page render, with no console errors and no CSP violations. The report tabs and the 3D replay have not been seen in a real browser yet: no match finished locally (the seeded ones and Kerem's own upload were deleted before completing).

## Decided by Kerem

- Stack: Vite + React dashboard, FastAPI, Alembic migrations.
- Database: SQLite, configured for performance (replaces the earlier Postgres/MySQL option).
- Two new top-level folders: `server/` (API + dashboard) and `inference/` (CV code + worker). Existing files (`src/`, `tools/`, `tests/`, ...) stay untouched as reference.
- All CV and model code lives on the inference machines; the server has none of it.
- The CV code may be changed inside `inference/` to fit this architecture.
- Deployment: the main server runs on Azure; the inference Macs sit in a local lab.
- Azure hosting: one VM with everything on its attached disk (FastAPI, the SQLite file, uploads, results), reverse proxy in front for HTTPS. File handling kept in one module so a move to Blob Storage stays contained.
- Sign-in: minimal but secure, one pre-created account with a hashed password.
- 3D replay: ported to a React component, not embedded.
- Working method: code is written by Opus/Sonnet subagents chosen per task; the main session coordinates and does architecture. This file is kept up to date.

## Designer's calls, awaiting approval with the design

The design doc is the source of truth; these are the calls Kerem should check while reading it.

- Macs pull jobs over outbound HTTPS; the queue is a SQLite table with heartbeats and a lease; every worker call carries an attempt number so a Mac that lost its lease cannot overwrite results.
- Calibration in the browser on a frame of the local file, with a preview of the fitted court; saving it is what queues the match. The worker does the authoritative fit and rejects a bad one.
- Chunked, resumable upload (not the single-request upload proposed earlier), used for videos and for results.
- Results are files, each run in its own folder; the newest succeeded job's folder is current. One JSON summary column per match; no per-shot tables.
- The worker delivers two videos, both H.264 at most 720 high: the annotated one, and a plain one for the 3D replay that is frame-for-frame what the pipeline read.
- The worker makes a corrected copy of an upload only when it has a variable frame rate, a rotation flag, a start offset or non-square pixels.
- One subprocess per job on the Mac; the work folder (with the tracking cache) survives a failed attempt.
- Dashboard: progress by polling; the report's drawings ported as React components, no chart library; the 3D replay keeps the existing three.js code in one module.
- No changes to detection or analysis algorithms; only fixes for a silent or late failure (calibration fit, four players never visible, players on court under 5 s).
- One account sees all matches.

## Open

- Video length: asked three times, unanswered. The design assumes clips of 10 s to 15 min and keeps jobs separate from matches so segments can come later.
- May the pipeline be installed and run once on this Mac for timings and fixtures: asked 2026-11-20, unanswered.
- Not answered: who the users are, what camera films the videos, how often videos are uploaded. The design assumes a single operator.
- Tracking speed and memory on Apple Silicon are unmeasured.
- Outbound data from Azure is billed beyond a free monthly allowance; check current pricing once video sizes are known.

## Next steps

1. Kerem approves the design, and confirms or changes the 15-minute clip assumption.
2. Write the implementation plan (superpowers writing-plans), then build with Opus/Sonnet subagents.
3. Early in the build: install `uv` and the pipeline's environment on a Mac and run `videos/final_f.mp4` once, to time the pipeline on Apple Silicon and to produce `report.json` / `scene.json` fixtures for the dashboard. Kerem was asked on 2026-11-20 whether this may be done on this Mac (several GB of downloads); no answer yet.

## Findings worth keeping

- `courtshot run` needs a calibration JSON. Today it comes from an OpenCV click window (`src/courtshot/calibration.py`); `court.calibration_from_keypoints` already builds one from clicked points.
- The tracked video is written with the `mp4v` codec (`src/courtshot/io/video.py`), which browsers don't play.
- The report numbers and the 3D scene are built as dicts (`report/match_report.py: build_report_data`, `report/scene3d.py: build_scene`) and then baked into HTML templates.
- `court.fit_homography` passes a RANSAC threshold of 3.0, which is in metres here, so a bad click is not rejected.
- `analysis/appearance.py` picks `cuda` or `cpu` only, so on a Mac the SigLIP embedding runs on the CPU unless changed.
- The pipeline assumes `time = frame / fps` everywhere; variable-frame-rate video (typical of phones) is not handled.
- A clip shorter than 5 seconds crashes the movement analysis (`movement.py`, rally window) when movement-based rallies are used.
- When four players are never positioned in one frame, the run fails late with a misleading "Is the court calibrated?" error.
- The 3D fit starts its process pool only in a process named `MainProcess`; inside a `multiprocessing` child it silently runs serially. A `subprocess` child is fine.
- 3D scene data: about 240,000 integers per minute at 25 fps, 1.0 to 1.2 MB of JSON per minute (estimate, nothing measured).
- The reference tests assume the package name `courtshot`; the copy keeps the name and lives in its own environment.
