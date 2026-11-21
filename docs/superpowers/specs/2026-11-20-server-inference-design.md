# CourtShot server + inference: design

Status: **awaiting Kerem's approval.** Reviewed twice (a builder's read and a technical risk review); their findings are folded in. One assumption needs Kerem's confirmation: section 3.

Date: 2026-11-20

## 1. Intent

Today CourtShot is a command-line pipeline run by hand on one machine. This project puts a small web product around it for a lab:

1. Sign in to a dashboard.
2. Upload a padel video and mark the court in the browser.
3. The video waits in a queue until one of the lab's Apple Silicon Macs picks it up and runs the pipeline.
4. The dashboard shows the match: statistics, the tracked video and the 3D replay.

The main server runs on one Azure VM. The Macs sit in the lab behind its network and only make outbound requests.

## 2. Goals, non-goals, anti-goals

**Goals**

- Upload from the dashboard that survives a dropped connection and a page reload.
- Court calibration in the browser, with a preview of the fitted court before anything is queued.
- A queue served by several Macs. A job is never lost, never run by two Macs at once, and never stuck without the dashboard saying so.
- A dashboard that shows every number the current HTML report shows, plus the tracked video and the 3D replay, in one consistent look.
- Minimal but sound sign-in: one pre-created account, hashed password.
- The server has no CV dependency at all.
- `inference/` holds the adapted CV code and the worker. A Mac runs it from a full clone of the repository.

**Non-goals for this version**

- Several users, roles, sign-up, password reset in the UI.
- Full matches cut into segments and merged (section 3).
- Statistics across matches; naming players.
- Changes to detection or analysis algorithms; the ball-labelling and training tools.
- Azure Blob Storage, WebSockets, a message broker.
- Automatic court detection.
- Windows or CUDA workers.

**Anti-goals: these make it a failure even if it runs**

- A job that sits in "processing" forever after its Mac died.
- Two attempts of one job overwriting each other's results, or a page showing files from two different runs.
- A CV library appearing in the server's dependencies.
- Any edit to the reference files (`src/`, `tools/`, `tests/`, `examples/`, the root `pyproject.toml` and `uv.lock`).
- Passwords or worker tokens stored in plain text; any video or result file reachable without signing in.
- The dashboard recomputing statistics its own way, so its numbers differ from the pipeline's.

## 3. Assumption to confirm: video length

Kerem was asked whether videos are clips or full matches and has not answered. This design assumes **clips: at least 10 seconds and at most 15 minutes**, both settings. One video is one job on one Mac.

Jobs are kept separate from matches in the data model, so segment jobs for full matches can be added later without reshaping it.

Why the limit matters:

- The pipeline keeps a whole video's tracking in memory and pickles it in one piece. Its footprint for a full match is unmeasured.
- The 3D scene data is about 240,000 integers per minute, 1.0 to 1.2 MB of JSON (an estimate; nothing has been run). Fifteen minutes is 15 to 18 MB, which a browser loads; ninety minutes is around 100 MB, which it should not.

## 4. Constraints

- **Server:** one Azure VM with an attached managed disk. A reverse proxy terminates HTTPS. The API runs as exactly one process (section 8.6).
- **Database:** SQLite on that disk, migrations with Alembic.
- **Workers:** Apple Silicon, PyTorch on MPS, outbound HTTPS only. Speed and memory on MPS are unmeasured. The development Mac is an M1 with 8 GB.
- **Stack:** FastAPI; Vite + React; `uv` for both Python projects.
- **Licence:** the reference code is GPLv3. The copy in `inference/` stays GPLv3. The dashboard parts ported from `report.html` and `scene3d.html` are derived from GPLv3 files too. This is a note for Kerem, not legal advice.
- **Working method:** code is written by Opus or Sonnet subagents; the main session coordinates. `handoff.md` at the repo root is kept current.

## 5. Approaches

Each choice was put to Kerem with the alternatives.

| Question | Considered | Chosen | Why |
|---|---|---|---|
| How server and Macs talk | (1) Macs pull over HTTP, server is the file hub. (2) Server pushes over WebSocket or a broker. (3) Shared storage. | **1**, settled by the Azure + lab deployment | The Macs need no open port. Only the server process opens the SQLite file. (2) adds a service to keep healthy for no gain at this scale. (3) is unsafe for SQLite and needs one network. |
| Hosting and files | (1) One VM, everything on its disk. (2) VM plus Blob Storage with signed links. (3) App Service or Container Apps. | **1**, Kerem | Same code on a laptop and in production. (3) keeps data on a network share, where SQLite is unsafe. File handling stays in one module so (2) remains a contained change. |
| Database | Postgres, MySQL, SQLite | **SQLite**, Kerem | One file, no service. |
| 3D replay | (1) Embed the existing page. (2) Port to a React component. (3) Leave it out. | **2**, Kerem | One consistent look. |
| Video length | Clips; full matches | **Clips, assumed** | Section 3. |

Calls made by the designer, open to correction:

- Results are stored as files, with one JSON summary column per match for the list page. No per-shot tables.
- The dashboard polls for progress.
- The 3D port keeps the existing three.js code in one imperative module inside a React component; it is not rewritten on another 3D library.
- Charts are the report's own drawings ported as React components. No chart library.
- The worker always delivers a second, unannotated video for the 3D replay (section 10), so the replay never depends on how the upload was encoded.

## 6. System overview

```
Browser ── HTTPS ──▶ Azure VM: reverse proxy ─▶ FastAPI (one process)
                                                  ├─ SQLite file   (data disk)
                                                  └─ match files   (data disk)
                                                        ▲
Lab Mac 1..n: worker ── outbound HTTPS, polling ────────┘
   └─ one subprocess per job: adapted CourtShot pipeline (MPS) + ffmpeg
```

Repository layout. Nothing outside the two new folders, `docs/` and `handoff.md` is touched.

```
server/
  pyproject.toml
  alembic.ini, migrations/
  app/            FastAPI application
  tests/
  web/            Vite + React + TypeScript dashboard
  deploy/         example reverse-proxy and service files
inference/
  pyproject.toml
  src/courtshot/  adapted copy of the CV pipeline (package keeps the name `courtshot`)
    worker/       job loop, HTTP client, job runner
  tests/
  deploy/         example launch file for a Mac
```

The two projects have separate environments and never import each other. They share two contracts: the worker HTTP protocol (section 9) and the result files (section 10). The fine-tuned ball model is not copied: the worker reads `checkpoints/padel_tracknetv3.pt` at the repo root by default, and the path is a setting.

## 7. Life of a match

1. **Choose a file.** The browser reads the file locally and shows a frame from it. It checks the extension, that it can decode the video, and that the length is inside the limits. If the browser cannot decode the file it says so and stops: calibration needs a frame.
2. **Upload and calibrate, at the same time.** The upload starts in the background in chunks. Meanwhile the user picks a frame with clear lines and clicks the court keypoints in a guided order: the same twelve points and order as the reference tool, with skip and undo. The browser fits the court to the clicks and draws the court lines back onto the frame, with the fit error. The clicks of an earlier match with the same frame size can be loaded as a starting point.
3. **Save the calibration.** The button is disabled while the fit error is above the limit or fewer than six points are placed. Saving is what queues the match: the server creates the first job in the same transaction as whichever comes last, the upload completing or the calibration being saved.
4. **Claim.** A Mac polling the server claims the job, downloads the video and the clicked points, and reports its stage and progress every 15 seconds.
5. **Process.** The Mac checks the video, builds and validates the calibration, and runs the pipeline in a subprocess.
6. **Deliver.** The Mac uploads the result files in chunks, then marks the job complete.
7. **View.** The match page shows the results.

Later actions on a match:

- **Run again** queues a new job with the same calibration. Allowed whenever no job is active.
- **Redo calibration** saves new points and queues a new job. Allowed whenever no job is active.
- **Cancel** stops the active job. **Delete** cancels it and removes the match and its files.
- **Resume an upload** after a reload: the match shows "uploading" and asks for the file again. The browser checks size, name and a hash of the first and last megabyte against what was stored at creation, then continues from the server's offset.

**Match status** is derived in one function from the stored facts, never stored on its own:

| Status | Condition |
|---|---|
| uploading | upload not complete |
| needs calibration | upload complete, no calibration |
| queued / processing | newest job is queued / running |
| ready | newest job succeeded |
| failed / cancelled | newest job failed / was cancelled |

A match also has **current results**: those of its newest succeeded job, if any. The match page always shows the current results when they exist, with a banner for the newest job when that job is something else ("a new run is processing", "the last run failed: …"). A failed second run therefore does not hide good earlier results.

A queued match shows "no worker online" when no worker has polled in the last 30 seconds.

## 8. Server

### 8.1 Data model

All timestamps are set by the server, in UTC.

| Table | Columns that matter |
|---|---|
| `users` | `username` (unique), `password_hash` |
| `sessions` | `token_hash` (unique), `user_id`, `created_at`, `last_seen_at`, `expires_at` |
| `workers` | `name` (unique), `token_hash`, `info` (JSON: host, chip, versions), `last_seen_at`, `revoked_at` |
| `matches` | `title`, `original_filename`, `source_ext`, `size_bytes`, `fingerprint` (hash of the first and last megabyte), `upload_complete`, `duration_s`, `frame_width`, `frame_height`, `calibration` (JSON: clicked points by name, frame size), `summary` (JSON for the list page), `created_at` |
| `jobs` | `match_id`, `status` (queued, running, succeeded, failed, cancelled), `attempt`, `not_before`, `worker_id`, `heartbeat_at`, `stage`, `progress` (0 to 1), `calibration` (copy taken when queued), `error`, `log_tail`, `created_at`, `started_at`, `finished_at` |

- The upload offset is the size of the file on disk; there is no column for it.
- `jobs.calibration` is what the run used. `matches.calibration` is the latest saved. They can differ after "redo calibration".
- A match has at most one job that is queued or running. A partial unique index enforces it, and the API answers a violation with a conflict, not a server error.

### 8.2 SQLite settings

Applied in the connection event, on every connection:

- `journal_mode=WAL`, `synchronous=NORMAL`, `foreign_keys=ON`, `busy_timeout=5000`
- `temp_store=MEMORY`, a 64 MB page cache, memory-mapped reads, a WAL size limit
- `PRAGMA optimize` when the process shuts down

**Two engines on the same file.** A session cannot know in advance whether it will write, and in WAL mode a transaction that starts by reading and then writes fails at once, without waiting for the busy timeout. So:

- A **write** engine whose transactions start with `BEGIN IMMEDIATE`. Writers queue on the busy timeout.
- A **read** engine with ordinary transactions, used by handlers that only read. Polling reads never wait behind an upload.

The mechanism is SQLAlchemy's documented recipe for the `sqlite3` driver: take over transaction control in the `connect` event and issue the `BEGIN` from the `begin` event.

Other rules:

- Plain synchronous sessions; one session stays on one thread.
- Every state change checks its precondition inside its own write transaction. In particular the lease sweep is a single `UPDATE … WHERE status = 'running' AND heartbeat_at < cutoff`, and heartbeat, chunk and complete check "running, this attempt" in the transaction that applies them.
- `sessions.last_seen_at` is updated at most once a minute, so polling does not turn every read into a write.
- Alembic runs in batch mode, and the models carry a naming convention for constraints. The partial unique index is created with an explicit `create_index` outside any batch block, because batch mode can drop its condition when it rebuilds a table.
- Tests: the migrations applied to an empty file produce the schema the models describe; the index's SQL in `sqlite_master` contains its condition.

### 8.3 Files

One module, `storage`, builds every path and does every file read and write. No client-supplied text becomes part of a path: the extension comes from a fixed list (mp4, mov, m4v, webm), and result names from the fixed list in section 10.

```
<data dir>/
  app.db
  matches/<match id>/
    source.<ext>              the upload ("source.part" until complete)
    runs/<job id>-<attempt>/  result files of one attempt of one job
```

- **Results are never moved.** Each attempt uploads into its own `runs/` folder. The current results are the folder of the newest succeeded job. The commit that marks a job succeeded, and stores the summary, is the switch: it is atomic, and a crash before it leaves the previous results current. Result URLs carry the job id, so a page cannot mix files of two runs.
- **Chunked upload** is one mechanism used twice, by the browser for videos and by workers for results. The client asks for the current offset, then sends chunks with their offsets. The server writes each chunk at its offset: an offset up to the current size is accepted and one beyond it is refused, so a retried chunk rewrites the same bytes. A lock per file covers the check and the write. The body is read in full, against a hard byte cap, before the lock is taken.
- **Caps:** chunk size; total size of a video; a size per result file name; a total per attempt. New uploads of either kind are refused when free disk space is below a configured reserve.
- **Reading** video and result files supports byte ranges. Content types come from the fixed lists, never from the file's content. CSV files are sent as downloads.
- **Cleanup sweep,** in the server process: removes match folders that have no row; `runs/` folders that are neither the current results nor the running attempt; uploads left incomplete for more than a set number of days.

### 8.4 Sign-in and security

- **Password:** Argon2id. The account is created and its password set from a command on the VM, never from the web.
- **Session:** a random token in a cookie that is `HttpOnly`, `Secure`, `SameSite=Strict` and, in production, carries the `__Host-` prefix. The database stores only a hash of the token. Sessions expire after 7 idle days and 30 days at most; signing out deletes the row.
- **Cross-site requests:** every cookie-authenticated request that changes state must carry an `Origin` header equal to the configured public origin. Worker routes take only the bearer token and ignore cookies, so the rule does not apply to them.
- **Guessing:** failed sign-ins are throttled per client address, and a global budget of failures slows all sign-ins down without locking the account. The throttle is checked before any hashing. Password hashing runs at most two at a time, which also bounds its memory. A sign-in for an unknown user costs the same time as for a known one.
- **Client address:** taken from the right-most entry of the forwarding header, and only when the request arrives from the configured proxy.
- **Workers:** each has its own random token, sent as a bearer token; the server stores a hash. A token is created by a command on the VM, shown once, and can be revoked.
- **Files:** every file route needs a session or a worker token.
- **Worker output is untrusted input.** A leaked worker token must not become script running in the dashboard. Payload values are rendered as text only; the port uses no `innerHTML` and no `dangerouslySetInnerHTML` on them, although the reference pages do.
- **Headers:** no content sniffing; no framing by other sites; a content security policy limited to the site itself, plus `blob:` for media and `data:` and `blob:` for images, which the local video preview needs.
- **Library floor:** byte-range support needs Starlette 0.39 or later, and versions before 0.49.1 have a known denial-of-service bug in range handling. The server pins Starlette at 0.49.1 or later.
- **Settings** come from environment variables: data directory, public origin, upload and length limits, calibration error limit, lease time, attempt limit, proxy address, free-space reserve.

### 8.5 Browser API

All under `/api`, all with the session cookie except sign-in.

| Route | Purpose |
|---|---|
| `POST /auth/login`, `POST /auth/logout`, `GET /auth/me` | session |
| `GET /config` | limits the browser needs: video length, upload size, chunk size, calibration error limit, accepted extensions |
| `GET /matches` | list: status, progress, summary, frame size, whether a calibration exists |
| `POST /matches` | create a match and its upload slot (title, file name, size, fingerprint, length, frame size) |
| `GET /matches/{id}` | detail: status, newest job (stage, progress, error, log tail, worker), calibration, current results (job id, files present) |
| `PATCH /matches/{id}` | rename |
| `DELETE /matches/{id}` | cancel any active job, delete rows and files |
| `GET` and `PATCH /matches/{id}/upload` | chunked upload (offset, then chunk) |
| `PUT /matches/{id}/calibration` | save clicked points and queue a run if the upload is complete; refused while a job is active |
| `POST /matches/{id}/jobs` | run again with the saved calibration; refused while a job is active |
| `POST /matches/{id}/cancel` | cancel the active job |
| `GET /matches/{id}/video` | the uploaded video, with ranges |
| `GET /matches/{id}/runs/{job id}/{name}` | a result file, with ranges |
| `GET /workers` | name, last seen, current job |

The server validates a calibration only structurally: known point names, numbers inside the frame, at least six points. It cannot fit a court, because that needs CV code; the browser previews the fit and the worker decides.

### 8.6 Deployment rules

- **One process.** No extra server workers, no pre-fork, no overlapping old and new process during a deploy. The sweeps and the sign-in throttle live in the process and would be duplicated or split.
- **Proxy:** request bodies at least as large as a chunk; compression for `application/json`; the forwarding header set; HSTS. `server/deploy/` holds a working example.
- **Backup:** SQLite's online backup command. A disk snapshot is a second line, not the first.

## 9. Worker protocol

All under `/api/worker`, all with the worker's bearer token. Every call about a job carries the attempt number in a header.

| Route | Purpose |
|---|---|
| `POST /claim` | Body: worker info. Returns nothing, or a job: job id, attempt number, match id, video size and extension, clicked points and frame size, limits. |
| `GET /jobs/{id}/video` | the video, with ranges |
| `POST /jobs/{id}/heartbeat` | Body: stage, progress. |
| `GET` and `PATCH /jobs/{id}/files/{name}` | chunked upload of one result file for this attempt |
| `POST /jobs/{id}/complete` | The server checks that the required files are present and marks the job succeeded. |
| `POST /jobs/{id}/fail` | Body: short reason, whether a retry makes sense, tail of the job log. |

Rules:

- **Claim** runs in one write transaction: pick the oldest queued job whose `not_before` has passed, mark it running for this worker, raise its attempt number. Two Macs cannot get the same job.
- **Lease.** A running job whose last heartbeat is older than the lease time (default 90 s) goes back to the queue. A sweep in the server process does this every few seconds, so it also happens when no Mac is polling. After three attempts the job fails with "the worker stopped responding".
- **Stale attempts.** If a call's attempt is not the job's current attempt, or the job is not running, the server answers with a conflict. One exception: `complete` for the attempt that already succeeded answers success again, so a lost response does not look like a lost lease.
- **What the worker does with refusals.** "Not found", "conflict" and "gone" all mean: kill the job. On "not found" or "gone" the worker also deletes its work folder.
- **Cancel and delete** take effect on the server at once; the Mac learns it at its next heartbeat.
- **Failures.** A reason the user must fix fails the job at once: bad calibration, four players never visible together, players on court too briefly, video too long or too short, frame size different from the calibration. A crash goes back to the queue with `not_before` a minute ahead, up to the attempt limit.
- **Polling:** idle workers ask every 5 seconds, with a little random spread.

## 10. Result files

| File | Required | Content |
|---|---|---|
| `report.json` | yes | The report payload: exactly what `build_report_data` produces today, plus `schema_version` and `pipeline_version`. Contract: `reference-maps/report-data.md`. |
| `tracked.mp4` | yes | The annotated video. H.264, at most 720 pixels high by default. |
| `plain.mp4` | yes | The same frames without annotations, same size and encoding. Frame *k* of this file is frame *k* of the pipeline. The 3D replay plays it. |
| `summary.json`, `positions.csv` | yes | Existing data files, offered as downloads. |
| `scene.json` | no | The 3D scene: exactly what `build_scene` produces, plus `schema_version`. Contract: `reference-maps/scene-data.md`. Absent when there is no ball track or 3D fit. |
| `shots.csv`, `ball_events.csv` | no | Exist only with ball analysis. |

- `schema_version` lets the dashboard say "made by an older version, run again" instead of drawing a payload it no longer understands.
- `pipeline_version` is the inference package version plus the git commit.
- The payload's own `title` is the work file's name; the dashboard ignores it and shows the match title.
- JSON is compressed on the fly by the reverse proxy. `scene.json` stays one file under the 15-minute limit.
- On completion the server reads `report.json` and stores the list-page numbers (length, rallies, shots, time in play) in `matches.summary`.

**Why `plain.mp4` exists.** The replay keeps the 3D in step with a video by `time = frame / fps`. The upload may have a variable frame rate, a start offset or a codec the viewer's browser cannot play, and the pipeline may have run on a corrected copy of it (section 11.2). A video rendered from the very frames the pipeline read is in step by construction and plays everywhere. It costs one more file of about the tracked video's size per match.

## 11. Inference

### 11.1 Worker

`python -m courtshot.worker` runs the loop: claim, run, deliver, repeat. One job at a time per Mac.

- **Work folder per match** on the Mac: the video, the caches the pipeline writes, the outputs, a log. It survives a failed attempt, so a retry on the same Mac reuses the tracking instead of redoing the slowest step. It is deleted after a successful delivery, when the server says the match is gone, or after seven days.
- **Subprocess per job.** The parent talks to the server; a fresh Python process runs the pipeline and writes progress as it goes. This frees all memory between jobs, keeps a crash away from the loop, and lets the 3D fit start its own process pool, which the reference code does only in a main process.
- **Process group.** The job runs in its own process group, whose id is written into the work folder. Cancel sends a termination signal to the group, waits briefly, then kills it; the fit pool and `ffmpeg` are in the group. At startup the worker kills any group left over from a previous run of itself.
- **Import discipline.** The job's main module imports no PyTorch at top level. Each process of the 3D-fit pool re-imports that module, and loading PyTorch several times over would exhaust an 8 GB Mac. The size of the fit pool is a worker setting, with a small default on small machines.
- **Outcome of a job** is told to the parent by exit code plus a `failure.json` that carries the reason and whether a retry makes sense.
- **Heartbeat** runs in the parent on its own thread, also during upload, so a long upload does not lose the lease.
- **Delivery** retries with growing pauses. A network failure during delivery never repeats the processing.
- **Setup command** downloads the YOLO weights and the SigLIP model ahead of time and checks that `ffmpeg`, `ffprobe` and MPS are present. The worker refuses to start without them. The paths of `ffmpeg` and `ffprobe` are settings, because a launch service does not see Homebrew's folder.
- **Staying awake:** the worker holds a wake lock while running. The README notes what this does not cover: a closed laptop lid, and a service that waits for a user to log in after a reboot.

Stages reported: download, prepare, tracking, appearance, analysis, 3D fit, render, upload. Overall progress is a weighted sum; the weights are a setting until real timings exist.

### 11.2 Preparing a video

1. `ffprobe` gives length, frame timing, rotation, start time and pixel shape. A length outside the limits fails the job.
2. The pipeline runs on the upload untouched unless one of these holds, in which case `ffmpeg` first writes a corrected copy at high quality and the pipeline runs on that:
   - **variable frame rate**, judged from the spread of packet timestamps over the first seconds, not from the two frame-rate fields, which mislead in both directions;
   - **a rotation flag**, looked for both in the stream tags and in the display matrix;
   - **a start time other than zero**;
   - **non-square pixels**, because the browser reports the displayed width and OpenCV decodes the stored one.
3. The decoded frame size must equal the frame size the calibration was clicked on. If not, the job fails with both sizes in the message.

Reason for the copy: the pipeline assumes `time = frame / fps` everywhere, so a phone video with a variable rate gives silently wrong speeds. Reason for making it only when needed: re-encoding can cost the ball detector a little.

### 11.3 Changes to the CV copy

Detection and analysis algorithms are not changed. The file-by-file list with line references is in `reference-maps/tracking-adaptation.md` and `reference-maps/analysis-adaptation.md`. In summary:

| Area | Change |
|---|---|
| Entry point | `CourtShot` takes a calibration object, a device, a work folder and a progress callback; always renders the videos; returns payloads instead of writing HTML. |
| Calibration | Built from the clicked points. The fit uses all points by least squares, because the current RANSAC threshold is in metres and rejects nothing. The worker rejects a calibration whose error exceeds the limit (default 0.25 m), whose points lie outside the frame, whose matrix cannot be inverted, or that is mirrored: near points must be below far points in the image, and left points left of right points. The browser applies the same fit and the same checks. |
| Devices | SigLIP runs on MPS; today it falls back to the CPU on a Mac. The CUDA-only autocast call goes. Every model gets its device passed explicitly. MPS fallback for unsupported operations is switched on. |
| Early failures | Right after the identity step the run stops with a clear reason when four players were never positioned together, or when players are on court for less than the rally window (5 s). Today the first fails late with a misleading message and the second crashes. |
| Progress | Callbacks in the tracking loop, the appearance pass, the 3D fit and the render. |
| Videos | One render pass pipes frames to two `ffmpeg` processes, annotated and plain. H.264 with fast start, scaled to an even size at most 720 high. `ffmpeg` messages go to the job log, never to an unread pipe. Each file is written under a temporary name, checked with `ffprobe`, then renamed. |
| Caches | Kept, inside the work folder. Every cache file and key (tracking, appearance, 3D fit) is written under a temporary name and renamed. A cache without its key is discarded, not trusted. The tracking key gains the package version. |
| Outputs | `report_payload()` and `scene_payload()` return the two JSON payloads; the CSV writers stay; the HTML templates and their writers are dropped. |
| Housekeeping | The pose thread pool is closed after a run. The weight download gets a timeout and a configurable folder. |
| Dropped | `cli.py`, the interactive `calibration.py`, the HTML templates, `run_match.sh`. The tools for labelling and training stay in the reference only. |
| Tests | Carried over as the analysis map lists; tests of dropped code are dropped. |

## 12. Dashboard

Vite, React, TypeScript, React Router, three.js from npm. No UI kit, no chart library, no state library.

**Look.** The report's colour variables, dark by default with a light theme. The 3D replay is restyled to the same variables.

**Brand.** `CourtShot-DesignDoc.html` at the repo root is a logo sheet that appeared during the design; the designer takes it as the brand for the dashboard, which Kerem should confirm. It agrees with the report's look and adds:

- the logo: a dark tile with a ball's arc, used as lockup with the wordmark, with a simplified mark below 120 px and one-colour versions;
- the wordmark: Inter Semibold, tracking −3.5%; Inter is the dashboard's typeface, self-hosted;
- the accent, "Volt" `#D4F542`: in the mark only the ball is Volt, and the arc is never recoloured;
- the dark version is the preferred one;
- the logo is not rotated, stretched, outlined or given a shadow, and keeps clear space of half the tile's width.

**Rule:** every number shown comes from `report.json`. The dashboard formats, and computes only what the report template computes today (sums, maxima, shares).

Supported browsers: current desktop Chrome, Edge and Safari. A video in a codec the browser cannot decode (HEVC outside Safari and hardware-backed Chrome) cannot be calibrated in that browser, and the upload page says so.

### 12.1 Pages

```
/login
/                      Matches
/matches/new           Upload and calibrate
/matches/:id           Match: Overview · Players · Shots · Rallies · Video · 3D replay · About
```

**Matches**

```
CourtShot                                                    [ New match ]   Sign out
Workers: lab-mac-1 idle · lab-mac-2 busy (club_tue)
──────────────────────────────────────────────────────────────────────────────────
Title          Uploaded     Length   Status                         Rallies  Shots
final_f        20 Nov       1:30     Ready                              6       41
club_tue       20 Nov       8:12     Processing · tracking  ▓▓▓░ 62%     –        –
court2_am      19 Nov       4:05     Failed · four players never…   [Run again]
court3_pm      19 Nov       6:40     Queued · no worker online
```

The list refreshes every few seconds while any match is unfinished.

**Upload and calibrate**

```
1 Choose video      2 Mark the court                       3 Save
[ file ]            ┌───────────────────────────────┐      Upload  ▓▓▓▓░ 71%
                    │   video frame, click points   │      Fit error 0.03 m
                    │   court lines drawn back      │      [ Save and queue ]
                    └───────────────────────────────┘
                    Click: net left (7 of 12)   Undo · Skip · Use points from: [earlier match]
```

**Match**, with no results yet: status, stage, progress, the worker's name; on failure the reason, the log tail, "Run again" and "Redo calibration".

**Match**, with results: the tabs, and a banner when the newest job is not the one on show. Overview, Players, Shots, Rallies and About are ports of the report's five sections, with the same content, drawings and empty states. `reference-maps/report-data.md` lists every element, its source fields and its drawing conventions: court heatmap, shot map, height strip and timeline.

- **Video:** a player for `tracked.mp4`, and a menu to download the data files.
- **3D replay:** loads `scene.json` only when the tab is opened, and plays `plain.mp4` beside it. The `ball` and `pose` arrays are converted to typed arrays right after parsing.

### 12.2 3D replay port

- React renders the controls, timeline, information panel and the video element.
- One module holds all three.js code behind a small interface: create with a host element, a video element and the scene data; setters for view, layers, speed and play state; seek; dispose.
- Playback time is published by subscription, not React state, so React does not re-render every frame.
- Disposal is complete: animation frame, video-frame callback, observers, listeners, every geometry, material and texture, then the renderer. The reference page never tears anything down; a tab in a dashboard must.
- Features and shortcuts match the reference page (`reference-maps/scene-data.md`, parts C and D), with one fix: the keyboard handler is scoped to the replay.

## 13. Testing

- **Server**
  - Sign-in: hash check, throttling before hashing, cookie flags, origin check on cookie routes and not on worker routes.
  - Queue: two claims at once get different jobs; lease expiry requeues; a heartbeat that lands with the sweep is not lost; a stale attempt is refused; a repeated `complete` succeeds; the attempt limit; the retry delay.
  - Upload: resume, a retried chunk, an offset beyond the size, the caps, two requests at once.
  - Files: no session, unknown name, path tricks, ranges, results of a second run replacing the first without mixing.
  - Schema: migrations against the models; the partial index's condition.
  - Anti-goals: the server's lock file contains no CV library; the reference paths are unchanged against the starting commit.
- **Inference:** the carried-over tests; calibration validation including the mirror check; the two payload functions on synthetic tracking; "the same input twice gives identical JSON"; both videos checked with `ffprobe`; the early failures; the worker against a stub server (claim, heartbeat, stop, stale attempt, upload resume, leftover process group).
- **Dashboard:** unit tests for the court fit, the formatters and the heatmap blur; every match tab rendered from fixture files, including the no-ball and no-3D cases; a payload containing markup renders as text.
- **End to end, once, on real data:** a local server, one worker on a Mac, `videos/final_f.mp4` from upload to a rendered match page. This is also the first timing of the pipeline on Apple Silicon. Its `report.json` and `scene.json` become the dashboard's fixtures.

## 14. Left to the builder

- Module and component breakdown inside the layout of section 6.
- Exact SQLite cache and memory-map sizes, chunk size, polling intervals, progress weights, size caps per result file.
- The mechanism of the sweeps and of the sign-in throttle.
- CSS organisation; the typeface.
- The form of the example proxy, service and launch files.
- The `ffmpeg` encoder settings and the timestamp tolerance for "variable frame rate".
- In the browser: draw the calibration frame only after the seek has finished and a video frame has been presented, or the canvas can be blank.

## 15. Risks

- **Speed and memory on Apple Silicon are unknown.** The end-to-end run measures them. If a 15-minute clip does not fit a lab Mac, the length limit comes down.
- **MPS operator support** for the YOLO, pose and SigLIP models is not established by the code. The fallback switch covers gaps at a speed cost. The worker can be forced to CPU.
- **Results differ across machines.** Identities and 3D fits can flip at thresholds between MPS and CUDA. The same Mac gives the same output twice; equality with the Windows results is not promised.
- **Nothing is validated against ground truth,** as the reference README says. The dashboard shows what the pipeline computes, with the data-quality tab.
- **Outbound data from Azure is billed** beyond a free monthly allowance, and every video is downloaded to the lab once per Mac that attempts it.
- **Three videos per match** (upload, tracked, plain) set the disk need on the VM.
