# CourtShot server API

The contract between the server, the dashboard and the Mac workers. Design: `docs/superpowers/specs/2026-11-20-server-inference-design.md`.

## Conventions

- JSON in and out unless stated. Errors are `{"detail": "<message>"}` with the HTTP status.
- Times are ISO 8601 UTC strings, e.g. `2026-11-20T21:14:03Z`. Ids are integers.
- Statuses used throughout: `401` no valid session or token, `403` wrong `Origin`, `404` unknown, `409` state conflict, `413` over a size cap, `422` invalid body, `429` throttled (with `Retry-After`), `507` free-space reserve reached.
- Browser routes use the session cookie. Every non-`GET` browser request must carry an `Origin` header equal to the configured public origin.
- Worker routes (`/api/worker/*`) use `Authorization: Bearer <token>` only and ignore cookies and `Origin`.

## Fixed lists

- **Video extensions:** `mp4`, `mov`, `m4v`, `webm`.
- **Calibration point names**, in clicking order: `bottom_left`, `bottom_right`, `top_left`, `top_right`, `net_left`, `net_right`, `service_near_left`, `service_near_center`, `service_near_right`, `service_far_left`, `service_far_center`, `service_far_right`.
- **Result files.** Required: `report.json`, `tracked.mp4`, `plain.mp4`, `summary.json`, `positions.csv`. Optional: `scene.json`, `shots.csv`, `ball_events.csv`.
- **Stages:** `download`, `prepare`, `tracking`, `appearance`, `analysis`, `ball3d`, `render`, `upload`.
- **Job status:** `queued`, `running`, `succeeded`, `failed`, `cancelled`.
- **Match status:** `uploading`, `needs_calibration`, `queued`, `processing`, `ready`, `failed`, `cancelled`.

## Shared shapes

```ts
type Calibration = {
  points: Record<string, [number, number]>;  // point name -> [x, y] in video pixels; at least 6
  frame_width: number;
  frame_height: number;
};

type Job = {
  id: number;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  attempt: number;
  stage: string | null;
  progress: number | null;        // 0..1, overall
  error: string | null;           // short reason when failed
  worker: string | null;          // worker name
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

type Results = {
  job_id: number;                 // the newest succeeded job
  files: string[];                // result file names present
  finished_at: string;
};

type Match = {
  id: number;
  title: string;
  original_filename: string;
  created_at: string;
  size_bytes: number;
  duration_s: number;
  frame_width: number;
  frame_height: number;
  status: "uploading" | "needs_calibration" | "queued" | "processing" | "ready" | "failed" | "cancelled";
  upload: { offset: number; complete: boolean };
  has_calibration: boolean;
  job: Job | null;                // newest job
  results: Results | null;        // current results, independent of the newest job
  summary: { duration_s: number | null; rallies: number | null; shots: number | null; time_in_play_s: number | null } | null;
};

type MatchDetail = Match & {
  fingerprint: string;
  calibration: Calibration | null;      // latest saved
  job: (Job & { log_tail: string | null }) | null;
};
```

## Browser routes

| Route | Body | Answer |
|---|---|---|
| `POST /api/auth/login` | `{username, password}` | `200 {username}` and the session cookie; `401`; `429` |
| `POST /api/auth/logout` | none | `204` |
| `GET /api/auth/me` | | `200 {username}`; `401` |
| `GET /api/config` | | `Config` below |
| `GET /api/matches` | | `200 {matches: Match[], workers_online: boolean}`, newest first |
| `POST /api/matches` | `{title, filename, size_bytes, fingerprint, duration_s, frame_width, frame_height}` | `201 MatchDetail`; `422` extension or length outside limits; `413` size; `507` |
| `GET /api/matches/{id}` | | `200 MatchDetail` |
| `PATCH /api/matches/{id}` | `{title}` | `200 MatchDetail` |
| `DELETE /api/matches/{id}` | | `204` |
| `GET /api/matches/{id}/upload` | | `200 {offset, size_bytes, complete, fingerprint}` |
| `PATCH /api/matches/{id}/upload?offset=N` | raw bytes, `application/octet-stream` | `200 {offset, complete}`; `409 {detail, offset}` when `N` is beyond the current size; `413` chunk or total too large |
| `PUT /api/matches/{id}/calibration` | `Calibration` | `200 MatchDetail`; `409` while a job is active; `422` |
| `POST /api/matches/{id}/jobs` | none | `201 MatchDetail`; `409` if a job is active, the upload is incomplete or there is no calibration |
| `POST /api/matches/{id}/cancel` | none | `200 MatchDetail`; `409` if no job is active |
| `GET /api/matches/{id}/video` | | the uploaded video; byte ranges |
| `GET /api/matches/{id}/runs/{job_id}/{name}` | | a result file of a succeeded job of this match; byte ranges; CSV as a download |
| `GET /api/workers` | | `200 {workers: Worker[]}` |

```ts
type Config = {
  min_video_seconds: number;          // default 10
  max_video_seconds: number;          // default 900
  max_upload_bytes: number;
  chunk_bytes: number;                // size the browser should send per chunk
  calibration_max_error_m: number;    // default 0.25
  min_calibration_points: number;     // 6
  extensions: string[];
  worker_online_seconds: number;      // 30
};

type Worker = {
  name: string;
  last_seen_at: string | null;
  online: boolean;
  job: { id: number; match_id: number; match_title: string; stage: string | null; progress: number | null } | null;
};
```

Rules:

- `title` is 1 to 200 characters after trimming; `filename` is 1 to 255 characters and its extension, case-insensitive, is one of the video extensions.
- `fingerprint` is the lowercase hex SHA-256 of the file's first megabyte followed by its last megabyte (the whole file if it is smaller than two megabytes).
- Chunk upload: the server writes the chunk at offset `N`. `N` up to the current size is accepted, so a retried chunk is harmless. A chunk that would end beyond `size_bytes` is refused with `413` before its offset is looked at. When the size reaches `size_bytes` the upload is complete. A `PATCH` on a complete upload answers `200 {offset: size_bytes, complete: true}`.
- The first job is created by the server, in the same transaction as whichever comes last: the upload completing or a calibration being saved. `PUT calibration` on a complete upload always queues a job. `POST jobs` runs again with the saved calibration.
- `workers_online` is true when any worker polled within `worker_online_seconds`.

## Worker routes

Every route about a job carries the header `X-Job-Attempt: <attempt>`.

| Route | Body | Answer |
|---|---|---|
| `POST /api/worker/claim` | `{info: {hostname, chip, memory_gb, device, pipeline_version}}` | `204` nothing to do; `200 Claim` |
| `GET /api/worker/jobs/{id}/video` | | the video; byte ranges |
| `POST /api/worker/jobs/{id}/heartbeat` | `{stage, progress}` | `200 {}` |
| `GET /api/worker/jobs/{id}/files/{name}` | | `200 {offset}` |
| `PATCH /api/worker/jobs/{id}/files/{name}?offset=N` | raw bytes | `200 {offset}`; `409 {detail, offset}`; `413`; `422` unknown name |
| `POST /api/worker/jobs/{id}/complete` | none | `200 {}`; `422` a required file is missing or `report.json` is not valid JSON |
| `POST /api/worker/jobs/{id}/fail` | `{error, retryable, log_tail}` | `200 {}` |

```ts
type Claim = {
  job_id: number;
  attempt: number;
  match_id: number;
  video: { size_bytes: number; ext: string };
  calibration: Calibration;            // the copy taken when the job was queued
  limits: { min_video_seconds: number; max_video_seconds: number; calibration_max_error_m: number };
  heartbeat_seconds: number;           // 15
  chunk_bytes: number;
};
```

Rules:

- On every job route: `404` when the job no longer exists; `409` when the job is not running or `X-Job-Attempt` is not its current attempt. One exception: `complete` for the attempt that already succeeded answers `200 {}` again.
- The worker treats `404`, `409` and `410` the same way: kill the job. On `404` and `410` it also deletes its work folder.
- `fail` with `retryable: true` puts the job back in the queue, not claimable for 60 seconds, until the attempt limit (3) is reached; then, and with `retryable: false`, the job is `failed` with `error`.
- A running job with no heartbeat for the lease time (90 s) goes back to the queue; after 3 attempts it fails with `the worker stopped responding`.
- `error` is at most 500 characters and `log_tail` at most 8,000.

## Additions to the result payloads

- `report.json`: the report payload plus top-level `schema_version: 1` and `pipeline_version: string`.
- `scene.json`: the scene payload plus top-level `schema_version: 1`.
- The dashboard ignores the report payload's `title`.
