# CourtShot inference worker

The adapted CourtShot pipeline (`src/courtshot`) and the worker that runs it on an Apple Silicon Mac
(`src/courtshot/worker`). The worker polls the CourtShot server over outbound HTTPS, claims one job at a
time, downloads the video, runs the pipeline in a subprocess and uploads the results. The Mac needs no
open port. Design: `docs/superpowers/specs/2026-11-20-server-inference-design.md`, sections 9 and 11;
protocol: `server/API.md`.

## Install

On the Mac, from a full clone of the repository (the worker reads the ball model from
`checkpoints/padel_tracknetv3.pt` at the repository root):

```sh
brew install uv ffmpeg
cd inference
uv sync
uv run python -m courtshot.worker setup
```

`setup` downloads the YOLO weights (to `~/.courtshot/weights/`, or `$COURTSHOT_WEIGHTS_DIR`) and SigLIP
(to the Hugging Face cache), then checks ffmpeg with libx264, ffprobe, PyTorch with MPS and the ball model,
and prints what it found. It retries each download three times; run it again if the network failed.
`python -m courtshot.worker check` runs the checks only, without downloading anything. The worker runs
the same checks at startup and refuses to start when one fails.

## Worker token

Each Mac has its own token. On the server VM:

```sh
cd server && uv run python -m app.cli create-worker lab-mac-1
```

The token is shown once; the server keeps only its hash. `revoke-worker` revokes it.

## Run

```sh
cd inference
export COURTSHOT_SERVER_URL=https://courtshot.example.org
export COURTSHOT_WORKER_TOKEN=...
uv run python -m courtshot.worker run
```

Stop it with Ctrl-C or SIGTERM: it stops claiming, kills the running job, reports it to the server as
retryable (the job goes back to the queue) and exits.

| Variable | Flag | Default |
|---|---|---|
| `COURTSHOT_SERVER_URL` | `--server-url` | required |
| `COURTSHOT_WORKER_TOKEN` | `--token` | required; prefer the variable, flags show in `ps` |
| `COURTSHOT_WORK_ROOT` | `--work-root` | `~/Library/Application Support/CourtShot/work` |
| `COURTSHOT_FFMPEG`, `COURTSHOT_FFPROBE` | `--ffmpeg`, `--ffprobe` | found on `PATH` at startup, else `/opt/homebrew/bin` |
| `COURTSHOT_DEVICE` | `--device` | detect (`mps`); `cpu` forces the CPU |
| `COURTSHOT_FIT_WORKERS` | `--fit-workers` | the pipeline's default, `min(2, cpu_count // 2)` |
| `COURTSHOT_BALL_MODEL` | `--ball-model` | `<repo>/checkpoints/padel_tracknetv3.pt` |
| `COURTSHOT_POLL_SECONDS` | `--poll-seconds` | `5` (with ±20 % spread; slower while the server is unreachable) |
| `COURTSHOT_STAGE_WEIGHTS` | `--stage-weights` | `download=5,prepare=3,tracking=45,appearance=8,analysis=4,ball3d=20,render=10,upload=5` |

The stage weights turn the per-stage progress into the overall percentage the dashboard shows; set only
the ones to change, e.g. `tracking=55,ball3d=10`.

## Run as a LaunchAgent

`deploy/com.courtshot.worker.plist` is an example. launchd expands neither `~` nor `$HOME` and does not
see Homebrew's folder, so every path in it is absolute.

```sh
cp deploy/com.courtshot.worker.plist ~/Library/LaunchAgents/
# edit: home folder, clone location, server URL, token
chmod 600 ~/Library/LaunchAgents/com.courtshot.worker.plist   # it holds the token
mkdir -p ~/Library/Logs/CourtShot
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.courtshot.worker.plist
```

- Restart after a `git pull` and `uv sync`: `launchctl kickstart -k gui/$(id -u)/com.courtshot.worker`
- Stop and unload: `launchctl bootout gui/$(id -u)/com.courtshot.worker`
- Logs: `~/Library/Logs/CourtShot/worker.log`

`KeepAlive` restarts the worker whenever it exits, including after a failed startup check (every 30 s).

## Keeping the Mac awake

While it runs, the worker holds `caffeinate -i -w <its pid>`: the Mac does not go to sleep when idle, and
the assertion ends with the worker. It does **not** cover:

- **A closed laptop lid.** A MacBook sleeps when its lid closes, unless it has power, an external display
  and an input device attached. Leave the lid open.
- **A reboot.** A LaunchAgent runs inside a user's login session, so after a reboot (a power cut, an
  update) nothing runs until someone logs in. Turn on automatic login (System Settings, Users & Groups,
  "Automatically log in as"); macOS offers it only with FileVault turned off.
- Sleep chosen by hand from the Apple menu, or a battery running empty.

## What happens to a job

- `<work root>/match-<id>/` holds the video (`source.<ext>`), a corrected copy when one is needed
  (`prepared.mp4`), the pipeline's caches (`cache/`) and one `attempt-<job>-<n>/` folder with `job.json`,
  `progress.json`, `job.log` and the results (`out/`).
- The folder survives a failed attempt, so a retry on the same Mac resumes the download and reuses the
  tracking. It is deleted after a successful delivery, when the server answers that the match is gone,
  and after seven days without use.
- The job runs as `python -m courtshot.worker.job_main` in its own process group. A cancel on the
  dashboard reaches the Mac at its next heartbeat (every 15 s); the worker then kills the whole group
  (job, 3D-fit processes, ffmpeg): SIGTERM, then SIGKILL after 10 s.
- The video is copied, corrected, only when it has a variable frame rate, a rotation flag, a start time
  other than 0 or non-square pixels. A video outside the length limits, or whose frame size differs from
  the one the court was marked on, fails at once with the reason.
- Uploads resume from the server's offset and are retried with growing pauses while the heartbeats
  keep the job's lease; a network failure during delivery never repeats the processing.

## Troubleshooting

- **First step:** `uv run python -m courtshot.worker check`, then `~/Library/Logs/CourtShot/worker.log`.
- **A job failed:** the dashboard shows the reason and the end of the job log. The full log is
  `job.log` in the match's work folder, which stays until the job succeeds or the folder is a week old.
- **`ffmpeg: not found` under launchd:** set `COURTSHOT_FFMPEG` and `COURTSHOT_FFPROBE` to absolute paths.
- **`PyTorch ... sees no MPS`:** check that `uv sync` installed an arm64 Python (`uv run python -c "import
  platform; print(platform.machine())"` prints `arm64`). `COURTSHOT_DEVICE=cpu` runs on the CPU, slowly.
- **`claim failed: 401`:** the token is wrong or revoked; create a new one.
- **`claim failed` with connection errors:** the worker keeps polling, slower, and recovers by itself.
- **The Mac runs out of memory** (jobs killed, `the job was killed (Killed: 9)`): set
  `COURTSHOT_FIT_WORKERS=1` and close other applications. Shorter clips need less.
- **Downloads fail in `setup`** (DNS errors): run it again; finished downloads are kept. After a complete
  setup, `HF_HUB_OFFLINE=1` stops transformers from contacting the Hub at all.
- **Leftover processes** after a crash of the worker are killed the next time it starts.
- **`another worker is using <work root>`:** only one worker runs per work folder (the LaunchAgent counts).
  Stop the other one first (`launchctl bootout ...`), or give this one its own `--work-root`.

## Tests

```sh
cd inference
PYTHONPATH=src uv run --no-project --with pytest --with requests python -m pytest tests/worker -q
```

The worker tests need only `requests`, `pytest` and ffmpeg: they run against an in-memory stub of the
server and fake pipelines (`tests/worker/stub_server.py`, `tests/worker/fakes.py`).
