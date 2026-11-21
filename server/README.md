# CourtShot server

The web side of CourtShot: the API, the job queue for the lab Macs, and the hosting of the
dashboard (`web/`). Design: `docs/superpowers/specs/2026-11-20-server-inference-design.md`.
HTTP contract: `API.md`. The server has no CV or ML dependency; the pipeline runs on the Macs
(`inference/`).

```
app/            FastAPI application
  main.py       app, lifespan, error handlers
  settings.py   COURTSHOT_* environment variables
  db.py         SQLite: write engine (BEGIN IMMEDIATE) and read engine
  models.py     tables
  storage.py    every path and file read/write
  auth.py       passwords, sessions, worker tokens, sign-in throttle
  jobs.py       the queue: claim, heartbeat, complete, fail, cancel, lease expiry
  matches.py    match changes: create, upload, calibrate, run again, delete
  views.py      response shapes and the match status
  sweeps.py     background lease sweep (5 s) and cleanup (10 min)
  web.py        security headers, dashboard files
  cli.py        administration commands
migrations/     Alembic
deploy/         nginx, systemd and environment examples
tests/
```

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11 or later.

```sh
cd server
uv sync
export COURTSHOT_DATA_DIR=$PWD/../.courtshot-data   # the database and every uploaded file
uv run python -m app.cli migrate                    # create or upgrade the database
uv run python -m app.cli create-user kerem          # asks for the password twice
uv run python -m app.cli create-worker lab-mac-1    # prints the worker's token once
```

The app never migrates by itself; it refuses to start on a database that is not migrated.
Run `migrate` after every update.

Other commands: `set-password <user>` (also signs that user out everywhere),
`revoke-worker <name>`, `list-workers`. `create-user` and `set-password` take
`--password-stdin` to read the password from one line of standard input. To replace a
worker's token, revoke it and run `create-worker` with the same name again.

## Run for development

```sh
export COURTSHOT_DATA_DIR=$PWD/../.courtshot-data
export COURTSHOT_SECURE_COOKIES=false                  # plain http on localhost
export COURTSHOT_PUBLIC_ORIGIN=http://localhost:5173   # the Vite dev server's origin
uv run uvicorn app.main:app --port 8000
```

`COURTSHOT_PUBLIC_ORIGIN` must equal the `Origin` header the browser sends: every request
that is not a GET is refused otherwise. With the Vite dev server proxying `/api` to port
8000 that is `http://localhost:5173`; with the built dashboard served by this app
(`cd web && npm ci && npm run build`) it is `http://localhost:8000`.

Tests:

```sh
cd server && uv run pytest -q
```

## Settings

All from environment variables prefixed `COURTSHOT_` (see `deploy/server.env.example`):
`DATA_DIR` (required), `PUBLIC_ORIGIN`, `MIN_VIDEO_SECONDS`, `MAX_VIDEO_SECONDS`,
`MAX_UPLOAD_BYTES`, `CHUNK_BYTES`, `MAX_CHUNK_BYTES`, `CALIBRATION_MAX_ERROR_M`,
`MIN_CALIBRATION_POINTS`, `LEASE_SECONDS`, `MAX_ATTEMPTS`, `RETRY_DELAY_SECONDS`,
`HEARTBEAT_SECONDS`, `WORKER_ONLINE_SECONDS`, `FREE_SPACE_RESERVE_BYTES`, `TRUSTED_PROXY`,
`SECURE_COOKIES`, `INCOMPLETE_UPLOAD_DAYS`, `WEB_DIST`.

## Deploy on an Azure VM

One Linux VM (Ubuntu LTS) with a managed data disk; nginx in front for HTTPS.

1. Attach a data disk, format it and mount it at `/var/lib/courtshot` (add it to
   `/etc/fstab`). The database, the uploads and the results all live there.
2. In the VM's network security group, open ports 80 and 443 only (and SSH from your
   address). The Macs only make outbound requests; they need no inbound rule.
3. Install the packages and a service user:
   ```sh
   sudo apt install nginx sqlite3 certbot python3-certbot-nginx
   curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh
   sudo useradd --system --home /var/lib/courtshot courtshot
   sudo chown courtshot: /var/lib/courtshot
   ```
4. Clone the repository to `/opt/courtshot`, then:
   ```sh
   cd /opt/courtshot/server
   sudo -u courtshot uv sync --frozen --no-dev
   (cd web && npm ci && npm run build)       # or copy a built web/dist
   ```
5. Settings: copy `deploy/server.env.example` to `/etc/courtshot/server.env`, set
   `COURTSHOT_PUBLIC_ORIGIN` to the site's `https://` address, keep
   `COURTSHOT_TRUSTED_PROXY=127.0.0.1`, and `chmod 640` it.
6. Database and accounts, as the service user with the same environment:
   ```sh
   sudo -u courtshot bash -c 'set -a; . /etc/courtshot/server.env; .venv/bin/python -m app.cli migrate'
   sudo -u courtshot bash -c 'set -a; . /etc/courtshot/server.env; .venv/bin/python -m app.cli create-user kerem'
   sudo -u courtshot bash -c 'set -a; . /etc/courtshot/server.env; .venv/bin/python -m app.cli create-worker lab-mac-1'
   ```
7. Service: copy `deploy/courtshot.service` to `/etc/systemd/system/`, then
   `sudo systemctl daemon-reload && sudo systemctl enable --now courtshot`.
8. Proxy: copy `deploy/nginx.conf` to `/etc/nginx/sites-available/courtshot`, set
   `server_name`, link it into `sites-enabled`, run `sudo certbot --nginx -d <name>`, then
   `sudo nginx -t && sudo systemctl reload nginx`.

Update: `git pull`, `uv sync --frozen --no-dev`, rebuild `web/dist`, run `migrate`, then
`sudo systemctl restart courtshot`.

### The single-process rule

Run exactly one server process: `--workers 1`, no gunicorn pre-fork, no second copy, no
blue-green overlap during a deploy (`systemctl restart` stops the old process first). The
lease sweep, the cleanup sweep and the sign-in throttle live in the process; a second one
would run the sweeps twice and split the throttle's counts. SQLite stays safe either way,
but this rule is what the design relies on.

### Backup

Use SQLite's online backup, which is consistent while the server runs:

```sh
sqlite3 /var/lib/courtshot/app.db ".backup '/var/backups/courtshot/app-$(date +%F).db'"
```

Copy `/var/lib/courtshot/matches/` with rsync after the database backup (files without a row
are removed by the next cleanup sweep after a restore). An Azure disk snapshot is a second
line of defence, not the first: a snapshot taken mid-write can hold a torn WAL.

## Notes

- Request bodies: video and result chunks are read against `MAX_CHUNK_BYTES` by the app;
  JSON bodies are capped by nginx's `client_max_body_size`.
- Files: every file route needs a session or a worker token; result files are served from
  `runs/<job id>-<attempt>/` of a succeeded job only, with content types from a fixed list.
- Cleanup every 10 minutes: match folders without a row, run folders that are neither the
  current results nor the running attempt, uploads still incomplete after
  `INCOMPLETE_UPLOAD_DAYS`, expired sessions.
