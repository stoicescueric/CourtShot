#!/usr/bin/env bash
# Run the whole CourtShot system on this Mac for testing: server, dashboard, one worker.
# Usage: scripts/local.sh help
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
LOCAL_USER="${LOCAL_USER:-admin}"
DATA_DIR="${DATA_DIR:-$REPO/.courtshot-data}"
WORKER_NAME=local-mac
LAN_IP=""
if [[ "${LAN:-}" == 1 ]]; then
  HOST=0.0.0.0
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
  [[ -n "$LAN_IP" ]] || { echo "LAN=1: no IP on en0 or en1" >&2; exit 1; }
  PUBLIC_ORIGIN="http://$LAN_IP:$PORT"
else
  PUBLIC_ORIGIN="http://127.0.0.1:$PORT"
fi
# Where this script itself talks to the server.
BASE="http://127.0.0.1:$PORT"
LOGS="$DATA_DIR/logs"
TOKEN_FILE="$DATA_DIR/worker.token"

die() { echo "error: $*" >&2; exit 1; }

server_env() {
  export COURTSHOT_DATA_DIR="$DATA_DIR" COURTSHOT_SECURE_COOKIES=false COURTSHOT_PUBLIC_ORIGIN="$PUBLIC_ORIGIN"
}

retry3() {
  local i
  for i in 1 2 3; do "$@" && return 0; echo "failed (try $i of 3): $*" >&2; sleep 3; done
  return 1
}

app_cli() { (cd "$REPO/server" && server_env && uv run python -m app.cli "$@"); }

cmd_setup() {
  local skip_models=0 a
  for a in "$@"; do [[ "$a" == --skip-models ]] && skip_models=1; done
  local t
  for t in uv npm ffmpeg ffprobe; do command -v "$t" >/dev/null || die "$t not found"; done
  mkdir -p "$LOGS" "$DATA_DIR/worker-work"
  chmod 700 "$DATA_DIR"

  echo "== uv sync"
  (cd "$REPO/server" && retry3 uv sync)
  (cd "$REPO/inference" && retry3 uv sync)
  echo "== dashboard"
  (cd "$REPO/server/web" && retry3 npm ci && npm run build)
  echo "== database"
  app_cli migrate

  echo "== user $LOCAL_USER"
  local pw="${LOCAL_PASSWORD:-}" out
  if [[ -z "$pw" ]]; then
    read -r -s -p "Password for $LOCAL_USER (min 10 chars, used only if the user is new): " pw; echo
  fi
  if out="$(printf '%s\n' "$pw" | app_cli create-user "$LOCAL_USER" --password-stdin 2>&1)"; then
    echo "$out"
  elif [[ "$out" == *exists* ]]; then
    echo "user $LOCAL_USER already exists"
  else
    echo "$out" >&2; die "create-user failed"
  fi

  echo "== worker $WORKER_NAME"
  if [[ -s "$TOKEN_FILE" ]]; then
    echo "token file exists, keeping it"
  else
    # a worker without a token file (lost file): replace it
    app_cli list-workers 2>/dev/null | grep -q "^$WORKER_NAME[[:space:]]" && app_cli revoke-worker "$WORKER_NAME" >/dev/null
    out="$(app_cli create-worker "$WORKER_NAME")" || die "create-worker failed"
    (umask 077; printf '%s\n' "$out" | tail -n 1 > "$TOKEN_FILE")
    chmod 600 "$TOKEN_FILE"
    echo "token saved to $TOKEN_FILE"
  fi

  if (( skip_models )); then echo "== models skipped"; else
    echo "== worker setup (model downloads)"
    (cd "$REPO/inference" && uv run python -m courtshot.worker setup)
  fi
  echo "Setup done. Next: scripts/local.sh up"
}

cmd_server() {
  server_env
  cd "$REPO/server"
  exec uv run uvicorn app.main:app --host "$HOST" --port "$PORT"
}

cmd_worker() {
  [[ -s "$TOKEN_FILE" ]] || die "no $TOKEN_FILE: run setup first"
  export COURTSHOT_SERVER_URL="$BASE"
  COURTSHOT_WORKER_TOKEN="$(cat "$TOKEN_FILE")"; export COURTSHOT_WORKER_TOKEN
  export COURTSHOT_WORK_ROOT="$DATA_DIR/worker-work"
  mkdir -p "$COURTSHOT_WORK_ROOT"
  cd "$REPO/inference"
  exec uv run python -m courtshot.worker run
}

server_up() { [[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/auth/me" || true)" == 401 ]]; }

print_lan_help() {
  [[ -n "$LAN_IP" ]] || return 0
  echo "LAN mode: the browser must use exactly $PUBLIC_ORIGIN (not localhost)."
  echo "Worker on another Mac (token: $TOKEN_FILE):"
  echo "  COURTSHOT_SERVER_URL=$PUBLIC_ORIGIN COURTSHOT_WORKER_TOKEN=<token> uv run python -m courtshot.worker run   (in inference/)"
}

cmd_up() {
  mkdir -p "$LOGS"
  [[ -s "$TOKEN_FILE" ]] || die "no $TOKEN_FILE: run setup first"
  server_up && die "something already answers on port $PORT"
  set -m   # each background job gets its own process group
  "$0" server >"$LOGS/server.log" 2>&1 & local spid=$!
  local wpid=""
  trap 'echo; echo "stopping..."; for p in $wpid $spid; do kill -TERM -- -"$p" 2>/dev/null || true; done; wait 2>/dev/null || true; exit 0' INT TERM
  local i
  for i in $(seq 1 60); do
    server_up && break
    kill -0 "$spid" 2>/dev/null || { tail -n 20 "$LOGS/server.log" >&2; die "server exited"; }
    sleep 1
  done
  server_up || die "server did not answer in 60 s"
  "$0" worker >"$LOGS/worker.log" 2>&1 & wpid=$!
  echo "Dashboard: $PUBLIC_ORIGIN   (sign in as $LOCAL_USER)"
  echo "Logs: $LOGS/server.log  $LOGS/worker.log"
  print_lan_help
  echo "Ctrl-C stops both."
  wait
}

ask_password() {
  if [[ -z "${LOCAL_PASSWORD:-}" ]]; then
    read -r -s -p "Password for $LOCAL_USER: " LOCAL_PASSWORD; echo
  fi
}

cmd_seed() {
  local video="$REPO/videos/final_f.mp4" calib="$REPO/videos/final_f_calibration.json" secs="" title="" pos=()
  while (( $# )); do
    case "$1" in
      --seconds) secs="${2:?--seconds needs a number}"; shift 2 ;;
      --title) title="${2:?--title needs a value}"; shift 2 ;;
      *) pos+=("$1"); shift ;;
    esac
  done
  (( ${#pos[@]} > 0 )) && video="${pos[0]}"
  (( ${#pos[@]} > 1 )) && calib="${pos[1]}"
  [[ -f "$video" ]] || die "no video $video"
  [[ -f "$calib" ]] || die "no calibration $calib"
  server_up || die "the server does not answer on $BASE"
  ask_password

  local tmp; tmp="$(mktemp -d)"
  trap "rm -rf '$tmp'" EXIT
  local name; name="$(basename "$video")"
  [[ -n "$title" ]] || title="seed ${name%.*}${secs:+ (${secs}s)}"
  if [[ -n "$secs" ]]; then
    local cut="$tmp/$name" dur
    ffmpeg -v error -y -ss 0 -i "$video" -t "$secs" -c copy "$cut"
    dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$cut")"
    if ! python3 -c "import sys; d=float(sys.argv[1]); s=float(sys.argv[2]); sys.exit(0 if abs(d-s)<=1.5 else 1)" "$dur" "$secs"; then
      echo "stream copy gave ${dur}s, re-encoding"
      ffmpeg -v error -y -i "$video" -t "$secs" -c:v libx264 -preset veryfast -crf 18 -an "$cut"
    fi
    video="$cut"
  fi

  local meta
  meta="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height:format=duration -of json "$video")"
  local jar="$tmp/jar" body
  curl -sf -c "$jar" -H "Origin: $PUBLIC_ORIGIN" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"username":sys.argv[1],"password":sys.argv[2]}))' "$LOCAL_USER" "$LOCAL_PASSWORD")" \
    "$BASE/api/auth/login" >/dev/null || die "sign-in failed"
  local chunk
  chunk="$(curl -sf -b "$jar" "$BASE/api/config" | python3 -c 'import json,sys; print(json.load(sys.stdin)["chunk_bytes"])')"

  body="$(python3 - "$video" "$title" "$meta" <<'PY'
import hashlib, json, os, sys
path, title, meta = sys.argv[1:4]
meta = json.loads(meta)
size = os.path.getsize(path)
h = hashlib.sha256()
with open(path, "rb") as f:
    if size < 2 * 2**20:
        h.update(f.read())
    else:
        h.update(f.read(2**20)); f.seek(size - 2**20); h.update(f.read())
s = meta["streams"][0]
print(json.dumps({"title": title, "filename": os.path.basename(path), "size_bytes": size,
  "fingerprint": h.hexdigest(), "duration_s": float(meta["format"]["duration"]),
  "frame_width": s["width"], "frame_height": s["height"]}))
PY
)"
  local id
  id="$(curl -sf -b "$jar" -H "Origin: $PUBLIC_ORIGIN" -H 'Content-Type: application/json' -d "$body" "$BASE/api/matches" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')" || die "creating the match failed (curl -v for details)"

  # Chunk files from split, not tail | head: with pipefail, head closing the pipe early kills tail with
  # SIGPIPE and fails the upload of any video larger than one chunk.
  local off=0 part
  split -b "$chunk" "$video" "$tmp/part."
  for part in "$tmp"/part.*; do
    curl -sf -b "$jar" -X PATCH -H "Origin: $PUBLIC_ORIGIN" -H 'Content-Type: application/octet-stream' \
      --data-binary @"$part" "$BASE/api/matches/$id/upload?offset=$off" >/dev/null || die "upload failed at offset $off"
    off=$(( off + $(wc -c <"$part") ))
  done

  python3 - "$calib" "$meta" >"$tmp/cal.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); s = json.loads(sys.argv[2])["streams"][0]
print(json.dumps({"points": c["labeled_keypoints"], "frame_width": s["width"], "frame_height": s["height"]}))
PY
  curl -sf -b "$jar" -X PUT -H "Origin: $PUBLIC_ORIGIN" -H 'Content-Type: application/json' --data-binary @"$tmp/cal.json" \
    "$BASE/api/matches/$id/calibration" >/dev/null || die "calibration rejected (curl -v for details)"
  echo "Match $id created: $PUBLIC_ORIGIN/matches/$id"
}

cmd_status() {
  if server_up; then echo "server: up on $BASE"; else echo "server: not answering on $BASE"; fi
  if server_up && [[ -n "${LOCAL_PASSWORD:-}" ]]; then
    local jar; jar="$(mktemp)"
    curl -sf -c "$jar" -H "Origin: $PUBLIC_ORIGIN" -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"username":sys.argv[1],"password":sys.argv[2]}))' "$LOCAL_USER" "$LOCAL_PASSWORD")" \
      "$BASE/api/auth/login" >/dev/null && curl -sf -b "$jar" "$BASE/api/workers" | python3 -m json.tool || echo "(could not read workers)"
    rm -f "$jar"
  else
    echo "workers: set LOCAL_PASSWORD to list them"
  fi
  local f
  for f in server worker; do
    [[ -f "$LOGS/$f.log" ]] && { echo "--- $f.log"; tail -n 8 "$LOGS/$f.log"; }
  done
  return 0
}

cmd_test() {
  (cd "$REPO/server" && uv run pytest -q)
  (cd "$REPO/server/web" && npm test)
  (cd "$REPO/inference" && uv run pytest -q)
  (cd "$REPO/inference" && PYTHONPATH=src uv run --no-project --with pytest --with requests python -m pytest tests/worker -q)
}

cmd_reset() {
  echo "This deletes $DATA_DIR (database, uploads, worker token, logs)."
  local a; read -r -p "Type 'delete' to confirm: " a
  [[ "$a" == delete ]] || die "not confirmed"
  rm -rf "$DATA_DIR"; echo "deleted"
}

cmd_help() {
  cat <<H
scripts/local.sh <command>   (from any directory)

  setup [--skip-models]   install, build the dashboard, migrate, create user and worker token, download models
  up                      server + worker in the background, logs in .courtshot-data/logs; Ctrl-C stops both
  server | worker         one of them in the foreground
  seed [video] [calib.json] [--seconds N] [--title T]
                          create a match through the API (default videos/final_f.mp4 + its calibration);
                          needs LOCAL_PASSWORD or asks. --seconds 30 takes about 10 min on this M1, full clip 30.
  status                  server, workers (needs LOCAL_PASSWORD), log tails
  test                    all four test suites
  reset                   delete the data folder after confirmation

Settings (environment): PORT=8000 HOST=127.0.0.1 LOCAL_USER=admin LOCAL_PASSWORD DATA_DIR=<repo>/.courtshot-data
  LAN=1  bind 0.0.0.0 and print the URL for browsers and other Macs' workers.

Testing flow: setup, up, open the URL and sign in, then 'seed --seconds 30' or upload and click the court in the browser.
H
}

main() {
  local c="${1:-help}"; shift || true
  case "$c" in
    setup|server|worker|up|seed|status|test|reset|help) "cmd_$c" "$@" ;;
    *) cmd_help; exit 1 ;;
  esac
}

# Sourcing the file only defines the functions (used for testing).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
