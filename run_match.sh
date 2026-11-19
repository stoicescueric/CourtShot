#!/bin/bash
# Full pipeline for one video: tracking -> tracked video -> report -> 3D replay.
#   bash run_match.sh videos/final_f.mp4
# Needs <name>_calibration.json next to the video (uv run python -m courtshot calibrate <video>).
# Results go to output/<name>/ : <name>_tracked.mp4, report.html, scene3d.html.
# Another ball model: BALL_MODEL=checkpoints/other.pt bash run_match.sh video.mp4
#
# This is `python -m courtshot run` plus a log file (logs/run_<name>.log).
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:?usage: bash run_match.sh video.mp4}"
NAME="$(basename "${VIDEO%.*}")"
mkdir -p logs
exec > >(tee -a "logs/run_$NAME.log") 2>&1
echo "=== start $(date) on $VIDEO"
# Keep a Mac awake during the run (caffeinate only exists on macOS).
if command -v caffeinate >/dev/null 2>&1; then caffeinate -dims -w $$ & fi

# Find uv: plain `uv` on macOS/Linux/Git Bash; under WSL the Windows build is only
# reachable as `uv.exe`.
UV="$(command -v uv || command -v uv.exe || true)"
[ -n "$UV" ] || { echo "uv not found - install it (https://docs.astral.sh/uv/) or run from Git Bash."; exit 1; }

# Values go in as arguments, not env vars: env vars don't cross from WSL to a
# Windows uv.exe. `python -u` replaces PYTHONUNBUFFERED for the same reason.
"$UV" run python -u -m courtshot run "$VIDEO" --ball-model "${BALL_MODEL:-checkpoints/padel_tracknetv3.pt}"
echo "=== done $(date)"
