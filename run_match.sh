#!/bin/bash
# Full pipeline for one video: tracking -> report -> 3D replay.
#   cd ~/padex && bash run_match.sh final_f.mp4
# Needs <name>_calibration.json next to the video (court calibration).
# Results go to output/<name>/ : tracked video (players, ball and ball speed),
# report.html, scene3d.html.
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:?usage: bash run_match.sh video.mp4}"
STEM="${VIDEO%.*}"
NAME="$(basename "$STEM")"
OUT="output/$NAME"
mkdir -p logs "$OUT"
exec > >(tee -a "logs/run_$NAME.log") 2>&1
echo "=== start $(date) on $VIDEO"
# Keep a Mac awake during the run (caffeinate only exists on macOS).
if command -v caffeinate >/dev/null 2>&1; then caffeinate -dims -w $$ & fi

# Find uv: plain `uv` on macOS/Linux/Git Bash; under WSL the Windows build is only
# reachable as `uv.exe`.
UV="$(command -v uv || command -v uv.exe || true)"
[ -n "$UV" ] || { echo "uv not found - install it (https://docs.astral.sh/uv/) or run from Git Bash."; exit 1; }

[ -f "${STEM}_calibration.json" ] || { echo "No ${STEM}_calibration.json - calibrate the court first."; exit 1; }
BEST="${BALL_MODEL:-checkpoints/padel_tracknetv3.pt}"
[ -f "$BEST" ] || { echo "No $BEST - the fine-tuned ball model is missing."; exit 1; }

# Tracking is reused from ${STEM}_tracking_cache.pkl when it was made from this
# video, calibration, ball model and settings; otherwise it is redone (the old
# cache is kept as ..._tracking_cache.old.pkl). Delete the cache to force it.
# Values go in as arguments, not env vars: env vars don't cross from WSL to a
# Windows uv.exe. `python -u` replaces PYTHONUNBUFFERED for the same reason.
"$UV" run python -u - "$VIDEO" "$BEST" "$OUT/${NAME}_tracked.mp4" <<'EOF'
import logging, os, sys
from stable_ids import StablePadex
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
video, best, out_video = sys.argv[1:4]
p = StablePadex(video, use_tracknet_v3=True, ball_model_path=best)
r = p.run()
print("=== ball speed (the 3D ball fit takes a minute or two per minute of play)", flush=True)
speeds = p.ball_speeds(r)   # makes the 3D fit if there is no saved one
fit = os.path.splitext(video)[0] + "_tracking_cache_ball3d.pkl"
# The tracked video is up to date if it was made after the tracking and the 3D fit it shows.
fresh = (p.tracking_from_cache and os.path.exists(out_video)
         and (not os.path.exists(fit) or os.path.getmtime(out_video) >= os.path.getmtime(fit)))
if fresh:
    print(f"Tracking and 3D fit reused; keeping {out_video}")
else:
    p.export_video(r, out_video, ball_speeds=speeds)
EOF

echo "=== report"
"$UV" run python analyze_match.py --cache "${STEM}_tracking_cache.pkl" --out "$OUT"
echo "=== 3D replay"
"$UV" run python -u export_scene3d.py "${STEM}_tracking_cache.pkl" --video "$VIDEO" --out "$OUT/scene3d.html"
echo "=== done $(date)"
echo "Report:    $OUT/report.html"
echo "3D replay: $OUT/scene3d.html"