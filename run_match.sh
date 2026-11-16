#!/bin/bash
# Full pipeline for one video: tracking -> report -> 3D replay.
#   cd ~/padex && bash run_match.sh final_f.mp4
# Needs <name>_calibration.json next to the video (court calibration).
# Results go to output/<name>/ : tracked video, report.html, scene3d.html.
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:?usage: bash run_match.sh video.mp4}"
STEM="${VIDEO%.*}"
NAME="$(basename "$STEM")"
OUT="output/$NAME"
mkdir -p logs "$OUT"
exec > >(tee -a "logs/run_$NAME.log") 2>&1
echo "=== start $(date) on $VIDEO"
caffeinate -dims -w $$ &

[ -f "${STEM}_calibration.json" ] || { echo "No ${STEM}_calibration.json - calibrate the court first."; exit 1; }
BEST="${BALL_MODEL:-checkpoints/padel_tracknetv3.pt}"
[ -f "$BEST" ] || { echo "No $BEST - the fine-tuned ball model is missing."; exit 1; }

if [ -f "${STEM}_tracking_cache.pkl" ]; then
    echo "Reusing ${STEM}_tracking_cache.pkl (delete it to re-track)"
else
    VIDEO="$VIDEO" BEST="$BEST" OUT="$OUT" NAME="$NAME" PYTHONUNBUFFERED=1 uv run python - <<'EOF'
import logging, os
from stable_ids import StablePadex
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
p = StablePadex(os.environ["VIDEO"], use_tracknet_v3=True, ball_model_path=os.environ["BEST"])
r = p.run()
p.export_video(r, f"{os.environ['OUT']}/{os.environ['NAME']}_tracked.mp4")
EOF
fi

uv run python analyze_match.py --cache "${STEM}_tracking_cache.pkl" --out "$OUT"
echo "=== 3D replay (the ball fit takes ~10 min per minute of play)"
PYTHONUNBUFFERED=1 uv run python export_scene3d.py "${STEM}_tracking_cache.pkl" --video "$VIDEO" --out "$OUT/scene3d.html"
echo "=== done $(date)"
echo "Report:    $OUT/report.html"
echo "3D replay: $OUT/scene3d.html"