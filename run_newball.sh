#!/bin/bash
# Re-run Padex on a match with the fine-tuned ball model and rebuild the report.
#   cd ~/padex && bash run_newball.sh            (uses match.mp4)
#   bash run_newball.sh other_match.mp4          (any video; calibrate it first)
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:-match.mp4}"
STEM="${VIDEO%.*}"
mkdir -p logs output
exec > >(tee -a logs/run_newball.log) 2>&1
echo "=== start $(date) on $VIDEO"
caffeinate -dims -w $$ &

BEST=checkpoints/padel_tracknetv3.pt
[ -f "$BEST" ] || { echo "No $BEST yet - train first."; exit 1; }

# Tracking results are cached per video; keep the old one so we can compare.
if [ -f "${STEM}_tracking_cache.pkl" ]; then
    mv -f "${STEM}_tracking_cache.pkl" "${STEM}_tracking_cache_oldball.pkl"
fi

VIDEO="$VIDEO" BEST="$BEST" PYTHONUNBUFFERED=1 uv run python - <<'EOF'
import logging, os
from stable_ids import StablePadex
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
video = os.environ["VIDEO"]
p = StablePadex(video, use_tracknet_v3=True, ball_model_path=os.environ["BEST"])
r = p.run()
p.export_video(r, f"output/{os.path.splitext(os.path.basename(video))[0]}_newball.mp4")
EOF

uv run python analyze_match.py --cache "${STEM}_tracking_cache.pkl" --out "output/analysis_newball"
echo "=== done $(date)"
echo "Report: output/analysis_newball/report.html"
