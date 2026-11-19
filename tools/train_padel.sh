#!/bin/bash
# Fine-tune the TrackNetV3 ball detector on PadelTracker100, then re-run Padex
# on videos/match.mp4 with the new model and rebuild the report.
#
# Run once from Terminal:   cd ~/padex && bash tools/train_padel.sh
# Everything is logged to logs/train_padel.log. Safe to re-run: finished
# steps (conversion, preparation) are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs output
exec > >(tee -a logs/train_padel.log) 2>&1
echo "=== start $(date)"

# Keep the Mac awake until this script ends.
caffeinate -dims -w $$ &

uv run python -c "import torch; print('torch', torch.__version__, '| Apple GPU (MPS):', torch.backends.mps.is_available())"

echo "=== 1/5 convert PadelTracker100 $(date)"
uv run python tools/convert_padeltracker100.py --src "$HOME/Downloads/padel-data-labels" --out data

echo "=== 2/5 prepare training frames $(date)"
uv run python tools/train_tracknetv3.py prepare --data data --cache cache

# Validation: the last 6 clips of the women's final (frames 30000-45933, about
# 9 minutes of play the model never trains on). Everything else is training data.
VAL=$(ls cache | grep -E 'FinalF.*_01[0-5]$' | paste -sd, - || true)
if [ -z "$VAL" ]; then VAL=$(ls cache | tail -4 | paste -sd, -); fi
echo "validation clips: $VAL"

echo "=== 3/5 baseline: badminton weights on padel $(date)"
uv run python tools/train_tracknetv3.py evaluate --cache cache --weights ckpts/TrackNet_best.pt \
    --val "$VAL" --batch-size 4 --workers 2 --val-stride 8

echo "=== 4/5 fine-tune $(date)"
uv run python tools/train_tracknetv3.py train --cache cache --weights ckpts/TrackNet_best.pt \
    --val "$VAL" --epochs 8 --batch-size 4 --workers 2 --stride 8 --val-stride 8 \
    --out checkpoints/padel_tracknetv3.pt

BEST=checkpoints/padel_tracknetv3.pt
[ -f "$BEST" ] || BEST=checkpoints/padel_tracknetv3_last.pt

echo "=== 5/5 re-run Padex on videos/match.mp4 with $BEST $(date)"
# The tracking cache made with the old ball model is set aside by Padex itself
# (as videos/match_tracking_cache.old.pkl), because the model is part of its key.
uv run python -u -m padex run videos/match.mp4 --ball-model "$BEST" --out output/match_newball

echo "=== done $(date)"
echo "Report: output/match_newball/report.html   Video: output/match_newball/match_tracked.mp4"
