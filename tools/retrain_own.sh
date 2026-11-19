#!/bin/bash
# Fine-tune the padel ball model further on your own labelled stretches.
#   cd ~/padex && bash tools/retrain_own.sh match          (clips data/match_01.., data/match_val)
#
# Starts from checkpoints/padel_tracknetv3.pt and writes a NEW model,
# checkpoints/padel_tracknetv3_own.pt - the current one is not touched.
# Trains on your clips plus 4 PadelTracker100 clips (so it doesn't forget),
# checks on your held-out clip and one PadelTracker100 clip it never trained on.
set -euo pipefail
cd "$(dirname "$0")/.."
PREFIX="${1:?usage: bash tools/retrain_own.sh <clip prefix, e.g. match>}"
mkdir -p logs
exec > >(tee -a "logs/retrain_$PREFIX.log") 2>&1
echo "=== start $(date)"
caffeinate -dims -w $$ &

ls -d data/"$PREFIX"_* >/dev/null 2>&1 || { echo "No data/${PREFIX}_* clips - label first (tools/find_gaps.py)."; exit 1; }
[ -d "data/${PREFIX}_val" ] || { echo "No data/${PREFIX}_val - label the held-out stretch too."; exit 1; }

echo "=== 1/4 prepare frames"
uv run python tools/train_tracknetv3.py prepare --data data --cache cache

# A small training set: your clips + a few PadelTracker100 clips, linked in.
OWN=cache_own_$PREFIX
rm -rf "$OWN" && mkdir -p "$OWN"
for d in cache/"$PREFIX"_*; do ln -s "../$d" "$OWN/$(basename "$d")"; done
for c in 000 003 006 009; do
    [ -d "cache/ptk_2022_BCN_FinalF_1_$c" ] && ln -s "../cache/ptk_2022_BCN_FinalF_1_$c" "$OWN/"
done
[ -d cache/ptk_2022_BCN_FinalF_1_012 ] && ln -s ../cache/ptk_2022_BCN_FinalF_1_012 "$OWN/"
VAL="${PREFIX}_val"
[ -d "$OWN/ptk_2022_BCN_FinalF_1_012" ] && VAL="$VAL,ptk_2022_BCN_FinalF_1_012"
echo "training on: $(ls "$OWN" | grep -v -E "_val$|_012$" | paste -sd' ' -)"
echo "checking on: $VAL"

echo "=== 2/4 current model on the check clips"
uv run python tools/train_tracknetv3.py evaluate --cache "$OWN" --weights checkpoints/padel_tracknetv3.pt \
    --val "$VAL" --batch-size 4 --workers 2 --val-stride 2

echo "=== 3/4 fine-tune"
uv run python tools/train_tracknetv3.py train --cache "$OWN" --weights checkpoints/padel_tracknetv3.pt \
    --val "$VAL" --epochs 6 --lr 3e-5 --batch-size 4 --workers 2 --stride 4 --val-stride 2 \
    --out checkpoints/padel_tracknetv3_own.pt

echo "=== 4/4 new model on the check clips"
uv run python tools/train_tracknetv3.py evaluate --cache "$OWN" --weights checkpoints/padel_tracknetv3_own.pt \
    --val "$VAL" --batch-size 4 --workers 2 --val-stride 2
echo "=== done $(date)"
echo "Compare the F1 in step 2 and step 4. If the new one is better on ${PREFIX}_val and not worse on the"
echo "PadelTracker100 clip, re-run your video with it:"
echo "    mv videos/${PREFIX}_tracking_cache.pkl videos/${PREFIX}_tracking_cache_v1.pkl"
echo "    BALL_MODEL=checkpoints/padel_tracknetv3_own.pt bash run_match.sh videos/${PREFIX}.mp4"