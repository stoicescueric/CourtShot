"""Find where the ball detector misses the ball during play, and print the
label_ball.py commands to label exactly those stretches.

    uv run python find_gaps.py match.mp4
    uv run python find_gaps.py match.mp4 --windows 8 --seconds 6

It reads <video>_tracking_cache.pkl, finds the rallies, scores every window by
how many frames inside play have no ball detection, and picks the worst
non-overlapping windows. The last window is held out for checking the new
model (label it, but it is never trained on).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from stable_ids import stabilize_tracking
import ball_track as bt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--cache", type=Path, default=None, help="default: <video>_tracking_cache.pkl")
    ap.add_argument("--windows", type=int, default=8, help="how many stretches to label (the last is held out)")
    ap.add_argument("--seconds", type=float, default=6.0, help="length of each stretch")
    ap.add_argument("--prefix", default=None, help="clip name prefix (default: video name)")
    a = ap.parse_args()

    cache = a.cache or a.video.with_name(a.video.stem + "_tracking_cache.pkl")
    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    raw = pickle.load(open(cache, "rb"))
    stable, _ = stabilize_tracking(raw)
    res = bt.run_on_tracking(stable, fps)

    det = np.zeros(n, bool)
    for bf in raw.ball_frames:
        if bf.bbox is not None and bf.frame_id < n:
            det[bf.frame_id] = True
    play = np.zeros(n, bool)
    for s0, s1, _ in res.rallies:
        play[int(s0 * fps):int(s1 * fps) + 1] = True
    if not play.any():
        play[:] = True
    miss = play & ~det
    print(f"{a.video.name}: {n} frames at {fps:.2f} fps; in play {play.mean():.0%} of the time; "
          f"ball missing in {miss.sum() / max(1, play.sum()):.0%} of in-play frames")

    W = int(round(a.seconds * fps))
    score = np.convolve(miss.astype(int), np.ones(W, int), "valid")      # score[i] = misses in [i, i+W)
    taken = np.zeros(n, bool)
    picks = []
    for i in np.argsort(score)[::-1]:
        if len(picks) == a.windows or score[i] == 0:
            break
        if taken[max(0, i - W // 4):i + W + W // 4].any():
            continue
        picks.append(int(i))
        taken[i:i + W] = True
    picks.sort()
    if not picks:
        print("No missing in-play detections found - nothing to label.")
        return

    prefix = a.prefix or a.video.stem
    # hold out the window with the median score, so the check is neither the easiest nor the hardest
    held = sorted(picks, key=lambda i: score[i])[len(picks) // 2] if len(picks) > 1 else None
    print(f"\nLabel these {len(picks)} stretches ({a.seconds:.0f} s each, about {W} frames). "
          "Enter accepts Padex's suggestion, so most frames take one key press.\n")
    k = 0
    for i in picks:
        name = f"{prefix}_val" if i == held else f"{prefix}_{k + 1:02d}"
        k += i != held
        t0 = i / fps
        print(f"# {int(t0 // 60)}:{t0 % 60:05.2f}-{int((t0 + a.seconds) // 60)}:{(t0 + a.seconds) % 60:05.2f}  "
              f"ball missing in {score[i]} of {W} frames" + ("   <- held out for checking" if i == held else ""))
        print(f"uv run python label_ball.py {a.video} --start {i} --end {i + W} --name {name} --suggest {cache}\n")
    print("Then: bash retrain_own.sh " + prefix)


if __name__ == "__main__":
    main()