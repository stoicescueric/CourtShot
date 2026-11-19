"""
Padex quickstart: one padel match video in, tracked video + report + 3D replay out.

Run from the padex folder:

    uv run python examples/quickstart.py path/to/match.mp4

The same from the command line:

    uv run python -m padex calibrate path/to/match.mp4
    uv run python -m padex run path/to/match.mp4
"""

import logging
import sys
from pathlib import Path

import padex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
video = Path(sys.argv[1] if len(sys.argv) > 1 else "videos/match.mp4")

# 1. Court calibration, once per camera position: a window opens, click the
#    court keypoints. It is saved next to the video as <name>_calibration.json
#    and found there on later runs.
if not video.with_name(video.stem + "_calibration.json").exists() and padex.interactive_calibrate(video) is None:
    sys.exit("Calibration cancelled.")

p = padex.Padex(video)

# 2. Everything in one go: writes output/<name>/<name>_tracked.mp4, report.html, scene3d.html
print(p.run())

# ... or step by step, to work with the data yourself:
analysis = p.analyse()  # tracking (cached next to the video) -> players, ball events, 3D flights
print(f"{analysis.fps:.0f} fps, {len(analysis.tracking.player_frames)} player frames")
if analysis.ball is not None:
    for shot in analysis.ball.shots[:10]:
        print(f"{shot.time_s:7.2f} s  {shot.player} ({shot.team})  landed at {shot.landing}")
