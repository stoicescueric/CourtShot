"""Print the tracker settings CourtShot actually uses. Run from the repository folder:
uv run python tools/check_tracker.py [video]
"""

import sys

import courtshot.tracking.player as player
from courtshot.io.video import VideoReader
from courtshot.tracking.player import YoloPersonTracker

print("courtshot code loaded from:", player.__file__)

tracker = YoloPersonTracker()
with VideoReader(sys.argv[1] if len(sys.argv) > 1 else "videos/match.mp4") as r:
    for _, _, frame in r.frames(end_frame=3):
        tracker.track(frame)

pred = tracker._model.predictor
t = pred.trackers[0]
print("tracker file:     ", pred.args.tracker)
print("detector conf:    ", pred.args.conf)
print("track_buffer:     ", t.args.track_buffer, "frames")
print("track_low_thresh: ", t.args.track_low_thresh)
print("new_track_thresh: ", t.args.new_track_thresh)
