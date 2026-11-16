"""Print the tracker settings Padex actually uses. Run from the padex folder:
    uv run python check_tracker.py
"""
import padex.tracking.player as player
from padex.io.video import VideoReader
from padex.tracking.player import YoloPlayerDetectionStrategy

print("padex code loaded from:", player.__file__)

s = YoloPlayerDetectionStrategy()
with VideoReader("match.mp4") as r:
    for frame_id, ts, frame in r.frames(end_frame=3):
        s.detect_with_tracking(frame, frame_id, ts)

pred = s._model.predictor
t = pred.trackers[0]
print("tracker file:     ", pred.args.tracker)
print("detector conf:    ", pred.args.conf)
print("track_buffer:     ", t.args.track_buffer, "frames")
print("track_low_thresh: ", t.args.track_low_thresh)
print("new_track_thresh: ", t.args.new_track_thresh)