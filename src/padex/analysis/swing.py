"""
What the hitter's body does around a hit, read from the pose keypoints the
tracker already saved (17 COCO points per player per frame).

The ball track alone gets two things wrong often enough to matter:

  * a ball that only passes a player in the image (a far-side bounce seen
    "above the head" of a player standing at the near net) is taken for a hit
    by that player, and then for an overhead;
  * the moment of a real hit is found a few frames late, by which time the
    ball is somewhere else in the hitter's box, so "contact height" is off.

The body settles both. Around a real hit an arm moves fast; around an overhead
an arm is up. Neither needs the ball.

Everything is measured in torso lengths (shoulder line to hip line), so it
means the same for a player filling half the frame and one 60 pixels tall.

    swing = PoseTracks(stable).swing(player, hit_frame, fps)
    swing.speed       peak wrist speed relative to the hips, torso lengths / s
    swing.arm_raise   how far an arm got above its shoulder, torso lengths
    swing.overhead    True / False / None (not seen well enough to say no)
    swing.still       True when the player was seen the whole time and no arm
                      moved: whatever the ball did, this player did not hit it

Players without keypoints around the hit (far players in some videos) give
None; callers fall back on the ball-only rules.

The thresholds were set on two broadcast clips (about 100 hits) checked by
eye: every overhead there has arm_raise >= 0.62, every other shot <= 0.35;
players wrongly given a hit moved their wrists at <= 4.4, real hitters >= 5.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
IDX = {n: i for i, n in enumerate(NAMES)}
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = (
    IDX[n]
    for n in (
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    )
)


@dataclass
class PoseSettings:
    before_s: float = 0.4  # look this long before the hit the ball track found...
    after_s: float = 0.2  # ...and this long after (found hits run late, not early)
    min_conf: float = 0.3  # keypoints less sure than this are ignored
    min_torso_box: float = 0.2  # torso at least this share of the box height (bent or turned bodies)
    min_frames: int = 6  # fewer frames with a body in the window = no reading
    max_step: int = 2  # speed is taken across at most this many frames
    max_speed: float = 40.0  # torso lengths / s; faster is a keypoint jump, not an arm
    overhead_raise: float = 0.5  # arm_raise at or above this = an arm is up (overhead)
    full_view: float = 0.75  # body seen in this share of the window = fit to rule things out
    still_speed: float = 5.0  # no wrist faster than this (and no arm up) = not swinging


@dataclass
class Swing:
    frames: int  # frames of the window with a usable body
    coverage: float  # ...as a share of the window
    speed: float | None  # peak wrist speed relative to the hips, torso lengths / s
    arm_raise: float | None  # highest an arm got above its own shoulder, torso lengths
    overhead: bool | None  # an arm was up; None = no, but the body was not seen throughout
    still: bool  # seen throughout, no arm up, no wrist moving: not a hitter


class PoseTracks:
    """Keypoints of every player by frame, from a tracking result."""

    def __init__(self, tracking, settings: PoseSettings | None = None):
        self.s = settings or PoseSettings()
        self.kp: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, float]] = {}
        for pf in tracking.player_frames:
            if not pf.keypoints:
                continue
            xy = np.full((len(NAMES), 2), np.nan)
            conf = np.zeros(len(NAMES))
            for k in pf.keypoints:
                i = IDX.get(k.name)
                if i is not None:
                    xy[i] = (k.x, k.y)
                    conf[i] = k.confidence
            self.kp[(pf.player_id, pf.frame_id)] = (xy, conf, float(pf.bbox.y2 - pf.bbox.y1))

    def _body(self, player: str, frame: int):
        """(hip centre, torso length, xy, ok-mask) or None when the trunk isn't seen."""
        got = self.kp.get((player, frame))
        if got is None:
            return None
        xy, conf, box_h = got
        ok = conf >= self.s.min_conf
        if not (ok[L_SH] and ok[R_SH] and ok[L_HIP] and ok[R_HIP]):
            return None
        hips = (xy[L_HIP] + xy[R_HIP]) / 2
        torso = float(np.linalg.norm((xy[L_SH] + xy[R_SH]) / 2 - hips))
        torso = max(torso, self.s.min_torso_box * box_h, 1e-3)
        return hips, torso, xy, ok

    def swing(self, player: str, frame: float, fps: float) -> Swing | None:
        s = self.s
        f0 = int(round(frame - s.before_s * fps))
        f1 = int(round(frame + s.after_s * fps))
        bodies = {f: b for f in range(f0, f1 + 1) if (b := self._body(player, f)) is not None}
        if len(bodies) < s.min_frames:
            return None
        scale = float(np.median([b[1] for b in bodies.values()]))

        # Arm height: the wrist above its shoulder, or twice the elbow above
        # the shoulder (the wrist blurs away in a fast swing, the elbow stays).
        # The second-highest frame counts, so one frame of misplaced keypoints
        # doesn't raise an arm.
        lifts = []
        rel: dict[int, dict[int, np.ndarray]] = {L_WR: {}, R_WR: {}}
        for f, (hips, _, xy, ok) in bodies.items():
            lift = None
            for wr, el, sh in ((L_WR, L_EL, L_SH), (R_WR, R_EL, R_SH)):
                if ok[wr]:
                    rel[wr][f] = (xy[wr] - hips) / scale
                    lift = max(lift or -np.inf, (xy[sh][1] - xy[wr][1]) / scale)
                if ok[el]:
                    lift = max(lift or -np.inf, 2.0 * (xy[sh][1] - xy[el][1]) / scale)
            if lift is not None:
                lifts.append(float(lift))
        arm_raise = sorted(lifts)[-2] if len(lifts) >= 2 else None

        speed = None
        for pts in rel.values():
            fr = sorted(pts)
            for a, b in zip(fr, fr[1:]):
                if b - a > s.max_step:
                    continue
                v = float(np.linalg.norm(pts[b] - pts[a]) * fps / (b - a))
                if v <= s.max_speed and (speed is None or v > speed):
                    speed = v

        coverage = len(bodies) / (f1 - f0 + 1)
        full = coverage >= s.full_view
        raised = arm_raise is not None and arm_raise >= s.overhead_raise
        # A raised arm is evidence whenever it is seen; "no arm was up" needs
        # the body seen throughout.
        overhead = True if raised else (False if full and arm_raise is not None else None)
        still = full and not raised and speed is not None and speed < s.still_speed
        return Swing(len(bodies), coverage, speed, arm_raise, overhead, still)
