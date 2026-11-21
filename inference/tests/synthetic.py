"""
A synthetic match for tests that need no detector: four players running about
on court for a few seconds, seen through the camera of final_f's calibration
scaled down to 480 x 270; optionally one rally with three hits and their
fitted 3D flights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from courtshot.analysis import ball_flight
from courtshot.analysis.ball_events import BallEvent, BallResult, Shot
from courtshot.analysis.identities import stabilize_tracking
from courtshot.analysis.match import MatchAnalysis
from courtshot.court import checked_calibration, court_to_pixel
from courtshot.schemas.tracking import BoundingBox, CourtCalibration, PlayerFrame, Position2D
from courtshot.tracking.pipeline import TrackingResult

REPO = Path(__file__).resolve().parents[2]
FPS = 25.0
W, H = 480, 270
SCALE = W / 1920
SPOTS = [(3.0, 4.0), (7.0, 4.0), (3.0, 16.0), (7.0, 16.0)]  # near pair, far pair


def points() -> dict[str, list[float]]:
    """final_f's clicked points, scaled to W x H."""
    saved = json.loads((REPO / "videos" / "final_f_calibration.json").read_text(encoding="utf-8"))
    return {n: [x * SCALE, y * SCALE] for n, (x, y) in saved["labeled_keypoints"].items()}


def job_calibration() -> dict:
    """The calibration as run_job receives it."""
    return {"points": points(), "frame_width": W, "frame_height": H}


def calibration() -> CourtCalibration:
    return checked_calibration(points(), W, H, 0.25)


def position(j: int, f: int) -> tuple[float, float]:
    """Player j at frame f: sways up to 1.5 m, fast enough to count as play."""
    t = f / FPS
    x0, y0 = SPOTS[j]
    return x0 + 1.0 * np.sin(2 * np.pi * t / 3 + j), y0 + 1.5 * np.sin(2 * np.pi * t / 2 + j)


def raw_tracking(n_frames: int = 250, players: int = 4) -> TrackingResult:
    """Detections as the tracker gives them (raw IDs P_001...), with boxes
    where the camera sees the players' feet."""
    cal = calibration()
    Hm = np.array(cal.homography_matrix)
    frames = []
    for f in range(n_frames):
        for j in range(players):
            x, y = position(j, f)
            u, v = court_to_pixel((x, y), Hm)
            h = 12.0 + 0.2 * v  # nearer players are bigger
            frames.append(
                PlayerFrame(
                    frame_id=f,
                    timestamp_ms=1000.0 * f / FPS,
                    player_id=f"P_{j + 1:03d}",
                    bbox=BoundingBox(x1=u - 0.2 * h, y1=v - h, x2=u + 0.2 * h, y2=v),
                    position=Position2D(x=float(x), y=float(y)),
                    confidence=0.9,
                )
            )
    return TrackingResult(player_frames=frames, ball_frames=[], calibration=cal)


def write_video(path: Path, n_frames: int = 250) -> Path:
    """A small video of the right size and length (its content is not looked at)."""
    from courtshot.io.video import VideoWriter

    rng = np.random.default_rng(0)
    with VideoWriter(path, FPS, (W, H), max_height=H) as writer:
        for _ in range(n_frames):
            writer.write(rng.integers(90, 110, (H, W, 3), dtype=np.uint8))
    return path


# ---------------------------------------------------------------------------
# a rally, with its 3D flights
# ---------------------------------------------------------------------------

# (time, hitter, launch position, launch velocity): near, far, near
HITS = [
    (2.0, "P_1", (3.0, 4.0, 1.0), (0.5, 11.0, 4.5)),
    (3.6, "P_3", (3.0, 16.0, 1.2), (1.0, -10.0, 4.0)),
    (5.2, "P_2", (7.0, 4.0, 1.0), (-1.0, 12.0, 3.5)),
]


def ball_analysis(stable: TrackingResult) -> tuple[BallResult, ball_flight.Ball3DResult]:
    """One rally of three hits, as the 2D step and the 3D fit would give it."""
    cal = stable.calibration
    homography = np.array(cal.homography_matrix)
    cam = ball_flight.camera_from_homography(homography, W, H, cal.court_keypoints_px, cal.court_keypoints_m)
    events, shots, flights = [], [], []
    for i, (t, hitter, x0, v0) in enumerate(HITS):
        end = HITS[i + 1][0] if i + 1 < len(HITS) else t + 1.4
        params = np.r_[x0, v0, ball_flight.DEFAULTS]
        fl = ball_flight.Flight(t, params, np.array([]), np.empty((0, 2)), 1.2, 0.1, end, FPS, hitter=hitter)
        path = fl.at(np.linspace(t, end, 80))
        k = int(np.argmax(np.diff(np.sign(path[:, 2] - 0.05)) != 0))  # first time near the floor
        bounce_t = t + (end - t) * k / 79
        landing = path[k]
        fl.bounces = [("ground", bounce_t, np.array([landing[0], landing[1], 0.0]))]
        flights.append(fl)
        px = tuple(cam.project(np.array([x0]))[0])
        team = "T_1" if hitter in ("P_1", "P_2") else "T_2"
        events.append(BallEvent(t * FPS, t, "hit", px, hitter, x0[:2]))
        bounce_px = tuple(cam.project(landing[None])[0])
        events.append(BallEvent(bounce_t * FPS, bounce_t, "ground", bounce_px, None, tuple(landing[:2])))
        shots.append(Shot(hitter, team, t, x0[:2], tuple(landing[:2]), bounce_t - t, 15.0, False, False, 0))
    ball = BallResult(0.9, 0.95, [], events, shots, rallies=[(2.0, 6.6, 3)])
    return ball, ball_flight.Ball3DResult(cam, flights, {})


def analysis(with_ball: bool = True) -> MatchAnalysis:
    raw = raw_tracking()
    stable, id_report = stabilize_tracking(raw)
    ball, ball3d = ball_analysis(stable) if with_ball else (None, None)
    return MatchAnalysis(Path("clip_tracking_cache.pkl"), raw, stable, id_report, FPS, ball, ball3d)
