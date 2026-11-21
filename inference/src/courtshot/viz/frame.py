"""
Drawing on video frames: player boxes and skeletons, the ball and its speed,
court lines, bounce marks, the shot label, a mini court and a stats panel.

Original author: Xingnan Zhu.
"""

from __future__ import annotations

import cv2
import numpy as np

from courtshot.court import COURT_MODEL
from courtshot.schemas.tracking import BallFrame, CourtCalibration, PlayerFrame
from courtshot.tracking.player import COCO_KEYPOINT_NAMES
from courtshot.viz.mini_court import TEAM_COLORS, MiniCourt

_DEFAULT_TEAM_COLOR = (200, 200, 200)
_BALL_COLOR = (59, 235, 255)  # yellow (BGR)

# Bounce mark colours by kind (BGR)
_BOUNCE_COLORS = {
    "ground": (0, 255, 0),  # green
    "wall": (0, 0, 255),  # red
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

# COCO skeleton: pairs of keypoint indices and their colours (BGR)
_SKELETON = {
    (0, 1): (255, 200, 50),
    (0, 2): (255, 200, 50),  # head
    (1, 3): (255, 200, 50),
    (2, 4): (255, 200, 50),
    (5, 6): (0, 255, 0),  # shoulders
    (5, 7): (0, 200, 255),
    (7, 9): (0, 200, 255),  # left arm
    (6, 8): (255, 0, 200),
    (8, 10): (255, 0, 200),  # right arm
    (5, 11): (0, 255, 0),
    (6, 12): (0, 255, 0),  # torso
    (11, 12): (0, 255, 0),
    (11, 13): (0, 255, 255),
    (13, 15): (0, 255, 255),  # left leg
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0),  # right leg
}
_KEYPOINT_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINT_NAMES)}
_MIN_KEYPOINT_CONFIDENCE = 0.3


def _ball_centre(ball_frame: BallFrame | None) -> tuple[int, int] | None:
    if ball_frame is None or ball_frame.bbox is None:
        return None
    b = ball_frame.bbox
    return int((b.x1 + b.x2) / 2), int((b.y1 + b.y2) / 2)


class FrameAnnotator:
    """All the frame drawing in one place. Every method draws in place."""

    def __init__(self) -> None:
        self._mini_court = MiniCourt()
        self._court_lines: tuple[CourtCalibration, list] | None = None  # (calibration, line ends in pixels)

    def draw_player_bboxes(self, frame: np.ndarray, player_frames: list[PlayerFrame]) -> None:
        """Boxes and player IDs, coloured by team."""
        for pf in player_frames:
            color = TEAM_COLORS.get(pf.team_id, _DEFAULT_TEAM_COLOR)
            x1, y1 = int(pf.bbox.x1), int(pf.bbox.y1)
            x2, y2 = int(pf.bbox.x2), int(pf.bbox.y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(pf.player_id, _FONT, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, pf.player_id, (x1 + 2, y1 - 4), _FONT, 0.5, (255, 255, 255), 1)

    def draw_pose_keypoints(self, frame: np.ndarray, player_frames: list[PlayerFrame]) -> None:
        """Skeleton and keypoints of each player."""
        for pf in player_frames:
            pts: dict[int, tuple[int, int]] = {}
            for kp in pf.keypoints:
                idx = _KEYPOINT_INDEX.get(kp.name)
                if idx is not None and kp.confidence > _MIN_KEYPOINT_CONFIDENCE:
                    pts[idx] = (int(kp.x), int(kp.y))
            for (i, j), color in _SKELETON.items():
                if i in pts and j in pts:
                    cv2.line(frame, pts[i], pts[j], color, 2, cv2.LINE_AA)
            for pt in pts.values():
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)
                cv2.circle(frame, pt, 4, (0, 0, 0), 1)

    def draw_ball(self, frame: np.ndarray, ball_frame: BallFrame | None) -> None:
        """A dot on the detected ball."""
        centre = _ball_centre(ball_frame)
        if centre is None:
            return
        cv2.circle(frame, centre, 8, _BALL_COLOR, -1)
        cv2.circle(frame, centre, 8, (255, 255, 255), 1)

    def draw_ball_speed(self, frame: np.ndarray, ball_frame: BallFrame | None, speed_kmh: float | None) -> None:
        """The ball's speed next to the ball."""
        centre = _ball_centre(ball_frame)
        if speed_kmh is None or centre is None:
            return

        h, w = frame.shape[:2]
        scale = max(0.5, h / 1080 * 0.7)
        thick = max(1, round(scale * 2))
        label = f"{speed_kmh:.0f} km/h"
        (tw, th), _ = cv2.getTextSize(label, _FONT, scale, thick)
        pad = max(3, round(th * 0.35))

        cx, cy = centre
        # to the right of the ball, a little below it (the shot label sits
        # above); on the left when there is no room on the right
        x = cx + 14
        if x + tw + pad > w:
            x = cx - 14 - tw
        y = min(max(cy + th + 6, th + pad), h - pad)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, label, (x, y), _FONT, scale, _BALL_COLOR, thick, cv2.LINE_AA)

    def draw_court_lines(self, frame: np.ndarray, calibration: CourtCalibration | None) -> None:
        """The court lines, projected into the image."""
        if calibration is None:
            return
        # the camera is fixed, so the lines are worked out once per calibration
        if self._court_lines is None or self._court_lines[0] is not calibration:
            try:
                H_inv = np.linalg.inv(np.array(calibration.homography_matrix))
            except np.linalg.LinAlgError:
                return
            kp = COURT_MODEL.KEYPOINTS
            ends = np.array([[kp[a], kp[b]] for a, b in COURT_MODEL.LINES], dtype=np.float64)
            px = cv2.perspectiveTransform(ends.reshape(-1, 1, 2), H_inv).reshape(-1, 2, 2)
            lines = [((int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1]))) for p in px]
            self._court_lines = (calibration, lines)
        for p1, p2 in self._court_lines[1]:
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

    def draw_court_keypoints(self, frame: np.ndarray, calibration: CourtCalibration | None) -> None:
        """The calibration's keypoints."""
        if calibration is None:
            return
        for px, py in calibration.court_keypoints_px:
            cv2.circle(frame, (int(px), int(py)), 5, (0, 255, 255), -1)

    def draw_mini_court(
        self,
        frame: np.ndarray,
        player_frames: list[PlayerFrame],
        ball_court: tuple[float, float] | None = None,
    ) -> None:
        """The mini court overlay; ball_court is the ball's court position."""
        self._mini_court.draw(frame, player_frames, ball_court)

    def draw_stats_panel(self, frame: np.ndarray, stats: dict[str, str | int | float]) -> None:
        """A semi-transparent panel of "name: value" lines in the top-left corner."""
        if not stats:
            return
        lines = [f"{k}: {v}" for k, v in stats.items()]
        line_h = 22
        panel_h = len(lines) * line_h + 20
        panel_w = 250

        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (15, 25 + i * line_h), _FONT, 0.5, (255, 255, 255), 1)

    def draw_shot_label(self, frame: np.ndarray, label: str | None, ball_frame: BallFrame | None) -> None:
        """The type of the shot just played, above the ball."""
        centre = _ball_centre(ball_frame)
        if not label or centre is None:
            return
        cx, cy = centre[0], centre[1] - 20
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.6, 2)
        cv2.rectangle(frame, (cx - 2, cy - th - 4), (cx + tw + 4, cy + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (cx, cy), _FONT, 0.6, (0, 255, 255), 2)

    def draw_bounces(self, frame: np.ndarray, bounces: list[tuple[tuple[float, float], str, float]]) -> None:
        """Fading marks where the ball bounced.

        bounces: (image position, kind, progress); kind is "ground" or
        "wall", progress runs from 0.0 (just happened) to 1.0 (about to go).
        """
        if not bounces:
            return
        overlay = frame.copy()
        for (u, v), kind, progress in bounces:
            cx, cy = int(u), int(v)
            alpha = 1.0 - progress
            color = _BOUNCE_COLORS.get(kind, (0, 165, 255))

            radius = int(12 + 8 * progress)  # outer ring, growing
            cv2.circle(overlay, (cx, cy), radius, color, max(1, int(3 * alpha)), cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), max(1, int(5 * alpha)), color, -1, cv2.LINE_AA)  # dot, shrinking

            if progress < 0.5:
                (tw, _), _ = cv2.getTextSize(kind, _FONT_SMALL, 1.0, 1)
                cv2.putText(overlay, kind, (cx - tw // 2, cy - radius - 6), _FONT_SMALL, 1.0, color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def draw_frame_number(self, frame: np.ndarray, frame_id: int) -> None:
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30), _FONT, 0.7, (255, 255, 255), 2)

    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        player_frames: list[PlayerFrame],
        ball_frame: BallFrame | None = None,
        calibration: CourtCalibration | None = None,
        shot_label: str | None = None,
        stats: dict[str, str | int | float] | None = None,
        bounces: list[tuple[tuple[float, float], str, float]] | None = None,
        ball_speed_kmh: float | None = None,
        ball_court: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Draw everything on the frame (in place) and return it.

        shot_label:     shot type to show above the ball
        stats:          lines for the stats panel
        bounces:        bounce marks, see draw_bounces()
        ball_speed_kmh: speed to write next to the ball
        ball_court:     the ball's position over the court (x, y in metres),
                        for the mini court
        """
        self.draw_court_lines(frame, calibration)
        self.draw_court_keypoints(frame, calibration)
        self.draw_player_bboxes(frame, player_frames)
        self.draw_pose_keypoints(frame, player_frames)
        self.draw_ball(frame, ball_frame)
        self.draw_ball_speed(frame, ball_frame, ball_speed_kmh)
        self.draw_bounces(frame, bounces or [])
        self.draw_shot_label(frame, shot_label, ball_frame)
        self.draw_mini_court(frame, player_frames, ball_court)
        if stats:
            self.draw_stats_panel(frame, stats)
        self.draw_frame_number(frame, frame_id)
        return frame
