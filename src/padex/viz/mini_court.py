"""
A small top-down court in a corner of the video frame, with the players and
the ball on it.

Original author: Xingnan Zhu.
"""

from __future__ import annotations

import cv2
import numpy as np

from padex.court import COURT_MODEL
from padex.schemas.tracking import PlayerFrame

# Team colours (BGR)
TEAM_COLORS = {
    "T_1": (229, 136, 30),  # blue
    "T_2": (53, 57, 229),  # red
}
_DEFAULT_COLOR = (200, 200, 200)
_BALL_COLOR = (59, 235, 255)  # yellow

_COURT_W = COURT_MODEL.COURT_WIDTH
_COURT_L = COURT_MODEL.COURT_LENGTH
_NET_Y = COURT_MODEL.NET_Y
_SERVICE_NEAR_Y = COURT_MODEL.SERVICE_NEAR_Y
_SERVICE_FAR_Y = COURT_MODEL.SERVICE_FAR_Y
_CENTER_X = _COURT_W / 2


class MiniCourt:
    """Draws a miniature 2D padel court on a video frame."""

    def __init__(self, width_px: int = 150, height_px: int = 300, margin: int = 10) -> None:
        self.width = width_px
        self.height = height_px
        self.margin = margin

    def _court_to_mini(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Court metres (0-10, 0-20) -> mini court pixels (y flipped: far end on top)."""
        px = int(x_m / _COURT_W * self.width)
        py = int((1.0 - y_m / _COURT_L) * self.height)
        return px, py

    def draw(
        self,
        frame: np.ndarray,
        player_frames: list[PlayerFrame],
        ball_court: tuple[float, float] | None = None,
        position: str = "bottom_right",
    ) -> None:
        """Draw the overlay on the frame (in place).

        ball_court: the ball's position over the court in metres, or None.
        position:   "bottom_right" or "top_right".
        """
        h, w = frame.shape[:2]
        total_w = self.width + 2 * self.margin
        total_h = self.height + 2 * self.margin
        x0 = w - total_w - 5
        y0 = 5 if position == "top_right" else h - total_h - 5

        # semi-transparent backdrop
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + total_w, y0 + total_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cx0, cy0 = x0 + self.margin, y0 + self.margin
        cv2.rectangle(frame, (cx0, cy0), (cx0 + self.width, cy0 + self.height), (30, 100, 30), -1)
        self._draw_lines(frame, cx0, cy0)

        for pf in player_frames:
            if pf.position is None:
                continue
            px, py = self._court_to_mini(pf.position.x, pf.position.y)
            color = TEAM_COLORS.get(pf.team_id, _DEFAULT_COLOR)
            cv2.circle(frame, (cx0 + px, cy0 + py), 5, color, -1)
            cv2.circle(frame, (cx0 + px, cy0 + py), 5, (255, 255, 255), 1)

        if ball_court is not None:
            bx, by = self._court_to_mini(min(max(ball_court[0], 0.0), _COURT_W), min(max(ball_court[1], 0.0), _COURT_L))
            cv2.circle(frame, (cx0 + bx, cy0 + by), 4, _BALL_COLOR, -1)

    def _draw_lines(self, frame: np.ndarray, cx0: int, cy0: int) -> None:
        """Court boundary, net, service lines, centre line."""
        white = (255, 255, 255)
        gray = (180, 180, 180)

        def line(x1m: float, y1m: float, x2m: float, y2m: float, color=white, thickness=1):
            p1x, p1y = self._court_to_mini(x1m, y1m)
            p2x, p2y = self._court_to_mini(x2m, y2m)
            cv2.line(frame, (cx0 + p1x, cy0 + p1y), (cx0 + p2x, cy0 + p2y), color, thickness)

        cv2.rectangle(frame, (cx0, cy0), (cx0 + self.width, cy0 + self.height), white, 1)
        line(0, _NET_Y, _COURT_W, _NET_Y, gray, 2)
        line(0, _SERVICE_NEAR_Y, _COURT_W, _SERVICE_NEAR_Y)
        line(0, _SERVICE_FAR_Y, _COURT_W, _SERVICE_FAR_Y)
        line(_CENTER_X, _SERVICE_NEAR_Y, _CENTER_X, _SERVICE_FAR_Y)
