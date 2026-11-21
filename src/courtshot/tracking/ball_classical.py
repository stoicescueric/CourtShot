"""
Ball detection without a neural network.

A padel ball in broadcast footage is a small blob, brighter than whatever is
behind it (blue or green court, dark glass, red or black walls), and it moves.
The camera is fixed. Those facts are enough to find most balls with a handful
of OpenCV operations per frame on the CPU, at hundreds of frames per second.

Per frame (candidates):

  1. Brightness channel: the green channel of a lightly blurred frame. The
     yellow ball is bright in it; blue and green courts, red walls and dark
     glass are darker (on the check clip green beat grey and max(G, R)).
  2. Signed three-frame difference. A pixel lights up in frame t when it is
     brighter than the same pixel in t-1 *and* in t+1. A static bright thing
     (lines, logos) cancels out, and a moving one leaves no ghost at its old
     or new position (a plain |t - (t-1)| difference leaves two blobs).
  3. White top-hat. Only bright structures smaller than the kernel survive,
     so a moving white shirt gives a much weaker response than the ball.
  4. Blobs. The response is thresholded; its connected components that are
     ball-sized and not too elongated become candidates, scored by their peak
     response. A candidate lying on a large moving region (a player: shoes,
     head, racket) is scored down: a ball in flight is a small moving region
     of its own. So is a candidate inside a player's box, when the boxes are
     given (the pipeline has them from the people detector): a shoe or a
     head of a player who barely moves is not a large moving region.

Over the whole video (which candidate is the ball):

  5. The candidates form a graph: an edge joins a candidate to one up to
     MAX_SKIP frames later if the ball could have got there (GATE). An edge
     costs more the further it is from the constant-velocity prediction
     (capped, so hits and bounces stay possible) and the more frames it
     skips; starting a new track costs RESTART. Each candidate earns its
     score. The best path through the video (dynamic programming, built up
     frame by frame in update() and read back in flush()) is the ball. A
     shoe or a head that moves smoothly but jumps away from the ball's path
     loses to the ball; frames the path skips report no ball.

A slow ball (wall bounce, top of a lob) moves too little for the frame
difference and is missed; a difference against the median background was
tried for it and made results worse (it lights up players standing still).
Downstream linking (courtshot.analysis.ball_events) bridges the gaps.

Unlike the TrackNet detector, all results come out of flush(): the path is
only known once the video has been seen. The work is done as frames arrive;
flush() only reads the path back.

All sizes are in pixels at PROC_H rows, so the detector behaves the same at
720p and 1080p.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from courtshot.schemas.tracking import BallFrame, BoundingBox

BALL_BOX_RADIUS = 4  # pixels; the box reported around the ball's centre (same as TrackNet's)


class ClassicalBallDetector:
    """Streaming ball detector: feed frames in order with update(), then
    flush() gives one BallFrame per frame."""

    PROC_H: int = 720  # frames are processed at this height (1080p is scaled down to 720)
    # -- candidates
    DIFF_THRESHOLD: int = 18  # brightness levels above both neighbouring frames
    TOPHAT_KERNEL: int = 15  # px; bright things larger than this are suppressed
    MIN_AREA: int = 2  # px
    MAX_AREA: int = 220  # px
    MAX_ELONGATION: float = 6.0  # bounding box long side / short side (motion blur stretches the ball)
    BODY_DIFF: int = 12  # brightness change that counts as motion, for finding moving bodies
    BODY_DILATE: int = 7  # px; joins a player's moving parts into one region
    BODY_AREA: int = 900  # px; a moving region larger than this is a body, not a ball
    BODY_PENALTY: float = 0.1  # score factor for a candidate on a moving body
    PLAYER_PENALTY: float = 0.3  # score factor for a candidate inside a player's box
    PLAYER_MARGIN: float = 0.05  # player boxes are grown by this x their height on each side
    MAX_RAW: int = 10  # candidates kept per frame
    MAX_CANDIDATES: int = 3  # candidates reported per frame
    # -- path
    SCORE_REF: float = 30.0  # a candidate earns score / SCORE_REF - 1 (weaker ones cost)
    SCORE_CAP: float = 3.0  # ...at most this much
    SCORE_FLOOR: float = -0.1  # ...and costs at most this much: less than SKIP_COST, so a weak candidate
    # right where the path goes (the ball by a player at a hit) is taken rather than jumped over
    GATE: float = 0.09  # max ball motion per frame, fraction of PROC_H
    MAX_SKIP: int = 5  # frames an edge can jump over
    SKIP_COST: float = 0.8  # per frame jumped over
    LINE_TOL: float = 0.06  # deviation from constant velocity per frame that costs 1, fraction of PROC_H
    BEND_CAP: float = 1.0  # most a bend (hit, bounce) costs
    RESTART: float = 3.5  # cost of starting a new track
    CONF_SCALE: float = 120.0  # score reported as confidence 1

    def __init__(self) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=3)  # brightness channels t-1, t, t+1
        self._players: deque[np.ndarray] = deque(maxlen=3)  # player boxes (x1, y1, x2, y2), processing px
        self._meta: list[tuple[int, float]] = []  # (frame_id, timestamp_ms) of every frame seen
        self._scale = 1.0
        self._kernel: np.ndarray | None = None
        self._body_kernel: np.ndarray | None = None
        self._dp = _PathDP(self)

    # -- public -----------------------------------------------------------

    def set_background(self, frame: np.ndarray | None) -> None:
        """Accepted so this detector can stand in for BallDetector; the frame
        difference needs no background (see the module docstring)."""

    def update(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
        players: list[BoundingBox] | None = None,
    ) -> list[BallFrame]:
        """Push one frame (and, optionally, the boxes of the people in it).
        Returns nothing: every result comes from flush()."""
        self._meta.append((frame_id, timestamp_ms))
        self._frames.append(self._channel(frame))
        boxes = np.array([(b.x1, b.y1, b.x2, b.y2) for b in players or []], dtype=float).reshape(-1, 4)
        if len(boxes):
            grow = self.PLAYER_MARGIN * (boxes[:, 3] - boxes[:, 1])
            boxes = (boxes + np.stack([-grow, -grow, grow, grow], axis=1)) / self._scale
        self._players.append(boxes)
        if len(self._frames) == 2:  # the first frame has no t-1: nothing found there
            self._dp.add(np.empty((0, 3)))
        elif len(self._frames) == 3:
            self._dp.add(self._detect(*self._frames, players=self._players[1]))
        return []

    def flush(self) -> list[BallFrame]:
        """One BallFrame per frame pushed since the last flush, in order."""
        while self._dp.n_frames < len(self._meta):  # the last frame has no t+1 (or a 1-frame video)
            self._dp.add(np.empty((0, 3)))
        path = self._dp.best_path()
        out = [self._to_ball_frame(self._dp.cands[t], path[t], fid, ts) for t, (fid, ts) in enumerate(self._meta)]
        self.reset()
        return out

    def reset(self) -> None:
        self._frames.clear()
        self._players.clear()
        self._meta.clear()
        self._dp = _PathDP(self)

    # -- candidates -------------------------------------------------------

    def _channel(self, frame: np.ndarray) -> np.ndarray:
        h = frame.shape[0]
        self._scale = h / self.PROC_H
        g = cv2.extractChannel(frame, 1)  # one channel first: resizing it is 3x cheaper than the frame
        if h != self.PROC_H:
            w = int(round(frame.shape[1] / self._scale))
            g = cv2.resize(g, (w, self.PROC_H), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g, (3, 3), 0)

    def _detect(self, prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray, players: np.ndarray) -> np.ndarray:
        """Ball-like blobs of frame `cur`: (n, 3) array of x, y, score in
        processing pixels, best first."""
        if self._kernel is None:
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.TOPHAT_KERNEL, self.TOPHAT_KERNEL))
            self._body_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.BODY_DILATE, self.BODY_DILATE))
        # cv2.subtract saturates at 0: only "brighter than" survives
        d = cv2.min(cv2.subtract(cur, prev), cv2.subtract(cur, nxt))
        resp = cv2.min(d, cv2.morphologyEx(cur, cv2.MORPH_TOPHAT, self._kernel))
        _, binary = cv2.threshold(resp, self.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 1:
            return np.empty((0, 3))
        # Moving bodies: everything that changed since the previous frame,
        # dilated so a player's limbs join into one region.
        _, moving = cv2.threshold(cv2.absdiff(cur, prev), self.BODY_DIFF, 255, cv2.THRESH_BINARY)
        moving = cv2.dilate(moving, self._body_kernel)
        _, body_labels, body_stats, _ = cv2.connectedComponentsWithStats(moving, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (self.MIN_AREA <= area <= self.MAX_AREA):
                continue
            if max(w, h) > self.MAX_ELONGATION * max(1, min(w, h)):
                continue
            peak = float(resp[y : y + h, x : x + w][labels[y : y + h, x : x + w] == i].max())
            cx, cy = centroids[i]
            body = body_labels[int(cy), int(cx)]
            if body and body_stats[body, cv2.CC_STAT_AREA] > self.BODY_AREA:
                peak *= self.BODY_PENALTY
            elif len(players) and np.any(
                (players[:, 0] <= cx) & (cx <= players[:, 2]) & (players[:, 1] <= cy) & (cy <= players[:, 3])
            ):
                peak *= self.PLAYER_PENALTY
            out.append((cx, cy, peak))
        if not out:
            return np.empty((0, 3))
        arr = np.array(out, dtype=float)
        return arr[np.argsort(-arr[:, 2], kind="stable")][: self.MAX_RAW]

    # -- output -----------------------------------------------------------

    def _to_ball_frame(self, cands: np.ndarray, chosen: int, frame_id: int, timestamp_ms: float) -> BallFrame:
        if chosen < 0:
            return BallFrame(frame_id=frame_id, timestamp_ms=timestamp_ms)
        order = [chosen] + [i for i in range(len(cands)) if i != chosen]
        s = self._scale
        reported = [
            (float(cands[i, 0] * s), float(cands[i, 1] * s), float(min(1.0, cands[i, 2] / self.CONF_SCALE)))
            for i in order[: self.MAX_CANDIDATES]
        ]
        x, y, conf = reported[0]
        r = BALL_BOX_RADIUS
        return BallFrame(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            bbox=BoundingBox(x1=x - r, y1=y - r, x2=x + r, y2=y + r),
            confidence=conf,
            candidates=reported,
        )


class _PathDP:
    """The best path through the candidates of all frames (step 5 of the
    module docstring), built one frame at a time."""

    def __init__(self, cfg: ClassicalBallDetector) -> None:
        self.c = cfg
        self.cands: list[np.ndarray] = []  # per frame: (n, 3) x, y, score
        self.best: list[np.ndarray] = []  # per frame: best path score ending at each candidate
        self.vel: list[np.ndarray] = []  # per frame: (n, 2) velocity on that path (nan = track starts here)
        self.back: list[np.ndarray] = []  # per frame: (n, 2) previous (frame, index) on the path; -1 = none
        self.top = 0.0  # best score of any path so far (0 = the empty path)
        self.top_at = (-1, -1)  # ...and where it ends

    @property
    def n_frames(self) -> int:
        return len(self.cands)

    def add(self, cands: np.ndarray) -> None:
        c = self.c
        gate, tol = c.GATE * c.PROC_H, c.LINE_TOL * c.PROC_H
        t = len(self.cands)
        n = len(cands)
        # start a new track after the best path so far
        best = np.full(n, self.top - c.RESTART)
        vel = np.full((n, 2), np.nan)
        back = np.tile(np.array(self.top_at, dtype=np.int64), (n, 1))
        if n:
            pos = cands[:, :2]
            cols = np.arange(n)
            for k in range(1, min(c.MAX_SKIP + 1, t) + 1):
                prev = self.cands[t - k]
                if not len(prev):
                    continue
                step = pos[None, :, :] - prev[:, None, :2]  # (m, n, 2) motion from each prev to each cand
                ok = np.hypot(step[..., 0], step[..., 1]) <= gate * k
                if not ok.any():
                    continue
                miss = step - self.vel[t - k][:, None, :] * k  # off the constant-velocity prediction
                dev = np.hypot(miss[..., 0], miss[..., 1])
                bend = np.where(np.isnan(dev), 0.0, np.minimum((dev / (tol * k)) ** 2, c.BEND_CAP))
                score = np.where(ok, self.best[t - k][:, None] - bend - c.SKIP_COST * (k - 1), -np.inf)
                i = np.argmax(score, axis=0)  # best predecessor of each candidate
                s = score[i, cols]
                better = s > best
                if better.any():
                    best[better] = s[better]
                    back[better, 0] = t - k
                    back[better, 1] = i[better]
                    vel[better] = step[i[better], cols[better]] / k
            best += np.clip(cands[:, 2] / c.SCORE_REF - 1.0, c.SCORE_FLOOR, c.SCORE_CAP)
        self.cands.append(cands)
        self.best.append(best)
        self.vel.append(vel)
        self.back.append(back)
        if n and best.max() > self.top:
            j = int(np.argmax(best))
            self.top, self.top_at = float(best[j]), (t, j)

    def best_path(self) -> list[int]:
        """Index of the chosen candidate per frame (-1 = no ball)."""
        path = [-1] * len(self.cands)
        t, j = self.top_at
        while t >= 0:
            path[t] = j
            t, j = (int(v) for v in self.back[t][j])
        return path
