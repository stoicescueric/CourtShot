"""
Ball post-processing for Padex: clean detections, find hits and bounces,
build shots.

Padex's ball detector gives at most one raw detection per frame. On padel
footage many of those are wrong (glass reflections, a spare ball lying on the
floor, heads), and the real ball is often missed. Padex then maps the airborne
ball onto the floor, which is where its bounce and shot detection go wrong.

This module works in IMAGE space instead, where a ball in flight follows a
smooth curve, and only uses the floor mapping where the ball really is on the
floor (at a ground bounce):

  1. Drop static detections (anything sitting on the same spot for seconds).
  2. Link detections into tracklets with a short-term motion prediction, so
     isolated false detections don't join the real ball's track.
  3. Split each tracklet into arcs that each fit a smooth quadratic curve.
     Where one arc ends and the next starts, something hit the ball.
  4. Classify each break:
       hit    - the ball is within reach of a player (expanded player box)
       ground - away from players, the ball gets a sudden upward kick in the
                image and the break point maps onto the court floor
       wall   - any other break (side/back glass or fence)
  5. Shots: from each hit to the next, with the first ground bounce as the
     landing spot (on the floor, so the court position is valid), flight
     time, average horizontal speed (hitter's position to landing spot), and
     whether the receiver volleyed it (hit before any ground bounce).

All thresholds scale with the frame height, so they mean the same at 720p and
4K. Results are only as good as the raw detections: check `coverage` first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class BallSettings:
    static_px: float = 0.006      # cell size for static-blob check, fraction of frame height
    static_s: float = 2.0         # a cell occupied this long in total = static object
    gate: float = 0.045           # linking gate per frame, fraction of frame height
    max_gap: int = 6              # frames a tracklet can go without a detection
    min_tracklet: int = 6         # shorter tracklets are treated as noise
    fit_tol: float = 0.006        # max deviation from one smooth arc, fraction of frame height
    min_arc: int = 4              # minimum detections per arc
    event_gap_s: float = 0.5      # arcs further apart than this are not joined by an event
    hit_reach_w: float = 0.6      # player box widened by this x its width on each side
    hit_reach_up: float = 0.5     # ...and raised by this x its height (racket overhead)
    min_change: float = 0.2       # velocity change / speed below this is not an event
    same_team_s: float = 1.0      # same-team "hits" closer than this: only the last is a hit
    volley_max_gap: int = 4       # frames; longer gaps in the track make "volley or not" unknown
    volley_net_m: float = 5.0     # a volley needs the receiver within this distance of the net
    hit_min_height: float = 0.2   # ball below this share of the player's box height
                                  # (0 = feet, 1 = head) is a bounce at their feet, not a hit
    merge_s: float = 0.12         # events closer than this are merged
    use_candidates: bool = False  # link over every candidate detection of a frame, not just the top one.
                                  # Off: on the PadelTracker100 check clips only ~3% of frames have a
                                  # second candidate and using them made the track slightly worse.
    rally_gap_s: float = 4.0      # no ball for this long ends a ball-rally (lobs can leave the frame for 3 s)


@dataclass
class Arc:
    t0: int
    t1: int                       # inclusive frame range covered by detections
    cu: np.ndarray                # quadratic coefficients for u(t - tc)
    cv: np.ndarray                # quadratic coefficients for v(t - tc)
    tc: float
    n: int

    def at(self, t: float) -> tuple[float, float]:
        x = t - self.tc
        return float(np.polyval(self.cu, x)), float(np.polyval(self.cv, x))

    def vel(self, t: float) -> tuple[float, float]:
        x = t - self.tc
        return float(np.polyval(np.polyder(self.cu), x)), float(np.polyval(np.polyder(self.cv), x))


@dataclass
class BallEvent:
    frame: float
    time_s: float
    kind: str                     # "hit", "ground", "wall"
    px: tuple[float, float]       # image position
    player: str | None = None     # hitter for "hit"
    court: tuple[float, float] | None = None  # floor position (ground bounces; hitter's feet for hits)
    bounce_kind: str = "wall"     # what this break would be if it were not a hit


@dataclass
class Shot:
    player: str
    team: str
    time_s: float
    hitter_pos: tuple[float, float] | None
    landing: tuple[float, float] | None       # first ground bounce after the hit
    flight_s: float | None                    # hit -> first ground bounce
    speed_ms: float | None                    # horizontal, hitter -> landing
    wall_after: bool                          # ball hit a wall before the next hit
    volleyed_by_next: bool | None             # next hit came before any ground bounce (None = not sure)
    rally: int


@dataclass
class BallResult:
    coverage: float               # frames with a raw detection (within the analysed span)
    kept: float                   # share of raw detections kept after cleaning
    arcs: list[Arc]
    events: list[BallEvent]
    shots: list[Shot]
    rallies: list[tuple[float, float, int]] = field(default_factory=list)  # start_s, end_s, hits
    track: dict[int, tuple[float, float]] = field(default_factory=dict)    # frame -> smoothed px
    det: dict[int, tuple[float, float]] = field(default_factory=dict)      # frame -> raw detection on the track


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def to_floor(H: np.ndarray | None, p: tuple[float, float]) -> tuple[float, float] | None:
    if H is None:
        return None
    q = cv2.perspectiveTransform(np.array([[p]], dtype=np.float64), H)[0, 0]
    return float(q[0]), float(q[1])


def fit_arc(t: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[Arc, float]:
    tc = float(t.mean())
    deg = 2 if len(t) >= 3 else 1
    cu = np.polyfit(t - tc, u, deg)
    cv_ = np.polyfit(t - tc, v, deg)
    if deg == 1:
        cu, cv_ = np.r_[0.0, cu], np.r_[0.0, cv_]
    res = np.hypot(np.polyval(cu, t - tc) - u, np.polyval(cv_, t - tc) - v)
    return Arc(int(t[0]), int(t[-1]), cu, cv_, tc, len(t)), float(res.max())


# ---------------------------------------------------------------------------
# pipeline steps
# ---------------------------------------------------------------------------


def drop_static(det: dict[int, list[tuple[float, float]]], frame_h: float, fps: float,
                s: BallSettings) -> dict[int, list[tuple[float, float]]]:
    cell = max(2.0, s.static_px * frame_h)

    def key(p):
        return int(p[0] // cell), int(p[1] // cell)

    counts: dict[tuple[int, int], int] = {}
    for ps in det.values():
        # a frame counts once per cell, however many candidates fall in it
        for k in {(kx + dx, ky + dy) for kx, ky in map(key, ps) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}:
            counts[k] = counts.get(k, 0) + 1
    limit = s.static_s * fps
    out = {f: [p for p in ps if counts[key(p)] < limit] for f, ps in det.items()}
    return {f: ps for f, ps in out.items() if ps}


def link_tracklets(det: dict[int, list[tuple[float, float]]], frame_h: float,
                   s: BallSettings) -> list[list[tuple[int, float, float]]]:
    """Greedy linking; several tracklets stay open so one false detection
    doesn't end the real ball's track.

    A frame can carry several candidate detections. Each open tracklet takes
    at most one of them (the best matches are made first); candidates nobody
    takes start their own tracklet.
    """
    gate = s.gate * frame_h
    open_: list[list[tuple[int, float, float]]] = []
    done: list[list[tuple[int, float, float]]] = []

    def predict(tr, f):
        pts = tr[-5:]
        if len(pts) == 1:
            return pts[-1][1], pts[-1][2]
        tt = np.array([p[0] for p in pts], float)
        uu = np.array([p[1] for p in pts])
        vv = np.array([p[2] for p in pts])
        deg = min(2, len(pts) - 1)
        return (float(np.polyval(np.polyfit(tt - tt[-1], uu, deg), f - tt[-1])),
                float(np.polyval(np.polyfit(tt - tt[-1], vv, deg), f - tt[-1])))

    for f in sorted(det):
        cands = det[f]
        still = []
        for tr in open_:
            (done if f - tr[-1][0] > s.max_gap else still).append(tr)
        open_ = still
        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(open_):
            pu, pv = predict(tr, f)
            allowed = gate * (1 + 0.5 * (f - tr[-1][0] - 1))
            for ci, (u, v) in enumerate(cands):
                d = float(np.hypot(pu - u, pv - v))
                if d <= allowed:
                    # prefer longer tracklets slightly, so the real ball wins ties
                    pairs.append((d / allowed - 0.02 * min(len(tr), 20), ti, ci))
        used_t: set[int] = set()
        used_c: set[int] = set()
        for _, ti, ci in sorted(pairs):
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            open_[ti].append((f, *cands[ci]))
        open_.extend([(f, u, v)] for ci, (u, v) in enumerate(cands) if ci not in used_c)
    done.extend(open_)
    return [tr for tr in done if len(tr) >= s.min_tracklet]


def split_arcs(tr: list[tuple[int, float, float]], frame_h: float, s: BallSettings) -> list[Arc]:
    tol = s.fit_tol * frame_h
    t = np.array([p[0] for p in tr], float)
    u = np.array([p[1] for p in tr])
    v = np.array([p[2] for p in tr])

    def rec(a: int, b: int) -> list[tuple[int, int]]:
        arc, err = fit_arc(t[a:b], u[a:b], v[a:b])
        if err <= tol or b - a < 2 * s.min_arc:
            return [(a, b)]
        # split where the curve bends most: best two-arc fit (coarse, then fine)
        def two_fit(k: int) -> float:
            return max(fit_arc(t[a:k], u[a:k], v[a:k])[1], fit_arc(t[k:b], u[k:b], v[k:b])[1])

        lo, hi = a + s.min_arc, b - s.min_arc
        step = max(1, (hi - lo) // 30)
        coarse = min(range(lo, hi + 1, step), key=two_fit)
        best_k = min(range(max(lo, coarse - step), min(hi, coarse + step) + 1), key=two_fit)
        return rec(a, best_k) + rec(best_k, b)

    parts = rec(0, len(t))
    # merge neighbours that fit one arc after all
    merged = [parts[0]]
    for a, b in parts[1:]:
        pa, _ = merged[-1]
        if fit_arc(t[pa:b], u[pa:b], v[pa:b])[1] <= tol:
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))
    return [fit_arc(t[a:b], u[a:b], v[a:b])[0] for a, b in merged]


def event_between(a: Arc, b: Arc) -> tuple[float, tuple[float, float]]:
    """Time and image point where arc a hands over to arc b."""
    lo, hi = a.t1, b.t0
    if hi <= lo:
        f = (lo + hi) / 2
    else:
        ts = np.linspace(lo, hi, max(3, int(4 * (hi - lo)) + 1))
        d = [np.hypot(*(np.subtract(a.at(x), b.at(x)))) for x in ts]
        f = float(ts[int(np.argmin(d))])
    pa, pb = a.at(f), b.at(f)
    return f, ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)


def players_near(frame: int, boxes: dict[int, list[tuple[str, tuple[float, float, float, float]]]],
                 p: tuple[float, float], s: BallSettings) -> tuple[str | None, float]:
    """Closest player whose reach (expanded box) contains p, and the ball's
    height within that player's box (0 = feet, 1 = head)."""
    best, best_d, best_rel = None, None, 0.0
    for df in (0, -1, 1, -2, 2):
        for name, (x1, y1, x2, y2) in boxes.get(frame + df, []):
            w, h = x2 - x1, y2 - y1
            if (x1 - s.hit_reach_w * w <= p[0] <= x2 + s.hit_reach_w * w
                    and y1 - s.hit_reach_up * h <= p[1] <= y2 + 0.3 * h):
                d = abs(p[0] - (x1 + x2) / 2) / w + abs(p[1] - (y1 + y2) / 2) / h
                if best_d is None or d < best_d:
                    best, best_d, best_rel = name, d, (y2 - p[1]) / max(h, 1e-6)
        if best:
            return best, best_rel
    return None, 0.0


def analyse_ball(
    ball_px: dict[int, tuple[float, float]],
    player_boxes: dict[int, list[tuple[str, tuple[float, float, float, float]]]],
    player_pos: dict[tuple[int, str], tuple[float, float]],
    teams: dict[str, str],
    H: np.ndarray | None,
    fps: float,
    frame_h: float,
    frame_span: tuple[int, int],
    s: BallSettings | None = None,
    candidates: dict[int, list[tuple[float, float]]] | None = None,
) -> BallResult:
    """Run the whole ball pipeline.

    ball_px:      frame -> raw detection centre in pixels
    candidates:   frame -> every ball-like detection of that frame (optional;
                  includes the one in ball_px). With s.use_candidates the
                  linking picks, per frame, the one that continues the track.
    player_boxes: frame -> [(player, (x1, y1, x2, y2))] in pixels (stable IDs)
    player_pos:   (frame, player) -> floor position in metres
    teams:        player -> team
    H:            pixel -> court-metres homography (floor)
    """
    s = s or BallSettings()
    n_frames = max(1, frame_span[1] - frame_span[0] + 1)
    coverage = len(ball_px) / n_frames

    raw = {f: [p] for f, p in ball_px.items()}
    for f, ps in ((candidates or {}) if s.use_candidates else {}).items():
        if ps:
            raw[f] = list(ps)
    det = drop_static(raw, frame_h, fps, s)
    tracklets = link_tracklets(det, frame_h, s)
    kept = len({p[0] for tr in tracklets for p in tr}) / max(1, len(raw))

    arcs: list[Arc] = []
    for tr in tracklets:
        arcs.extend(split_arcs(tr, frame_h, s))
    arcs.sort(key=lambda a: a.t0)
    # overlapping arcs (two tracklets alive at once): keep the longer one
    clean: list[Arc] = []
    for a in arcs:
        if clean and a.t0 <= clean[-1].t1:
            if a.n > clean[-1].n:
                clean[-1] = a
            continue
        clean.append(a)
    arcs = clean

    track: dict[int, tuple[float, float]] = {}
    for a in arcs:
        for f in range(a.t0, a.t1 + 1):
            track[f] = a.at(f)

    # the raw detection each tracked frame is based on (nearest to the track)
    on_track: dict[int, tuple[float, float]] = {}
    for f, p in track.items():
        if f in det:
            q = min(det[f], key=lambda c: (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2)
            if np.hypot(q[0] - p[0], q[1] - p[1]) <= s.gate * frame_h:
                on_track[f] = q

    events: list[BallEvent] = []
    max_gap = s.event_gap_s * fps
    for a, b in zip(arcs, arcs[1:]):
        if b.t0 - a.t1 > max_gap:
            continue
        f, p = event_between(a, b)
        fi = int(round(f))
        vb, va = a.vel(f), b.vel(f)
        speed = max(np.hypot(*vb), np.hypot(*va), 1e-6)
        du, dv = (va[0] - vb[0]) / speed, (va[1] - vb[1]) / speed
        if np.hypot(du, dv) < s.min_change and b.t0 - a.t1 <= 2:
            continue  # the curve just needed two pieces; nothing hit the ball
        hitter, rel = players_near(fi, player_boxes, p, s)
        court = to_floor(H, p)
        on_floor = court is not None and -0.5 <= court[0] <= 10.5 and -0.5 <= court[1] <= 20.5
        # Gravity always pulls the image track downwards (+v). A floor bounce
        # gives it a sudden upward kick (vertical image velocity drops); a
        # back-wall bounce does the opposite, a side-wall bounce flips u.
        bounce = "ground" if (dv < -0.15 and on_floor) else "wall"
        if hitter and rel >= s.hit_min_height:
            events.append(BallEvent(f, f / fps, "hit", p, hitter, player_pos.get((fi, hitter)), bounce))
        elif bounce == "ground":
            events.append(BallEvent(f, f / fps, "ground", p, None, court, bounce))
        else:
            events.append(BallEvent(f, f / fps, "wall", p, None, None, bounce))

    # Serves (and any first hit after the ball reappears): there is no arc
    # before them, so no break to find. If the first arc of a ball-rally
    # starts within a player's reach, that player hit it.
    prev_end = -np.inf
    for a in arcs:
        if (a.t0 - prev_end) / fps >= s.rally_gap_s:
            p0 = a.at(a.t0)
            hitter, _ = players_near(a.t0, player_boxes, p0, s)
            if hitter:
                events.append(BallEvent(a.t0, a.t0 / fps, "hit", p0, hitter,
                                        player_pos.get((a.t0, hitter)), "wall"))
        prev_end = a.t1
    events.sort(key=lambda e: e.time_s)

    merged: list[BallEvent] = []
    for e in events:
        if merged and e.time_s - merged[-1].time_s < s.merge_s:
            if e.kind == "hit" and merged[-1].kind != "hit":
                merged[-1] = e  # a hit explains the break better than a bounce
            continue
        merged.append(e)
    events = merged

    # Padel rule: the teams hit alternately. Several "hits" by the same team
    # in quick succession are a bounce near that player followed by the real
    # hit, so only the last of each run counts; the others become bounces.
    for i, e in enumerate(events):
        if e.kind != "hit":
            continue
        nxt = next((x for x in events[i + 1:] if x.kind == "hit"), None)
        if (nxt and teams.get(nxt.player) == teams.get(e.player)
                and nxt.time_s - e.time_s < s.same_team_s):
            e.kind = e.bounce_kind
            if e.kind == "ground":
                e.court = to_floor(H, e.px)
            else:
                e.court = None
            e.player = None

    # ball rallies: arcs separated by less than rally_gap_s
    rallies: list[tuple[float, float, int]] = []
    groups: list[list[Arc]] = []
    for a in arcs:
        if groups and (a.t0 - groups[-1][-1].t1) / fps < s.rally_gap_s:
            groups[-1].append(a)
        else:
            groups.append([a])
    for g in groups:
        t0, t1 = g[0].t0 / fps, g[-1].t1 / fps
        hits = sum(1 for e in events if e.kind == "hit" and t0 <= e.time_s <= t1)
        rallies.append((t0, t1, hits))

    def rally_of(ts: float) -> int:
        for i, (a0, a1, _) in enumerate(rallies):
            if a0 - 0.2 <= ts <= a1 + 0.2:
                return i
        return -1

    shots: list[Shot] = []
    hits = [e for e in events if e.kind == "hit"]
    for i, h in enumerate(hits):
        r = rally_of(h.time_s)
        nxt = hits[i + 1] if i + 1 < len(hits) and rally_of(hits[i + 1].time_s) == r else None
        end = nxt.time_s if nxt else (rallies[r][1] if r >= 0 else h.time_s)
        between = [e for e in events if h.time_s < e.time_s < end and e.kind != "hit"]
        grounds = [e for e in between if e.kind == "ground"]
        # A shot has to land on the other half. A "ground" bounce on the
        # hitter's own half right after the hit is a misread, so skip it.
        if h.court is not None:
            own_near = h.court[1] < 10.0
            ground = next((e for e in grounds if (e.court[1] >= 10.0) == own_near), None)
        else:
            ground = grounds[0] if grounds else None
        flight = ground.time_s - h.time_s if ground else None
        speed = None
        if ground and h.court and flight and flight > 0:
            speed = float(np.hypot(ground.court[0] - h.court[0], ground.court[1] - h.court[1]) / flight)
        # Volley: only call it when the whole flight to the next hit was seen
        # and no floor bounce was found; with gaps in the track it's unknown.
        volleyed = None
        if nxt is not None:
            if grounds:
                volleyed = False
            else:
                f0, f1 = int(h.frame), int(nxt.frame)
                seen = [f for f in range(f0, f1 + 1) if f in track]
                longest_gap = max((b - a for a, b in zip([f0] + seen, seen + [f1])), default=f1 - f0)
                # Volleys are played near the net; far from it, a missing
                # bounce is more likely a missed detection than a volley.
                near_net = nxt.court is not None and abs(nxt.court[1] - 10.0) <= s.volley_net_m
                if longest_gap <= s.volley_max_gap and near_net:
                    volleyed = True
        shots.append(Shot(
            player=h.player, team=teams.get(h.player, "?"), time_s=h.time_s, hitter_pos=h.court,
            landing=ground.court if ground else None, flight_s=flight, speed_ms=speed,
            wall_after=any(e.kind == "wall" for e in between),
            volleyed_by_next=volleyed, rally=r,
        ))

    return BallResult(coverage, kept, arcs, events, shots, rallies, track, on_track)


def run_on_tracking(tracking, fps: float, s: BallSettings | None = None) -> BallResult | None:
    """Convenience wrapper for a Padex TrackingResult whose players already
    carry stable IDs (see stable_ids.stabilize_tracking).

    Returns None when there are no ball detections or no calibration.
    """
    from stable_ids import SLOT_NAMES, SLOT_TEAMS

    cal = tracking.calibration
    ball_px = {
        bf.frame_id: ((bf.bbox.x1 + bf.bbox.x2) / 2, (bf.bbox.y1 + bf.bbox.y2) / 2)
        for bf in tracking.ball_frames if bf.bbox is not None
    }
    if not ball_px or cal is None:
        return None
    # caches made before the detector kept several candidates have none
    cands = {bf.frame_id: [(c[0], c[1]) for c in getattr(bf, "candidates", None) or []]
             for bf in tracking.ball_frames}
    boxes: dict[int, list[tuple[str, tuple[float, float, float, float]]]] = {}
    pos: dict[tuple[int, str], tuple[float, float]] = {}
    for pf in tracking.player_frames:
        if pf.player_id not in SLOT_NAMES:
            continue
        b = pf.bbox
        boxes.setdefault(pf.frame_id, []).append((pf.player_id, (b.x1, b.y1, b.x2, b.y2)))
        if pf.position is not None:
            pos[(pf.frame_id, pf.player_id)] = (pf.position.x, pf.position.y)
    frames = [bf.frame_id for bf in tracking.ball_frames]
    return analyse_ball(
        ball_px, boxes, pos, dict(zip(SLOT_NAMES, SLOT_TEAMS)),
        np.array(cal.homography_matrix, dtype=np.float64), fps,
        float(cal.frame_height), (min(frames), max(frames)), s,
        candidates={f: c for f, c in cands.items() if c},
    )
