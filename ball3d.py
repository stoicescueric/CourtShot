"""
3D ball trajectories from one fixed camera.

A single camera only tells you the direction to the ball, not its distance.
Between contacts, though, the ball follows known physics (gravity + air
drag), so fitting that motion to the 2D detections pins down the depth.

Pipeline (works on the output of ball_track.py):

  1. Camera: build the full pinhole camera (focal length, rotation,
     position) from the floor calibration Padex already has. Assumes square
     pixels and the principal point at the image centre.
  2. For every flight (from one hit ball_track found to the next), simulate
     the ball with gravity, air drag and bounces off the floor and the glass
     walls, and fit its launch position/velocity plus how bouncy the floor
     and walls are, so the simulated flight, projected into the image,
     matches every detection of that flight.
  3. Soft constraints: at the hit the ball is near the hitter and at a
     plausible height; at the next hit, near the next hitter.
  4. Output per flight: 3D path, bounces (floor and wall, with positions),
     launch speed, max height, fit quality and a height uncertainty.

Units: metres, seconds. Court frame: x 0..10 across, y 0..20 along, z up,
net at y = 10.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from scipy.optimize import least_squares

G = 9.81
NET_Y = 10.0
NET_TOP = 0.88       # net height in the middle (0.92 m at the posts)
# Padel ball: ~57 g, ~6.5 cm diameter, Cd ~0.55 (similar to a tennis ball)
DRAG_K = 0.5 * 1.2 * 0.55 * np.pi * 0.0325**2 / 0.057  # 1/m


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    K: np.ndarray        # 3x3 intrinsics
    R: np.ndarray        # world -> camera rotation
    t: np.ndarray        # world -> camera translation
    width: int
    height: int

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    def project(self, X: np.ndarray) -> np.ndarray:
        """(N,3) world points -> (N,2) pixels."""
        Xc = X @ self.R.T + self.t
        z = np.maximum(Xc[:, 2:3], 1e-2)   # points behind the camera land far away, not at infinity
        uv = Xc[:, :2] / z
        return uv * self.focal + self.K[:2, 2]

    def ray(self, px: tuple[float, float]) -> np.ndarray:
        d = np.linalg.solve(self.K, np.array([px[0], px[1], 1.0]))
        d = self.R.T @ d
        return d / np.linalg.norm(d)

    def point_at_height(self, px: tuple[float, float], z: float) -> np.ndarray:
        """Where the viewing ray through px crosses the plane at height z."""
        c, d = self.centre, self.ray(px)
        s = (z - c[2]) / d[2] if abs(d[2]) > 1e-9 else 1e9
        return c + s * d


def _decompose(H_m2px: np.ndarray, f: float, cx: float, cy: float):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    A = np.linalg.solve(K, H_m2px)
    lam = 1.0 / np.linalg.norm(A[:, 0])
    if (lam * A[:, 2])[2] < 0:  # camera must see the court in front of it
        lam = -lam
    r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
    R = np.stack([r1, r2, np.cross(r1, r2)], axis=1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return K, R, t


def camera_from_homography(H_px2m: np.ndarray, width: int, height: int,
                           floor_px=None, floor_m=None) -> Camera:
    """Full camera from a pixel->floor homography.

    The focal length comes from requiring the floor axes to be perpendicular
    and equally scaled, then (if floor points are given) is refined by
    minimising their reprojection error.
    """
    H = np.linalg.inv(H_px2m)
    H = H / H[2, 2]
    cx, cy = width / 2, height / 2
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    h = T @ H
    h1, h2 = h[:, 0], h[:, 1]
    cands = []
    den = h1[2] * h2[2]
    if abs(den) > 1e-12:
        cands.append(-(h1[0] * h2[0] + h1[1] * h2[1]) / den)
    den = h1[2] ** 2 - h2[2] ** 2
    if abs(den) > 1e-12:
        cands.append((h2[0] ** 2 + h2[1] ** 2 - h1[0] ** 2 - h1[1] ** 2) / den)
    f2 = [c for c in cands if c > 0]
    f = float(np.sqrt(np.median(f2))) if f2 else float(width)
    K, R, t = _decompose(H, f, cx, cy)
    cam = Camera(K, R, t, width, height)
    if floor_px is not None and floor_m is not None and len(floor_px) >= 4:
        world = np.c_[np.asarray(floor_m, float), np.zeros(len(floor_m))]
        cam = refine_camera(cam, np.asarray(floor_px, float), world)
    return cam


def refine_camera(cam: Camera, px: np.ndarray, world: np.ndarray) -> Camera:
    """Refine focal length + pose on known points (floor and/or at heights)."""
    rvec0 = cv2.Rodrigues(cam.R)[0].ravel()
    p0 = np.r_[np.log(cam.focal), rvec0, cam.t]

    def build(p):
        f = np.exp(p[0])
        K = np.array([[f, 0, cam.K[0, 2]], [0, f, cam.K[1, 2]], [0, 0, 1.0]])
        R = cv2.Rodrigues(p[1:4])[0]
        return Camera(K, R, p[4:7], cam.width, cam.height)

    def res(p):
        return (build(p).project(world) - px).ravel()

    sol = least_squares(res, p0, method="lm" if len(px) * 2 >= 7 else "trf")
    return build(sol.x)


# ---------------------------------------------------------------------------
# flight model: gravity + drag + bounces off the floor and the walls
# ---------------------------------------------------------------------------
#
# A "flight" is everything between one hit and the next. The ball's whole
# journey (including floor and wall bounces) follows from 11 numbers: where
# and how fast it left the racket (6) and how the floor and walls bounce it
# (5, fitted per flight because real bounces vary with spin and surface). Fitting those to every
# detection of the flight is far better constrained than fitting short
# arcs one at a time: the bounces pin down the depth.

PARAM_NAMES = ("x", "y", "z", "vx", "vy", "vz", "e_floor", "keep_floor", "e_back", "e_side", "keep_wall")
LO = np.array([-1.0, -1.0, 0.0, -45, -45, -30, 0.45, 0.55, 0.35, 0.15, 0.4])
HI = np.array([11.0, 21.0, 4.0, 45, 45, 30, 0.92, 1.0, 0.95, 0.95, 1.0])
DEFAULTS = np.array([0.72, 0.85, 0.7, 0.6, 0.9])   # bounce parameters to start from
ROLL_VZ = 0.6                                    # m/s: slower floor impacts count as rolling, not bounces
# e_back: bounciness off the back glass; e_side: off the side walls (part glass,
# part mesh fence, so usually deader); keep_wall: share of the along-the-wall
# speed kept in a wall bounce (friction).


def simulate_batch(P: np.ndarray, n_steps: int, dt: float, sample_steps: np.ndarray,
                   record: bool = False, net: bool = False):
    """Simulate B flights at once.

    P: (B, 9) parameters. Returns positions at sample_steps, shape (B, S, 3);
    with record=True (B must be 1) also the list of bounces; with net=True
    also the lowest height at which each flight crossed the net (inf if it
    never did).
    """
    B = P.shape[0]
    x, v = P[:, 0:3].copy(), P[:, 3:6].copy()
    ef, kf, ew = P[:, 6], P[:, 7], P[:, 8]
    es = P[:, 9] if P.shape[1] > 9 else ew
    kw = P[:, 10] if P.shape[1] > 10 else np.ones(B)
    out = np.zeros((B, len(sample_steps), 3))
    want: dict[int, list[int]] = {}
    for i, st in enumerate(sample_steps):
        want.setdefault(int(st), []).append(i)   # several samples may share a step
    events = []
    g = np.array([0.0, 0.0, -G])
    net_z = np.full(B, np.inf)
    for k in range(n_steps + 1):
        if k in want:
            for i in want[k]:
                out[:, i] = x
        if k == n_steps:
            break
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        v = v + (g - DRAG_K * sp * v) * dt
        y_before = x[:, 1].copy()
        x = x + v * dt
        if net:
            crossed = (y_before - NET_Y) * (x[:, 1] - NET_Y) < 0
            if crossed.any():
                net_z[crossed] = np.minimum(net_z[crossed], x[crossed, 2])
        # floor
        hit = (x[:, 2] < 0) & (v[:, 2] < 0)
        if hit.any():
            roll = hit & (v[:, 2] > -ROLL_VZ)       # too slow to bounce: the ball rolls
            bounce = hit & ~roll
            x[roll, 2] = 0.0
            v[roll, 2] = 0.0
            x[bounce, 2] = -x[bounce, 2]
            v[bounce, 2] = -ef[bounce] * v[bounce, 2]
            v[bounce, :2] *= kf[bounce, None]
            if record and bounce[0]:
                events.append(("ground", (k + 1) * dt, x[0].copy()))
        # walls (glass box 10 x 20)
        for ax, lo, hi in ((0, 0.0, 10.0), (1, 0.0, 20.0)):
            lo_hit = (x[:, ax] < lo) & (v[:, ax] < 0)
            hi_hit = (x[:, ax] > hi) & (v[:, ax] > 0)
            for m, edge in ((lo_hit, lo), (hi_hit, hi)):
                if m.any():
                    x[m, ax] = 2 * edge - x[m, ax]
                    v[m, ax] = -(es[m] if ax == 0 else ew[m]) * v[m, ax]
                    other = 1 - ax
                    v[m, other] *= kw[m]
                    v[m, 2] *= kw[m]
                    if record and m[0]:
                        events.append(("wall", (k + 1) * dt, x[0].copy()))
    if record:
        return out, events
    return (out, net_z) if net else out


@dataclass
class Flight:
    hit_time_s: float                 # when the ball left the racket
    params: np.ndarray                # 11 numbers, see PARAM_NAMES
    frames: np.ndarray                # detection frames used
    detections: np.ndarray            # (N,2) pixels
    rms_px: float
    height_sd: float                  # 1-sigma height uncertainty mid-flight (m)
    end_time_s: float
    fps: float
    hitter: str | None = None
    bounces: list = field(default_factory=list)   # [(kind, time_s, xyz)]
    ok: bool = True

    def _steps(self, times_s):
        dt = 1.0 / (self.fps * 8)
        idx = np.round((np.asarray(times_s) - self.hit_time_s) / dt).astype(int)
        return idx, dt

    def at(self, times_s) -> np.ndarray:
        idx, dt = self._steps(times_s)
        idx = np.clip(idx, 0, None)
        return simulate_batch(self.params[None], int(idx.max()), dt, idx)[0]

    @property
    def launch_speed(self) -> float:
        return float(np.linalg.norm(self.params[3:6]))

    @property
    def max_height(self) -> float:
        ts = np.linspace(self.hit_time_s, self.end_time_s, 60)
        return float(self.at(ts)[:, 2].max())

    @property
    def landing(self):
        g = [b for b in self.bounces if b[0] == "ground"]
        return g[0] if g else None


@dataclass
class FitSettings:
    pixel_sd: float = 2.0             # detection noise (pixels)
    hit_xy_sd: float = 1.5            # ball within ~1.5 m of the hitter's feet at contact...
    hit_z: float = 1.2                # ...at about 1.2 m...
    hit_z_sd: float = 1.0             # ...give or take 1 m
    min_points: int = 6
    max_flight_s: float = 3.5
    outlier_px: float = 8.0           # detections this far from ball_track's 2D track are dropped
    bad_rms_px: float = 5.0           # fits worse than this are flagged not ok
    substeps: int = 8                 # simulation steps per video frame
    use_net: bool = True              # ball must clear the net when it crosses it in the air
    loss: str = "soft_l1"             # robust: a few wrong detections don't bend the whole flight
    n_starts: int = 6                 # best starting guesses refined by the optimiser
    max_nfev: int = 120
    f_scale: float = 2.0              # residuals beyond ~2 sigma (4 px) count less


def _residuals_batch(P, cam, px, sample_steps, n_steps, dt, s, hit_xy, end_xy, end_step):
    """Residual vectors for a batch of parameter sets, shape (B, M)."""
    steps = np.r_[sample_steps, end_step] if end_step is not None else sample_steps
    if s.use_net:
        pos, net_z = simulate_batch(P, n_steps, dt, steps, net=True)
    else:
        pos, net_z = simulate_batch(P, n_steps, dt, steps), None
    det = pos[:, : len(sample_steps)]
    B = P.shape[0]
    proj = np.stack([cam.project(det[b]) for b in range(B)])
    r = [((proj - px[None]) / s.pixel_sd).reshape(B, -1)]
    x0 = P[:, :3]
    if hit_xy is not None:
        r.append((x0[:, :2] - np.asarray(hit_xy)[None]) / s.hit_xy_sd)
    r.append(((x0[:, 2] - s.hit_z) / s.hit_z_sd)[:, None])
    if end_step is not None:
        q = pos[:, -1]
        if end_xy is not None:
            r.append((q[:, :2] - np.asarray(end_xy)[None]) / s.hit_xy_sd)
        r.append(((q[:, 2] - s.hit_z) / s.hit_z_sd)[:, None])
    if net_z is not None:
        # a ball crossing the net in the air must clear the tape
        under = np.where(np.isfinite(net_z), np.maximum(0.0, NET_TOP + 0.03 - net_z), 0.0)
        r.append((under / 0.05)[:, None])
    out = np.concatenate(r, axis=1)
    return np.nan_to_num(out, nan=1e4, posinf=1e4, neginf=-1e4)


def _candidates(cam, first_px, hit_xy, opp_far: bool | None):
    """Starting guesses: contact height x landing target x time to landing."""
    P = []
    for z0 in (0.4, 0.9, 1.5, 2.2, 3.0):
        X0 = cam.point_at_height(tuple(first_px), z0)
        if not np.all(np.isfinite(X0)):
            continue
        if hit_xy is not None and np.hypot(*(X0[:2] - hit_xy)) > 4:
            continue
        halves = [opp_far] if opp_far is not None else [True, False]
        for far in halves:
            for tx in (1.5, 5.0, 8.5):
                for ty in ((12.5, 15.5, 18.5) if far else (1.5, 4.5, 7.5)):
                    for T in (0.45, 0.75, 1.1, 1.6, 2.2):
                        V = np.array([(tx - X0[0]) / T, (ty - X0[1]) / T, (0 - X0[2] + 0.5 * G * T**2) / T])
                        P.append(np.r_[X0, V, DEFAULTS])
    return np.clip(np.array(P), LO, HI) if P else np.empty((0, 9))


def _kink_candidates(cam, t: np.ndarray, px: np.ndarray, n_kinks: int = 3) -> np.ndarray:
    """Starting guesses from the 2D track: the sharpest kinks are likely floor
    bounces, so aim the launch at the kink's floor position, arriving at the
    kink's time."""
    if len(t) < 7:
        return np.empty((0, len(LO)))
    turn = np.zeros(len(t))
    for k in range(2, len(t) - 2):
        v1 = (px[k] - px[k - 2]) / max(t[k] - t[k - 2], 1e-3)
        v2 = (px[k + 2] - px[k]) / max(t[k + 2] - t[k], 1e-3)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            turn[k] = 1 - float(np.dot(v1, v2) / (n1 * n2)) + abs(n2 - n1) / (n1 + n2)
    P = []
    for k in np.argsort(turn)[::-1][:n_kinks]:
        if turn[k] <= 0.05 or t[k] <= 0.08:
            continue
        L = cam.point_at_height(tuple(px[k]), 0.0)
        if not np.all(np.isfinite(L)) or not (-1 < L[0] < 11 and -1 < L[1] < 21):
            continue
        T = t[k]
        for z0 in (0.3, 0.8, 1.4, 2.2, 3.0):
            X0 = cam.point_at_height(tuple(px[0]), z0)
            if not np.all(np.isfinite(X0)):
                continue
            V = np.array([(L[0] - X0[0]) / T, (L[1] - X0[1]) / T, (0 - X0[2] + 0.5 * G * T**2) / T])
            P.append(np.r_[X0, V, DEFAULTS])
    return np.clip(np.array(P), LO, HI) if P else np.empty((0, len(LO)))


def fit_flight(cam: Camera, frames: np.ndarray, px: np.ndarray, fps: float, hit_time_s: float,
               end_time_s: float, hit_xy, end_xy, s: FitSettings) -> Flight | None:
    if len(frames) < s.min_points:
        return None
    dt = 1.0 / (fps * s.substeps)
    sample_steps = np.round((frames / fps - hit_time_s) / dt).astype(int)
    sample_steps = np.clip(sample_steps, 0, None)
    end_step = int(round((end_time_s - hit_time_s) / dt)) if end_xy is not None or end_time_s else None
    n_steps = int(max(sample_steps.max(), end_step or 0))
    opp_far = None if hit_xy is None else bool(hit_xy[1] < 10)

    C = _candidates(cam, px[0], hit_xy, opp_far)
    K = _kink_candidates(cam, frames / fps - hit_time_s, px)
    if len(K):
        C = np.vstack([C, K]) if len(C) else K
    if len(C) == 0:
        return None
    args = (cam, px, sample_steps, n_steps, dt, s, hit_xy, end_xy, end_step)
    costs = []
    for i in range(0, len(C), 256):
        R = _residuals_batch(C[i:i + 256], *args)
        costs.append(np.sum(R**2, axis=1))
    costs = np.concatenate(costs)
    starts = C[np.argsort(costs)[: s.n_starts]]

    def fun(p):
        return _residuals_batch(p[None], *args)[0]

    npar = len(LO)

    def jac(p):
        h = np.where(np.arange(npar) // 3 == 1, 1e-2, 1e-3)
        Pb = np.repeat(p[None], npar + 1, axis=0)
        for i in range(npar):
            Pb[i + 1, i] += h[i]
        Rb = _residuals_batch(Pb, *args)
        return ((Rb[1:] - Rb[0]) / h[:, None]).T

    best = None
    for p0 in starts:
        sol = least_squares(fun, p0, jac=jac, bounds=(LO, HI), method="trf", max_nfev=s.max_nfev,
                            loss=s.loss, f_scale=s.f_scale)
        if best is None or sol.cost < best.cost:
            best = sol
    p = best.x
    pos, events = simulate_batch(p[None], n_steps, dt, sample_steps, record=True)
    proj = cam.project(pos[0])
    err = np.hypot(*(proj - px).T)
    # quality: rms over the best 90% of detections, so one bad label doesn't fail a good flight
    good = np.sort(err)[: max(1, int(np.ceil(0.9 * len(err))))]
    rms = float(np.sqrt(np.mean(good**2)))

    # height uncertainty mid-flight from the fit's Jacobian
    try:
        J = best.jac
        cov = np.linalg.pinv(J.T @ J)
        mid = np.array([sample_steps[len(sample_steps) // 2]])
        Pb = np.repeat(p[None], npar + 1, axis=0)
        h = 1e-3
        for i in range(npar):
            Pb[i + 1, i] += h
        zz = simulate_batch(Pb, int(mid[0]), dt, mid)[:, 0, 2]
        gz = (zz[1:] - zz[0]) / h
        height_sd = float(np.sqrt(max(gz @ cov @ gz, 0)))
    except np.linalg.LinAlgError:
        height_sd = float("nan")

    bounces = [(k, hit_time_s + t, xyz) for k, t, xyz in events]
    return Flight(hit_time_s, p, frames, px, rms, height_sd, end_time_s, fps, None, bounces,
                  ok=rms <= s.bad_rms_px)


# ---------------------------------------------------------------------------
# whole match
# ---------------------------------------------------------------------------


@dataclass
class Ball3DResult:
    camera: Camera
    flights: list[Flight]
    positions: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> xyz (ok flights only)


class _FlightSolver:
    """Everything needed to fit the flight after one hit. Picklable, so the
    hits of a match can be fitted in separate processes."""

    def __init__(self, cam: Camera, fps: float, s: FitSettings, ball_px: dict, track: dict,
                 breaks: list, player_pos: dict, player_boxes: dict | None):
        self.cam, self.fps, self.s = cam, fps, s
        self.ball_px, self.track, self.breaks = ball_px, track, breaks
        self.player_pos, self.player_boxes = player_pos, player_boxes

    def detections(self, t_a, t_b):
        fps, s, ball_px, track = self.fps, self.s, self.ball_px, self.track
        f0, f1 = int(np.ceil(t_a * fps)) + 1, int(np.floor(t_b * fps)) - 1
        fr = np.array([f for f in range(f0, f1 + 1) if f in ball_px and f in track])
        if len(fr) == 0:
            return fr, np.empty((0, 2))
        px = np.array([ball_px[f] for f in fr], float)
        tr = np.array([track[f] for f in fr], float)
        keep = np.hypot(*(px - tr).T) <= s.outlier_px
        return fr[keep], px[keep]

    def who_hit(self, t):
        """Player within reach of the 2D ball position at time t, if any."""
        if self.player_boxes is None:
            return None
        f = int(round(t * self.fps))
        p = self.track.get(f)
        if p is None:
            return None
        from ball_track import BallSettings, players_near
        name, _ = players_near(f, self.player_boxes, p, BallSettings())
        return name

    def fit(self, t_a, t_b, hitter, end_xy):
        fr, px = self.detections(t_a, t_b)
        hit_xy = self.player_pos.get((int(round(t_a * self.fps)), hitter)) if hitter else None
        fl = fit_flight(self.cam, fr, px, self.fps, t_a, t_b, hit_xy, end_xy, self.s)
        if fl is not None:
            fl.hitter = hitter
        return fl

    def solve(self, t_a, t_b, hitter, end_xy, depth=0):
        fps, player_pos = self.fps, self.player_pos
        fl = self.fit(t_a, t_b, hitter, end_xy)
        if fl is None or fl.ok or depth >= 2:
            return [fl] if fl is not None else []
        cands = [t for t in self.breaks if t_a + 0.3 < t < t_b - 0.3]
        # prefer breaks within a player's reach (likely missed hits)
        cands.sort(key=lambda t: (self.who_hit(t) is None, abs(t - (t_a + t_b) / 2)))
        best, best_cost = None, np.inf
        for t in cands[:4]:
            mid_hitter = self.who_hit(t)
            mid_xy = player_pos.get((int(round(t * fps)), mid_hitter)) if mid_hitter else None
            left = self.fit(t_a, t, hitter, mid_xy)
            right = self.fit(t, t_b, mid_hitter, end_xy)
            if left is None or right is None:
                continue
            cost = left.rms_px ** 2 * len(left.frames) + right.rms_px ** 2 * len(right.frames)
            if left.ok and right.ok and cost < best_cost:
                best, best_cost = (t, mid_hitter, mid_xy), cost
        if best is None:
            return [fl]
        t, mh, mxy = best
        return self.solve(t_a, t, hitter, mxy, depth + 1) + self.solve(t, t_b, mh, end_xy, depth + 1)


_worker_solver: _FlightSolver | None = None


def _init_worker(solver: _FlightSolver) -> None:
    global _worker_solver
    _worker_solver = solver


def _solve_job(job) -> list:
    return _worker_solver.solve(*job)


@contextmanager
def _importable_main():
    """Let worker processes start when the calling script was piped in
    (``python - <<EOF``, as run_match.sh does).

    On Windows and macOS a worker re-runs the main script from its path. A
    piped script has no real path ("<stdin>"), the worker dies on start and
    the pool waits for ever. The workers only need this module, so hide the
    fake path while they are started.
    """
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    hide = path is not None and getattr(main, "__spec__", None) is None and not os.path.isfile(path)
    if hide:
        del main.__file__
    try:
        yield
    finally:
        if hide:
            main.__file__ = path


def _default_workers() -> int:
    """Processes for the flight fits: $PADEX_WORKERS, else half the logical
    CPUs (roughly the physical cores; the fit is pure number crunching)."""
    env = os.environ.get("PADEX_WORKERS", "").strip()
    if env:
        return max(1, int(env))
    return max(1, (os.cpu_count() or 2) // 2)


def reconstruct(ball_result, ball_px: dict[int, tuple[float, float]], player_pos: dict,
                cam: Camera, fps: float, s: FitSettings | None = None,
                player_boxes: dict | None = None, workers: int | None = None) -> Ball3DResult:
    """Fit a 3D flight between every pair of consecutive hits ball_track found.

    A flight that can't be explained by one launch (bad fit) usually hides
    a hit the 2D step missed. Such flights are split at the most likely
    point (a break in the 2D track, preferably within a player's reach) and
    each half is fitted again; the split is kept if both halves fit.

    The hits are independent of each other, so they are fitted in ``workers``
    processes at once (default: see _default_workers; 1 = no extra processes).
    The result is the same whatever the number of workers.
    """
    s = s or FitSettings()
    hits = sorted((e for e in ball_result.events if e.kind == "hit"), key=lambda e: e.time_s)
    track = ball_result.track  # frame -> 2D track position (pixels)
    breaks = sorted({e.time_s for e in ball_result.events if e.kind != "hit"}
                    | {a.t0 / fps for a in ball_result.arcs} | {a.t1 / fps for a in ball_result.arcs})
    solver = _FlightSolver(cam, fps, s, ball_px, track, breaks, player_pos, player_boxes)

    jobs = []
    for i, h in enumerate(hits):
        nxt = hits[i + 1] if i + 1 < len(hits) else None
        if nxt is not None and nxt.time_s - h.time_s <= s.max_flight_s * 2:
            end_t = nxt.time_s
            end_xy = player_pos.get((int(round(nxt.frame)), nxt.player))
        else:
            last = max((f for f in track if h.frame <= f <= h.frame + s.max_flight_s * fps), default=None)
            if last is None:
                continue
            end_t, end_xy = last / fps, None
        jobs.append((h.time_s, end_t, h.player, end_xy))

    workers = min(workers or _default_workers(), len(jobs))
    # Worker processes re-import the calling script, so only the main process
    # may start them.
    if workers > 1 and multiprocessing.current_process().name == "MainProcess":
        with _importable_main(), ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                                     initargs=(solver,)) as pool:
            solved = list(pool.map(_solve_job, jobs))
    else:
        solved = [solver.solve(*job) for job in jobs]
    flights: list[Flight] = [fl for part in solved for fl in part]

    positions: dict[int, np.ndarray] = {}
    for fl in flights:
        if not fl.ok:
            continue
        fr = np.arange(int(np.ceil(fl.hit_time_s * fps)), int(np.floor(fl.end_time_s * fps)) + 1)
        if len(fr):
            for f, p in zip(fr, fl.at(fr / fps)):
                positions[int(f)] = p
    return Ball3DResult(cam, flights, positions)


def frame_speeds(result: Ball3DResult, fps: float) -> dict[int, float]:
    """The ball's speed in km/h at every video frame covered by a well-fitted
    flight: frame -> speed.

    It is the speed along the fitted 3D path (not across the image), so it
    does not depend on where the ball is on screen. Frames between flights,
    and flights whose fit is poor, have no entry.
    """
    out: dict[int, float] = {}
    for fl in result.flights:
        if not fl.ok:
            continue
        fr = np.arange(int(np.ceil(fl.hit_time_s * fps)), int(np.floor(fl.end_time_s * fps)) + 1)
        if len(fr) == 0:
            continue
        # half a frame either side of each frame: a speed for the frame
        # itself, and none of them reaching across the hit that starts the
        # flight or the one that ends it
        t = np.clip(np.r_[fr / fps - 0.5 / fps, fr[-1] / fps + 0.5 / fps], fl.hit_time_s, fl.end_time_s)
        pos = fl.at(t)
        dt = np.diff(t)
        ok = dt > 1e-6
        speed = np.full(len(fr), np.nan)
        speed[ok] = np.linalg.norm(np.diff(pos, axis=0)[ok], axis=1) / dt[ok]
        for f, v in zip(fr, speed):
            if np.isfinite(v):
                out[int(f)] = float(v * 3.6)
    return out


def run_on_tracking(tracking, ball_result, fps: float, s: FitSettings | None = None,
                    workers: int | None = None) -> Ball3DResult | None:
    """Convenience wrapper: tracking = Padex TrackingResult with stable IDs."""
    cal = tracking.calibration
    if cal is None or ball_result is None:
        return None
    H = np.array(cal.homography_matrix, float)
    floor_px = cal.court_keypoints_px or None
    floor_m = cal.court_keypoints_m or None
    cam = camera_from_homography(H, cal.frame_width, cal.frame_height, floor_px, floor_m)
    # pixel settings are tuned on 720p; scale them with the frame height
    s = s or FitSettings()
    k = cal.frame_height / 720.0
    s = replace(s, pixel_sd=s.pixel_sd * k, outlier_px=s.outlier_px * k, bad_rms_px=s.bad_rms_px * k)
    ball_px = {bf.frame_id: ((bf.bbox.x1 + bf.bbox.x2) / 2, (bf.bbox.y1 + bf.bbox.y2) / 2)
               for bf in tracking.ball_frames if bf.bbox is not None}
    # where the 2D step chose another candidate of the frame, fit to that one
    ball_px.update(getattr(ball_result, "det", None) or {})
    pos = {(pf.frame_id, pf.player_id): (pf.position.x, pf.position.y)
           for pf in tracking.player_frames if pf.position is not None}
    boxes: dict[int, list] = {}
    for pf in tracking.player_frames:
        b = pf.bbox
        boxes.setdefault(pf.frame_id, []).append((pf.player_id, (b.x1, b.y1, b.x2, b.y2)))
    return reconstruct(ball_result, ball_px, pos, cam, fps, s, boxes, workers)
