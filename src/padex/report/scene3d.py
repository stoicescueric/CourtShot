"""
The 3D replay: court, players and ball in one self-contained HTML page.

Players are lifted to 3D like this: the pose model only gives 2D keypoints,
so each skeleton is placed on the vertical plane that stands on the player's
floor position and faces the camera. Seen from the broadcast angle it is
exact up to that flattening; from the side the bodies look flat.

The ball comes from the fitted 3D flights (padex.analysis.ball_flight).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from padex.analysis import ball_flight as b3
from padex.analysis.match import MatchAnalysis

TEMPLATE = Path(__file__).resolve().parent / "templates" / "scene3d.html"

JOINTS = [
    "head",
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
]
HEAD_PARTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
MIN_CONF = 0.35
NONE = -32000  # "no value" marker in the integer arrays


def T(p):
    """Court coordinates (x across, y along, z up; metres) -> viewer (cm, y up)."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0] - 5.0, p[..., 2], 10.0 - p[..., 1]], axis=-1)


def gauss_smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth along axis 0, ignoring NaNs."""
    if sigma <= 0:
        return a
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    ok = np.isfinite(a)
    num = np.zeros_like(a, dtype=float)
    den = np.zeros_like(a, dtype=float)
    pad = lambda x: np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1))
    A, M = pad(np.where(ok, a, 0.0)), pad(ok.astype(float))
    for i, w in enumerate(k):
        num += w * A[i : i + len(a)]
        den += w * M[i : i + len(a)]
    out = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
    return np.where(ok | (den > 0.3), out, np.nan)


def fill_short_gaps(a: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over NaN runs up to max_gap frames (axis 0, column-wise)."""
    a = a.copy()
    flat = a.reshape(len(a), -1)
    idx = np.arange(len(a))
    for c in range(flat.shape[1]):
        col = flat[:, c]
        good = np.isfinite(col)
        if good.sum() < 2:
            continue
        g = idx[good]
        for lo, hi in zip(g[:-1], g[1:]):
            if 1 < hi - lo <= max_gap + 1:
                col[lo + 1 : hi] = np.interp(idx[lo + 1 : hi], [lo, hi], [col[lo], col[hi]])
    return flat.reshape(a.shape)


def pose_matches_box(pf, min_iou: float = 0.4) -> bool:
    """The pose model sometimes hands a player another frame's or another
    person's keypoints (about 1 frame in 10 here). Keep a pose only when the
    box around its confident keypoints overlaps the player's detection box."""
    k = np.array([(q.x, q.y) for q in pf.keypoints if q.confidence >= MIN_CONF])
    if len(k) < 8:
        return False
    b = pf.bbox
    kx1, ky1, kx2, ky2 = k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()
    iw = max(0.0, min(b.x2, kx2) - max(b.x1, kx1))
    ih = max(0.0, min(b.y2, ky2) - max(b.y1, ky1))
    inter = iw * ih
    union = (b.x2 - b.x1) * (b.y2 - b.y1) + (kx2 - kx1) * (ky2 - ky1) - inter
    return union > 0 and inter / union >= min_iou


def lift_players(stable, cam: b3.Camera, n_frames: int, hits: list) -> list[dict]:
    C = cam.centre
    players = []
    ids = sorted({pf.player_id for pf in stable.player_frames})
    for pid in ids:
        pfs = [pf for pf in stable.player_frames if pf.player_id == pid and pf.frame_id < n_frames]
        team = next((pf.team_id for pf in pfs if pf.team_id), None)
        floor = np.full((n_frames, 2), np.nan)
        kp = np.full((n_frames, len(JOINTS), 2), np.nan)
        for pf in pfs:
            if pf.position is not None:
                floor[pf.frame_id] = (pf.position.x, pf.position.y)
            if not pf.keypoints or not pose_matches_box(pf):
                continue
            k = {q.name: q for q in pf.keypoints}
            head = [(k[n].x, k[n].y, k[n].confidence) for n in HEAD_PARTS if n in k and k[n].confidence >= MIN_CONF]
            if head:
                h = np.array(head)
                kp[pf.frame_id, 0] = np.average(h[:, :2], axis=0, weights=h[:, 2])
            b = pf.bbox
            mx, my = 0.25 * (b.x2 - b.x1), 0.15 * (b.y2 - b.y1)
            for j, name in enumerate(JOINTS[1:], start=1):
                q = k.get(name)
                if (
                    q is not None
                    and q.confidence >= MIN_CONF
                    and b.x1 - mx <= q.x <= b.x2 + mx
                    and b.y1 - my <= q.y <= b.y2 + my
                ):
                    kp[pf.frame_id, j] = (q.x, q.y)
        present = np.isfinite(floor[:, 0])
        floor = gauss_smooth(fill_short_gaps(floor, 6), 1.5)
        kp = gauss_smooth(fill_short_gaps(kp, 8), 1.0)

        # lift: keypoint ray meets the vertical plane through the feet, facing the camera
        pose = np.full((n_frames, len(JOINTS), 3), np.nan)
        for f in np.flatnonzero(present & np.isfinite(floor[:, 0])):
            F = np.array([floor[f, 0], floor[f, 1], 0.0])
            nrm = F - C
            nrm[2] = 0
            nrm /= max(np.linalg.norm(nrm), 1e-6)
            for j in range(len(JOINTS)):
                if not np.all(np.isfinite(kp[f, j])):
                    continue
                d = cam.ray(tuple(kp[f, j]))
                den = float(nrm @ d)
                if abs(den) < 1e-6:
                    continue
                X = C + d * float(nrm @ (F - C)) / den
                X[2] = max(X[2], 0.03)
                pose[f, j] = X

        # frames with a position but no pose: carry the last pose shape along
        rel_last = None
        default = np.array(
            [
                [0, 0, 1.68],
                [-0.2, 0, 1.45],
                [0.2, 0, 1.45],
                [-0.25, 0, 1.15],
                [0.25, 0, 1.15],
                [-0.25, 0, 0.9],
                [0.25, 0, 0.9],
                [-0.12, 0, 0.95],
                [0.12, 0, 0.95],
                [-0.13, 0, 0.5],
                [0.13, 0, 0.5],
                [-0.14, 0, 0.06],
                [0.14, 0, 0.06],
            ]
        )
        for f in range(n_frames):
            if not present[f] or not np.isfinite(floor[f, 0]):
                continue
            F = np.array([floor[f, 0], floor[f, 1], 0.0])
            ok = np.all(np.isfinite(pose[f]), axis=1)
            if ok.sum() >= 9:
                rel = pose[f] - F
                if rel_last is not None:
                    rel[~ok] = rel_last[~ok]
                else:
                    rel[~ok] = default[~ok]
                pose[f] = F + rel
                rel_last = rel
            else:
                rel = rel_last if rel_last is not None else default
                p = F + rel
                p[ok] = pose[f, ok]
                pose[f] = p

        # racket hand: the wrist nearer the ball at this player's hits
        votes = {"left": 0, "right": 0}
        for e in hits:
            if e.player != pid:
                continue
            f = int(round(e.frame))
            if 0 <= f < n_frames and np.all(np.isfinite(kp[f, 5:7])):
                dl = np.hypot(*(kp[f, 5] - e.px))
                dr = np.hypot(*(kp[f, 6] - e.px))
                votes["left" if dl < dr else "right"] += 1
        hand = "left" if votes["left"] > votes["right"] else "right"

        q = np.where(np.isfinite(pose), np.round(T(pose) * 100), NONE).astype(int)
        players.append(
            {
                "id": pid,
                "team": team or "?",
                "hand": hand,
                "pose": q.reshape(-1).tolist(),
            }
        )
    return players


def build_scene(stable, res2d, r3, fps: float) -> dict:
    n = max(pf.frame_id for pf in stable.player_frames) + 1
    n = max(n, len(stable.ball_frames))
    cam = r3.camera
    hits = [e for e in res2d.events if e.kind == "hit"]

    ball = np.full((n, 4), NONE, int)
    flights = []
    for fl in sorted(r3.flights, key=lambda f: f.hit_time_s):
        f0 = int(np.ceil(fl.hit_time_s * fps))
        f1 = int(np.floor(fl.end_time_s * fps))
        fr = np.arange(max(f0, 0), min(f1, n - 1) + 1)
        if len(fr) < 2:
            continue
        P = fl.at(fr / fps)
        ball[fr, :3] = np.round(T(P) * 100).astype(int)
        ball[fr, 3] = len(flights)
        ts = np.linspace(fl.hit_time_s, fl.end_time_s, 240)
        PP = fl.at(ts)
        cross = np.flatnonzero(np.diff(np.sign(PP[:, 1] - b3.NET_Y)) != 0)
        net_h = float(PP[cross[0] + 1, 2]) if len(cross) else None
        flights.append(
            {
                "t0": round(fl.hit_time_s, 3),
                "t1": round(fl.end_time_s, 3),
                "ok": bool(fl.ok),
                "hitter": fl.hitter,
                "speed_kmh": round(fl.launch_speed * 3.6, 1),
                "max_h": round(fl.max_height, 2),
                "net_h": None if net_h is None else round(net_h, 2),
                "rms_px": round(float(fl.rms_px), 1),
                "h_sd": round(float(fl.height_sd), 2) if np.isfinite(fl.height_sd) else None,
                "bounces": [
                    {"kind": k, "t": round(t, 3), "p": [round(v, 3) for v in T(xyz)]} for k, t, xyz in fl.bounces
                ],
            }
        )

    # broadcast camera in viewer coordinates
    Tv = lambda v: np.array([v[0], v[2], -v[1]])
    R = cam.R
    cam_json = {
        "pos": T(cam.centre).round(3).tolist(),
        "right": Tv(R[0]).round(5).tolist(),
        "up": Tv(-R[1]).round(5).tolist(),
        "back": Tv(-R[2]).round(5).tolist(),
        "fov": round(float(np.degrees(2 * np.arctan(cam.height / 2 / cam.focal))), 3),
        "aspect": round(cam.width / cam.height, 4),
    }
    return {
        "fps": fps,
        "n": int(n),
        "camera": cam_json,
        "players": lift_players(stable, cam, n, hits),
        "ball": ball.reshape(-1).tolist(),
        "flights": flights,
        "hits": [{"t": round(e.time_s, 3), "player": e.player} for e in hits],
        "rallies": [list(map(float, r[:2])) for r in res2d.rallies],
        "none": NONE,
    }


def write_scene(
    analysis: MatchAnalysis,
    out: str | Path,
    video: str | Path | None = None,
    fragment: bool = False,
    also_json: bool = False,
) -> Path:
    """Write the 3D replay page for an analysed match.

    video:     the match video shown next to the 3D (the page loads it by its
               path relative to ``out``); None = the page asks for one
    fragment:  write the page without <html>/<head> (for hosts that add them)
    also_json: also write the scene data next to the page, as .json

    Raises ValueError when there is no 3D ball fit to show.
    """
    if analysis.ball is None or analysis.ball3d is None:
        raise ValueError("no court calibration or ball track in this cache; cannot build a 3D scene")
    scene = build_scene(analysis.tracking, analysis.ball, analysis.ball3d, analysis.fps)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene["video"] = None
    if video is not None:
        video = Path(video)
        if video.suffix.lower() in (".mp4", ".mov", ".m4v", ".webm") and video.exists():
            scene["video"] = Path(os.path.relpath(video.resolve(), out.parent.resolve())).as_posix()
    data = json.dumps(scene, separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*SCENE_DATA*/null", data)
    if not fragment:
        html = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + html
            + "\n</html>\n"
        )
    out.write_text(html, encoding="utf-8")
    if also_json:
        out.with_suffix(".json").write_text(data, encoding="utf-8")
    return out
