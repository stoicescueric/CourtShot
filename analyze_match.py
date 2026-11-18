"""
Match report for a Padex match.

Reads the tracking cache Padex already saved (no video processing), applies
stable player IDs, and writes:

    output/analysis/report.html    the report: open it in a browser
    output/analysis/positions.csv  smoothed per-frame player data
    output/analysis/shots.csv      one row per shot
    output/analysis/ball_events.csv
    output/analysis/summary.json   the numbers in the report

Usage, from the padex folder (after at least one run_stable.py / run_match.sh run):

    uv run python analyze_match.py
    uv run python analyze_match.py --cache match_tracking_cache.pkl --out output/analysis

What is in it:
    movement   distance, speed, sprints, where each player stands (heatmaps),
               net / back time, team formations: from player positions only
    rallies    from the ball track when the ball is seen often enough, else
               from movement
    shots      type, speed, contact height, direction, target, landing spot:
               from the 2D ball events and the fitted 3D flights (match_stats.py)

The 3D ball fit is the slow part (a minute or two per minute of play). It is
saved next to the tracking cache and reused by export_scene3d.py; --no-3d
skips it (shot speeds are then averages to the landing spot, and lobs are
found by time in the air only).

Coordinates: court metres, x = 0..10 across, y = 0..20 along, net at y = 10.
"Own half" views flip each team so its back wall is at the bottom and the
net at the top, whichever end the team is playing from.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

import match_stats
from stable_ids import (
    SLOT_NAMES,
    SLOT_TEAMS,
    StableIdConfig,
    cache_appearance,
    frame_rate,
    stabilize_tracking,
)

try:
    import ball_track
except ImportError:  # ball_track.py not next to this script
    ball_track = None


@dataclass
class Settings:
    """All thresholds in one place. Metres, seconds, m/s."""

    smooth_window: float = 0.4    # Savitzky-Golay window for positions/speed
    still_speed: float = 0.5      # below this a player counts as standing (no distance)
    max_speed: float = 10.0       # faster than this = tracking glitch, ignored
    sprint_speed: float = 4.0     # a sprint is at least this fast...
    sprint_min: float = 0.3       # ...for at least this long
    peak_window: float = 0.25     # top speed = best average over this window
    rally_speed: float = 2.5      # rally if the fastest player is above this...
    rally_share: float = 0.1      # ...for at least this share of...
    rally_window: float = 5.0     # ...a sliding window of this length
    rally_min: float = 3.0        # shorter "rallies" are dropped
    rally_merge: float = 2.5      # pauses shorter than this don't end a rally
    net_zone: float = 4.0         # within this of the net = at the net
    back_zone: float = 7.0        # beyond this from the net = at the back (service line ~6.95 m)
    side_by_side: float = 2.0     # partners less than this apart in depth = side by side
    state_min: float = 0.5        # a team formation must last this long to count
    ball_rally_coverage: float = 0.4  # use ball-based rallies when the ball is seen at least this often
    rally_source: str = "movement"    # set at run time


NET_Y = 10.0
TEAMS = {"T_1": (0, 1), "T_2": (2, 3)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of consecutive True values, end exclusive."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def rolling_nanmean(a: np.ndarray, n: int) -> np.ndarray:
    n = max(1, n)
    ok = ~np.isnan(a)
    k = np.ones(n)
    s = np.convolve(np.where(ok, a, 0.0), k, mode="same")
    c = np.convolve(ok.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def pct(x: float) -> str:
    return "–" if x is None or np.isnan(x) else f"{100 * x:.0f}%"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_tracks(cache: Path, gap_fill: float):
    with open(cache, "rb") as f:  # your own Padex cache file
        raw = pickle.load(f)
    stable, id_report = stabilize_tracking(
        raw, StableIdConfig(max_gap_fill=gap_fill), cache_appearance(cache))
    pfs = [pf for pf in stable.player_frames if pf.player_id in SLOT_NAMES]
    if not pfs:
        raise SystemExit("No player positions in the cache. Is the court calibrated?")

    f0 = min(pf.frame_id for pf in pfs)
    f1 = max(pf.frame_id for pf in pfs)
    n = f1 - f0 + 1
    pos = np.full((n, 4, 2), np.nan)
    real = np.zeros((n, 4), dtype=bool)
    col = {name: i for i, name in enumerate(SLOT_NAMES)}
    stamps: dict[int, float] = {}
    for pf in pfs:
        i, j = pf.frame_id - f0, col[pf.player_id]
        pos[i, j] = (pf.position.x, pf.position.y)
        real[i, j] = pf.confidence > 0
        stamps[pf.frame_id] = pf.timestamp_ms / 1000.0

    fids = np.array(sorted(stamps))
    fps = frame_rate(raw)
    t = stamps[int(fids[0])] + (np.arange(n) + f0 - fids[0]) / fps
    return pos, real, t, fps, f0, id_report, stable


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------


def smooth(pos: np.ndarray, fps: float, s: Settings):
    """Savitzky-Golay smoothing per player, separately on each continuous stretch."""
    win = max(5, int(round(s.smooth_window * fps)) | 1)
    sm = np.full_like(pos, np.nan)
    vel = np.full_like(pos, np.nan)
    for j in range(pos.shape[1]):
        p = pos[:, j]
        for a, b in runs(~np.isnan(p[:, 0])):
            seg = p[a:b]
            if b - a >= win:
                sm[a:b, j] = savgol_filter(seg, win, 2, axis=0)
                vel[a:b, j] = savgol_filter(seg, win, 2, deriv=1, delta=1 / fps, axis=0)
            elif b - a >= 3:
                sm[a:b, j] = seg
                vel[a:b, j] = np.gradient(seg, 1 / fps, axis=0)
            else:
                sm[a:b, j] = seg
    speed = np.linalg.norm(vel, axis=2)
    speed[speed > s.max_speed] = np.nan
    return sm, speed


def detect_rallies(speed: np.ndarray, fps: float, s: Settings):
    """Rallies from movement: someone keeps making fast moves.

    Between points players walk (about 1-1.5 m/s). In a rally they don't move
    all the time, but someone bursts above walking pace every few seconds.
    So the signal is the share of a sliding window in which the fastest
    player is above rally_speed.
    """
    with np.errstate(all="ignore"):
        fastest = np.where(np.all(np.isnan(speed), axis=1), np.nan, np.nanmax(speed, axis=1))
    fast = (np.nan_to_num(fastest) > s.rally_speed).astype(float)
    activity = rolling_nanmean(fast, int(s.rally_window * fps))
    active = activity >= s.rally_share

    spans = runs(active)
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] < s.rally_merge * fps:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    rallies = [(a, b) for a, b in merged if b - a >= s.rally_min * fps]
    in_rally = np.zeros(len(speed), dtype=bool)
    for a, b in rallies:
        in_rally[a:b] = True
    return rallies, in_rally, activity


# ---------------------------------------------------------------------------
# tactics
# ---------------------------------------------------------------------------


def team_half(sm: np.ndarray, team: tuple[int, int]) -> np.ndarray:
    """-1 if the team is on the near half (y < 10), +1 if far, NaN if unseen."""
    with warnings.catch_warnings():
        # frames where neither player was seen: the mean is NaN, as documented
        warnings.simplefilter("ignore", RuntimeWarning)
        y = np.nanmean(sm[:, list(team), 1], axis=1)
    return np.where(np.isnan(y), np.nan, np.where(y < NET_Y, -1.0, 1.0))


def own_half_coords(sm: np.ndarray, half: np.ndarray, j: int):
    """(lateral, depth) in the player's own-half view.

    lateral: 0 = the player's left side looking at the net, 10 = right.
    depth:   distance from the net, 0 = net, 10 = back wall.
    The camera is behind the near baseline, so the near team faces the same
    way as the camera (their left = x 0) and the far team faces the camera.
    """
    x, y = sm[:, j, 0], sm[:, j, 1]
    lateral = np.where(half < 0, x, 10.0 - x)
    depth = np.abs(y - NET_Y)
    return lateral, depth


def formation_runs(state: np.ndarray, rallies, fps: float, s: Settings):
    """Count net losses / regains inside rallies from a per-frame state array.

    state: +1 both at net, -1 both at back, 0 anything else, NaN unknown.
    """
    min_len = int(s.state_min * fps)
    lost, regained, recover_times = 0, 0, []
    for a, b in rallies:
        seq = []
        st = state[a:b]
        for val in (1.0, -1.0):
            for ra, rb in runs(st == val):
                if rb - ra >= min_len:
                    seq.append((ra, rb, val))
        seq.sort()
        for (pa, _, pv), (na, _, nv) in zip(seq, seq[1:]):
            if pv == 1 and nv == -1:
                lost += 1
            elif pv == -1 and nv == 1:
                regained += 1
                recover_times.append((na - pa) / fps)
    return lost, regained, recover_times


def analyse(pos, real, t, fps, s: Settings, rally_times=None):
    """rally_times: optional [(start_s, end_s)] from the ball, used instead of
    movement-based rally detection."""
    sm, speed = smooth(pos, fps, s)
    rallies, in_rally, activity = detect_rallies(speed, fps, s)
    if rally_times:
        rallies = []
        for a_s, b_s in rally_times:
            a = int(np.searchsorted(t, a_s))
            b = int(np.searchsorted(t, b_s, side="right"))
            if b - a >= 2:
                rallies.append((a, min(b, len(t))))
        in_rally = np.zeros(len(t), dtype=bool)
        for a, b in rallies:
            in_rally[a:b] = True
    dt = 1.0 / fps

    halves = {tm: team_half(sm, idx) for tm, idx in TEAMS.items()}
    own = {}
    players = {}
    for j, name in enumerate(SLOT_NAMES):
        team = SLOT_TEAMS[j]
        lateral, depth = own_half_coords(sm, halves[team], j)
        own[name] = (lateral, depth)
        sp = speed[:, j]
        seen = ~np.isnan(pos[:, j, 0])
        moving = np.nan_to_num(sp) >= s.still_speed
        peak = np.nanmax(rolling_nanmean(sp, int(s.peak_window * fps))) if np.any(~np.isnan(sp)) else np.nan
        sprints = sum(
            1 for a, b in runs(np.nan_to_num(sp) >= s.sprint_speed) if b - a >= s.sprint_min * fps
        )
        r = in_rally & seen
        # 1 m squares of the own half; "covered" = at least half a second spent there
        cells = np.zeros((10, 10))
        if r.any():
            cells, _, _ = np.histogram2d(lateral[r], depth[r], bins=[np.linspace(0, 10, 11)] * 2)
        players[name] = {
            "team": team,
            "tracked": float(seen.mean()),
            "interpolated": float((seen & ~real[:, j]).sum() / max(1, seen.sum())),
            "distance_m": float(np.nansum(np.where(moving, sp, 0.0)) * dt),
            "distance_in_rallies_m": float(np.nansum(np.where(moving & in_rally, sp, 0.0)) * dt),
            "top_speed_ms": float(peak),
            "sprints": sprints,
            "avg_dist_to_net_m": float(np.nanmean(depth[r])) if r.any() else np.nan,
            "time_at_net": float(np.mean(depth[r] < s.net_zone)) if r.any() else np.nan,
            "time_in_middle": (float(np.mean((depth[r] >= s.net_zone) & (depth[r] <= s.back_zone)))
                               if r.any() else np.nan),
            "time_at_back": float(np.mean(depth[r] > s.back_zone)) if r.any() else np.nan,
            "court_coverage": float(np.mean(cells >= 0.5 * fps)),
            "time_left_side": float(np.mean(lateral[r] < 5.0)) if r.any() else np.nan,
        }

    teams = {}
    for tm, (j0, j1) in TEAMS.items():
        d0, d1 = own[SLOT_NAMES[j0]][1], own[SLOT_NAMES[j1]][1]
        both = in_rally & ~np.isnan(d0) & ~np.isnan(d1)
        net0, net1 = d0 < s.net_zone, d1 < s.net_zone
        back0, back1 = d0 > s.back_zone, d1 > s.back_zone
        state = np.full(len(t), np.nan)
        state[both] = 0.0
        state[both & net0 & net1] = 1.0
        state[both & back0 & back1] = -1.0
        split = both & ((net0 & back1) | (back0 & net1))
        gap = np.linalg.norm(sm[:, j0] - sm[:, j1], axis=1)
        depth_gap = np.abs(d0 - d1)
        lost, regained, rec = formation_runs(state, rallies, fps, s)
        n = max(1, both.sum())
        sync = (
            float(np.corrcoef(d0[both], d1[both])[0, 1]) if both.sum() > 10 else np.nan
        )
        teams[tm] = {
            "players": [SLOT_NAMES[j0], SLOT_NAMES[j1]],
            "both_at_net": float((state == 1).sum() / n),
            "both_at_back": float((state == -1).sum() / n),
            "split": float(split.sum() / n),
            "in_transition": float(((state == 0) & ~split).sum() / n),
            "avg_partner_distance_m": float(np.nanmean(gap[both])) if both.any() else np.nan,
            "side_by_side": float(np.mean(depth_gap[both] < s.side_by_side)) if both.any() else np.nan,
            "depth_sync": sync,
            "net_lost": lost,
            "net_regained": regained,
            "median_time_to_regain_s": float(np.median(rec)) if rec else np.nan,
            "_state": state,
        }

    rally_list = [
        {"start_s": float(t[a]), "end_s": float(t[b - 1]), "duration_s": float((b - a) / fps)}
        for a, b in rallies
    ]
    return sm, speed, in_rally, activity, rallies, rally_list, players, teams, own


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def ball_stats(ball) -> tuple[dict, dict]:
    """Per-player and per-team shot statistics from the ball analysis."""
    shots = ball.shots
    # volley[i]: True / False / None (unknown: the ball wasn't seen for the whole previous flight)
    volley: list[bool | None] = [None] * len(shots)
    for i in range(1, len(shots)):
        prev = shots[i - 1]
        if prev.rally == shots[i].rally:
            volley[i] = prev.volleyed_by_next

    def summarise(idx: list[int]) -> dict:
        sel = [shots[i] for i in idx]
        landed = [x for x in sel if x.landing]
        deep = [abs(x.landing[1] - NET_Y) > 6.95 for x in landed]
        flights = [x.flight_s for x in sel if x.flight_s]
        speeds = [x.speed_ms for x in sel if x.speed_ms]
        return {
            "shots": len(sel),
            "volleys": int(sum(1 for i in idx if volley[i] is True)),
            "volley_share": (float(np.mean([volley[i] for i in idx if volley[i] is not None]))
                             if any(volley[i] is not None for i in idx) else np.nan),
            "landed": len(landed),
            "deep_share": float(np.mean(deep)) if deep else np.nan,
            "median_flight_s": float(np.median(flights)) if flights else np.nan,
            "median_speed_ms": float(np.median(speeds)) if speeds else np.nan,
        }

    per_player = {n: summarise([i for i, x in enumerate(shots) if x.player == n]) for n in SLOT_NAMES}
    per_team = {tm: summarise([i for i, x in enumerate(shots) if x.team == tm]) for tm in TEAMS}
    return per_player, per_team


HEAT_BINS = 40   # heatmap cells per 10 m (0.25 m squares)


def clean(x):
    """JSON-ready: NaN and infinities become null, numpy numbers plain ones."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def build_report_data(title: str, t, fps, activity, rally_list, players, teams, own, in_rally,
                      id_report, s: Settings, ball=None, stats=None, has_3d=False) -> dict:
    """Everything report.html shows, as one JSON-ready dict."""
    edges = np.linspace(0, 10, HEAT_BINS + 1)
    player_rows = []
    for j, name in enumerate(SLOT_NAMES):
        lateral, depth = own[name]
        ok = in_rally & ~np.isnan(lateral)
        # heat[row][col]: row 0 = at the net, last row = back wall; col 0 = the player's left
        heat, _, _ = np.histogram2d(depth[ok], lateral[ok], bins=[edges, edges])
        player_rows.append({
            "id": name, "team": SLOT_TEAMS[j], "move": players[name],
            "shots": stats["players"][name] if stats else None,
            "heat": heat.astype(int).tolist(), "heat_seconds": float(ok.sum() / fps),
        })
    team_rows = [{"id": tm, "players": v["players"], "move": v,
                  "shots": stats["teams"][tm] if stats else None} for tm, v in teams.items()]
    step = max(1, int(round(fps / 2)))
    merged = sum(len(v) for v in id_report.get("tracker_ids_merged", {}).values())
    return clean({
        "title": title,
        "duration_s": float(t[-1] - t[0]),
        "t0": float(t[0]),
        "fps": fps,
        "settings": {"net_zone": s.net_zone, "back_zone": s.back_zone, "still_speed": s.still_speed,
                     "sprint_speed": s.sprint_speed, "rally_source": s.rally_source,
                     "service_line": match_stats.SERVICE_LINE},
        "players": player_rows,
        "teams": team_rows,
        "movement_rallies": rally_list,
        "match": stats["match"] if stats else None,
        "rallies": stats["rallies"] if stats else None,
        "shots": stats["shots"] if stats else [],
        "shot_types": list(match_stats.SHOT_TYPES),
        "shot_labels": match_stats.SHOT_LABELS,
        "attacking_types": sorted(match_stats.ATTACKING),
        "timeline": {"dt": step / fps, "activity": [float(a) for a in activity[::step]],
                     "threshold": s.rally_share},
        "quality": {
            "ball_coverage": ball.coverage if ball is not None else None,
            "ball_kept": ball.kept if ball is not None else None,
            "has_ball": ball is not None, "has_3d": has_3d,
            "tracker_ids_merged": merged,
            "interpolated_frames": id_report.get("interpolated_frames", 0),
            "appearance_used": bool(id_report.get("appearance_used")),
            "swaps_fixed": id_report.get("swaps_fixed", []),
        },
    })


def write_report(out: Path, data: dict) -> None:
    """report_template.html with the data embedded: one self-contained page."""
    template = (Path(__file__).resolve().parent / "report_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (out / "report.html").write_text(template.replace("/*REPORT_DATA*/null", payload), encoding="utf-8")


def write_csv(out: Path, t, f0, pos, sm, speed, real, in_rally) -> None:
    with open(out / "positions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "player", "team", "x", "y", "x_smooth", "y_smooth",
                    "speed_ms", "detected", "in_rally"])
        for i in range(len(t)):
            for j, name in enumerate(SLOT_NAMES):
                if np.isnan(pos[i, j, 0]):
                    continue
                w.writerow([f0 + i, f"{t[i]:.3f}", name, SLOT_TEAMS[j],
                            f"{pos[i, j, 0]:.3f}", f"{pos[i, j, 1]:.3f}",
                            f"{sm[i, j, 0]:.3f}", f"{sm[i, j, 1]:.3f}",
                            "" if np.isnan(speed[i, j]) else f"{speed[i, j]:.2f}",
                            int(real[i, j]), int(in_rally[i])])


def write_ball_csv(out: Path, ball) -> None:
    with open(out / "ball_events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "kind", "player", "court_x", "court_y", "image_u", "image_v"])
        for e in ball.events:
            w.writerow([f"{e.time_s:.3f}", e.kind, e.player or "",
                        "" if not e.court else f"{e.court[0]:.2f}", "" if not e.court else f"{e.court[1]:.2f}",
                        f"{e.px[0]:.1f}", f"{e.px[1]:.1f}"])


def write_shots_csv(out: Path, shots: list[dict]) -> None:
    cols = ["time_s", "rally", "player", "team", "type", "speed_kmh", "speed_source", "contact_height_m",
            "depth_m", "zone", "before", "direction", "target", "apex_m", "air_time_s", "wall_after", "attacking"]

    def pair(p) -> list[str]:
        return ["", ""] if not p else [f"{p[0]:.2f}", f"{p[1]:.2f}"]

    with open(out / "shots.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + ["hitter_x", "hitter_y", "landing_x", "landing_y", "landing_source"])
        for x in shots:
            w.writerow(["" if x[c] is None else x[c] for c in cols]
                       + pair(x["contact"]) + pair(x["landing"]) + [x["landing_source"] or ""])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="match_tracking_cache.pkl", type=Path)
    ap.add_argument("--out", default="output/analysis", type=Path)
    ap.add_argument("--gap-fill", default=1.5, type=float, help="max gap to interpolate, seconds")
    ap.add_argument("--no-ball", action="store_true", help="skip everything that needs the ball")
    ap.add_argument("--no-3d", action="store_true", help="skip the 3D ball fit (faster; rougher shot data)")
    args = ap.parse_args()
    if not args.cache.exists():
        raise SystemExit(f"{args.cache} not found. Run run_stable.py first (it creates the cache).")

    s = Settings()
    pos, real, t, fps, f0, id_report, stable = load_tracks(args.cache, args.gap_fill)
    ball = None
    if ball_track is not None and not args.no_ball:
        ball = ball_track.run_on_tracking(stable, fps)
    # The ball marks rallies much more precisely than movement, when it is seen often enough.
    use_ball_rallies = ball is not None and ball.coverage >= s.ball_rally_coverage and len(ball.rallies) > 0
    rally_times = [(a, b) for a, b, _ in ball.rallies] if use_ball_rallies else None
    sm, speed, in_rally, activity, rallies, rally_list, players, teams, own = analyse(
        pos, real, t, fps, s, rally_times)
    s.rally_source = "ball" if use_ball_rallies else "movement"

    ball3d, stats = None, None
    if ball is not None:
        if not args.no_3d:
            import export_scene3d   # the 3D fit and its saved copy live there

            ball3d = export_scene3d.ball3d_for(args.cache, stable, ball, fps, id_report)
        stats = match_stats.match_stats(ball, stable, dict(zip(SLOT_NAMES, SLOT_TEAMS)), ball3d, fps,
                                        video_end_s=float(t[-1]))

    args.out.mkdir(parents=True, exist_ok=True)
    title = args.cache.stem.replace("_tracking_cache", "")
    data = build_report_data(title, t, fps, activity, rally_list, players, teams, own, in_rally,
                             id_report, s, ball, stats, has_3d=ball3d is not None)
    write_report(args.out, data)
    write_csv(args.out, t, f0, pos, sm, speed, real, in_rally)
    summary = {
        "fps": fps, "duration_s": float(t[-1] - t[0]), "rallies": rally_list, "players": players,
        "teams": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in teams.items()},
    }
    if ball is not None:
        write_ball_csv(args.out, ball)
        write_shots_csv(args.out, stats["shots"])
        per_player, per_team = ball_stats(ball)
        summary["ball"] = {"coverage": ball.coverage, "kept": ball.kept, "shots": len(ball.shots),
                           "rallies": [{"start_s": a, "end_s": b, "hits": h} for a, b, h in ball.rallies],
                           "players": per_player, "teams": per_team}
        summary["stats"] = {k: stats[k] for k in ("match", "rallies", "players", "teams")}
    (args.out / "summary.json").write_text(json.dumps(clean(summary), indent=2))

    print(f"{len(rally_list)} rallies, {sum(r['duration_s'] for r in rally_list):.0f} s in play")
    for n, p in players.items():
        print(f"{n}: {p['distance_m']:.0f} m, top {p['top_speed_ms']:.1f} m/s, "
              f"{p['sprints']} sprints, at net {pct(p['time_at_net'])}")
    for tm, v in teams.items():
        print(f"{tm}: both at net {pct(v['both_at_net'])}, both back {pct(v['both_at_back'])}, "
              f"net lost {v['net_lost']}x, regained {v['net_regained']}x")
    if ball is not None:
        print(f"ball: detected in {pct(ball.coverage)} of frames, {len(ball.shots)} shots, "
              f"{len(ball.rallies)} ball-rallies")
        kinds = {k: sum(v["by_type"][k] for v in stats["teams"].values()) for k in match_stats.SHOT_TYPES}
        print("shot types: " + ", ".join(f"{k.replace('_', '/')} {n}" for k, n in kinds.items() if n))
    print(f"\nReport: {args.out / 'report.html'}")


if __name__ == "__main__":
    main()
