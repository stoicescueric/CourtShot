"""
How the players move: distance, speed, sprints, where each one stands, net and
back time, team formations. Everything here comes from the player positions
alone (stable IDs, court metres).

Coordinates: court metres, x = 0..10 across, y = 0..20 along, net at y = 10.
"Own half" views flip each team so its back wall is at the bottom and the
net at the top, whichever end the team is playing from.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from courtshot.analysis.identities import SLOT_NAMES, SLOT_TEAMS

NET_Y = 10.0
TEAMS = {"T_1": (0, 1), "T_2": (2, 3)}  # team -> columns of its players in the position arrays


@dataclass
class Settings:
    """All thresholds in one place. Metres, seconds, m/s."""

    smooth_window: float = 0.4  # Savitzky-Golay window for positions/speed
    still_speed: float = 0.5  # below this a player counts as standing (no distance)
    max_speed: float = 10.0  # faster than this = tracking glitch, ignored
    sprint_speed: float = 4.0  # a sprint is at least this fast...
    sprint_min: float = 0.3  # ...for at least this long
    peak_window: float = 0.25  # top speed = best average over this window
    rally_speed: float = 2.5  # rally if the fastest player is above this...
    rally_share: float = 0.1  # ...for at least this share of...
    rally_window: float = 5.0  # ...a sliding window of this length
    rally_min: float = 3.0  # shorter "rallies" are dropped
    rally_merge: float = 2.5  # pauses shorter than this don't end a rally
    net_zone: float = 4.0  # within this of the net = at the net
    back_zone: float = 7.0  # beyond this from the net = at the back (service line ~6.95 m)
    side_by_side: float = 2.0  # partners less than this apart in depth = side by side
    state_min: float = 0.5  # a team formation must last this long to count
    ball_rally_coverage: float = 0.4  # use ball-based rallies when the ball is seen at least this often
    rally_source: str = "movement"  # set at run time


@dataclass
class Movement:
    """Result of analyse(). Arrays are per frame; axis 1 is the player (SLOT_NAMES order)."""

    smoothed: np.ndarray  # (n, 4, 2) smoothed positions
    speed: np.ndarray  # (n, 4) m/s
    in_rally: np.ndarray  # (n,) bool
    activity: np.ndarray  # (n,) share of the sliding window with a fast player
    rallies: list[tuple[int, int]]  # frame-index spans, end exclusive
    rally_list: list[dict]  # the same in seconds
    players: dict[str, dict]  # per-player totals
    teams: dict[str, dict]  # per-team totals
    own: dict[str, tuple[np.ndarray, np.ndarray]]  # player -> (lateral, depth) in the own-half view


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


def player_arrays(stable, fps: float):
    """Positions of the four players as arrays.

    Returns (pos, real, t, f0): pos (n, 4, 2) court metres with NaN where a
    player is not seen, real (n, 4) False for interpolated frames, t (n,)
    seconds, f0 the frame number of row 0.

    Raises ValueError when the tracking has no player with a stable ID.
    """
    pfs = [pf for pf in stable.player_frames if pf.player_id in SLOT_NAMES]
    if not pfs:
        raise ValueError("No player positions in the tracking. Is the court calibrated?")

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

    first = min(stamps)
    t = stamps[first] + (np.arange(n) + f0 - first) / fps
    return pos, real, t, f0


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

    merged: list[list[int]] = []
    for a, b in runs(active):
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


def analyse(pos, real, t, fps, s: Settings, rally_times=None) -> Movement:
    """Movement and formation statistics from the position arrays of
    player_arrays().

    rally_times: optional [(start_s, end_s)] from the ball, used instead of
    movement-based rally detection.
    """
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
        sprints = sum(1 for a, b in runs(np.nan_to_num(sp) >= s.sprint_speed) if b - a >= s.sprint_min * fps)
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
            "time_in_middle": (
                float(np.mean((depth[r] >= s.net_zone) & (depth[r] <= s.back_zone))) if r.any() else np.nan
            ),
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
        sync = float(np.corrcoef(d0[both], d1[both])[0, 1]) if both.sum() > 10 else np.nan
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
        }

    rally_list = [
        {"start_s": float(t[a]), "end_s": float(t[b - 1]), "duration_s": float((b - a) / fps)} for a, b in rallies
    ]
    return Movement(sm, speed, in_rally, activity, rallies, rally_list, players, teams, own)
