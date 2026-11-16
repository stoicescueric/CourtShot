"""
Player-movement analysis for a Padex match.

Reads the tracking cache Padex already saved (no video processing), applies
stable player IDs, and writes:

    output/analysis/report.html    charts + tables, open in a browser
    output/analysis/positions.csv  smoothed per-frame data for your own analysis
    output/analysis/summary.json   the numbers in the report

Usage, from the padex folder (after at least one run_stable.py run):

    uv run python analyze_match.py
    uv run python analyze_match.py --cache match_tracking_cache.pkl --out output/analysis

Everything here uses player positions only. The ball is not used, because
its court position is not reliable in this version of Padex.

Coordinates: court metres, x = 0..10 across, y = 0..20 along, net at y = 10.
"Own half" views flip each team so its back wall is at the bottom and the
net at the top, whichever end the team is playing from.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

from stable_ids import SLOT_NAMES, SLOT_TEAMS, StableIdConfig, stabilize_tracking

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
TEAMS_PLAYERS = {"T_1": ["P_1", "P_2"], "T_2": ["P_3", "P_4"]}


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


def num(x: float, unit: str = "", digits: int = 1) -> str:
    return "–" if x is None or np.isnan(x) else f"{x:.{digits}f}{unit}"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_tracks(cache: Path, gap_fill: float):
    with open(cache, "rb") as f:  # your own Padex cache file
        raw = pickle.load(f)
    stable, id_report = stabilize_tracking(raw, StableIdConfig(max_gap_fill=gap_fill))
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
    secs = np.array([stamps[f] for f in fids])
    if len(fids) > 1:
        rates = np.diff(fids) / np.diff(secs)
        fps = float(np.median(rates[np.isfinite(rates) & (rates > 0)]))
    else:
        fps = 30.0
    t = secs[0] + (np.arange(n) + f0 - fids[0]) / fps
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
    with np.errstate(all="ignore"):
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

PLAYER_COLORS = {"P_1": "#378ADD", "P_2": "#85B7EB", "P_3": "#D85A30", "P_4": "#F0997B"}


def half_court_shapes(row: int, col: int) -> list[dict]:
    """Own-half court lines: back wall at the bottom (y=0), net at the top (y=10)."""
    xref = "x" if (row, col) == (1, 1) else f"x{(row - 1) * 2 + col}"
    yref = xref.replace("x", "y")
    line = {"color": "rgba(120,120,120,0.9)", "width": 1}
    service = 10 - 6.95
    return [
        {"type": "rect", "x0": 0, "x1": 10, "y0": 0, "y1": 10, "xref": xref, "yref": yref, "line": line},
        {"type": "line", "x0": 0, "x1": 10, "y0": service, "y1": service, "xref": xref, "yref": yref, "line": line},
        {"type": "line", "x0": 5, "x1": 5, "y0": service, "y1": 10, "xref": xref, "yref": yref, "line": line},
        {"type": "line", "x0": 0, "x1": 10, "y0": 10, "y1": 10, "xref": xref, "yref": yref,
         "line": {"color": "rgba(120,120,120,0.9)", "width": 3, "dash": "dash"}},
    ]


def fig_heatmaps(own, in_rally) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.12, vertical_spacing=0.12,
        subplot_titles=[f"{n} ({SLOT_TEAMS[i]})" for i, n in enumerate(SLOT_NAMES)],
    )
    edges_x = np.linspace(0, 10, 21)
    edges_y = np.linspace(0, 10, 21)
    shapes = []
    for i, name in enumerate(SLOT_NAMES):
        r, c = i // 2 + 1, i % 2 + 1
        lateral, depth = own[name]
        ok = in_rally & ~np.isnan(lateral)
        h, _, _ = np.histogram2d(lateral[ok], 10 - depth[ok], bins=[edges_x, edges_y])
        h = 100 * h / max(1, h.sum())
        fig.add_trace(
            go.Heatmap(
                x=(edges_x[:-1] + edges_x[1:]) / 2, y=(edges_y[:-1] + edges_y[1:]) / 2,
                z=h.T, colorscale="Blues", showscale=False,
                hovertemplate="%{z:.1f}% of rally time<extra></extra>",
            ),
            row=r, col=c,
        )
        fig.update_xaxes(range=[0, 10], showticklabels=False, row=r, col=c)
        fig.update_yaxes(range=[0, 10], showticklabels=False, scaleanchor=f"x{i + 1 if i else ''}", row=r, col=c)
        shapes += half_court_shapes(r, c)
    fig.update_layout(height=720, margin={"l": 20, "r": 20, "t": 40, "b": 20}, shapes=shapes,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def fig_timeline(t, activity, rallies, s: Settings) -> go.Figure:
    fig = go.Figure()
    for a, b in rallies:
        fig.add_vrect(x0=t[a], x1=t[b - 1], fillcolor="rgba(29,158,117,0.15)", line_width=0)
    fig.add_trace(go.Scatter(x=t, y=100 * activity, mode="lines", line={"color": "#534AB7", "width": 1},
                             hovertemplate="%{x:.1f}s: %{y:.0f}%<extra></extra>"))
    fig.add_hline(y=100 * s.rally_share, line_dash="dot", line_color="#888780",
                  annotation_text="rally threshold", annotation_position="top left")
    fig.update_layout(height=260, margin={"l": 50, "r": 20, "t": 20, "b": 40},
                      xaxis_title="Time (s)", yaxis_title="% of time someone is fast", showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def fig_formations(teams) -> go.Figure:
    names = list(teams)
    parts = [("both_at_net", "Both at net", "#1D9E75"), ("split", "Split (one up, one back)", "#EF9F27"),
             ("in_transition", "In between", "#B4B2A9"), ("both_at_back", "Both at back", "#D85A30")]
    fig = go.Figure()
    for key, label, color in parts:
        fig.add_trace(go.Bar(y=names, x=[100 * teams[n][key] for n in names], name=label,
                             orientation="h", marker_color=color,
                             hovertemplate=label + ": %{x:.0f}%<extra></extra>"))
    fig.update_layout(barmode="stack", height=200, margin={"l": 50, "r": 20, "t": 10, "b": 40},
                      xaxis={"title": "% of rally time", "range": [0, 100]},
                      legend={"orientation": "h", "y": -0.45},
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


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


def fig_landing(ball) -> go.Figure:
    """Where each team's shots land, seen from behind the hitter."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=[f"{tm} shots land here" for tm in TEAMS])
    line = {"color": "rgba(120,120,120,0.9)", "width": 1}
    shapes = []
    for c, tm in enumerate(TEAMS, start=1):
        xs, ys, names = [], [], []
        for sh in ball.shots:
            if sh.team != tm or not sh.landing:
                continue
            x, y = sh.landing
            xs.append(x if y > NET_Y else 10 - x)  # lateral as the hitter sees it
            ys.append(abs(y - NET_Y))               # distance past the net
            names.append(sh.player)
        for name in sorted(set(names)):
            fig.add_trace(go.Scatter(
                x=[x for x, n in zip(xs, names) if n == name], y=[y for y, n in zip(ys, names) if n == name],
                mode="markers", name=name, marker={"size": 10, "color": PLAYER_COLORS[name], "line": {"width": 0}},
                hovertemplate=name + ": %{y:.1f} m past the net<extra></extra>"),
                row=1, col=c)
        xr = "x" if c == 1 else "x2"
        yr = xr.replace("x", "y")
        shapes += [
            {"type": "rect", "x0": 0, "x1": 10, "y0": 0, "y1": 10, "xref": xr, "yref": yr, "line": line},
            {"type": "line", "x0": 0, "x1": 10, "y0": 6.95, "y1": 6.95, "xref": xr, "yref": yr, "line": line},
            {"type": "line", "x0": 5, "x1": 5, "y0": 0, "y1": 6.95, "xref": xr, "yref": yr, "line": line},
            {"type": "line", "x0": 0, "x1": 10, "y0": 0, "y1": 0, "xref": xr, "yref": yr,
             "line": {"color": "rgba(120,120,120,0.9)", "width": 3, "dash": "dash"}},
        ]
        fig.update_xaxes(range=[-0.3, 10.3], showticklabels=False, row=1, col=c)
        fig.update_yaxes(range=[-0.3, 10.3], showticklabels=False, scaleanchor=xr, row=1, col=c)
    fig.update_layout(height=420, margin={"l": 20, "r": 20, "t": 40, "b": 20}, shapes=shapes,
                      legend={"orientation": "h", "y": -0.05},
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def html_table(headers, rows) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def ball_section(ball, landing_div: str) -> str:
    if ball_track is None:
        return "<h2>Ball and shots</h2><p class='note'>ball_track.py not found next to this script.</p>"
    if ball is None:
        return ("<h2>Ball and shots</h2><p class='note'>No ball detections or no court calibration in "
                "the cache, so there is no ball section.</p>")
    per_player, per_team = ball_stats(ball)
    hits = [r[2] for r in ball.rallies]
    kpis = f"""<div class="kpis">
 <div class="kpi"><b>{pct(ball.coverage)}</b><span>frames with a ball detection</span></div>
 <div class="kpi"><b>{pct(ball.kept)}</b><span>detections kept after cleaning</span></div>
 <div class="kpi"><b>{len(ball.shots)}</b><span>shots found</span></div>
 <div class="kpi"><b>{len(ball.rallies)}</b><span>rallies (from the ball)</span></div>
 <div class="kpi"><b>{num(float(np.median(hits)) if hits else np.nan, '', 0)}</b><span>median shots per rally</span></div>
</div>"""
    prow = [[n, SLOT_TEAMS[SLOT_NAMES.index(n)], v["shots"], f"{v['volleys']} ({pct(v['volley_share'])})",
             v["landed"], pct(v["deep_share"]), num(v["median_flight_s"], " s", 2), num(v["median_speed_ms"], " m/s")]
            for n, v in per_player.items()]
    trow = [[tm, " + ".join(TEAMS_PLAYERS[tm]), v["shots"], f"{v['volleys']} ({pct(v['volley_share'])})",
             v["landed"], pct(v["deep_share"]), num(v["median_flight_s"], " s", 2), num(v["median_speed_ms"], " m/s")]
            for tm, v in per_team.items()]
    heads = ["Shots", "Volleys", "Landing found", "Deep", "Flight time", "Shot speed"]
    shot_rows = [[i + 1, num(x.time_s, " s"), x.player, x.team,
                  "–" if not x.landing else f"({x.landing[0]:.1f}, {x.landing[1]:.1f})",
                  num(x.flight_s or np.nan, " s", 2), num(x.speed_ms or np.nan, " m/s"),
                  {True: "yes", False: "no", None: "–"}[x.volleyed_by_next]]
                 for i, x in enumerate(ball.shots)]
    low = ("<p class='note'><b>Ball detection coverage is low</b>, so expect missed shots. Improving the ball "
           "model (fine-tuning) will help more than tuning this analysis.</p>") if ball.coverage < 0.5 else ""
    return f"""<h2>Ball and shots</h2>
<p class="note">Hits are found where the cleaned ball track changes direction within a player's reach;
the hitter is that player. The landing spot is the first floor bounce after a hit, which is the one point
where the ball's court position is reliable. Deep = landed beyond the service line. Volleys = hits made
before the ball touched the floor; the share counts only shots where the whole incoming flight was seen.
Shot speed = average horizontal speed from the hitter to the landing
spot. Wall bounces are detected too (see ball_events.csv) but are less reliable, so they aren't summarised.</p>
{kpis}{low}
<div class="scroll">{html_table(["Player", "Team"] + heads, prow)}</div>
<div class="scroll" style="margin-top:12px">{html_table(["Team", "Players"] + heads, trow)}</div>
<h2>Where shots land</h2>
<p class="note">Each team's landing spots, seen from behind the hitter: net (dashed) at the bottom, the
opponents' back wall at the top, the hitter's left on the left.</p>
{landing_div}
<h2>All shots</h2>
<div class="scroll">{html_table(["#", "Time", "Player", "Team", "Landing (x, y)", "Flight", "Speed",
    "Volleyed back"], shot_rows)}</div>"""


def write_report(out: Path, t, activity, rallies, rally_list, players, teams, own, in_rally,
                 id_report, s: Settings, ball=None) -> None:
    total = t[-1] - t[0]
    rally_time = sum(r["duration_s"] for r in rally_list)

    player_rows = [
        [n, p["team"], num(p["distance_m"], " m", 0), num(p["distance_in_rallies_m"], " m", 0),
         num(p["top_speed_ms"], " m/s"), p["sprints"], num(p["avg_dist_to_net_m"], " m"),
         pct(p["time_at_net"]), pct(p["time_left_side"]), pct(p["tracked"]), pct(p["interpolated"])]
        for n, p in players.items()
    ]
    team_rows = [
        [tm, " + ".join(v["players"]), pct(v["both_at_net"]), pct(v["split"]), pct(v["both_at_back"]),
         num(v["avg_partner_distance_m"], " m"), pct(v["side_by_side"]), num(v["depth_sync"], "", 2),
         v["net_lost"], v["net_regained"], num(v["median_time_to_regain_s"], " s")]
        for tm, v in teams.items()
    ]
    rally_rows = [[i + 1, num(r["start_s"], " s"), num(r["end_s"], " s"), num(r["duration_s"], " s")]
                  for i, r in enumerate(rally_list)]

    figs = [fig_timeline(t, activity, rallies, s), fig_formations(teams), fig_heatmaps(own, in_rally)]
    if ball is not None and ball.shots:
        figs.append(fig_landing(ball))
    divs = [f.to_html(full_html=False, include_plotlyjs=(i == 0), config={"displaylogo": False})
            for i, f in enumerate(figs)]
    ball_html = ball_section(ball, divs[3] if len(divs) > 3 else "")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Match report</title>
<style>
:root {{ --fg:#1f1e1c; --muted:#6b6a65; --line:#dedcd5; --bg:#fbfaf7; --card:#fff; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#ecebe6; --muted:#a3a19a; --line:#3a3936; --bg:#1c1b19; --card:#252422; }} }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--fg); background:var(--bg);
       max-width:1000px; margin:0 auto; padding:24px 16px 64px; line-height:1.55; }}
h1 {{ font-size:24px; font-weight:600; margin:0 0 4px; }} h2 {{ font-size:18px; font-weight:600; margin:36px 0 8px; }}
p.note, li {{ color:var(--muted); font-size:14px; }}
.kpis {{ display:flex; flex-wrap:wrap; gap:12px; margin:20px 0; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:140px; }}
.kpi b {{ display:block; font-size:22px; font-weight:600; }} .kpi span {{ color:var(--muted); font-size:13px; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; background:var(--card); }}
th, td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:right; white-space:nowrap; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
th {{ color:var(--muted); font-weight:500; }}
</style></head><body>
<h1>Match report</h1>
<p class="note">Movement sections use player positions only. The ball section uses cleaned ball detections.</p>
<div class="kpis">
 <div class="kpi"><b>{total / 60:.1f} min</b><span>footage analysed</span></div>
 <div class="kpi"><b>{len(rally_list)}</b><span>rallies detected</span></div>
 <div class="kpi"><b>{rally_time / 60:.1f} min</b><span>ball in play (est.)</span></div>
 <div class="kpi"><b>{num(np.median([r['duration_s'] for r in rally_list]) if rally_list else np.nan, ' s')}</b><span>median rally</span></div>
</div>

<h2>Rally detection</h2>
<p class="note">{"<b>Rallies below come from the ball track</b> (start of the first shot to the last ball "
"movement); the purple line is the movement signal, shown for comparison. " if s.rally_source == "ball" else ""}
Green bands are rallies. The line shows, over a sliding {s.rally_window:.0f} s window,
how much of the time at least one player moves faster than {s.rally_speed} m/s (faster than walking).
Check a few bands against the video. If rallies are split or merged, change <code>rally_share</code>,
<code>rally_speed</code> or <code>rally_merge</code> in <code>Settings</code>.</p>
{divs[0]}

<h2>Teams</h2>
<p class="note">During rallies only. At the net = closer than {s.net_zone} m to the net; at the back = farther
than {s.back_zone} m. Depth sync is the correlation of the partners' distance to the net (1 = they move up and
back together). Net lost / regained count changes between "both at net" and "both at back" lasting at least
{s.state_min} s; time to regain runs from losing the net to getting both players back up.</p>
{divs[1]}
<div class="scroll">{html_table(["Team", "Players", "Both at net", "Split", "Both at back", "Partner gap",
    "Side by side", "Depth sync", "Net lost", "Net regained", "Time to regain"], team_rows)}</div>

<h2>Players</h2>
<p class="note">Distance and speed come from smoothed positions; standing still (under {s.still_speed} m/s)
adds no distance. Top speed is the best {s.peak_window} s average. Left side is the share of rally time on
the left half of their own court, looking at the net (in padel usually the backhand/revés player).
"Interpolated" is the share of tracked time filled in during short detection gaps.</p>
<div class="scroll">{html_table(["Player", "Team", "Distance", "In rallies", "Top speed", "Sprints",
    "Avg to net", "At net", "Left side", "Tracked", "Interpolated"], player_rows)}</div>

<h2>Where each player plays</h2>
<p class="note">Own-half view: back wall at the bottom, net (dashed) at the top, the player's left on the left.
Each team is flipped so the view is the same whichever end they play from. Rally time only.</p>
{divs[2]}

{ball_html}

<h2>Rallies (from {s.rally_source})</h2>
<div class="scroll">{html_table(["#", "Start", "End", "Duration"], rally_rows)}</div>

<h2>Caveats</h2>
<ul>
<li>Positions are each player's feet mapped through your court calibration, so their accuracy is your
reprojection error plus bounding-box jitter (roughly 0.2–0.4 m).</li>
<li>Rallies come from the {s.rally_source}. Movement-based rallies can be 1–2 s off at the edges,
can split a point that ends with everyone still, and can merge points if players jog between them.</li>
<li>Identity mapping merged {sum(len(v) for v in id_report.get('tracker_ids_merged', {}).values())} raw
tracker IDs into 4 players and filled {id_report.get('interpolated_frames', 0)} frames.</li>
</ul>
</body></html>"""
    (out / "report.html").write_text(html, encoding="utf-8")


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
    with open(out / "shots.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "player", "team", "rally", "hitter_x", "hitter_y", "landing_x", "landing_y",
                    "flight_s", "speed_ms", "wall_after", "volleyed_by_next"])
        for x in ball.shots:
            w.writerow([f"{x.time_s:.3f}", x.player, x.team, x.rally,
                        *(["", ""] if not x.hitter_pos else [f"{v:.2f}" for v in x.hitter_pos]),
                        *(["", ""] if not x.landing else [f"{v:.2f}" for v in x.landing]),
                        "" if x.flight_s is None else f"{x.flight_s:.2f}",
                        "" if x.speed_ms is None else f"{x.speed_ms:.1f}",
                        int(x.wall_after), "" if x.volleyed_by_next is None else int(x.volleyed_by_next)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="match_tracking_cache.pkl", type=Path)
    ap.add_argument("--out", default="output/analysis", type=Path)
    ap.add_argument("--gap-fill", default=1.5, type=float, help="max gap to interpolate, seconds")
    ap.add_argument("--no-ball", action="store_true", help="skip the ball section")
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

    args.out.mkdir(parents=True, exist_ok=True)
    write_report(args.out, t, activity, rallies, rally_list, players, teams, own, in_rally, id_report, s, ball)
    write_csv(args.out, t, f0, pos, sm, speed, real, in_rally)
    summary = {
        "fps": fps, "duration_s": float(t[-1] - t[0]), "rallies": rally_list, "players": players,
        "teams": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in teams.items()},
    }
    if ball is not None:
        write_ball_csv(args.out, ball)
        per_player, per_team = ball_stats(ball)
        summary["ball"] = {"coverage": ball.coverage, "kept": ball.kept, "shots": len(ball.shots),
                           "rallies": [{"start_s": a, "end_s": b, "hits": h} for a, b, h in ball.rallies],
                           "players": per_player, "teams": per_team}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

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
    print(f"\nReport: {args.out / 'report.html'}")


if __name__ == "__main__":
    main()
