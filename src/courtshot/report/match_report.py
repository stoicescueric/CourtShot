"""
The match report: one self-contained HTML page plus the data behind it.

    report.html      the report: open it in a browser
    positions.csv    smoothed per-frame player data
    shots.csv        one row per shot
    ball_events.csv  hits and bounces
    summary.json     the numbers in the report

What is in it:
    movement   distance, speed, sprints, where each player stands (heatmaps),
               net / back time, team formations: from player positions only
               (courtshot.analysis.movement)
    rallies    from the ball track when the ball is seen often enough, else
               from movement
    shots      type, speed, contact height, direction, target, landing spot:
               from the 2D ball events and the fitted 3D flights
               (courtshot.analysis.shots)

Without the 3D fit, shot speeds are averages to the landing spot and lobs are
found by time in the air only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from courtshot.analysis import shots as shot_stats
from courtshot.analysis.identities import SLOT_NAMES, SLOT_TEAMS
from courtshot.analysis.match import MatchAnalysis
from courtshot.analysis.movement import NET_Y, TEAMS, Movement, Settings, analyse, player_arrays

TEMPLATE = Path(__file__).resolve().parent / "templates" / "report.html"
HEAT_BINS = 40  # heatmap cells per 10 m (0.25 m squares)


def pct(x: float) -> str:
    return "–" if x is None or np.isnan(x) else f"{100 * x:.0f}%"


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


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------


def ball_stats(ball) -> tuple[dict, dict]:
    """Per-player and per-team shot statistics from the 2D ball analysis."""
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
        deep = [abs(x.landing[1] - NET_Y) > shot_stats.SERVICE_LINE for x in landed]
        flights = [x.flight_s for x in sel if x.flight_s]
        speeds = [x.speed_ms for x in sel if x.speed_ms]
        return {
            "shots": len(sel),
            "volleys": int(sum(1 for i in idx if volley[i] is True)),
            "volley_share": (
                float(np.mean([volley[i] for i in idx if volley[i] is not None]))
                if any(volley[i] is not None for i in idx)
                else np.nan
            ),
            "landed": len(landed),
            "deep_share": float(np.mean(deep)) if deep else np.nan,
            "median_flight_s": float(np.median(flights)) if flights else np.nan,
            "median_speed_ms": float(np.median(speeds)) if speeds else np.nan,
        }

    per_player = {n: summarise([i for i, x in enumerate(shots) if x.player == n]) for n in SLOT_NAMES}
    per_team = {tm: summarise([i for i, x in enumerate(shots) if x.team == tm]) for tm in TEAMS}
    return per_player, per_team


def build_report_data(
    title: str, t, fps, mv: Movement, id_report, s: Settings, ball=None, stats=None, has_3d=False
) -> dict:
    """Everything report.html shows, as one JSON-ready dict."""
    edges = np.linspace(0, 10, HEAT_BINS + 1)
    player_rows = []
    for j, name in enumerate(SLOT_NAMES):
        lateral, depth = mv.own[name]
        ok = mv.in_rally & ~np.isnan(lateral)
        # heat[row][col]: row 0 = at the net, last row = back wall; col 0 = the player's left
        heat, _, _ = np.histogram2d(depth[ok], lateral[ok], bins=[edges, edges])
        player_rows.append(
            {
                "id": name,
                "team": SLOT_TEAMS[j],
                "move": mv.players[name],
                "shots": stats["players"][name] if stats else None,
                "heat": heat.astype(int).tolist(),
                "heat_seconds": float(ok.sum() / fps),
            }
        )
    team_rows = [
        {"id": tm, "players": v["players"], "move": v, "shots": stats["teams"][tm] if stats else None}
        for tm, v in mv.teams.items()
    ]
    step = max(1, int(round(fps / 2)))
    merged = sum(len(v) for v in id_report.get("tracker_ids_merged", {}).values())
    return clean(
        {
            "title": title,
            "duration_s": float(t[-1] - t[0]),
            "t0": float(t[0]),
            "fps": fps,
            "settings": {
                "net_zone": s.net_zone,
                "back_zone": s.back_zone,
                "still_speed": s.still_speed,
                "sprint_speed": s.sprint_speed,
                "rally_source": s.rally_source,
                "service_line": shot_stats.SERVICE_LINE,
            },
            "players": player_rows,
            "teams": team_rows,
            "movement_rallies": mv.rally_list,
            "match": stats["match"] if stats else None,
            "rallies": stats["rallies"] if stats else None,
            "shots": stats["shots"] if stats else [],
            "shot_types": list(shot_stats.SHOT_TYPES),
            "shot_labels": shot_stats.SHOT_LABELS,
            "attacking_types": sorted(shot_stats.ATTACKING),
            "timeline": {
                "dt": step / fps,
                "activity": [float(a) for a in mv.activity[::step]],
                "threshold": s.rally_share,
            },
            "quality": {
                "ball_coverage": ball.coverage if ball is not None else None,
                "ball_kept": ball.kept if ball is not None else None,
                "has_ball": ball is not None,
                "has_3d": has_3d,
                "tracker_ids_merged": merged,
                "interpolated_frames": id_report.get("interpolated_frames", 0),
                "appearance_used": bool(id_report.get("appearance_used")),
                "swaps_fixed": id_report.get("swaps_fixed", []),
            },
        }
    )


# ---------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------


def write_report(out: Path, data: dict) -> None:
    """The report template with the data embedded: one self-contained page."""
    template = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (out / "report.html").write_text(template.replace("/*REPORT_DATA*/null", payload), encoding="utf-8")


def write_csv(out: Path, t, f0, pos, mv: Movement, real) -> None:
    sm, speed = mv.smoothed, mv.speed
    with open(out / "positions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["frame", "time_s", "player", "team", "x", "y", "x_smooth", "y_smooth", "speed_ms", "detected", "in_rally"]
        )
        for i in range(len(t)):
            for j, name in enumerate(SLOT_NAMES):
                if np.isnan(pos[i, j, 0]):
                    continue
                w.writerow(
                    [
                        f0 + i,
                        f"{t[i]:.3f}",
                        name,
                        SLOT_TEAMS[j],
                        f"{pos[i, j, 0]:.3f}",
                        f"{pos[i, j, 1]:.3f}",
                        f"{sm[i, j, 0]:.3f}",
                        f"{sm[i, j, 1]:.3f}",
                        "" if np.isnan(speed[i, j]) else f"{speed[i, j]:.2f}",
                        int(real[i, j]),
                        int(mv.in_rally[i]),
                    ]
                )


def write_ball_csv(out: Path, ball) -> None:
    with open(out / "ball_events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "kind", "player", "court_x", "court_y", "image_u", "image_v"])
        for e in ball.events:
            w.writerow(
                [
                    f"{e.time_s:.3f}",
                    e.kind,
                    e.player or "",
                    "" if not e.court else f"{e.court[0]:.2f}",
                    "" if not e.court else f"{e.court[1]:.2f}",
                    f"{e.px[0]:.1f}",
                    f"{e.px[1]:.1f}",
                ]
            )


def write_shots_csv(out: Path, shots: list[dict]) -> None:
    cols = [
        "time_s",
        "rally",
        "player",
        "team",
        "type",
        "speed_kmh",
        "speed_source",
        "contact_height_m",
        "depth_m",
        "zone",
        "before",
        "direction",
        "target",
        "apex_m",
        "air_time_s",
        "wall_after",
        "attacking",
        "overhead",
        "confidence",
    ]

    def pair(p) -> list[str]:
        return ["", ""] if not p else [f"{p[0]:.2f}", f"{p[1]:.2f}"]

    with open(out / "shots.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + ["hitter_x", "hitter_y", "landing_x", "landing_y", "landing_source"])
        for x in shots:
            w.writerow(
                ["" if x[c] is None else x[c] for c in cols]
                + pair(x["contact"])
                + pair(x["landing"])
                + [x["landing_source"] or ""]
            )


def write_match_report(analysis: MatchAnalysis, out: str | Path, settings: Settings | None = None) -> dict:
    """Write the report and its data files into ``out``.

    Returns {"summary": what went into summary.json, "stats": the shot
    statistics or None, "ball": the 2D ball result or None}.

    Raises ValueError when the tracking has no player positions.
    """
    s = settings or Settings()
    out = Path(out)
    stable, ball, ball3d, fps = analysis.tracking, analysis.ball, analysis.ball3d, analysis.fps
    pos, real, t, f0 = player_arrays(stable, fps)

    # The ball marks rallies much more precisely than movement, when it is seen often enough.
    use_ball_rallies = ball is not None and ball.coverage >= s.ball_rally_coverage and len(ball.rallies) > 0
    rally_times = [(a, b) for a, b, _ in ball.rallies] if use_ball_rallies else None
    mv = analyse(pos, real, t, fps, s, rally_times)
    s.rally_source = "ball" if use_ball_rallies else "movement"

    stats = None
    if ball is not None:
        stats = shot_stats.match_stats(
            ball, stable, dict(zip(SLOT_NAMES, SLOT_TEAMS)), ball3d, fps, video_end_s=float(t[-1])
        )

    out.mkdir(parents=True, exist_ok=True)
    title = analysis.cache.stem.replace("_tracking_cache", "")
    data = build_report_data(title, t, fps, mv, analysis.id_report, s, ball, stats, has_3d=ball3d is not None)
    write_report(out, data)
    write_csv(out, t, f0, pos, mv, real)
    summary = {
        "fps": fps,
        "duration_s": float(t[-1] - t[0]),
        "rallies": mv.rally_list,
        "players": mv.players,
        "teams": mv.teams,
    }
    if ball is not None:
        write_ball_csv(out, ball)
        write_shots_csv(out, stats["shots"])
        per_player, per_team = ball_stats(ball)
        summary["ball"] = {
            "coverage": ball.coverage,
            "kept": ball.kept,
            "shots": len(ball.shots),
            "rallies": [{"start_s": a, "end_s": b, "hits": h} for a, b, h in ball.rallies],
            "players": per_player,
            "teams": per_team,
        }
        summary["stats"] = {k: stats[k] for k in ("match", "rallies", "players", "teams")}
    (out / "summary.json").write_text(json.dumps(clean(summary), indent=2))
    return {"summary": summary, "stats": stats, "ball": ball}


def print_summary(result: dict) -> None:
    """The headline numbers of write_match_report()'s result, on the console."""
    summary, stats, ball = result["summary"], result["stats"], result["ball"]
    rallies = summary["rallies"]
    print(f"{len(rallies)} rallies, {sum(r['duration_s'] for r in rallies):.0f} s in play")
    for n, p in summary["players"].items():
        print(
            f"{n}: {p['distance_m']:.0f} m, top {p['top_speed_ms']:.1f} m/s, "
            f"{p['sprints']} sprints, at net {pct(p['time_at_net'])}"
        )
    for tm, v in summary["teams"].items():
        print(
            f"{tm}: both at net {pct(v['both_at_net'])}, both back {pct(v['both_at_back'])}, "
            f"net lost {v['net_lost']}x, regained {v['net_regained']}x"
        )
    if ball is not None:
        print(
            f"ball: detected in {pct(ball.coverage)} of frames, {len(ball.shots)} shots, "
            f"{len(ball.rallies)} ball-rallies"
        )
        kinds = {k: sum(v["by_type"][k] for v in stats["teams"].values()) for k in shot_stats.SHOT_TYPES}
        print("shot types: " + ", ".join(f"{k.replace('_', '/')} {n}" for k, n in kinds.items() if n))
