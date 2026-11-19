"""
Tests for padex.analysis.shots (shot types, direction, rally endings, totals)
and for the report data padex.report.match_report builds from them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from padex.analysis import movement
from padex.analysis import shots as ms
from padex.analysis.ball_events import BallEvent, BallResult, Shot
from padex.report import match_report as am
from padex.schemas.tracking import BoundingBox, PlayerFrame, Position2D

R = ms.ShotRules()
TEAMS = {"P_1": "T_1", "P_2": "T_1", "P_3": "T_2", "P_4": "T_2"}


def kind(**f) -> str:
    base = {
        "first_in_rally": False,
        "time_s": 20.0,
        "height_rel": 0.5,
        "speed_kmh": 60.0,
        "depth": 7.5,
        "before": "bounce",
        "apex": 1.5,
        "air_time": 0.8,
        "landing_depth": 7.0,
        "answered_overhead": False,
    }
    base.update(f)
    return ms.classify(base, R)


class TestClassify:
    def test_serve_is_the_first_hit_from_the_service_line(self):
        assert kind(first_in_rally=True, before="serve", depth=7.0) == "serve"

    def test_first_hit_right_at_the_start_of_the_video_is_not_a_serve(self):
        assert kind(first_in_rally=True, before="serve", depth=7.0, time_s=0.4) != "serve"

    def test_first_hit_from_the_net_is_not_a_serve(self):
        assert kind(first_in_rally=True, before="serve", depth=2.0) != "serve"

    def test_overhead_is_a_smash_when_fast_else_bandeja(self):
        assert kind(height_rel=1.2, speed_kmh=110.0) == "smash"
        assert kind(height_rel=1.2, speed_kmh=55.0) == "bandeja_vibora"
        assert kind(height_rel=1.2, speed_kmh=None) == "bandeja_vibora"

    def test_bajada_is_hard_and_high_off_the_back_wall(self):
        assert kind(before="wall", height_rel=0.9, depth=7.0, speed_kmh=80.0) == "bajada"
        assert kind(before="wall", height_rel=0.4, depth=7.0, speed_kmh=80.0) == "drive"
        assert kind(before="wall", height_rel=0.9, depth=7.0, speed_kmh=30.0) == "drive"

    def test_lob_by_apex_by_air_time_or_by_the_answer(self):
        assert kind(apex=5.5) == "lob"
        assert kind(apex=None, air_time=2.1) == "lob"
        assert kind(apex=None, air_time=None, answered_overhead=True) == "lob"
        assert kind(apex=2.0, air_time=0.9) == "drive"

    def test_an_overhead_seen_in_the_pose_is_never_a_lob(self):
        assert kind(overhead=True, overhead_by_pose=True, apex=6.0, speed_kmh=40.0) == "bandeja_vibora"
        assert kind(height_rel=1.3, apex=3.0, speed_kmh=40.0) == "bandeja_vibora"

    def test_ball_above_the_head_that_then_climbs_like_a_lob_is_a_lob(self):
        assert kind(height_rel=1.3, apex=6.0, speed_kmh=40.0) == "lob"

    def test_volley_is_taken_before_the_bounce_near_the_net(self):
        assert kind(before="volley", depth=3.0) == "volley"
        assert kind(before="volley", depth=8.5) == "other"

    def test_chiquita_is_soft_and_short_from_the_back(self):
        assert kind(depth=6.0, speed_kmh=35.0, landing_depth=3.0) == "chiquita"
        assert kind(depth=6.0, speed_kmh=35.0, landing_depth=8.0) == "drive"
        assert kind(depth=6.0, speed_kmh=70.0, landing_depth=3.0) == "drive"

    def test_unseen_bounce_goes_by_where_the_hitter_stands(self):
        assert kind(before=None, depth=3.0) == "volley"
        assert kind(before=None, depth=8.0) == "drive"
        assert kind(before=None, depth=5.0) == "other"
        assert kind(before=None, depth=None) == "other"

    def test_every_result_is_a_known_type(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            got = kind(
                first_in_rally=bool(rng.integers(2)),
                time_s=float(rng.uniform(0, 30)),
                height_rel=float(rng.uniform(0, 1.6)),
                speed_kmh=float(rng.uniform(5, 140)),
                depth=float(rng.uniform(0, 10)),
                before=[None, "serve", "volley", "bounce", "wall"][rng.integers(5)],
                apex=float(rng.uniform(0.5, 8)),
                air_time=float(rng.uniform(0.2, 3)),
                landing_depth=float(rng.uniform(0, 10)),
                answered_overhead=bool(rng.integers(2)),
            )
            assert got in ms.SHOT_TYPES


class TestDirection:
    def test_cross_line_and_middle(self):
        assert ms._direction((2.0, 3.0), (8.0, 16.0), R) == "cross"
        assert ms._direction((2.0, 3.0), (1.5, 16.0), R) == "line"
        assert ms._direction((2.0, 3.0), (5.5, 16.0), R) == "middle"

    def test_unknown_without_both_ends(self):
        assert ms._direction(None, (5.0, 16.0), R) is None
        assert ms._direction((2.0, 3.0), None, R) is None


# ---------------------------------------------------------------------------
# a small made-up rally: serve, lob, smash (unreturned)
# ---------------------------------------------------------------------------

FPS = 25.0
SPOT = {"P_1": (3.0, 3.0), "P_2": (7.0, 3.0), "P_3": (3.0, 17.0), "P_4": (7.0, 13.5)}


def _stable() -> SimpleNamespace:
    frames = []
    for f in range(0, 400):
        for i, (pid, (x, y)) in enumerate(SPOT.items()):
            # every player's box: 100 px tall, feet at v = 500
            frames.append(
                PlayerFrame(
                    frame_id=f,
                    timestamp_ms=1000 * f / FPS,
                    player_id=pid,
                    team_id=TEAMS[pid],
                    bbox=BoundingBox(x1=100 + 200 * i, y1=400, x2=140 + 200 * i, y2=500),
                    position=Position2D(x=x, y=y),
                    confidence=0.9,
                )
            )
    return SimpleNamespace(player_frames=frames)


def _hit(t: float, player: str, ball_v: float) -> BallEvent:
    return BallEvent(t * FPS, t, "hit", (120.0, ball_v), player, SPOT[player])


def _rally() -> BallResult:
    events = [
        _hit(4.0, "P_1", 450.0),  # ball at half height: serve
        BallEvent(4.9 * FPS, 4.9, "ground", (0, 0), None, (6.5, 15.0)),
        _hit(5.4, "P_4", 470.0),  # low, answered by an overhead: lob
        _hit(7.6, "P_2", 380.0),  # 20 px above the head: overhead
        BallEvent(8.1 * FPS, 8.1, "ground", (0, 0), None, (2.0, 16.0)),
    ]
    shots = [
        Shot("P_1", "T_1", 4.0, SPOT["P_1"], (6.5, 15.0), 0.9, 12.0, False, False, 0),
        Shot("P_4", "T_2", 5.4, SPOT["P_4"], None, None, None, False, True, 0),
        Shot("P_2", "T_1", 7.6, SPOT["P_2"], (2.0, 16.0), 0.5, 28.0, False, None, 0),
    ]
    return BallResult(0.9, 0.95, [], events, shots, rallies=[(4.0, 8.5, 3)])


def _flight(t: float, hitter: str, speed_ms: float, apex: float, bounces, ok: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        hit_time_s=t,
        hitter=hitter,
        ok=ok,
        launch_speed=speed_ms,
        max_height=apex,
        bounces=bounces,
        landing=next((b for b in bounces if b[0] == "ground"), None),
    )


def _ball3d() -> SimpleNamespace:
    return SimpleNamespace(
        flights=[
            _flight(4.0, "P_1", 16.0, 1.6, [("ground", 4.9, np.array([6.5, 15.0, 0.0]))]),
            _flight(5.4, "P_4", 14.0, 5.8, []),
            _flight(
                7.6,
                "P_2",
                31.0,
                2.6,
                [("ground", 8.1, np.array([2.0, 16.0, 0.0])), ("wall", 8.5, np.array([1.0, 20.0, 1.0]))],
            ),
        ]
    )


class TestBuildShots:
    def test_types_speeds_and_context(self):
        shots = ms.build_shots(_rally(), _stable(), TEAMS, _ball3d(), FPS)
        assert [s.type for s in shots] == ["serve", "lob", "smash"]
        assert [s.before for s in shots] == ["serve", "bounce", "volley"]
        assert shots[0].speed_source == "3d" and abs(shots[0].speed_kmh - 57.6) < 0.1
        assert abs(shots[2].contact_height_m - 1.2 * ms.PLAYER_HEIGHT) < 0.01
        assert shots[1].apex_m == 5.8
        assert [s.attacking for s in shots] == [False, False, True]

    def test_direction_and_target(self):
        shots = ms.build_shots(_rally(), _stable(), TEAMS, _ball3d(), FPS)
        assert shots[0].direction == "cross" and shots[0].target == "P_4"  # from x = 3 to x = 6.5
        assert shots[1].landing is None and shots[1].target == "P_2"
        assert shots[1].direction == "line"  # no landing seen: aimed at P_2, on the same side (x = 7)
        assert shots[2].direction == "cross" and shots[2].target is None

    def test_without_3d_speed_is_the_average_to_the_landing(self):
        shots = ms.build_shots(_rally(), _stable(), TEAMS, None, FPS)
        assert shots[0].speed_source == "2d" and abs(shots[0].speed_kmh - 43.2) < 0.1
        assert shots[1].speed_kmh is None
        assert shots[1].type == "lob"  # still found: the opponents answered with an overhead
        assert shots[2].speed_source == "2d" and shots[2].type == "smash"  # 28 m/s to the landing spot

    def test_landing_on_the_hitters_own_half_is_dropped(self):
        ball = _rally()
        ball.shots[0].landing = (4.0, 6.0)
        shots = ms.build_shots(ball, _stable(), TEAMS, None, FPS)
        assert shots[0].landing is None


class TestEndingsAndTotals:
    def test_winner_when_the_last_shot_bounces_on_the_other_side(self):
        shots = ms.build_shots(_rally(), _stable(), TEAMS, _ball3d(), FPS)
        end = ms.rally_endings(shots, _rally().rallies, _ball3d(), video_end_s=30.0)[0]
        assert (end["outcome"], end["last_player"], end["last_type"], end["shots"]) == ("winner", "P_2", "smash", 3)

    def test_error_when_it_never_reaches_their_floor(self):
        b3 = _ball3d()
        b3.flights[2].bounces = [("wall", 8.0, np.array([5.0, 20.0, 2.0]))]  # their back wall first: out
        shots = ms.build_shots(_rally(), _stable(), TEAMS, b3, FPS)
        assert ms.rally_endings(shots, _rally().rallies, b3, 30.0)[0]["outcome"] == "error"
        b3.flights[2].bounces = [("ground", 7.9, np.array([6.0, 8.0, 0.0]))]  # own half
        assert ms.rally_endings(shots, _rally().rallies, b3, 30.0)[0]["outcome"] == "error"

    def test_cut_when_the_video_ends_with_the_rally(self):
        shots = ms.build_shots(_rally(), _stable(), TEAMS, _ball3d(), FPS)
        assert ms.rally_endings(shots, _rally().rallies, _ball3d(), video_end_s=9.0)[0]["outcome"] == "cut"

    def test_unknown_without_a_good_flight(self):
        shots = ms.build_shots(_rally(), _stable(), TEAMS, None, FPS)
        assert ms.rally_endings(shots, _rally().rallies, None, 30.0)[0]["outcome"] == "unknown"

    def test_totals(self):
        out = ms.match_stats(_rally(), _stable(), TEAMS, _ball3d(), FPS, video_end_s=30.0)
        t1, t2 = out["teams"]["T_1"], out["teams"]["T_2"]
        assert (t1["shots"], t2["shots"]) == (2, 1)
        assert t1["by_type"]["serve"] == 1 and t1["by_type"]["smash"] == 1 and t2["by_type"]["lob"] == 1
        assert t1["attacking_share"] == 0.5 and t1["winners"] == 1 and t2["winners"] == 0
        assert out["players"]["P_2"]["max_speed_kmh"] == 111.6
        assert out["players"]["P_3"]["shots"] == 0 and out["players"]["P_3"]["avg_speed_kmh"] is None
        assert out["match"] == {
            "rallies": 1,
            "shots": 3,
            "avg_rally_shots": 3.0,
            "avg_rally_s": 4.5,
            "longest_rally_shots": 3,
            "longest_rally_s": 4.5,
            "time_in_play_s": 4.5,
        }
        json.dumps(am.clean(out), allow_nan=False)  # everything in it can go into the report


class TestReportData:
    def _data(self, stats):
        n, fps = 200, FPS
        t = np.arange(n) / fps
        pos = np.full((n, 4, 2), np.nan)
        for j, (x, y) in enumerate(SPOT.values()):
            pos[:, j] = (x, y)
        s = movement.Settings()
        mv = movement.analyse(pos, np.ones((n, 4), bool), t, fps, s, rally_times=[(1.0, 7.0)])
        return am.build_report_data(
            "clip",
            t,
            fps,
            mv,
            {"tracker_ids_merged": {}, "appearance_used": True, "swaps_fixed": []},
            s,
            ball=_rally() if stats else None,
            stats=stats,
            has_3d=bool(stats),
        )

    def test_report_data_is_plain_json(self):
        stats = ms.match_stats(_rally(), _stable(), TEAMS, _ball3d(), FPS, video_end_s=30.0)
        data = self._data(stats)
        text = json.dumps(data, allow_nan=False)
        assert json.loads(text)["shots"][2]["type"] == "smash"
        assert [p["id"] for p in data["players"]] == ["P_1", "P_2", "P_3", "P_4"]

    def test_heatmap_puts_the_player_where_they_stood(self):
        data = self._data(None)
        heat = np.array(data["players"][3]["heat"])  # P_4 at x = 7, 3.5 m from the net
        assert heat.shape == (am.HEAT_BINS, am.HEAT_BINS)
        row, col = np.unravel_index(heat.argmax(), heat.shape)
        assert row == int(3.5 / 10 * am.HEAT_BINS)
        # far team: seen from behind, x = 7 is on their left (10 - 7 = 3 m from their left wall)
        assert col in (11, 12)  # 3.0 m sits on the edge between two 0.25 m cells
        assert heat.sum() > 0 and data["players"][3]["move"]["time_at_net"] == 1.0

    def test_report_without_ball_data(self):
        data = self._data(None)
        assert data["shots"] == [] and data["match"] is None and data["players"][0]["shots"] is None

    def test_report_page_is_written_with_the_data_in_it(self, tmp_path):
        stats = ms.match_stats(_rally(), _stable(), TEAMS, _ball3d(), FPS, video_end_s=30.0)
        am.write_report(tmp_path, self._data(stats))
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert "/*REPORT_DATA*/null" not in html
        start = html.index("const D = ") + len("const D = ")
        data = json.loads(html[start : html.index(";\n", start)])
        assert data["title"] == "clip" and len(data["shots"]) == 3
