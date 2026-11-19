"""
Tests for courtshot.analysis.swing (what the hitter's body did around a hit) and for the
two places that use it: ball_events drops "hits" by players standing still,
shots takes overhead / not overhead from the pose.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from courtshot.analysis import ball_events as bt
from courtshot.analysis import shots as ms
from courtshot.analysis import swing as sp
from courtshot.schemas.tracking import BoundingBox, PlayerFrame, PoseKeypoint, Position2D

FPS = 25.0
HIT = 20  # frame of the hit in every test


def _body(
    frame: int,
    wrist=(0.3, 0.8),
    elbow=(0.3, 0.4),
    scale: float = 100.0,
    origin=(500.0, 300.0),
    wrist_conf: float = 0.9,
    player: str = "P_1",
) -> PlayerFrame:
    """A player seen from behind. Units of the offsets: torso lengths, from the
    middle of the shoulder line, y down. The right arm is the one described;
    the left one hangs."""
    ox, oy = origin

    def kp(name, x, y, conf=0.9):
        return PoseKeypoint(name=name, x=ox + x * scale, y=oy + y * scale, confidence=conf)

    pts = [
        kp("nose", 0, -0.3),
        kp("left_shoulder", -0.2, 0),
        kp("right_shoulder", 0.2, 0),
        kp("left_elbow", -0.3, 0.4),
        kp("left_wrist", -0.3, 0.8),
        kp("right_elbow", *elbow),
        kp("right_wrist", *wrist, conf=wrist_conf),
        kp("left_hip", -0.15, 1.0),
        kp("right_hip", 0.15, 1.0),
    ]
    return PlayerFrame(
        frame_id=frame,
        timestamp_ms=1000 * frame / FPS,
        player_id=player,
        team_id="T_1",
        bbox=BoundingBox(x1=ox - 0.5 * scale, y1=oy - 0.5 * scale, x2=ox + 0.5 * scale, y2=oy + 2.2 * scale),
        position=Position2D(x=5.0, y=4.0),
        confidence=0.9,
        keypoints=pts,
    )


def _swing(frames) -> sp.Swing | None:
    return sp.PoseTracks(SimpleNamespace(player_frames=list(frames))).swing("P_1", HIT, FPS)


def _standing(**kw):
    return [_body(f, **kw) for f in range(0, 40)]


def _forehand(**kw):
    """The wrist sweeps across the body at waist height around the hit."""
    return [_body(f, wrist=(float(np.clip(0.9 - 0.3 * (f - HIT + 3), -0.9, 0.9)), 0.7), **kw) for f in range(0, 40)]


def _overhead(**kw):
    return [
        _body(
            f,
            wrist=(0.3, -0.9) if abs(f - HIT) < 4 else (0.3, 0.8),
            elbow=(0.35, -0.3) if abs(f - HIT) < 4 else (0.3, 0.4),
            **kw,
        )
        for f in range(0, 40)
    ]


class TestSwing:
    def test_standing_player_is_still_and_not_overhead(self):
        sw = _swing(_standing())
        assert sw.still and sw.overhead is False

    def test_waist_high_swing_is_a_hit_but_not_an_overhead(self):
        sw = _swing(_forehand())
        assert not sw.still and sw.overhead is False
        assert sw.speed > 5.0

    def test_arm_up_is_an_overhead(self):
        sw = _swing(_overhead())
        assert sw.overhead is True and not sw.still

    def test_raised_elbow_counts_when_the_wrist_is_not_seen(self):
        sw = _swing(_overhead(wrist_conf=0.05))
        assert sw.overhead is True

    def test_same_reading_for_a_small_far_player(self):
        near, far = _swing(_forehand()), _swing(_forehand(scale=22.0))
        assert abs(near.speed - far.speed) < 1e-6
        assert abs(near.arm_raise - far.arm_raise) < 1e-6

    def test_body_seen_only_part_of_the_time_cannot_rule_things_out(self):
        sw = _swing(f for f in _standing() if f.frame_id % 2 == 0)
        assert sw.overhead is None and not sw.still

    def test_a_raised_arm_counts_even_when_seen_part_of_the_time(self):
        sw = _swing(f for f in _overhead() if f.frame_id % 2 == 0)
        assert sw.overhead is True

    def test_no_keypoints_no_reading(self):
        bare = [f.model_copy(update={"keypoints": []}) for f in _standing()]
        assert _swing(bare) is None


# ---------------------------------------------------------------------------
# ball_events: a ball passing above a player who is standing still
# ---------------------------------------------------------------------------


def _passing_ball(swing):
    """The ball flies right, turns sharply above P_1's head at frame 20 and
    flies on up and to the left. P_1's box: x 480..520, y 300..400."""
    ball = {f: (300.0 + 10 * f, 290.0) for f in range(0, 21)}
    ball.update({f: (500.0 - 8 * (f - 20), 290.0 - 9 * (f - 20)) for f in range(21, 40)})
    boxes = {f: [("P_1", (480.0, 300.0, 520.0, 400.0))] for f in range(0, 40)}
    pos = {(f, "P_1"): (5.0, 4.0) for f in range(0, 40)}
    return bt.analyse_ball(ball, boxes, pos, {"P_1": "T_1"}, np.eye(3), FPS, 720.0, (0, 39), swing=swing)


class TestStandingPlayerDidNotHit:
    def test_without_pose_the_turn_is_a_hit(self):
        assert [e.kind for e in _passing_ball(None).events] == ["hit"]

    def test_standing_player_is_not_given_the_hit(self):
        still = sp.Swing(16, 1.0, 2.0, -0.4, False, True)
        res = _passing_ball(lambda player, frame: still)
        assert [e.kind for e in res.events] == ["wall"] and not res.shots

    def test_swinging_player_keeps_the_hit(self):
        swinging = sp.Swing(16, 1.0, 15.0, 0.9, True, False)
        assert [e.kind for e in _passing_ball(lambda player, frame: swinging).events] == ["hit"]

    def test_skipped_player_is_passed_over(self):
        boxes = {20: [("P_1", (480.0, 300.0, 520.0, 400.0)), ("P_2", (470.0, 310.0, 540.0, 420.0))]}
        s = bt.BallSettings()
        assert bt.players_near(20, boxes, (500.0, 330.0), s)[0] == "P_1"
        assert bt.players_near(20, boxes, (500.0, 330.0), s, {"P_1"})[0] == "P_2"


# ---------------------------------------------------------------------------
# shots: overhead from the pose
# ---------------------------------------------------------------------------


def _one_shot(frames, ball_v: float):
    """One hit by P_1 at frame 20, the ball at image height ball_v."""
    hit = bt.BallEvent(HIT, HIT / FPS, "hit", (500.0, ball_v), "P_1", (5.0, 4.0))
    shot = bt.Shot("P_1", "T_1", HIT / FPS, (5.0, 4.0), None, None, None, False, None, 0)
    ball = bt.BallResult(0.9, 0.9, [], [hit], [shot], rallies=[(0.0, 2.0, 1)])
    return ms.build_shots(ball, SimpleNamespace(player_frames=list(frames)), {"P_1": "T_1"}, None, FPS)[0]


class TestOverheadFromPose:
    # the test body's box: top at y = 250, feet at y = 520

    def test_ball_above_the_head_of_a_waist_high_swing_is_not_an_overhead(self):
        s = _one_shot(_forehand(), ball_v=200.0)
        assert not s.overhead and s.type != "bandeja_vibora"
        assert s.contact_height_m <= ms.PLAYER_HEIGHT

    def test_arm_up_is_an_overhead_wherever_the_ball_was_found(self):
        s = _one_shot(_overhead(), ball_v=400.0)
        assert s.overhead and s.type == "bandeja_vibora"
        assert s.contact_height_m >= ms.PLAYER_HEIGHT

    def test_without_pose_the_ball_decides(self):
        bare = [f.model_copy(update={"keypoints": []}) for f in _standing()]
        assert _one_shot(bare, ball_v=200.0).overhead
        assert not _one_shot(bare, ball_v=400.0).overhead

    def test_only_an_overhead_answer_a_second_later_makes_a_lob(self):
        def first_type(gap_s: float) -> str:
            frames = [f.model_copy(update={"keypoints": []}) for f in _standing()]
            frames += [
                f.model_copy(update={"keypoints": [], "frame_id": f.frame_id + 30}) for f in _standing(player="P_3")
            ]
            t2 = HIT / FPS + gap_s
            hits = [
                bt.BallEvent(HIT, HIT / FPS, "hit", (500.0, 400.0), "P_1", (5.0, 4.0)),
                bt.BallEvent(t2 * FPS, t2, "hit", (500.0, 200.0), "P_3", (5.0, 16.0)),
            ]
            shots = [
                bt.Shot("P_1", "T_1", HIT / FPS, (5.0, 4.0), None, None, None, False, None, 0),
                bt.Shot("P_3", "T_2", t2, (5.0, 16.0), None, None, None, False, None, 0),
            ]
            ball = bt.BallResult(0.9, 0.9, [], hits, shots, rallies=[(0.0, 4.0, 2)])
            out = ms.build_shots(ball, SimpleNamespace(player_frames=frames), {"P_1": "T_1", "P_3": "T_2"}, None, FPS)
            assert out[1].overhead
            return out[0].type

        assert first_type(1.6) == "lob"
        assert first_type(0.5) != "lob"

    def test_confidence_counts_what_was_seen(self):
        assert ms._confidence("drive", "bounce", True, True) == "high"
        assert ms._confidence("drive", "bounce", True, False) == "medium"
        assert ms._confidence("drive", None, False, True) == "low"
        assert ms._confidence("other", "bounce", True, True) == "low"
