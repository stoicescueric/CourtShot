"""
Tests for the appearance check on player identities (appearance.py and
StablePlayerIds._fix_swaps), on synthetic tracks and synthetic looks: no
video and no image model needed.
"""

from __future__ import annotations

import numpy as np

from courtshot.analysis import appearance as ap
from courtshot.analysis import identities as si
from courtshot.schemas.tracking import BoundingBox, PlayerFrame, Position2D

FPS = 25.0
N_FRAMES = 1000  # 40 s
# Four players standing well apart: two per side of the net.
SPOTS = {1: (2.5, 4.0), 2: (7.5, 4.0), 3: (2.5, 16.0), 4: (7.5, 16.0)}


def _frames(tracker_ids_at=lambda f: {1: 1, 2: 2, 3: 3, 4: 4}) -> list[PlayerFrame]:
    """Each person p stands on SPOTS[p]; tracker_ids_at(frame) says which raw
    tracker ID the detector gave person p in that frame."""
    out = []
    for f in range(N_FRAMES):
        for person, tid in tracker_ids_at(f).items():
            x, y = SPOTS[person]
            out.append(
                PlayerFrame(
                    frame_id=f,
                    timestamp_ms=1000.0 * f / FPS,
                    player_id=f"P_{tid:03d}",
                    bbox=BoundingBox(x1=100.0 * x, y1=10.0 * y, x2=100.0 * x + 40, y2=10.0 * y + 90),
                    position=Position2D(x=x, y=y),
                    confidence=0.9,
                )
            )
    return out


def _looks(person_of_tid_at, look_of_person, noise: float = 0.05, seed: int = 0) -> ap.Appearance:
    """Appearance samples, 5 a second, for tracker IDs 1..4: the look of the
    person behind each tracker ID at that time, plus noise."""
    rng = np.random.default_rng(seed)
    frames, tids, vecs = [], [], []
    for f in range(0, N_FRAMES, 5):
        for tid in (1, 2, 3, 4):
            frames.append(f)
            tids.append(tid)
            vecs.append(look_of_person[person_of_tid_at(f, tid)] + noise * rng.standard_normal(16))
    return ap.Appearance(np.array(frames), np.array(tids), np.array(vecs), FPS)


def _distinct_looks() -> dict[int, np.ndarray]:
    rng = np.random.default_rng(1)
    return {p: rng.standard_normal(16) for p in (1, 2, 3, 4)}


def _slot_of_spot(frames: list[PlayerFrame]) -> dict[int, dict[tuple[float, float], str]]:
    out: dict[int, dict[tuple[float, float], str]] = {}
    for pf in frames:
        out.setdefault(pf.frame_id, {})[(pf.position.x, pf.position.y)] = pf.player_id
    return out


class TestAppearanceLookup:
    def test_vector_is_the_tracks_average_near_that_time(self):
        looks = _distinct_looks()
        app = _looks(lambda f, tid: tid, looks, noise=0.0)
        v1, v2 = app.vector(100, "P_001"), app.vector(100, "P_002")
        assert v1 is not None and v2 is not None
        assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5
        assert float(v1 @ app.vector(500, "P_001")) > 0.99  # same person, another time
        assert float(v1 @ v2) < 0.9  # another person

    def test_unknown_or_untracked_detection_has_no_vector(self):
        app = _looks(lambda f, tid: tid, _distinct_looks())
        assert app.vector(100, "P_099") is None
        assert app.vector(100, "P_det_000") is None

    def test_no_vector_far_from_any_sample(self):
        app = ap.Appearance(np.array([0, 5]), np.array([1, 1]), np.ones((2, 4)) + np.eye(4)[:2], FPS)
        assert app.vector(3, "P_001") is not None
        assert app.vector(500, "P_001") is None


class TestFixSwaps:
    """The 'motion step' is fooled by making the tracker hand two people each
    other's tracker ID at the same instant they jump to each other's spot:
    position, motion and tracker ID all say nothing happened."""

    SWAP_AT = 400

    def _swapped_frames(self, a: int, b: int) -> list[PlayerFrame]:
        # Persons a and b trade places at SWAP_AT. The detections are the same
        # as if nobody had moved (same spots, same tracker IDs): only the
        # looks, from _person_of_tid, tell that another person stands there.
        return _frames()

    def _person_of_tid(self, a: int, b: int):
        def person(f: int, tid: int) -> int:
            if f >= self.SWAP_AT and tid in (a, b):
                return b if tid == a else a
            return tid

        return person

    def test_without_appearance_the_swap_goes_unnoticed(self):
        frames = self._swapped_frames(1, 2)
        out = si.StablePlayerIds().apply(frames)
        slots = _slot_of_spot(out)
        assert slots[10][SPOTS[1]] == slots[900][SPOTS[1]] == "P_1"

    def test_swap_between_players_who_look_different_is_undone(self):
        frames = self._swapped_frames(1, 2)
        app = _looks(self._person_of_tid(1, 2), _distinct_looks())
        mapper = si.StablePlayerIds()
        slots = _slot_of_spot(mapper.apply(frames, app))
        # before: person 1 on spot 1 is P_1; after: person 2 stands on spot 1
        assert slots[10][SPOTS[1]] == "P_1" and slots[10][SPOTS[2]] == "P_2"
        assert slots[900][SPOTS[1]] == "P_2" and slots[900][SPOTS[2]] == "P_1"
        assert slots[900][SPOTS[3]] == "P_3" and slots[900][SPOTS[4]] == "P_4"
        fixed = mapper.report["swaps_fixed"]
        assert len(fixed) == 1 and fixed[0]["players"] == ["P_1", "P_2"]
        assert abs(fixed[0]["frame"] - self.SWAP_AT) <= 2 * FPS

    def test_swap_across_the_net_is_undone(self):
        frames = self._swapped_frames(2, 3)
        app = _looks(self._person_of_tid(2, 3), _distinct_looks())
        slots = _slot_of_spot(si.StablePlayerIds().apply(frames, app))
        assert slots[900][SPOTS[2]] == "P_3" and slots[900][SPOTS[3]] == "P_2"

    def test_correct_tracks_are_left_alone(self):
        app = _looks(lambda f, tid: tid, _distinct_looks())
        mapper = si.StablePlayerIds()
        slots = _slot_of_spot(mapper.apply(_frames(), app))
        assert mapper.report["swaps_fixed"] == []
        assert mapper.report["appearance_used"] is True
        for f in (10, 500, 990):
            assert [slots[f][SPOTS[p]] for p in (1, 2, 3, 4)] == ["P_1", "P_2", "P_3", "P_4"]

    def test_lookalike_partners_are_never_switched(self):
        looks = _distinct_looks()
        looks[2] = looks[1].copy()  # persons 1 and 2 in identical kit
        for seed in range(5):
            app = _looks(self._person_of_tid(1, 2), looks, noise=0.3, seed=seed)
            mapper = si.StablePlayerIds()
            mapper.apply(self._swapped_frames(1, 2), app)
            assert mapper.report["swaps_fixed"] == []

    def test_a_look_that_changes_for_a_moment_is_not_a_swap(self):
        # persons 1 and 2 look like each other for 2 s (pose, lighting), then normal again
        def person(f: int, tid: int) -> int:
            if 400 <= f < 450 and tid in (1, 2):
                return 3 - tid
            return tid

        mapper = si.StablePlayerIds()
        mapper.apply(_frames(), _looks(person, _distinct_looks()))
        assert mapper.report["swaps_fixed"] == []

    def test_gap_filling_uses_the_corrected_identities(self):
        frames = [pf for pf in self._swapped_frames(1, 2) if not 600 <= pf.frame_id < 610]
        app = _looks(self._person_of_tid(1, 2), _distinct_looks())
        slots = _slot_of_spot(si.StablePlayerIds().apply(frames, app))
        assert slots[605][SPOTS[1]] == "P_2"  # interpolated frame, after the swap
