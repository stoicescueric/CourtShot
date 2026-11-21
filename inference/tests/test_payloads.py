"""
Tests for report_payload and scene_payload, and the data files, on a
synthetic match (no video, no models).
"""

from __future__ import annotations

import csv
import json

import pytest

from courtshot.report.match_report import report_payload, write_data_files
from courtshot.report.scene3d import JOINTS, NONE, scene_payload
from tests import synthetic

REPORT_KEYS = {
    "title",
    "duration_s",
    "t0",
    "fps",
    "settings",
    "players",
    "teams",
    "movement_rallies",
    "match",
    "rallies",
    "shots",
    "shot_types",
    "shot_labels",
    "attacking_types",
    "timeline",
    "quality",
}
SCENE_KEYS = {"fps", "n", "camera", "players", "ball", "flights", "hits", "rallies", "none", "video"}


@pytest.fixture(scope="module")
def with_ball():
    return synthetic.analysis(with_ball=True)


@pytest.fixture(scope="module")
def without_ball():
    return synthetic.analysis(with_ball=False)


def _json(data: dict) -> str:
    return json.dumps(data, separators=(",", ":"), allow_nan=False)


class TestReportPayload:
    def test_contract_shape(self, with_ball):
        data = json.loads(_json(report_payload(with_ball).payload))
        assert set(data) == REPORT_KEYS
        assert data["title"] == "clip"
        assert [p["id"] for p in data["players"]] == ["P_1", "P_2", "P_3", "P_4"]
        assert [p["team"] for p in data["players"]] == ["T_1", "T_1", "T_2", "T_2"]
        assert [t["id"] for t in data["teams"]] == ["T_1", "T_2"]
        assert all(len(p["heat"]) == 40 and len(p["heat"][0]) == 40 for p in data["players"])
        assert data["fps"] == synthetic.FPS and data["duration_s"] == pytest.approx(249 / synthetic.FPS)

    def test_ball_analysis_is_in_it(self, with_ball):
        data = report_payload(with_ball).payload
        assert [s["player"] for s in data["shots"]] == ["P_1", "P_3", "P_2"]
        assert [s["index"] for s in data["shots"]] == [0, 1, 2]
        assert data["match"]["shots"] == 3 and len(data["rallies"]) == 1
        assert data["quality"]["has_ball"] and data["quality"]["has_3d"]
        assert all(p["shots"] is not None for p in data["players"])

    def test_without_ball(self, without_ball):
        data = report_payload(without_ball).payload
        assert data["shots"] == [] and data["match"] is None and data["rallies"] is None
        assert data["settings"]["rally_source"] == "movement"
        assert all(p["shots"] is None for p in data["players"])
        assert not data["quality"]["has_ball"] and not data["quality"]["has_3d"]

    def test_players_who_move_have_distance_and_rallies(self, without_ball):
        data = report_payload(without_ball).payload
        assert all(p["move"]["distance_m"] > 5 for p in data["players"])
        assert data["movement_rallies"]


class TestDataFiles:
    def test_with_ball(self, with_ball, tmp_path):
        names = write_data_files(tmp_path, report_payload(with_ball))
        assert names == ["summary.json", "positions.csv", "ball_events.csv", "shots.csv"]
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert {"fps", "duration_s", "rallies", "players", "teams", "ball", "stats"} <= set(summary)
        with open(tmp_path / "shots.csv", newline="") as f:
            assert len(list(csv.DictReader(f))) == 3
        with open(tmp_path / "positions.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4 * 250 and {r["player"] for r in rows} == {"P_1", "P_2", "P_3", "P_4"}

    def test_without_ball(self, without_ball, tmp_path):
        assert write_data_files(tmp_path, report_payload(without_ball)) == ["summary.json", "positions.csv"]
        assert sorted(p.name for p in tmp_path.iterdir()) == ["positions.csv", "summary.json"]
        assert "ball" not in json.loads((tmp_path / "summary.json").read_text())


class TestScenePayload:
    def test_contract_shape(self, with_ball):
        scene = json.loads(_json(scene_payload(with_ball)))
        assert set(scene) == SCENE_KEYS
        n = scene["n"]
        assert n == 250 and scene["video"] is None and scene["none"] == NONE
        assert len(scene["ball"]) == 4 * n
        assert [p["id"] for p in scene["players"]] == ["P_1", "P_2", "P_3", "P_4"]
        assert all(len(p["pose"]) == n * len(JOINTS) * 3 for p in scene["players"])
        assert {"pos", "right", "up", "back", "fov", "aspect"} == set(scene["camera"])

    def test_flights_and_ball_frames(self, with_ball):
        scene = scene_payload(with_ball)
        assert [f["hitter"] for f in scene["flights"]] == ["P_1", "P_3", "P_2"]
        assert [h["player"] for h in scene["hits"]] == ["P_1", "P_3", "P_2"]
        owned = {scene["ball"][4 * f + 3] for f in range(scene["n"])} - {NONE}
        assert owned == {0, 1, 2}
        frame = int(2.5 * synthetic.FPS)  # mid first flight: above the floor, in centimetres
        assert scene["ball"][4 * frame + 3] == 0 and scene["ball"][4 * frame + 1] > 50
        assert scene["flights"][0]["bounces"][0]["kind"] == "ground"

    def test_none_without_ball_or_3d(self, without_ball, with_ball):
        assert scene_payload(without_ball) is None
        no_3d = synthetic.analysis(with_ball=True)
        no_3d.ball3d = None
        assert scene_payload(no_3d) is None


class TestSameInputSameJson:
    def test_report_and_scene_twice(self):
        first, second = synthetic.analysis(), synthetic.analysis()
        assert _json(report_payload(first).payload) == _json(report_payload(second).payload)
        assert _json(report_payload(first).summary) == _json(report_payload(second).summary)
        assert _json(scene_payload(first)) == _json(scene_payload(second))
