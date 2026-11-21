"""
Tests for the ffmpeg-backed VideoWriter: browser-playable H.264 files with
even dimensions, at most max_height high, written under a temporary name and
checked with ffprobe before they appear.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from courtshot.io.video import VideoWriter, output_size, probe_video


def _frames(n: int, w: int, h: int):
    for i in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, : (i * 7) % w] = (40, 200, 90)  # something that changes
        yield frame


def _write(path, n: int, w: int, h: int, max_height: int = 720, **tools: str) -> VideoWriter:
    with VideoWriter(path, fps=25.0, frame_size=(w, h), max_height=max_height, **tools) as writer:
        for frame in _frames(n, w, h):
            writer.write(frame)
    return writer


class TestOutputSize:
    def test_scaled_down_to_the_limit(self):
        assert output_size(1920, 1080, 720) == (1280, 720)

    def test_small_videos_are_not_scaled_up(self):
        assert output_size(640, 360, 720) == (640, 360)

    def test_odd_sizes_become_even(self):
        w, h = output_size(641, 361, 720)
        assert w % 2 == 0 and h % 2 == 0 and h <= 361

    def test_odd_limit(self):
        w, h = output_size(1920, 1080, 719)
        assert (w % 2, h % 2) == (0, 0) and h <= 719


class TestVideoWriter:
    @pytest.mark.parametrize(
        ("w", "h", "max_height", "out_w", "out_h"),
        [(1280, 720, 360, 640, 360), (321, 241, 720, 320, 240), (200, 120, 720, 200, 120)],
    )
    def test_h264_yuv420p_even_and_at_most_max_height(self, tmp_path, w, h, max_height, out_w, out_h):
        out = tmp_path / "plain.mp4"
        writer = _write(out, 12, w, h, max_height)
        info = probe_video(out)
        assert info["codec_name"] == "h264" and info["pix_fmt"] == "yuv420p"
        assert (info["width"], info["height"]) == (out_w, out_h) == writer.output_size
        assert info["width"] % 2 == 0 and info["height"] % 2 == 0 and info["height"] <= max_height
        assert info["nb_read_packets"] == writer.frame_count == 12

    def test_no_temporary_file_left(self, tmp_path):
        _write(tmp_path / "tracked.mp4", 5, 64, 48)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["tracked.mp4"]

    def test_a_file_that_fails_the_check_is_removed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("courtshot.io.video.probe_video", lambda path, ffprobe: {"codec_name": "mpeg4"})
        with pytest.raises(RuntimeError, match="came out wrong"):
            _write(tmp_path / "plain.mp4", 3, 64, 48)
        assert list(tmp_path.iterdir()) == []

    def test_failure_leaves_no_file(self, tmp_path):
        out = tmp_path / "tracked.mp4"
        with pytest.raises(RuntimeError, match="boom"), VideoWriter(out, 25.0, (64, 48)) as writer:
            writer.write(np.zeros((48, 64, 3), np.uint8))
            raise RuntimeError("boom")
        assert list(tmp_path.iterdir()) == []

    def test_frame_of_the_wrong_size_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="expected"), VideoWriter(tmp_path / "x.mp4", 25.0, (64, 48)) as writer:
            writer.write(np.zeros((40, 64, 3), np.uint8))
        assert list(tmp_path.iterdir()) == []

    def test_ffmpeg_messages_go_to_stderr(self, tmp_path, monkeypatch, capfd):
        # an encoder setting ffmpeg rejects: the writer fails and says so
        monkeypatch.setattr("courtshot.io.video.output_size", lambda w, h, m: (63, 47))
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            _write(tmp_path / "bad.mp4", 3, 64, 48)
        assert "divisible by 2" in capfd.readouterr().err  # ffmpeg's reason reaches the job log
        assert list(tmp_path.iterdir()) == []

    def test_the_given_tools_win_over_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COURTSHOT_FFMPEG", str(tmp_path / "no-such-ffmpeg"))
        monkeypatch.setenv("COURTSHOT_FFPROBE", str(tmp_path / "no-such-ffprobe"))
        ffprobe = shutil.which("ffprobe")
        _write(tmp_path / "plain.mp4", 3, 64, 48, ffmpeg=shutil.which("ffmpeg"), ffprobe=ffprobe)
        assert probe_video(tmp_path / "plain.mp4", ffprobe)["nb_read_packets"] == 3

    def test_ffmpeg_path_comes_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COURTSHOT_FFMPEG", str(tmp_path / "no-such-ffmpeg"))
        with pytest.raises(FileNotFoundError):
            VideoWriter(tmp_path / "x.mp4", 25.0, (64, 48))
