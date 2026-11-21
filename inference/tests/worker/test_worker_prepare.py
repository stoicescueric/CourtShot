"""prepare.py's decisions on small real videos made with ffmpeg."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FFMPEG, FFPROBE, LIMITS, make_video, remux

from courtshot.errors import JobFailure
from courtshot.worker.prepare import corrections, prepare_video, probe


def prepare(source: Path, width: int, height: int, limits: dict = LIMITS) -> Path:
    calibration = {"points": {}, "frame_width": width, "frame_height": height}
    return prepare_video(source, source.with_name("prepared.mp4"), calibration, limits,
                         ffmpeg=FFMPEG, ffprobe=FFPROBE, on_progress=lambda fraction: None)


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_video(tmp_path_factory.mktemp("plain") / "plain.mp4", seconds=3)


def test_constant_rate_video_is_used_untouched(plain: Path) -> None:
    assert corrections(probe(FFPROBE, plain)) == []
    assert prepare(plain, 160, 120) == plain
    assert not plain.with_name("prepared.mp4").exists()


@pytest.mark.parametrize(
    ("kind", "reason", "size"),
    [
        ("variable", "variable frame rate", (160, 120)),
        ("rotated", "rotated 90 degrees", (120, 160)),
        ("offset", "starts at 2.000 s", (160, 120)),
        ("anamorphic", "non-square pixels (4:3)", (213, 120)),
    ],
)
def test_corrected_copy(plain: Path, tmp_path: Path, kind: str, reason: str, size: tuple[int, int]) -> None:
    source = tmp_path / f"{kind}.mp4"
    if kind == "variable":  # every fifth frame dropped from a 30 fps clip, timestamps kept
        make_video(source, seconds=3, rate=30, filters="select='not(eq(mod(n\\,5)\\,2))'",
                   extra=("-fps_mode", "passthrough"))
    elif kind == "rotated":
        remux(plain, source, "-display_rotation:v:0", "90")
    elif kind == "offset":
        remux(plain, source, after_input=("-output_ts_offset", "2"))
    else:
        make_video(source, seconds=3, filters="setsar=4/3")

    assert corrections(probe(FFPROBE, source)) == [reason]
    fractions: list[float] = []
    prepared = prepare_video(source, tmp_path / "prepared.mp4", {"frame_width": size[0], "frame_height": size[1]},
                             LIMITS, ffmpeg=FFMPEG, ffprobe=FFPROBE, on_progress=fractions.append)

    assert prepared == tmp_path / "prepared.mp4"
    info = probe(FFPROBE, prepared)
    assert corrections(info) == []
    assert (info.width, info.height) == size
    assert fractions[-1] == 1.0
    if kind == "variable":
        assert info.rate == 30  # the measured rate, not the 24 frames a second on average


def test_corrected_copy_is_reused(tmp_path: Path, plain: Path) -> None:
    source = remux(plain, tmp_path / "rotated.mp4", "-display_rotation:v:0", "90")
    first = prepare(source, 120, 160)
    written = first.stat().st_mtime_ns
    assert prepare(source, 120, 160) == first
    assert first.stat().st_mtime_ns == written  # the tracking cache keyed on this file stays valid


def test_too_short(tmp_path: Path) -> None:
    short = make_video(tmp_path / "short.mp4", seconds=0.5)
    with pytest.raises(JobFailure) as failure:
        prepare(short, 160, 120)
    assert failure.value.retryable is False
    assert "0.5 s long" in failure.value.message


def test_too_long(plain: Path) -> None:
    with pytest.raises(JobFailure, match="limits are 1 s to 2 s"):
        prepare(plain, 160, 120, {**LIMITS, "max_video_seconds": 2})


def test_frame_size_must_match_the_calibration(plain: Path) -> None:
    with pytest.raises(JobFailure, match="160x120 pixels but the court was marked on a 1280x720 frame"):
        prepare(plain, 1280, 720)


def test_not_a_video(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video" * 100)
    with pytest.raises(JobFailure, match="cannot be read"):
        prepare(junk, 160, 120)
