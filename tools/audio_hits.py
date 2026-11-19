"""
Ball-impact times from a video's audio track.

A racket hit (and, more quietly, a bounce off the glass or the floor) is a
short, sharp, broadband click. This module finds those clicks:

  1. Decode the audio to mono with ffmpeg.
  2. Spectral flux: how much the spectrum's energy in the 1.5-9 kHz band rises
     from one 12 ms step to the next. Speech and crowd noise change slowly
     and sit lower; impacts jump.
  3. An onset is a local peak of the flux that stands well above the
     surrounding second of audio (median + k x spread), so the threshold
     follows the noise level of the venue.

It does not know WHAT was hit, only WHEN, and it is not wired into the shot
detection: on broadcast audio (commentary, crowd, music) many onsets are not
ball impacts. Use --compare to see how well the sounds line up with the hits
found in the video before relying on it for a given recording.

    uv run python tools/audio_hits.py videos/match.mp4            # prints the onsets
    uv run python tools/audio_hits.py videos/match.mp4 --compare  # against the video hits in the tracking cache

Returns nothing useful for videos without an audio track (has_audio() is False).
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import stft


@dataclass
class AudioSettings:
    sr: int = 22050
    n_fft: int = 1024
    hop: int = 256  # 11.6 ms steps
    band_hz: tuple[float, float] = (1500.0, 9000.0)
    window_s: float = 1.0  # length of the "surrounding audio" the threshold adapts to
    k: float = 6.0  # onset = flux above median + k x spread
    min_gap_s: float = 0.10  # two impacts are never closer than this


def has_audio(video: str | Path) -> bool:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    return out.returncode == 0 and bool(out.stdout.strip())


def load_audio(video: str | Path, sr: int = 22050) -> np.ndarray:
    """Mono float32 samples in [-1, 1]; empty if the video has no audio."""
    if not has_audio(video):
        return np.zeros(0, np.float32)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def onset_strength(y: np.ndarray, s: AudioSettings) -> tuple[np.ndarray, np.ndarray]:
    """(times_s, strength): strength is the band flux in units of the local
    spread above the local median, so 0 = typical, s.k = onset threshold."""
    freqs, times, Z = stft(y, fs=s.sr, nperseg=s.n_fft, noverlap=s.n_fft - s.hop, boundary=None)
    band = (freqs >= s.band_hz[0]) & (freqs <= s.band_hz[1])
    logmag = np.log1p(100.0 * np.abs(Z[band]))
    flux = np.maximum(np.diff(logmag, axis=1), 0.0).sum(axis=0)
    flux = np.r_[0.0, flux]
    win = max(3, int(s.window_s * s.sr / s.hop) | 1)
    med = median_filter(flux, size=win, mode="nearest")
    spread = median_filter(np.abs(flux - med), size=win, mode="nearest") + 1e-6
    return times, (flux - med) / spread


def detect_onsets(y: np.ndarray, s: AudioSettings | None = None) -> list[tuple[float, float]]:
    """[(time_s, strength)] of impact-like sounds, in time order."""
    s = s or AudioSettings()
    if len(y) < s.n_fft * 2:
        return []
    times, z = onset_strength(y, s)
    gap = max(1, int(round(s.min_gap_s * s.sr / s.hop)))
    peak = (z == maximum_filter1d(z, size=2 * gap + 1, mode="nearest")) & (z >= s.k)
    return [(float(times[i]), float(z[i])) for i in np.flatnonzero(peak)]


def onsets_for_video(video: str | Path, s: AudioSettings | None = None) -> list[tuple[float, float]]:
    s = s or AudioSettings()
    return detect_onsets(load_audio(video, s.sr), s)


def estimate_offset(onsets_s: np.ndarray, events_s: np.ndarray, tol_s: float = 0.05, search_s: float = 0.3) -> float:
    """Constant audio-minus-video delay that lines up most events with an onset."""
    if len(onsets_s) == 0 or len(events_s) == 0:
        return 0.0
    best, best_n = 0.0, -1
    for off in np.arange(-search_s, search_s + 1e-9, 0.01):
        idx = np.clip(np.searchsorted(onsets_s, events_s + off), 1, len(onsets_s) - 1)
        d = np.minimum(np.abs(onsets_s[idx] - events_s - off), np.abs(onsets_s[idx - 1] - events_s - off))
        n = int((d <= tol_s).sum())
        if n > best_n or (n == best_n and abs(off) < abs(best)):
            best, best_n = float(off), n
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--compare", action="store_true", help="compare with video hits from <video>_tracking_cache.pkl")
    ap.add_argument("--k", type=float, default=AudioSettings.k)
    a = ap.parse_args()
    if not has_audio(a.video):
        raise SystemExit(f"{a.video} has no audio track.")
    s = AudioSettings(k=a.k)
    onsets = onsets_for_video(a.video, s)
    print(f"{len(onsets)} impact-like sounds")
    if not a.compare:
        for t, z in onsets:
            print(f"{t:8.2f} s  strength {z:5.1f}")
        return

    import cv2

    from courtshot.analysis.match import analyse_cache

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    cache = a.video.with_name(a.video.stem + "_tracking_cache.pkl")
    res = analyse_cache(cache, fps=fps, fit_3d=False).ball
    if res is None:
        raise SystemExit("No ball detections or no calibration in the cache.")
    on = np.array([t for t, _ in onsets])
    hits = np.array([e.time_s for e in res.events if e.kind == "hit"])
    off = estimate_offset(on, hits)

    def share_matched(times: np.ndarray, tol_s: float = 0.06) -> float:
        if len(times) == 0 or len(on) < 2:
            return float("nan")
        idx = np.clip(np.searchsorted(on, times + off), 1, len(on) - 1)
        d = np.minimum(np.abs(on[idx] - times - off), np.abs(on[idx - 1] - times - off))
        return float((d <= tol_s).mean())

    print(f"audio delay vs video: {off * 1000:+.0f} ms")
    for kind in ("hit", "ground", "wall"):
        t = np.array([e.time_s for e in res.events if e.kind == kind])
        print(f"{kind:6s}: {share_matched(t):.0%} of {len(t)} have a sound within 60 ms")
    if len(hits):
        rnd = np.random.default_rng(0).uniform(hits.min(), hits.max(), 5000)
        print(f"random moments: {share_matched(rnd):.0%}  (the hit share only means something well above this)")


if __name__ == "__main__":
    main()
