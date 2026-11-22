"""
What each player looks like, for telling the four players apart.

courtshot.analysis.identities follows the players by position and motion. That is enough
while they are in view, but it cannot tell who is who after they were out of
sight for a while (a changeover, a walk to the bench, a long occlusion) or
after partners swap sides unseen. Looks can: this module turns crops of the
detected players into appearance vectors with a pretrained image model
(SigLIP, the model Roboflow's sports projects use for the same job), and
the identity step compares them with what each of P_1..P_4 looked like before.

The vectors are made once per video and saved next to the tracking cache as
<video>_tracking_cache_appearance.npz, so the report and the 3D replay (which
only read the caches) get the same player identities as the tracked video.

    app = appearance_for("match_tracking_cache.pkl", video="match.mp4")
    stable, report = stabilize_tracking(raw, appearance=app)

Only a few crops per second are embedded, and only of detections the tracker
gave an ID. A detection's appearance is the average of its tracker ID's
vectors close to it in time, so every frame of a track has one.
"""

from __future__ import annotations

import json
import logging
import re
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL = "google/siglip-base-patch16-224"
SAMPLES_PER_SECOND = 5.0  # crops embedded per player per second
MIN_BOX_HEIGHT = 24  # pixels; smaller detections say nothing about looks
# The part of the detection box that is embedded: the middle of the body. The
# edges of a box are mostly court, glass and, when players overlap, the other
# player; the bottom is floor and feet.
CROP_X = (0.2, 0.8)
CROP_Y = (0.05, 0.8)


def _tracker_id(player_id: str) -> int | None:
    m = re.fullmatch(r"P_(\d+)", player_id)
    return int(m.group(1)) if m else None


class Appearance:
    """Appearance vectors of sampled detections, looked up by (frame, tracker ID)."""

    def __init__(
        self, frames: np.ndarray, track_ids: np.ndarray, vectors: np.ndarray, fps: float, window_s: float = 0.6
    ) -> None:
        self.fps = float(fps)
        self.window = max(1, int(round(window_s * fps)))
        v = vectors.astype(np.float32)
        # Most of a raw vector is what all crops share ("a padel player on a
        # court"). Taking the average out leaves what tells the players apart.
        if len(v):
            v = v - v.mean(axis=0, keepdims=True)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        self._by_track: dict[int, tuple[list[int], np.ndarray]] = {}
        order = np.lexsort((frames, track_ids))
        frames, track_ids, v = frames[order], track_ids[order], v[order]
        for tid in np.unique(track_ids):
            sel = track_ids == tid
            self._by_track[int(tid)] = (frames[sel].tolist(), v[sel])
        self._memo: dict[tuple[int, int], np.ndarray | None] = {}

    def __len__(self) -> int:
        return sum(len(f) for f, _ in self._by_track.values())

    def vector(self, frame_id: int, player_id: str) -> np.ndarray | None:
        """Unit appearance vector of this detection, or None if its tracker ID
        has no sample within the time window."""
        tid = _tracker_id(player_id)
        if tid is None or tid not in self._by_track:
            return None
        key = (frame_id, tid)
        if key not in self._memo:
            frames, vecs = self._by_track[tid]
            a = bisect_left(frames, frame_id - self.window)
            b = bisect_left(frames, frame_id + self.window + 1)
            if a == b:
                self._memo[key] = None
            else:
                m = vecs[a:b].mean(axis=0)
                self._memo[key] = m / max(float(np.linalg.norm(m)), 1e-9)
        return self._memo[key]


# ---------------------------------------------------------------------------
# making the vectors
# ---------------------------------------------------------------------------


class _Embedder:
    def __init__(self, device: str | None = None, batch: int = 64) -> None:
        import torch
        from transformers import SiglipVisionModel
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # the text half of the checkpoint is not loaded, by design
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.startswith("cuda")
        model = SiglipVisionModel.from_pretrained(MODEL).to(self.device).eval()
        self.model = model.half() if self.half else model
        self.batch = batch

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        """BGR crops -> (N, D) float32 vectors (mean of the image tokens)."""
        torch = self.torch
        out = []
        for i in range(0, len(crops), self.batch):
            imgs = [
                cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA if c.shape[0] > 224 else cv2.INTER_CUBIC)
                for c in crops[i : i + self.batch]
            ]
            x = torch.from_numpy(np.stack(imgs)[..., ::-1].copy()).to(self.device)
            x = x.permute(0, 3, 1, 2).float() / 127.5 - 1.0  # SigLIP: mean 0.5, std 0.5
            with torch.inference_mode():
                h = self.model(pixel_values=x.half() if self.half else x).last_hidden_state
            out.append(h.float().mean(dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 768), np.float32)


def body_crop(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if h < MIN_BOX_HEIGHT or w < 4:
        return None
    H, W = frame.shape[:2]
    a, b = int(max(0, x1 + CROP_X[0] * w)), int(min(W, x1 + CROP_X[1] * w))
    c, d = int(max(0, y1 + CROP_Y[0] * h)), int(min(H, y1 + CROP_Y[1] * h))
    if b - a < 4 or d - c < 8:
        return None
    return frame[c:d, a:b]


def extract(video: str | Path, tracking, fps: float | None = None, embedder=None) -> dict:
    """Embed crops of the tracked detections, SAMPLES_PER_SECOND frames a second.

    Returns the arrays that make an Appearance: frames, track_ids, vectors, fps.
    """
    by_frame: dict[int, list] = defaultdict(list)
    for pf in tracking.player_frames:
        tid = _tracker_id(pf.player_id)
        if tid is not None and pf.confidence > 0:
            by_frame[pf.frame_id].append((tid, (pf.bbox.x1, pf.bbox.y1, pf.bbox.x2, pf.bbox.y2)))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / SAMPLES_PER_SECOND)))
    embedder = embedder or _Embedder()

    frames: list[int] = []
    tids: list[int] = []
    vectors: list[np.ndarray] = []
    crops: list[np.ndarray] = []

    def flush() -> None:
        if crops:
            vectors.append(embedder(crops).astype(np.float16))
            crops.clear()

    last = max(by_frame, default=-1)
    fid = 0
    while fid <= last:
        if fid % step == 0 and fid in by_frame:
            ok, frame = cap.read()
            if not ok:
                break
            for tid, box in by_frame[fid]:
                crop = body_crop(frame, box)
                if crop is not None:
                    frames.append(fid)
                    tids.append(tid)
                    crops.append(crop.copy())
            if len(crops) >= 256:
                flush()
        elif not cap.grab():  # skip the frame without decoding it to an image
            break
        fid += 1
    flush()
    cap.release()
    return {
        "frames": np.array(frames, np.int32),
        "track_ids": np.array(tids, np.int32),
        "vectors": np.concatenate(vectors) if vectors else np.empty((0, 768), np.float16),
        "fps": np.float32(fps),
    }


# ---------------------------------------------------------------------------
# saved next to the tracking cache
# ---------------------------------------------------------------------------


def sidecar(tracking_cache: str | Path) -> Path:
    p = Path(tracking_cache)
    return p.with_name(p.stem + "_appearance.npz")


def _key(tracking_cache: Path) -> dict:
    from courtshot.tracking.cache import file_sha1

    return {
        "tracking_sha1": file_sha1(tracking_cache),
        "model": MODEL,
        "samples_per_second": SAMPLES_PER_SECOND,
        "crop": [list(CROP_X), list(CROP_Y)],
    }


def appearance_for(tracking_cache: str | Path, video: str | Path | None = None, tracking=None) -> Appearance | None:
    """The saved appearance vectors that belong to this tracking cache.

    If there are none (or they were made from other tracking) and ``video``
    is given, they are made now and saved. Without a video, returns None and
    the caller falls back to position and motion only.
    """
    tracking_cache = Path(tracking_cache)
    path = sidecar(tracking_cache)
    key = _key(tracking_cache)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if json.loads(str(z["key"])) == key:
                return Appearance(z["frames"], z["track_ids"], z["vectors"], float(z["fps"]))
        logger.info("%s was made from other tracking or settings", path.name)
    if video is None or not Path(video).exists():
        logger.warning(
            "No appearance data for %s (it is made by `courtshot run`, which has the video); "
            "player IDs use position and motion only.",
            tracking_cache.name,
        )
        return None
    if tracking is None:
        from courtshot.tracking.cache import load_tracking

        tracking = load_tracking(tracking_cache)
    logger.info("Appearance: embedding player crops from %s ...", Path(video).name)
    data = extract(video, tracking)
    np.savez_compressed(path, key=json.dumps(key), **data)
    logger.info("Appearance: %d crops embedded, saved %s", len(data["frames"]), path.name)
    return Appearance(data["frames"], data["track_ids"], data["vectors"], float(data["fps"]))
