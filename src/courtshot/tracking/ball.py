"""
Ball detection with TrackNetV3.

TrackNetV3 is a U-Net that looks at a background frame plus SEQ_LEN
consecutive frames at once and gives one heatmap per frame. A padel ball is a
few blurred pixels in broadcast footage; seeing it move is what makes it
findable.

The detector reports at most one ball per frame (plus up to two look-alikes,
see BallFrame.candidates). It knows nothing about trajectories: wrong
detections (reflections, a spare ball on the floor) are sorted out later, in
courtshot.analysis.ball_events.

Model architecture: qaz812345/TrackNetV3 (MIT). Weights: fine-tuned on padel
with tools/train_tracknetv3.py.

Original author: Xingnan Zhu.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np

from courtshot.schemas.tracking import BallFrame, BoundingBox
from courtshot.tracking.device import detect_device, use_half_precision

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

BALL_BOX_RADIUS = 4  # pixels; the box reported around the ball's centre


def build_tracknet_v3(in_dim: int = 9, out_dim: int = 3) -> torch.nn.Module:
    """The TrackNetV3 network (U-Net with skip connections).

      Encoder:
        down_block_1: Double2DConv(in_dim -> 64)  + MaxPool
        down_block_2: Double2DConv(64 -> 128)     + MaxPool
        down_block_3: Triple2DConv(128 -> 256)    + MaxPool
        bottleneck:   Triple2DConv(256 -> 512)
      Decoder (Upsample + concat skip connections):
        up_block_1: Triple2DConv(512+256 -> 256)
        up_block_2: Double2DConv(256+128 -> 128)
        up_block_3: Double2DConv(128+64 -> 64)
        predictor:  Conv2d(64 -> out_dim) + Sigmoid

    Layer names match the original checkpoints.
    """
    import torch
    import torch.nn as nn

    class _Conv2DBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding="same", bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.bn(self.conv(x)))

    class _Double2DConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.conv_1 = _Conv2DBlock(in_ch, out_ch)
            self.conv_2 = _Conv2DBlock(out_ch, out_ch)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv_2(self.conv_1(x))

    class _Triple2DConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.conv_1 = _Conv2DBlock(in_ch, out_ch)
            self.conv_2 = _Conv2DBlock(out_ch, out_ch)
            self.conv_3 = _Conv2DBlock(out_ch, out_ch)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv_3(self.conv_2(self.conv_1(x)))

    class TrackNetV3(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_block_1 = _Double2DConv(in_dim, 64)
            self.down_block_2 = _Double2DConv(64, 128)
            self.down_block_3 = _Triple2DConv(128, 256)
            self.bottleneck = _Triple2DConv(256, 512)
            self.up_block_1 = _Triple2DConv(768, 256)
            self.up_block_2 = _Double2DConv(384, 128)
            self.up_block_3 = _Double2DConv(192, 64)
            self.predictor = nn.Conv2d(64, out_dim, kernel_size=(1, 1))
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.down_block_1(x)
            x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)
            x2 = self.down_block_2(x)
            x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)
            x3 = self.down_block_3(x)
            x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)
            x = self.bottleneck(x)
            x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)
            x = self.up_block_1(x)
            x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)
            x = self.up_block_2(x)
            x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)
            x = self.up_block_3(x)
            x = self.predictor(x)
            return self.sigmoid(x)

    return TrackNetV3()


class BallDetector:
    """Streaming ball detector: feed frames in order, get BallFrames back.

    The model looks at windows of SEQ_LEN consecutive frames. A window is run
    every ``stride`` frames, so each frame is part of SEQ_LEN / stride
    overlapping windows; its heatmap is the average over all of them, and the
    model has seen the frames after it as well as those before. Results
    therefore come SEQ_LEN - 1 frames late: update() returns the frames that
    are now final, flush() the rest.

    stride 1 runs the model once per frame (every frame is in SEQ_LEN
    windows). stride 2 halves the work and still shows every frame to the
    model four times, at four different places in the window.

    Input:  27 channels = background frame + SEQ_LEN frames, at INFER_W x INFER_H
    Output: SEQ_LEN sigmoid heatmaps, one per frame of the window
    """

    INFER_W: int = 512
    INFER_H: int = 288
    HEATMAP_THRESHOLD: float = 0.5
    MAX_BLOB_AREA: int = 200  # larger blobs are not a ball (at INFER_W x INFER_H)
    MAX_CANDIDATES: int = 3  # blobs kept per frame (the ball plus look-alikes)
    SEQ_LEN: int = 8

    def __init__(self, model_path: str, device: str | None = None, stride: int = 1) -> None:
        if not 1 <= stride <= self.SEQ_LEN:
            raise ValueError(f"stride must be 1..{self.SEQ_LEN}, got {stride}")
        self.model_path = str(model_path)
        self.stride = stride
        self._device_str = device
        self._model = None
        self._frame_buffer = None  # the last frames as one tensor on the device, newest last
        self._bg_source: np.ndarray | None = None  # from set_background()
        self._bg_tensor = None
        self._seen = 0  # frames pushed since the last reset
        self._last_window = -1  # how many frames had been seen when the model last ran
        # frames not yet returned: [frame_id, timestamp_ms, shape, sum of heatmaps, how many]
        self._pending: deque[list] = deque()

    # -- public -----------------------------------------------------------

    def set_background(self, frame: np.ndarray | None) -> None:
        """Use this frame (e.g. the median of the video) as the background.

        It is kept across reset(). With None, the first frame seen is used.
        """
        self._bg_source = frame
        self._bg_tensor = None

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        """A video frame at the size the model takes. Separate from push() so
        that it can be done ahead of time, on another thread."""
        self._ensure_model()
        # INTER_AREA: the same downscaling the training frames were made with
        return cv2.resize(frame, (self.INFER_W, self.INFER_H), interpolation=cv2.INTER_AREA)

    def update(self, frame: np.ndarray, frame_id: int, timestamp_ms: float) -> list[BallFrame]:
        """Push one frame; returns the frames whose result is now final (none
        or one, SEQ_LEN - 1 frames behind the input)."""
        return self.push(self.prepare(frame), frame.shape, frame_id, timestamp_ms)

    def push(self, resized: np.ndarray, shape: tuple, frame_id: int, timestamp_ms: float) -> list[BallFrame]:
        """update() for a frame prepare() has already resized; ``shape`` is
        the shape of the video frame it came from."""
        return self.push_many([resized], shape, [frame_id], [timestamp_ms])

    def push_many(
        self, resized: list[np.ndarray], shape: tuple, frame_ids: list[int], timestamps_ms: list[float]
    ) -> list[BallFrame]:
        """push() for several consecutive frames at once. The result is the
        same as pushing them one by one; the frames go to the GPU together,
        their windows run as one batch and the results come back together,
        which is a good deal faster when the GPU is shared with other work."""
        for frame_id, timestamp_ms in zip(frame_ids, timestamps_ms):
            self._pending.append([frame_id, timestamp_ms, shape, None, 0])
        self._buffer(resized)
        before, self._seen = self._seen, self._seen + len(resized)
        ends = [
            n for n in range(before + 1, self._seen + 1) if n >= self.SEQ_LEN and (n - self.SEQ_LEN) % self.stride == 0
        ]
        if ends:
            self._run_windows(ends)
        return self._finish(len(self._pending) - (self.SEQ_LEN - 1))

    def flush(self) -> list[BallFrame]:
        """The frames still waiting for later windows that will never come."""
        if self._pending and self._seen >= self.SEQ_LEN and self._last_window != self._seen:
            self._run_windows([self._seen])  # with a stride, the last frames may be in no window yet
        out = self._finish(len(self._pending))
        self._frame_buffer = None  # whatever comes next is another run of frames
        self._seen = 0
        self._last_window = -1
        return out

    def reset(self) -> None:
        self._frame_buffer = None
        self._bg_tensor = None
        self._seen = 0
        self._last_window = -1
        self._pending.clear()

    def candidates(self, heatmap: np.ndarray, orig_shape: tuple) -> list[tuple[float, float, float]]:
        """Ball-sized blobs of a heatmap as (x, y, conf) in original pixels,
        largest first, at most MAX_CANDIDATES.

        The first one is the detection reported. The others are kept because
        the largest blob is sometimes a reflection or a spare ball while the
        real ball is a smaller blob in the same frame; trajectory linking can
        then pick the one that continues the track.
        """
        if heatmap.max() < self.HEATMAP_THRESHOLD:
            return []
        h_orig, w_orig = orig_shape[:2]
        scale_x = w_orig / self.INFER_W
        scale_y = h_orig / self.INFER_H

        binary = (heatmap >= self.HEATMAP_THRESHOLD).astype(np.uint8)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        order = [int(k) + 1 for k in np.argsort(-areas, kind="stable") if areas[k] <= self.MAX_BLOB_AREA]
        out: list[tuple[float, float, float]] = []
        for label in order[: self.MAX_CANDIDATES]:
            cx, cy = centroids[label]
            conf = float(heatmap[labels == label].max())
            out.append((float(cx * scale_x), float(cy * scale_y), conf))
        return out

    # -- internals --------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch

        device = self._device_str or detect_device()
        self._torch_device = torch.device(device)
        self._half = use_half_precision(device)

        model = build_tracknet_v3(in_dim=(self.SEQ_LEN + 1) * 3, out_dim=self.SEQ_LEN)
        state = torch.load(self.model_path, map_location="cpu", weights_only=False)
        # A checkpoint trained at another resolution says so (train_tracknetv3.py --size)
        if isinstance(state, dict) and "infer_size" in state:
            cls = type(self)
            self.INFER_W, self.INFER_H = (int(v) for v in state["infer_size"])
            scale = (self.INFER_W * self.INFER_H) / (cls.INFER_W * cls.INFER_H)
            self.MAX_BLOB_AREA = int(round(cls.MAX_BLOB_AREA * scale))
        # plain state_dict, or wrapped in a checkpoint
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        model.load_state_dict(state)
        model.eval()
        model.to(self._torch_device, memory_format=torch.channels_last)
        if self._half:
            model.half()
        self._model = model
        logger.info("TrackNetV3 loaded on %s (%dx%d)", device, self.INFER_W, self.INFER_H)

    def _to_tensor(self, images: np.ndarray) -> torch.Tensor:
        """(N, H, W, 3) uint8 images -> (N, 3, H, W) float tensor in 0..1 on the device."""
        import torch

        t = torch.from_numpy(images).to(self._torch_device).permute(0, 3, 1, 2)
        return (t.half() if self._half else t.float()) / 255.0

    def _buffer(self, resized: list[np.ndarray]) -> None:
        """Keep resized frames for the windows they will be in. Frames stay
        on the device, so each one is converted and uploaded once instead of
        once per window."""
        import torch

        self._ensure_model()
        if self._bg_tensor is None:
            background = resized[0] if self._bg_source is None else self.prepare(self._bg_source)
            self._bg_tensor = self._to_tensor(background[None])[0]
        new = self._to_tensor(np.stack(resized))
        old = self._frame_buffer
        # the frames before these that their windows reach back to
        self._frame_buffer = new if old is None else torch.cat([old[-(self.SEQ_LEN - 1) :], new])

    def _infer(self, ends: list[int]):
        """Heatmaps (len(ends), SEQ_LEN, H, W): one window per entry of
        ``ends``, each ending with the frame that was the ends[i]-th pushed."""
        import torch

        frames = self._frame_buffer
        windows = []
        for end in ends:
            last = len(frames) - (self._seen - end)  # one past the window's last frame in the buffer
            window = frames[last - self.SEQ_LEN : last].reshape(-1, *frames.shape[2:])
            windows.append(torch.cat([self._bg_tensor, window], dim=0))
        with torch.inference_mode():
            out = self._model(torch.stack(windows).contiguous(memory_format=torch.channels_last))
        return out.float()

    def _run_windows(self, ends: list[int]) -> None:
        """Run the model on the windows ending at these frames and add each
        frame's heatmap to what other windows gave for it."""
        heatmaps = self._infer(ends)
        self._last_window = ends[-1]
        first_pending = self._seen - len(self._pending)  # how many frames were pushed before the oldest waiting one
        for end, window in zip(ends, heatmaps):
            for k in range(self.SEQ_LEN):
                entry = self._pending[end - self.SEQ_LEN + k - first_pending]
                entry[3] = window[k] if entry[3] is None else entry[3] + window[k]
                entry[4] += 1

    def _finish(self, count: int) -> list[BallFrame]:
        """The ``count`` oldest waiting frames, each from the average of its heatmaps."""
        entries = [self._pending.popleft() for _ in range(max(0, count))]
        averaged = [total / n for _, _, _, total, n in entries if n]
        if averaged and not isinstance(averaged[0], np.ndarray):
            import torch

            averaged = list(torch.stack(averaged).cpu().numpy())  # one trip from the GPU for all of them
        heatmaps = iter(averaged)
        out = []
        for frame_id, timestamp_ms, shape, _, n in entries:
            if n == 0:  # in no window: fewer than SEQ_LEN frames were ever seen
                out.append(BallFrame(frame_id=frame_id, timestamp_ms=timestamp_ms))
            else:
                out.append(self._to_ball_frame(next(heatmaps), shape, frame_id, timestamp_ms))
        return out

    def _to_ball_frame(self, heatmap: np.ndarray, orig_shape: tuple, frame_id: int, timestamp_ms: float) -> BallFrame:
        candidates = self.candidates(heatmap, orig_shape)
        if not candidates:
            return BallFrame(frame_id=frame_id, timestamp_ms=timestamp_ms)
        x, y, confidence = candidates[0]
        r = BALL_BOX_RADIUS
        return BallFrame(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            bbox=BoundingBox(x1=float(x - r), y1=float(y - r), x2=float(x + r), y2=float(y + r)),
            confidence=float(confidence),
            candidates=candidates,
        )
