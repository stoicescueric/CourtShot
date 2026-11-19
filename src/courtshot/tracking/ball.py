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

    Each frame is part of up to SEQ_LEN overlapping windows; its heatmap is
    the average over all of them, so the model has seen the frames after it
    as well as those before. Results therefore come SEQ_LEN - 1 frames late:
    update() returns the frames that are now final, flush() the rest.

    Input:  27 channels = background frame + SEQ_LEN frames, at INFER_W x INFER_H
    Output: SEQ_LEN sigmoid heatmaps, one per frame of the window
    """

    INFER_W: int = 512
    INFER_H: int = 288
    HEATMAP_THRESHOLD: float = 0.5
    MAX_BLOB_AREA: int = 200  # larger blobs are not a ball (at INFER_W x INFER_H)
    MAX_CANDIDATES: int = 3  # blobs kept per frame (the ball plus look-alikes)
    SEQ_LEN: int = 8

    def __init__(self, model_path: str, device: str | None = None) -> None:
        self.model_path = str(model_path)
        self._device_str = device
        self._model = None
        self._frame_buffer: list = []  # last SEQ_LEN frames, as tensors on the device
        self._bg_source: np.ndarray | None = None  # from set_background()
        self._bg_frame: np.ndarray | None = None
        self._bg_tensor = None
        self._windows: list[np.ndarray] = []  # model outputs of the last SEQ_LEN windows
        self._pending: list[tuple[int, float, tuple]] = []  # (frame_id, timestamp_ms, shape) not yet returned

    # -- public -----------------------------------------------------------

    def set_background(self, frame: np.ndarray | None) -> None:
        """Use this frame (e.g. the median of the video) as the background.

        It is kept across reset(). With None, the first frame seen is used.
        """
        self._bg_source = frame
        self._bg_frame = None
        self._bg_tensor = None

    def update(self, frame: np.ndarray, frame_id: int, timestamp_ms: float) -> list[BallFrame]:
        """Push one frame; returns the frames whose result is now final (none
        or one, SEQ_LEN - 1 frames behind the input)."""
        self._pending.append((frame_id, timestamp_ms, frame.shape))
        heatmaps = self._forward(frame)
        if heatmaps is None:
            return []
        self._windows.append(heatmaps)
        if len(self._windows) > self.SEQ_LEN:
            self._windows.pop(0)
        return [self._finish_oldest(0)]

    def flush(self) -> list[BallFrame]:
        """The frames still waiting for later windows that will never come."""
        out = []
        k = 1  # position of the oldest pending frame in the newest window
        while self._pending:
            if self._windows and k < self.SEQ_LEN:
                out.append(self._finish_oldest(k))
            else:  # fewer than SEQ_LEN frames were ever seen
                fid, ts, _ = self._pending.pop(0)
                out.append(BallFrame(frame_id=fid, timestamp_ms=ts))
            k += 1
        self._windows.clear()
        return out

    def reset(self) -> None:
        self._frame_buffer.clear()
        self._bg_frame = None
        self._bg_tensor = None
        self._windows.clear()
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
        model.to(self._torch_device)
        self._model = model
        logger.info("TrackNetV3 loaded on %s (%dx%d)", device, self.INFER_W, self.INFER_H)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 image -> (3, H, W) float tensor in 0..1 on the device."""
        import torch

        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return t.to(self._torch_device).float() / 255.0

    def _forward(self, frame: np.ndarray) -> np.ndarray | None:
        """Push one frame; heatmaps (SEQ_LEN, H, W) for the last SEQ_LEN frames,
        or None until SEQ_LEN frames have been seen."""
        self._ensure_model()
        import torch

        # INTER_AREA: the same downscaling the training frames were made with
        resized = cv2.resize(frame, (self.INFER_W, self.INFER_H), interpolation=cv2.INTER_AREA)

        if self._bg_frame is None:
            if self._bg_source is not None:
                self._bg_frame = cv2.resize(self._bg_source, (self.INFER_W, self.INFER_H), interpolation=cv2.INTER_AREA)
            else:
                self._bg_frame = resized

        # Frames are kept on the device as float tensors, so each one is
        # converted and uploaded once instead of once per window it is in.
        if self._bg_tensor is None:
            self._bg_tensor = self._to_tensor(self._bg_frame)
        self._frame_buffer.append(self._to_tensor(resized))
        if len(self._frame_buffer) > self.SEQ_LEN:
            self._frame_buffer.pop(0)
        if len(self._frame_buffer) < self.SEQ_LEN:
            return None

        tensor = torch.cat([self._bg_tensor] + self._frame_buffer, dim=0).unsqueeze(0)
        with torch.inference_mode(), torch.autocast("cuda", enabled=self._half):
            out = self._model(tensor)  # (1, SEQ_LEN, H, W)
        return out[0].float().cpu().numpy()

    def _finish_oldest(self, k: int) -> BallFrame:
        """Average the heatmaps of the frame at position k of the newest
        window: it is at position k + j of the window j steps older."""
        fid, ts, shape = self._pending.pop(0)
        maps = [self._windows[-1 - j][k + j] for j in range(len(self._windows)) if k + j < self.SEQ_LEN]
        return self._to_ball_frame(np.mean(maps, axis=0), shape, fid, ts)

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
