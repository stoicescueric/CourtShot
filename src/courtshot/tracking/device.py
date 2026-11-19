"""
Project: CourtShot
File Created: 2026-03-05
Author: Xingnan Zhu
File Name: device.py
Description:
    Hardware device auto-detection for model inference.
    Tries GPU (CUDA / MPS) first, falls back to CPU.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_cached_device: str | None = None


def detect_device() -> str:
    """Auto-detect the best available device for inference.

    Priority: cuda > mps > cpu.
    Result is cached after first call.

    Returns:
        Device string compatible with ultralytics YOLO: "cuda", "mps", or "cpu".
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    # Try CUDA (NVIDIA GPU)
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("Using CUDA device: %s", device_name)
            _cached_device = "cuda"
            return _cached_device
    except ImportError:
        pass

    # Try MPS (Apple Silicon GPU)
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using Apple MPS (Metal Performance Shaders)")
            _cached_device = "mps"
            return _cached_device
    except ImportError:
        pass

    logger.info("No GPU detected, using CPU")
    _cached_device = "cpu"
    return _cached_device


def use_half_precision(device: str | None) -> bool:
    """Whether to run models in half precision (FP16) on this device.

    FP16 roughly halves GPU inference time on CUDA with near-identical
    detections. It is off on CPU and MPS, where it is slow or unreliable.
    Set COURTSHOT_FP16=0 to force full precision.
    """
    if os.environ.get("COURTSHOT_FP16", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    return str(device or "").startswith("cuda")


def use_pose_closeups_only() -> bool:
    """Whether pose comes only from close-ups of the detected players, without
    the full-frame pose pass (COURTSHOT_POSE_CLOSEUPS=1). Off by default: tracking
    runs faster, but the near players get a usable pose a little less often.
    """
    return os.environ.get("COURTSHOT_POSE_CLOSEUPS", "0").strip().lower() in ("1", "true", "yes", "on")


def use_threads() -> bool:
    """Whether independent models may run on worker threads at the same time.

    Results are identical either way; set COURTSHOT_THREADS=0 to run everything
    on one thread (e.g. to debug or to profile a single model).
    """
    return os.environ.get("COURTSHOT_THREADS", "1").strip().lower() not in ("0", "false", "no", "off")
