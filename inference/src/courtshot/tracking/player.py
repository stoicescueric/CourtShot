"""
Player detection, short-term tracking and pose.

Per frame:
  1. YOLO finds people; ByteTrack gives each box a tracker ID. Those IDs are
     short-lived (they break at occlusions and crossings); the four lasting
     identities are made later, in courtshot.analysis.identities.
  2. The bottom-centre of each box (the feet) is mapped onto the court.
     People more than COURT_MARGIN_M outside it are dropped: spectators,
     line judges and most reflections in the glass.
  3. The most confident `max_players` are kept.
  4. YOLO-Pose gives 17 COCO keypoints per kept player.

Original author: Xingnan Zhu.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from courtshot.schemas.tracking import BoundingBox, PlayerFrame, PoseKeypoint, Position2D
from courtshot.tracking.device import detect_device, use_half_precision, use_pose_closeups_only, use_threads

logger = logging.getLogger(__name__)

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass
class RawDetection:
    """One person found in a frame."""

    bbox: BoundingBox
    confidence: float
    track_id: int | None = None


# ---------------------------------------------------------------------------
# people: YOLO + ByteTrack
# ---------------------------------------------------------------------------


class YoloPersonTracker:
    """People in a frame, with tracker IDs (ultralytics YOLO + ByteTrack)."""

    PERSON_CLASS_ID = 0
    CONFIDENCE = 0.1  # low on purpose: ByteTrack uses the weak boxes too

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        imgsz: int = 1280,
        tracker: str | None = None,
    ) -> None:
        if model_path is None:
            from courtshot.weights import get_weight_path

            model_path = str(get_weight_path("yolo26m.pt"))
        self.model_path = model_path
        self.device = device or detect_device()
        self.half = use_half_precision(self.device)
        self.imgsz = imgsz
        # Tracker config (ultralytics yaml): argument, else $COURTSHOT_TRACKER,
        # else courtshot_bytetrack.yaml. courtshot_botsort.yaml adds appearance (ReID).
        # Both ship next to this module; a file of your own is found by its path.
        name = tracker or os.environ.get("COURTSHOT_TRACKER", "courtshot_bytetrack.yaml")
        shipped = Path(__file__).with_name(name)
        self.tracker = name if Path(name).exists() or not shipped.exists() else str(shipped)
        self._model = self._load_model()

    def _load_model(self):
        from ultralytics import YOLO

        return YOLO(self.model_path)

    def track(self, frame: np.ndarray) -> list[RawDetection]:
        """Detect and track; call with the frames of one video in order."""
        results = self._model.track(
            frame,
            conf=self.CONFIDENCE,
            classes=[self.PERSON_CLASS_ID],
            persist=True,
            tracker=self.tracker,
            verbose=False,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
        )
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        return [
            RawDetection(
                bbox=BoundingBox(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3])),
                confidence=float(confs[i]),
                track_id=int(track_ids[i]) if track_ids is not None else None,
            )
            for i, b in enumerate(xyxy)
        ]

    def reset(self) -> None:
        """Forget the tracks (new video)."""
        self._model = self._load_model()


# ---------------------------------------------------------------------------
# pose: YOLO-Pose
# ---------------------------------------------------------------------------


def iou_matrix(bboxes: list[BoundingBox], det_xyxy: np.ndarray) -> np.ndarray:
    """IoU of every box in bboxes with every row of det_xyxy."""
    if len(bboxes) == 0 or len(det_xyxy) == 0:
        return np.zeros((len(bboxes), len(det_xyxy)), dtype=np.float64)
    a = np.array([[b.x1, b.y1, b.x2, b.y2] for b in bboxes], dtype=np.float64)[:, None, :]
    d = np.asarray(det_xyxy, dtype=np.float64)[None, :, :4]
    inter = np.clip(np.minimum(a[..., 2], d[..., 2]) - np.maximum(a[..., 0], d[..., 0]), 0.0, None) * np.clip(
        np.minimum(a[..., 3], d[..., 3]) - np.maximum(a[..., 1], d[..., 1]), 0.0, None
    )
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    union = area_a + area_d - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


class YoloPoseEstimator:
    """17 COCO keypoints per player (YOLO-Pose).

    One pass over the full frame, matched to the player boxes; players it
    finds no pose for (small, far side of the net, behind glass) get a second
    look at a close-up of just that player.
    """

    MIN_IOU: float = 0.3  # a pose belongs to a box it overlaps at least this much
    CROP_PAD: float = 0.25  # close-up margin around the player box, as a share of its size
    CROP_IMGSZ: int = 384
    CROP_CONFIDENCE: float = 0.1

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = 0.3,
        device: str | None = None,
        imgsz: int = 1280,
        crop_fallback: bool = True,
    ) -> None:
        if model_path is None:
            from courtshot.weights import get_weight_path

            model_path = str(get_weight_path("yolo26m-pose.pt"))
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device or detect_device()
        self.half = use_half_precision(self.device)
        self.imgsz = imgsz
        self.crop_fallback = crop_fallback
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)

    def predict_full_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Full-frame pose pass: (boxes xyxy, keypoints xy, keypoint conf) of
        everyone found, or None if nobody was.

        It does not need the player boxes, so it can run at the same time as
        player detection; hand the result to estimate() as ``full_frame``.
        """
        self._ensure_model()
        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
        )
        has_poses = (
            bool(results)
            and results[0].keypoints is not None
            and results[0].keypoints.conf is not None
            and results[0].boxes is not None
            and len(results[0].boxes) > 0
        )
        if not has_poses:
            return None
        return (
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].keypoints.xy.cpu().numpy(),  # (N, 17, 2)
            results[0].keypoints.conf.cpu().numpy(),  # (N, 17)
        )

    _NOT_RUN = object()

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[BoundingBox],
        full_frame=_NOT_RUN,
    ) -> list[list[PoseKeypoint]]:
        """Keypoints per bbox (empty where none were found). ``full_frame`` is
        the result of predict_full_frame() for this frame if it was already
        run; otherwise it is run here."""
        if not bboxes:
            return []
        self._ensure_model()
        if full_frame is self._NOT_RUN:
            full_frame = self.predict_full_frame(frame)

        result: list[list[PoseKeypoint]] = [[] for _ in bboxes]
        if full_frame is not None:
            from scipy.optimize import linear_sum_assignment

            det_xyxy, kpts_xy, kpts_conf = full_frame
            iou = iou_matrix(bboxes, det_xyxy)
            for r, c in zip(*linear_sum_assignment(1.0 - iou)):
                if iou[r, c] >= self.MIN_IOU:
                    result[r] = self._to_keypoints(kpts_xy[c], kpts_conf[c])

        missing = [i for i, kps in enumerate(result) if not kps]
        if missing and self.crop_fallback:
            for i, kps in zip(missing, self._estimate_from_crops(frame, [bboxes[i] for i in missing])):
                result[i] = kps
        return result

    @staticmethod
    def _to_keypoints(xy: np.ndarray, conf: np.ndarray, dx: float = 0.0, dy: float = 0.0) -> list[PoseKeypoint]:
        return [
            PoseKeypoint(name=name, x=float(xy[k, 0]) + dx, y=float(xy[k, 1]) + dy, confidence=float(conf[k]))
            for k, name in enumerate(COCO_KEYPOINT_NAMES)
            if float(conf[k]) > 0.0
        ]

    def _estimate_from_crops(self, frame: np.ndarray, bboxes: list[BoundingBox]) -> list[list[PoseKeypoint]]:
        """Pose for each bbox from a padded crop around it (one batched pass)."""
        h_img, w_img = frame.shape[:2]
        crops, origins = [], []
        for b in bboxes:
            pw, ph = self.CROP_PAD * (b.x2 - b.x1), self.CROP_PAD * (b.y2 - b.y1)
            x1, y1 = int(max(0, b.x1 - pw)), int(max(0, b.y1 - ph))
            x2, y2 = int(min(w_img, b.x2 + pw)), int(min(h_img, b.y2 + ph))
            crops.append(None if x2 - x1 < 8 or y2 - y1 < 8 else frame[y1:y2, x1:x2])
            origins.append((x1, y1))

        out: list[list[PoseKeypoint]] = [[] for _ in bboxes]
        valid = [i for i, c in enumerate(crops) if c is not None]
        if not valid:
            return out
        results = self._model.predict(
            [crops[i] for i in valid],
            conf=self.CROP_CONFIDENCE,
            verbose=False,
            device=self.device,
            half=self.half,
            imgsz=self.CROP_IMGSZ,
        )
        for i, res in zip(valid, results):
            if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None or res.keypoints.conf is None:
                continue
            x0, y0 = origins[i]
            b = bboxes[i]
            local = BoundingBox(x1=b.x1 - x0, y1=b.y1 - y0, x2=b.x2 - x0, y2=b.y2 - y0)
            ious = iou_matrix([local], res.boxes.xyxy.cpu().numpy())[0]
            j = int(np.argmax(ious))
            if ious[j] >= self.MIN_IOU:
                out[i] = self._to_keypoints(
                    res.keypoints.xy[j].cpu().numpy(),
                    res.keypoints.conf[j].cpu().numpy(),
                    dx=x0,
                    dy=y0,
                )
        return out

    def reset(self) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# the two together
# ---------------------------------------------------------------------------


def foot_position(bbox: BoundingBox) -> tuple[float, float]:
    """Where a player stands in the image: bottom-centre of the box."""
    return (bbox.x1 + bbox.x2) / 2.0, bbox.y2


class PlayerDetector:
    """People on the court in each frame, as PlayerFrames.

    tracker: anything with track(frame) -> list[RawDetection] and reset()
    pose:    anything with estimate(frame, bboxes, full_frame=...) and,
             optionally, predict_full_frame(frame); None for no pose
    """

    COURT_MARGIN_M: float = 1.0
    POSE_SMOOTHING: float = 0.6  # weight of the new frame in the keypoint average

    def __init__(
        self,
        tracker=None,
        pose=None,
        max_players: int = 4,
        pose_closeups_only: bool | None = None,
    ) -> None:
        # How many in-court detections to keep per frame (most confident
        # first). 4 suits a consumer that takes them as the players; the
        # identity step wants more to choose from, so that an umpire or a
        # reflection scoring higher than a far player doesn't push that
        # player out.
        self.max_players = max_players
        # Pose from close-ups of the players only, skipping the full-frame
        # pose pass (COURTSHOT_POSE_CLOSEUPS=1): faster, slightly worse poses for
        # the near players.
        self.pose_closeups_only = use_pose_closeups_only() if pose_closeups_only is None else pose_closeups_only
        self.tracker = tracker or YoloPersonTracker()
        self.pose = pose
        # Run the full-frame pose pass alongside detection (COURTSHOT_THREADS=0: off)
        self.parallel_pose = use_threads()
        self._pose_pool: ThreadPoolExecutor | None = None
        self._pose_buffer: dict[str, dict[str, tuple[float, float, float]]] = {}

    def detect_and_track(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
        homography_matrix: np.ndarray | None = None,
    ) -> list[PlayerFrame]:
        """The players in this frame; call with the frames of a video in order."""
        # The full-frame pose pass does not depend on the detections, so it
        # runs on a worker thread while the main thread detects and tracks.
        pose_future = None
        predict_pose = None if self.pose_closeups_only else getattr(self.pose, "predict_full_frame", None)
        if predict_pose is not None and self.parallel_pose:
            if self._pose_pool is None:
                self._pose_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="courtshot-pose")
            pose_future = self._pose_pool.submit(predict_pose, frame)

        try:
            raw = self.tracker.track(frame)
        except BaseException:
            if pose_future is not None:
                pose_future.exception()  # wait: never leave the worker running
            raise
        return self._build_player_frames(raw, frame, frame_id, timestamp_ms, homography_matrix, pose_future)

    def close(self) -> None:
        """Stop the pose worker thread; call at the end of a run."""
        if self._pose_pool is not None:
            self._pose_pool.shutdown(wait=True)
            self._pose_pool = None

    def reset(self) -> None:
        """Forget tracks and pose history (new video)."""
        self.tracker.reset()
        if self.pose is not None:
            self.pose.reset()
        self._pose_buffer.clear()

    # -- internals --------------------------------------------------------

    def _build_player_frames(
        self,
        raw_detections: list[RawDetection],
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: float,
        homography_matrix: np.ndarray | None,
        pose_future: Future | None = None,
    ) -> list[PlayerFrame]:
        # Always collect the worker's result, so its errors surface here and
        # it never overlaps with the next frame.
        pose_kwargs = {}
        if pose_future is not None:
            pose_kwargs["full_frame"] = pose_future.result()
        elif self.pose_closeups_only:
            pose_kwargs["full_frame"] = None  # every player goes to the close-up pass

        if not raw_detections:
            return []

        # 1. court positions; people outside the court are dropped
        positioned: list[tuple[RawDetection, Position2D | None]] = []
        for det in raw_detections:
            pos = None
            if homography_matrix is not None:
                court_pt = self._pixel_to_court(det.bbox, homography_matrix)
                if court_pt is None:
                    continue
                pos = Position2D(x=court_pt[0], y=court_pt[1])
            positioned.append((det, pos))

        # 2. the most confident max_players
        if len(positioned) > self.max_players:
            positioned.sort(key=lambda t: t[0].confidence, reverse=True)
            positioned = positioned[: self.max_players]

        # 3. pose
        poses: list[list[PoseKeypoint]] | None = None
        if self.pose is not None:
            poses = self.pose.estimate(frame, [det.bbox for det, _ in positioned], **pose_kwargs)

        # 4. PlayerFrames
        player_frames: list[PlayerFrame] = []
        for i, (det, pos) in enumerate(positioned):
            tracked = det.track_id is not None
            player_id = f"P_{det.track_id:03d}" if tracked else f"P_det_{i:03d}"
            keypoints = poses[i] if poses and i < len(poses) else []
            if keypoints and tracked:
                keypoints = self._smooth_keypoints(player_id, keypoints)
            player_frames.append(
                PlayerFrame(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    player_id=player_id,
                    bbox=det.bbox,
                    position=pos,
                    confidence=det.confidence,
                    keypoints=keypoints,
                )
            )
        return player_frames

    def _smooth_keypoints(self, player_id: str, keypoints: list[PoseKeypoint]) -> list[PoseKeypoint]:
        """Exponential moving average of a tracked player's keypoints."""
        alpha = self.POSE_SMOOTHING
        prev = self._pose_buffer.get(player_id, {})
        smoothed: list[PoseKeypoint] = []
        new_buf: dict[str, tuple[float, float, float]] = {}
        for kp in keypoints:
            if kp.name in prev:
                px, py, pc = prev[kp.name]
                sx = alpha * kp.x + (1 - alpha) * px
                sy = alpha * kp.y + (1 - alpha) * py
                sc = alpha * kp.confidence + (1 - alpha) * pc
            else:
                sx, sy, sc = kp.x, kp.y, kp.confidence
            new_buf[kp.name] = (sx, sy, sc)
            smoothed.append(PoseKeypoint(name=kp.name, x=sx, y=sy, confidence=sc))
        self._pose_buffer[player_id] = new_buf
        return smoothed

    def _pixel_to_court(self, bbox: BoundingBox, H: np.ndarray) -> tuple[float, float] | None:
        """The feet on the court, in metres; None if outside it (plus margin)."""
        pt = np.array([[foot_position(bbox)]], dtype=np.float64)
        x, y = (float(v) for v in cv2.perspectiveTransform(pt, H)[0, 0])
        margin = self.COURT_MARGIN_M
        if x < -margin or x > 10.0 + margin or y < -margin or y > 20.0 + margin:
            return None
        return max(0.0, min(10.0, x)), max(0.0, min(20.0, y))
