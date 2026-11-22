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
from courtshot.tracking.device import use_half_precision, use_pose_closeups_only, use_threads

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


def letterbox(frame: np.ndarray, size: int, stride: int = 32) -> tuple[np.ndarray, float, int, int]:
    """The frame as YOLO takes it: longer side scaled to ``size``, grey bars
    up to a multiple of ``stride``. Returns (image, scale, left, top); a point
    (x, y) of the frame is at (x * scale + left, y * scale + top) in the image.

    The same picture ultralytics makes from a frame, so the detections are
    the same; made here so that a batch can be prepared on several threads.
    """
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    if (new_w, new_h) != (w, h):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = (-new_w) % stride, (-new_h) % stride
    left, top = pad_w // 2, pad_h // 2
    out = np.full((new_h + pad_h, new_w + pad_w, 3), 114, np.uint8)
    out[top : top + new_h, left : left + new_w] = frame
    return out, scale, left, top


class YoloPersonTracker:
    """People in a frame, with tracker IDs (ultralytics YOLO + ByteTrack).

    track() takes one frame, track_batch() several consecutive ones (faster:
    the detector sees them together). Use one or the other for a video; each
    keeps its own tracks.
    """

    PERSON_CLASS_ID = 0
    CONFIDENCE = 0.1  # low on purpose: ByteTrack uses the weak boxes too
    TRACKER_FRAME_RATE = 30  # what ultralytics gives its trackers; track_buffer counts frames at this rate

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
        self.device = device
        self.half = use_half_precision(device)
        self.imgsz = imgsz
        # Tracker config (ultralytics yaml): argument, else $COURTSHOT_TRACKER,
        # else courtshot_bytetrack.yaml. courtshot_botsort.yaml adds appearance (ReID).
        # Both ship next to this module; a file of your own is found by its path.
        name = tracker or os.environ.get("COURTSHOT_TRACKER", "courtshot_bytetrack.yaml")
        shipped = Path(__file__).with_name(name)
        self.tracker = name if Path(name).exists() or not shipped.exists() else str(shipped)
        self._model = self._load_model()
        self._byte_tracker = None  # track_batch()'s tracker, made on first use
        self._byte_tracker_usable: bool | None = None
        self._prepare_pool: ThreadPoolExecutor | None = None

    def _load_model(self):
        from ultralytics import YOLO

        return YOLO(self.model_path)

    def track_batch(self, frames: list[np.ndarray]) -> list[list[RawDetection]]:
        """Detect and track in several consecutive frames of one video (all
        the same size); one list of detections per frame.

        The detector runs on the frames together and ByteTrack then takes
        them one by one, which gives what track() gives frame by frame. Other
        trackers (BoT-SORT needs the detector's features) go through track().
        """
        detections = self.detect_batch(frames)
        if detections is None:
            return [self.track(frame) for frame in frames]
        return [self.track_detected(d, frames[0].shape) for d in detections]

    def detect_batch(self, frames: list[np.ndarray]) -> list[np.ndarray] | None:
        """The people in several frames (all the same size), not yet tracked:
        per frame an array with a row x1, y1, x2, y2, confidence, class for
        each. None when the tracker in use can't take detections made ahead
        (anything but ByteTrack); track() is the way then.
        """
        if not self._use_byte_tracker():
            return None
        if not frames:
            return []
        import torch

        if self._prepare_pool is None and use_threads():
            self._prepare_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="courtshot-letterbox")
        prepare = self._prepare_pool.map if self._prepare_pool is not None else map
        boxed = list(prepare(lambda frame: letterbox(frame, self.imgsz), frames))
        _, scale, left, top = boxed[0]
        batch = torch.from_numpy(np.stack([b[0] for b in boxed])).to(self.device)
        batch = batch.flip(-1).permute(0, 3, 1, 2).contiguous()  # BGR, HWC -> RGB, CHW
        batch = (batch.half() if self.half else batch.float()) / 255.0
        results = self._model.predict(
            batch,
            conf=self.CONFIDENCE,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
        )

        h, w = frames[0].shape[:2]
        out: list[np.ndarray] = []
        for result in results:
            data = result.boxes.data.float().cpu().numpy().copy()
            data[:, [0, 2]] = np.clip((data[:, [0, 2]] - left) / scale, 0, w)
            data[:, [1, 3]] = np.clip((data[:, [1, 3]] - top) / scale, 0, h)
            out.append(data)
        return out

    def track_detected(self, detections: np.ndarray, shape: tuple) -> list[RawDetection]:
        """Tracker IDs for the detections detect_batch() made in one frame
        (``shape``: that of the frame). Call it for every frame, in order. It
        needs no GPU, so it can run on another thread while the detector is
        busy with the next frames."""
        from ultralytics.engine.results import Boxes

        # rows: x1, y1, x2, y2, ID, confidence, class, index of the detection
        tracks = self._byte_tracker.update(Boxes(detections, shape[:2]))
        if len(tracks) == 0:  # as ultralytics does: the detections, without IDs
            return [self._detection(row[:4], row[4], None) for row in detections]
        return [self._detection(row[:4], row[5], int(row[4])) for row in tracks]

    def _use_byte_tracker(self) -> bool:
        if self._byte_tracker_usable is None:
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import YAML, IterableSimpleNamespace
            from ultralytics.utils.checks import check_yaml

            cfg = IterableSimpleNamespace(**YAML.load(check_yaml(self.tracker)))
            self._byte_tracker_usable = cfg.tracker_type == "bytetrack"
            if self._byte_tracker_usable:
                self._byte_tracker = BYTETracker(args=cfg, frame_rate=self.TRACKER_FRAME_RATE)
        return self._byte_tracker_usable

    @staticmethod
    def _detection(xyxy, confidence, track_id: int | None) -> RawDetection:
        return RawDetection(
            bbox=BoundingBox(x1=float(xyxy[0]), y1=float(xyxy[1]), x2=float(xyxy[2]), y2=float(xyxy[3])),
            confidence=float(confidence),
            track_id=track_id,
        )

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
        self._byte_tracker = None
        self._byte_tracker_usable = None


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
    CLOSEUP_SIZE: tuple[int, int] = (CROP_IMGSZ, CROP_IMGSZ)  # width, height of a close-up in estimate_closeups()
    WEIGHTS = "yolo26m-pose.pt"

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

            model_path = str(get_weight_path(self.WEIGHTS))
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.half = use_half_precision(device)
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

    def predict_full_frames(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
        """predict_full_frame() for several frames of one size in one pass."""
        if not frames:
            return []
        self._ensure_model()
        import torch

        boxed = [letterbox(frame, self.imgsz) for frame in frames]
        _, scale, left, top = boxed[0]
        batch = torch.from_numpy(np.stack([b[0] for b in boxed])).to(self.device or "cpu")
        batch = batch.flip(-1).permute(0, 3, 1, 2).contiguous()  # BGR, HWC -> RGB, CHW
        batch = (batch.half() if self.half else batch.float()) / 255.0
        results = self._model.predict(
            batch,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
        )
        out: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
        for res in results:
            if res.keypoints is None or res.keypoints.conf is None or res.boxes is None or len(res.boxes) == 0:
                out.append(None)
                continue
            back = np.array([left, top], np.float32)
            xyxy = (res.boxes.xyxy.float().cpu().numpy() - np.tile(back, 2)) / scale
            xy = (res.keypoints.xy.float().cpu().numpy() - back) / scale
            out.append((xyxy, xy, res.keypoints.conf.float().cpu().numpy()))
        return out

    def assign(self, bboxes: list[BoundingBox], full_frame) -> list[list[PoseKeypoint]]:
        """The poses of a full-frame pass given to the player boxes they
        belong to (empty for a box with none)."""
        result: list[list[PoseKeypoint]] = [[] for _ in bboxes]
        if full_frame is not None and bboxes:
            from scipy.optimize import linear_sum_assignment

            det_xyxy, kpts_xy, kpts_conf = full_frame
            iou = iou_matrix(bboxes, det_xyxy)
            for r, c in zip(*linear_sum_assignment(1.0 - iou)):
                if iou[r, c] >= self.MIN_IOU:
                    result[r] = self._to_keypoints(kpts_xy[c], kpts_conf[c])
        return result

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

        result = self.assign(bboxes, full_frame)
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

    def estimate_closeups(self, views: list[tuple[np.ndarray, BoundingBox]]) -> list[list[PoseKeypoint]]:
        """Keypoints of one player per (frame, box), each from a close-up of
        that player, all in one batched pass. The views may come from
        different frames (courtshot.analysis.poses uses it that way).

        Every close-up is brought to CLOSEUP_SIZE, so a far player is seen
        larger than in the full frame and batches always have the same shape.
        """
        out: list[list[PoseKeypoint]] = [[] for _ in views]
        width, height = self.CLOSEUP_SIZE
        images, placed = [], []
        for i, (frame, b) in enumerate(views):
            h_img, w_img = frame.shape[:2]
            pw, ph = self.CROP_PAD * (b.x2 - b.x1), self.CROP_PAD * (b.y2 - b.y1)
            x0, y0 = int(max(0, b.x1 - pw)), int(max(0, b.y1 - ph))
            x1, y1 = int(min(w_img, b.x2 + pw)), int(min(h_img, b.y2 + ph))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            scale = min(width / (x1 - x0), height / (y1 - y0))
            new_w, new_h = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
            left, top = (width - new_w) // 2, (height - new_h) // 2
            image = np.full((height, width, 3), 114, np.uint8)
            image[top : top + new_h, left : left + new_w] = cv2.resize(
                frame[y0:y1, x0:x1], (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            images.append(image)
            placed.append((i, x0, y0, scale, left, top))
        if not images:
            return out

        self._ensure_model()
        import torch

        batch = torch.from_numpy(np.stack(images)).to(self.device or "cpu")
        batch = batch.flip(-1).permute(0, 3, 1, 2).contiguous()  # BGR, HWC -> RGB, CHW
        batch = (batch.half() if self.half else batch.float()) / 255.0
        results = self._model.predict(
            batch,
            conf=self.CROP_CONFIDENCE,
            verbose=False,
            device=self.device,
            half=self.half,
            imgsz=max(width, height),
        )
        for (i, x0, y0, scale, left, top), res in zip(placed, results):
            if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None or res.keypoints.conf is None:
                continue
            b = views[i][1]
            in_closeup = BoundingBox(
                x1=(b.x1 - x0) * scale + left,
                y1=(b.y1 - y0) * scale + top,
                x2=(b.x2 - x0) * scale + left,
                y2=(b.y2 - y0) * scale + top,
            )
            ious = iou_matrix([in_closeup], res.boxes.xyxy.cpu().numpy())[0]
            j = int(np.argmax(ious))
            if ious[j] >= self.MIN_IOU:
                xy = (res.keypoints.xy[j].float().cpu().numpy() - (left, top)) / scale + (x0, y0)
                out[i] = self._to_keypoints(xy, res.keypoints.conf[j].float().cpu().numpy())
        return out

    def reset(self) -> None:
        self._model = None


# ---------------------------------------------------------------------------
# the two together
# ---------------------------------------------------------------------------


def foot_position(bbox: BoundingBox) -> tuple[float, float]:
    """Where a player stands in the image: bottom-centre of the box."""
    return (bbox.x1 + bbox.x2) / 2.0, bbox.y2


def players_between(
    before: list[PlayerFrame], after: list[PlayerFrame], frame_id: int, timestamp_ms: float, share: float
) -> list[PlayerFrame]:
    """The players in a frame the detector skipped, from the frames it saw
    just before and after it: every tracked player that is in both, ``share``
    (0..1) of the way from one to the other. At 60 frames a second a player
    moves a few centimetres between two frames, far less than the detector's
    own jitter, so a straight line is as good as detecting again.
    """
    later = {pf.player_id: pf for pf in after}
    out: list[PlayerFrame] = []
    for a in before:
        b = later.get(a.player_id)
        if b is None or a.player_id.startswith("P_det_"):  # P_det_: no tracker ID, not the same person for sure
            continue

        def mix(u: float, v: float) -> float:
            return u + share * (v - u)

        position = None
        if a.position is not None and b.position is not None:
            position = Position2D(x=mix(a.position.x, b.position.x), y=mix(a.position.y, b.position.y))
        out.append(
            PlayerFrame(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                player_id=a.player_id,
                bbox=BoundingBox(
                    x1=mix(a.bbox.x1, b.bbox.x1),
                    y1=mix(a.bbox.y1, b.bbox.y1),
                    x2=mix(a.bbox.x2, b.bbox.x2),
                    y2=mix(a.bbox.y2, b.bbox.y2),
                ),
                position=position,
                confidence=mix(a.confidence, b.confidence),
            )
        )
    return out


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

    def detect_and_track_batch(
        self,
        frames: list[np.ndarray],
        frame_ids: list[int],
        timestamps_ms: list[float],
        homography_matrix: np.ndarray | None = None,
    ) -> list[list[PlayerFrame]]:
        """detect_and_track() for several consecutive frames at once; one
        list of players per frame. The detector sees the frames together,
        which is what makes this faster."""
        if hasattr(self.tracker, "track_batch"):
            raw = self.tracker.track_batch(frames)
        else:
            raw = [self.tracker.track(frame) for frame in frames]
        return [
            self._build_player_frames(dets, frame, frame_id, timestamp_ms, homography_matrix)
            for dets, frame, frame_id, timestamp_ms in zip(raw, frames, frame_ids, timestamps_ms)
        ]

    # The same in two steps, for running them on two threads: detect_batch()
    # is the GPU's work, track_detected() the tracker's.

    def detect_batch(self, frames: list[np.ndarray]) -> list | None:
        """The detector's part of detect_and_track_batch(): what it found in
        each frame, to be given to track_detected(). None when the tracker
        (or a pose estimator, which needs the frames) rules out the split."""
        if self.pose is not None or not hasattr(self.tracker, "detect_batch"):
            return None
        return self.tracker.detect_batch(frames)

    def track_detected(
        self,
        detections,
        shape: tuple,
        frame_id: int,
        timestamp_ms: float,
        homography_matrix: np.ndarray | None = None,
    ) -> list[PlayerFrame]:
        """The players of one frame from what detect_batch() found in it;
        call for every frame, in order."""
        raw = self.tracker.track_detected(detections, shape)
        return self._build_player_frames(raw, None, frame_id, timestamp_ms, homography_matrix)

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
