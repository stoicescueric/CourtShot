"""
Fine-tune Padex's TrackNetV3 ball detector on padel footage.

Padex's TrackNetV3 support expects weights from the original TrackNetV3
project (trained on badminton shuttlecocks), but Padex's own download link
for them is broken (404). Get the original checkpoint instead:

    https://github.com/qaz812345/TrackNetV3  ->  TrackNetV3_ckpts.zip (Google Drive)
    unzip, use ckpts/TrackNet_best.pt  (MIT licence)

This script starts from that checkpoint, continues training on labelled padel
clips, and saves a checkpoint Padex can load:

    padex.process("match.mp4", use_tracknet_v3=True,
                  ball_model_path="checkpoints/padel_tracknetv3.pt")

Input matches Padex's inference exactly: 512x288 BGR frames, a background
frame followed by 8 consecutive frames (27 channels), one sigmoid heatmap per
frame out.

-----------------------------------------------------------------------------
Data format (one folder per clip):

    data/
      clip_001/
        video.mp4            or  frames/000000.jpg, 000001.jpg, ...
        labels.csv           columns: frame,visibility,x,y
      clip_002/
        ...

  frame       0-based frame index within the clip
  visibility  1 if the ball is visible, 0 if not (x,y ignored then)
  x, y        ball centre in pixels of the original video/frames
  Frames missing from labels.csv count as "ball not visible", unless the
  file also has explicit visibility-0 rows (as label_ball.py writes): then
  missing frames are treated as unlabelled and skipped.

Your own footage: label it with label_ball.py, which writes this layout.

PadelTracker100 needs a small converter into this layout (see
convert_padeltracker100() at the bottom; fill it in once you have seen how the
download is organised). Your own labelled clips use the same layout.

-----------------------------------------------------------------------------
Steps (from the padex folder):

  1. uv run python train_tracknetv3.py prepare --data data --cache cache
       Extracts frames at 512x288 as JPEG and a median background per clip.
       Roughly 30 KB per frame, so ~3 GB for 100k frames. Run once.

  2. uv run python train_tracknetv3.py evaluate --cache cache --weights ckpts/TrackNet_best.pt
       Scores the badminton weights on your validation clips (baseline).

  3. uv run python train_tracknetv3.py train --cache cache --weights ckpts/TrackNet_best.pt --epochs 10
       Fine-tunes, evaluates after every epoch, keeps the best checkpoint
       (by F1) at checkpoints/padel_tracknetv3.pt.

Validation clips: by default the last 15% of clips (sorted by name). Pass
--val clip_a,clip_b to choose them; use clips from a different match than the
training clips, otherwise the score is optimistic.

Options worth trying once the basics work:

  --size 1024x576   (all three commands; use a separate --cache folder)
       Train and run at twice the resolution. At 512x288 a ball from a 1080p
       video is only 2-3 pixels wide; at 1024x576 it keeps its shape. About
       4x slower and 4x the memory. The checkpoint remembers its size, so
       Padex runs it at the right resolution without further settings.

  --occlusion-aug 0.3   (train)
       In that share of training windows, hide the ball in 1-2 frames under
       a patch copied from next to it, while still asking the model for its
       position. Teaches it to carry the ball through short occlusions
       (behind a player, a glass post) from the frames before and after.

Hardware: a CUDA GPU is much faster. Apple Silicon (MPS) works; start with
--batch-size 2 or 4. On CPU use it only to check the pipeline (--max-steps).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from padex.tracking.ball import TrackNetV3BallDetectionStrategy, _build_tracknet_v3

import builtins
import functools

print = functools.partial(builtins.print, flush=True)  # show progress live, even through tee

BASE_W, BASE_H = TrackNetV3BallDetectionStrategy.INFER_W, TrackNetV3BallDetectionStrategy.INFER_H  # 512x288
W, H = BASE_W, BASE_H
SEQ = TrackNetV3BallDetectionStrategy.SEQ_LEN  # 8
RADIUS = 2.5  # target disk radius in 512x288 pixels (same as the original TrackNetV3 training)


def set_size(size: str | None) -> None:
    """--size WxH: resolution of the frames the model sees (default 512x288)."""
    global W, H, RADIUS
    if not size:
        return
    try:
        w, h = (int(v) for v in size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--size must look like 1024x576, got {size!r}")
    if w % 8 or h % 8:
        raise SystemExit("--size: width and height must be multiples of 8")
    W, H = w, h
    RADIUS = 2.5 * w / BASE_W


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Plain state_dict at 512x288 (as before); with the size attached otherwise."""
    if (W, H) == (BASE_W, BASE_H):
        torch.save(model.state_dict(), path)
    else:
        torch.save({"model": model.state_dict(), "infer_size": [W, H]}, path)


# ---------------------------------------------------------------------------
# prepare: decode once into small JPEGs + background
# ---------------------------------------------------------------------------


def iter_source_frames(clip: Path):
    """Yield (index, BGR frame) from video.mp4 or a frames/ folder."""
    video = next((p for p in clip.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}), None)
    if video is not None:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open {video}")
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
        cap.release()
        return
    frames_dir = clip / "frames"
    if not frames_dir.is_dir():
        raise SystemExit(f"{clip}: no video file and no frames/ folder")
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for i, p in enumerate(files):
        yield i, cv2.imread(str(p))


def read_labels(path: Path) -> tuple[dict[int, tuple[float, float]], set[int]]:
    """(frame -> (x, y) for visible frames, set of every frame that has a row)."""
    out, rows = {}, set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            k = int(row["frame"])
            rows.add(k)
            if int(float(row["visibility"])) > 0:
                out[k] = (float(row["x"]), float(row["y"]))
    return out, rows


def prepare(data: Path, cache: Path, quality: int = 92) -> None:
    clips = sorted(p for p in data.iterdir() if p.is_dir() and (p / "labels.csv").exists())
    if not clips:
        raise SystemExit(f"No clip folders with labels.csv found in {data}")
    cache.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        out = cache / clip.name
        if (out / "meta.json").exists():
            print(f"{clip.name}: already prepared, skipping")
            continue
        (out / "frames").mkdir(parents=True, exist_ok=True)
        labels, rows = read_labels(clip / "labels.csv")
        orig_w = orig_h = None
        n = 0
        for i, frame in iter_source_frames(clip):
            if frame is None:
                raise SystemExit(f"{clip.name}: unreadable frame {i}")
            orig_h, orig_w = frame.shape[:2]
            small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / "frames" / f"{i:06d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, quality])
            n = i + 1
        if n < SEQ:
            print(f"{clip.name}: only {n} frames, need at least {SEQ}; skipping")
            continue

        # Median background from up to 60 frames spread over the clip.
        picks = np.linspace(0, n - 1, min(60, n)).astype(int)
        stack = np.stack([cv2.imread(str(out / "frames" / f"{i:06d}.jpg")) for i in picks])
        cv2.imwrite(str(out / "background.jpg"), np.median(stack, axis=0).astype(np.uint8))

        sx, sy = W / orig_w, H / orig_h
        balls = {str(i): [x * sx, y * sy] for i, (x, y) in labels.items() if i < n}
        # If labels.csv lists "no ball" frames explicitly (label_ball.py does),
        # frames with no row at all are unlabelled, not "no ball": windows
        # containing them are skipped in training.
        explicit = len(rows) > len(labels)
        unknown = sorted(set(range(n)) - rows) if explicit else []
        meta = {"frames": n, "orig_size": [orig_w, orig_h], "size": [W, H], "balls": balls, "unknown": unknown}
        (out / "meta.json").write_text(json.dumps(meta))
        note = f", {len(unknown)} unlabelled (skipped)" if unknown else ""
        print(f"{clip.name}: {n} frames, {len(balls)} with ball{note} ({orig_w}x{orig_h})")


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


@dataclass
class Clip:
    name: str
    root: Path
    n: int
    orig_size: tuple[int, int]
    balls: dict[int, tuple[float, float]]
    unknown: frozenset[int] = frozenset()


def load_clips(cache: Path) -> list[Clip]:
    clips = []
    for d in sorted(p for p in cache.iterdir() if (p / "meta.json").exists()):
        m = json.loads((d / "meta.json").read_text())
        if tuple(m.get("size", (BASE_W, BASE_H))) != (W, H):
            raise SystemExit(f"{d} was prepared at {'x'.join(map(str, m.get('size', (BASE_W, BASE_H))))}, "
                             f"not {W}x{H}. Use a separate --cache folder per --size.")
        clips.append(Clip(d.name, d, m["frames"], tuple(m["orig_size"]),
                          {int(k): tuple(v) for k, v in m["balls"].items()},
                          frozenset(m.get("unknown", []))))
    if not clips:
        raise SystemExit(f"Nothing prepared in {cache}. Run the prepare step first.")
    return clips


def split_clips(clips: list[Clip], val_names: str | None) -> tuple[list[Clip], list[Clip]]:
    if val_names:
        names = {v.strip() for v in val_names.split(",")}
        missing = names - {c.name for c in clips}
        if missing:
            raise SystemExit(f"Unknown validation clips: {sorted(missing)}")
        return [c for c in clips if c.name not in names], [c for c in clips if c.name in names]
    if len(clips) == 1:
        raise SystemExit("Only one clip: pass more clips, or split it into two folders.")
    k = max(1, round(0.15 * len(clips)))
    return clips[:-k], clips[-k:]


def target_heatmap(xy: tuple[float, float] | None) -> np.ndarray:
    """Binary disk around the ball (1 inside, 0 outside).

    Binary rather than a soft Gaussian, so the network is trained to push the
    ball region above Padex's fixed 0.5 detection threshold.
    """
    hm = np.zeros((H, W), np.float32)
    if xy is None:
        return hm
    x, y = xy
    if not (0 <= x < W and 0 <= y < H):
        return hm
    yy, xx = np.ogrid[:H, :W]
    hm[(xx - x) ** 2 + (yy - y) ** 2 <= RADIUS**2] = 1.0
    return hm


class BallWindows(Dataset):
    """Windows of SEQ consecutive frames, plus a background frame."""

    def __init__(self, clips: list[Clip], stride: int, train: bool, occlusion_aug: float = 0.0) -> None:
        self.clips = clips
        self.train = train
        self.occlusion_aug = occlusion_aug
        self._size = (W, H, RADIUS)  # DataLoader workers re-import this module with the defaults
        self.index = [(ci, s) for ci, c in enumerate(clips) for s in range(0, c.n - SEQ + 1, stride)
                      if not any(s + i in c.unknown for i in range(SEQ))]
        self._bg = {c.name: cv2.imread(str(c.root / "background.jpg")) for c in clips}

    def __len__(self) -> int:
        return len(self.index)

    def _frame(self, clip: Clip, i: int) -> np.ndarray:
        img = cv2.imread(str(clip.root / "frames" / f"{i:06d}.jpg"))
        if img is None:
            raise RuntimeError(f"Missing frame {i} in {clip.name}")
        return img

    @staticmethod
    def _occlude(frames: list[np.ndarray], targets: list) -> list[np.ndarray]:
        """Hide the ball in 1-2 consecutive frames under a patch copied from
        just beside it. The target stays, so the model has to place the ball
        from the neighbouring frames. Only inner frames of the window, where
        there is context on both sides."""
        inner = [i for i in range(1, SEQ - 1) if targets[i] is not None]
        if not inner:
            return frames
        first = random.choice(inner)
        frames = list(frames)
        for i in range(first, min(first + random.randint(1, 2), SEQ - 1)):
            if targets[i] is None:
                break
            half = int(round(random.uniform(2.0, 4.0) * RADIUS))
            x, y = int(round(targets[i][0])), int(round(targets[i][1]))
            ang = random.uniform(0, 2 * math.pi)
            dist = random.uniform(2.5, 5.0) * half
            sx, sy = int(round(x + dist * math.cos(ang))), int(round(y + dist * math.sin(ang)))
            if not (half <= x < W - half and half <= y < H - half
                    and half <= sx < W - half and half <= sy < H - half):
                continue
            img = frames[i].copy()
            img[y - half:y + half + 1, x - half:x + half + 1] = \
                frames[i][sy - half:sy + half + 1, sx - half:sx + half + 1]
            frames[i] = img
        return frames

    def __getitem__(self, k: int):
        global W, H, RADIUS
        W, H, RADIUS = self._size
        ci, s = self.index[k]
        clip = self.clips[ci]
        frames = [self._frame(clip, s + i) for i in range(SEQ)]
        targets = [clip.balls.get(s + i) for i in range(SEQ)]

        # Padex uses the first frame it sees as background; the original
        # TrackNetV3 uses a median. Train on both so either works.
        if self.train and random.random() < 0.5:
            bg = self._frame(clip, random.randrange(clip.n))
        else:
            bg = self._bg[clip.name]

        if self.train and random.random() < self.occlusion_aug:
            frames = self._occlude(frames, targets)

        imgs = [bg] + frames
        if self.train:
            if random.random() < 0.5:  # horizontal flip
                imgs = [im[:, ::-1] for im in imgs]
                targets = [None if t is None else (W - 1 - t[0], t[1]) for t in targets]
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-20, 20)    # brightness
            imgs = [np.clip(im.astype(np.float32) * alpha + beta, 0, 255) for im in imgs]

        x = np.concatenate([np.asarray(im, np.float32) for im in imgs], axis=2) / 255.0
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        y = torch.from_numpy(np.stack([target_heatmap(t) for t in targets]))
        has = torch.tensor([t is not None for t in targets])
        pts = torch.tensor([[-1.0, -1.0] if t is None else list(t) for t in targets])
        return x, y, has, pts, ci


# ---------------------------------------------------------------------------
# model, loss, evaluation
# ---------------------------------------------------------------------------


def pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str | None, rgb_to_bgr: bool | None = None) -> torch.nn.Module:
    """Build the Padex TrackNetV3 and load weights.

    path:       None for random init; the original TrackNetV3 checkpoint
                (ckpts/TrackNet_best.pt); or a checkpoint saved by this script.
    rgb_to_bgr: Padex feeds OpenCV (BGR) frames. The original TrackNetV3
                checkpoint (recognisable by its 'param_dict' entry) was
                trained on RGB (its dataset code loads frames with PIL), so its
                first layer is reordered to BGR by default.
    """
    model = _build_tracknet_v3(in_dim=(SEQ + 1) * 3, out_dim=SEQ)
    if path is None:
        print("No --weights given: starting from random weights. Training will need many more "
              "epochs; the original TrackNetV3 checkpoint is a much better start (see the docstring).")
        return model
    state = torch.load(path, map_location="cpu", weights_only=False)
    original = isinstance(state, dict) and "param_dict" in state
    if original:
        pd = state["param_dict"]
        if pd.get("seq_len", SEQ) != SEQ or pd.get("bg_mode", "concat") != "concat":
            raise SystemExit(f"Checkpoint has seq_len={pd.get('seq_len')}, bg_mode={pd.get('bg_mode')}; "
                             f"Padex needs seq_len={SEQ}, bg_mode=concat (use TrackNet_best.pt).")
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    if rgb_to_bgr is None:
        rgb_to_bgr = original
    if rgb_to_bgr:
        k = "down_block_1.conv_1.conv.weight"
        w = state[k]
        order = [3 * g + c for g in range(w.shape[1] // 3) for c in (2, 1, 0)]
        state[k] = w[:, order].clone()
        print("Reordered first-layer input channels RGB -> BGR to match Padex.")
    model.load_state_dict(state)
    print(f"Loaded weights: {path}")
    return model


def wbce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Weighted BCE from TrackNetV3: down-weights easy pixels (mostly background)."""
    p = pred.clamp(1e-6, 1 - 1e-6)
    loss = -((1 - p) ** 2 * target * torch.log(p) + p**2 * (1 - target) * torch.log(1 - p))
    return loss.mean()


def heatmap_to_point(hm: np.ndarray, thr: float, max_area: int) -> tuple[float, float] | None:
    """Same post-processing as Padex: largest blob above threshold, centroid."""
    if hm.max() < thr:
        return None
    binary = (hm >= thr).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if areas[best - 1] > max_area:
        return None
    return float(cents[best][0]), float(cents[best][1])


@torch.no_grad()
def evaluate(model, loader, clips: list[Clip], device, tol_px: float, limit: int | None = None) -> dict:
    """Precision / recall / F1 with Padex's post-processing.

    A detection is correct if within tol_px of the label, measured in the
    clip's ORIGINAL resolution (so tolerance means the same thing for any
    video size).
    """
    model.eval()
    thr = TrackNetV3BallDetectionStrategy.HEATMAP_THRESHOLD
    max_area = round(TrackNetV3BallDetectionStrategy.MAX_BLOB_AREA * (W * H) / (BASE_W * BASE_H))
    tp = fp = fn = tn = 0
    errors = []
    for b, (x, _, has, pts, ci) in enumerate(loader):
        if limit is not None and b >= limit:
            break
        out = model(x.to(device)).cpu().numpy()
        # Padex uses the last channel (newest frame) of each window.
        for i in range(out.shape[0]):
            clip = clips[int(ci[i])]
            sx, sy = clip.orig_size[0] / W, clip.orig_size[1] / H
            det = heatmap_to_point(out[i, -1], thr, max_area)
            truth = pts[i, -1].tolist() if bool(has[i, -1]) else None
            if det is None and truth is None:
                tn += 1
            elif det is None:
                fn += 1
            elif truth is None:
                fp += 1
            else:
                d = math.hypot((det[0] - truth[0]) * sx, (det[1] - truth[1]) * sy)
                if d <= tol_px:
                    tp += 1
                    errors.append(d)
                else:
                    fp += 1
                    fn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "median_error_px": float(np.median(errors)) if errors else float("nan")}


def fmt(m: dict) -> str:
    return (f"P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  "
            f"(tp {m['tp']}, fp {m['fp']}, fn {m['fn']}, tn {m['tn']}, "
            f"median error {m['median_error_px']:.1f}px)")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def make_loaders(args, clips):
    train_clips, val_clips = split_clips(clips, args.val)
    print(f"train clips: {[c.name for c in train_clips]}")
    print(f"val clips:   {[c.name for c in val_clips]}")
    train_ds = BallWindows(train_clips, stride=args.stride, train=True,
                           occlusion_aug=getattr(args, "occlusion_aug", 0.0))
    val_ds = BallWindows(val_clips, stride=args.val_stride, train=False)
    kw = {"num_workers": args.workers, "persistent_workers": args.workers > 0}
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)
    return train_dl, val_dl, val_clips


def cmd_evaluate(args) -> None:
    device = pick_device(args.device)
    clips = load_clips(args.cache)
    _, val_dl, val_clips = make_loaders(args, clips)
    model = load_model(args.weights, args.rgb_to_bgr).to(device)
    print("validation:", fmt(evaluate(model, val_dl, val_clips, device, args.tol, args.max_val_batches)))


def cmd_train(args) -> None:
    random.seed(0)
    torch.manual_seed(0)
    device = pick_device(args.device)
    print(f"device: {device}")
    clips = load_clips(args.cache)
    train_dl, val_dl, val_clips = make_loaders(args, clips)
    model = load_model(args.weights, args.rgb_to_bgr).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = len(train_dl) if args.max_steps is None else min(len(train_dl), args.max_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * steps_per_epoch))

    base = evaluate(model, val_dl, val_clips, device, args.tol, args.max_val_batches)
    print("before fine-tuning:", fmt(base))
    best = base["f1"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    history = [{"epoch": 0, **base}]

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, (x, y, *_rest) in enumerate(train_dl, 1):
            pred = model(x.to(device))
            loss = wbce_loss(pred, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            if step % 50 == 0 or step == steps_per_epoch:
                rate = step / (time.time() - t0)
                print(f"epoch {epoch} step {step}/{steps_per_epoch}  loss {running / step:.5f}  "
                      f"{rate:.1f} it/s  eta {(steps_per_epoch - step) / max(rate, 1e-9) / 60:.0f} min")
            if step >= steps_per_epoch:
                break

        m = evaluate(model, val_dl, val_clips, device, args.tol, args.max_val_batches)
        history.append({"epoch": epoch, "loss": running / steps_per_epoch, **m})
        print(f"epoch {epoch}:", fmt(m))
        save_checkpoint(model, args.out.with_name(args.out.stem + "_last.pt"))
        if m["f1"] > best:
            best = m["f1"]
            save_checkpoint(model, args.out)
            print(f"  new best, saved {args.out}")
        (args.out.parent / "history.json").write_text(json.dumps(history, indent=2))

    if not args.out.exists():
        print("No epoch beat the starting weights; nothing saved as best (see *_last.pt).")
    print(f"\nUse it in Padex with:\n  use_tracknet_v3=True, ball_model_path=\"{args.out}\"")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="extract 512x288 frames + backgrounds")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--size", help="frame size the model sees, e.g. 1024x576 (default 512x288)")

    for name in ("train", "evaluate"):
        q = sub.add_parser(name)
        q.add_argument("--cache", type=Path, default=Path("cache"))
        q.add_argument("--size", help="frame size the model sees, e.g. 1024x576 (default 512x288); "
                       "the cache must have been prepared with the same --size")
        q.add_argument("--val", help="comma-separated validation clip names")
        q.add_argument("--weights", help="checkpoint to start from (original ckpts/TrackNet_best.pt, "
                       "or one saved by this script); omit for random init")
        q.add_argument("--rgb-to-bgr", dest="rgb_to_bgr", action="store_true", default=None,
                       help="force reordering first-layer channels RGB->BGR")
        q.add_argument("--no-rgb-to-bgr", dest="rgb_to_bgr", action="store_false",
                       help="never reorder (default: only for the original checkpoint)")
        q.add_argument("--batch-size", type=int, default=4)
        q.add_argument("--workers", type=int, default=2)
        q.add_argument("--device", help="cuda, mps or cpu (default: best available)")
        q.add_argument("--tol", type=float, default=10.0, help="hit tolerance in original pixels")
        q.add_argument("--max-val-batches", type=int, default=None)
        q.add_argument("--val-stride", type=int, default=4,
                       help="score every Nth frame of the validation clips (1 = all, slower)")
        q.add_argument("--stride", type=int, default=SEQ // 2,
                       help="frames between training windows (smaller = more, overlapping windows)")
        if name == "train":
            q.add_argument("--epochs", type=int, default=10)
            q.add_argument("--lr", type=float, default=1e-4)
            q.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch (quick tests)")
            q.add_argument("--occlusion-aug", type=float, default=0.0,
                           help="share of training windows where the ball is hidden in 1-2 frames (try 0.3)")
            q.add_argument("--out", type=Path, default=Path("checkpoints/padel_tracknetv3.pt"))

    args = ap.parse_args()
    set_size(args.size)
    if args.cmd == "prepare":
        prepare(args.data, args.cache)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
    else:
        cmd_train(args)


# ---------------------------------------------------------------------------
# PadelTracker100 -> data/ layout
# ---------------------------------------------------------------------------


def convert_padeltracker100(src: Path, dst: Path) -> None:
    """Convert the PadelTracker100 download into data/<clip>/{video|frames, labels.csv}.

    Not written yet: the exact file layout of the download wasn't available
    when this script was made. Open the included
    explore-padeltracker100-annotations.ipynb, note where the frames (or
    videos) and the ball coordinates live, and fill this in. Each output clip
    needs consecutive frames, so split at cuts/replays. Keep the two matches
    in separate clips so one can be used for validation.
    """
    raise NotImplementedError("Fill in once the PadelTracker100 layout is known.")


if __name__ == "__main__":
    main()
