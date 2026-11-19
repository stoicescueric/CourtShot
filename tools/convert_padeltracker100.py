"""
Convert PadelTracker100 into the clip layout train_tracknetv3.py reads.

PadelTracker100 has, per match, a full 1920x1080@30 video and a COCO-style
ball file (<match>_ball.json: one image per video frame, an annotation with
a bbox when the ball is visible). This script cuts each match into clips of
--clip-frames frames and writes, per clip:

    data/ptk_<match>_<NNN>/frames/000000.jpg ...   (512x288, ready for training)
    data/ptk_<match>_<NNN>/labels.csv              (frame,visibility,x,y in 512x288 pixels)

Frames with a trusted label get a row (visibility 0 = no ball). Frames that
can't be trusted get no row, so training skips them: frames inside long
stretches with no ball labels at all (the men's final is only labelled for
its first ~12 minutes), frames with two ball boxes, and frames the label
file doesn't list. Clips with nothing labelled are not written.
Frames are written small to save disk; train_tracknetv3.py's prepare step
then just copies them.

Usage (from the padex folder):
    uv run python tools/convert_padeltracker100.py --src ptk --out data
    uv run python tools/convert_padeltracker100.py --src ptk --out data --list   # only show what was found
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2

W, H = 512, 288


def frame_number(file_name: str) -> int:
    """Frame index from a COCO file_name such as 'frame_000123.jpg'."""
    nums = re.findall(r"\d+", Path(file_name).stem)
    if not nums:
        raise ValueError(f"No frame number in {file_name!r}")
    return int(nums[-1])


def load_ball_centres(path: Path) -> tuple[dict[int, tuple[float, float]], set[int], int]:
    """(frame -> ball centre in original pixels, frames listed, highest frame number).

    Frames with more than one ball box are left out (they can't be trusted
    as "the" ball), and so are frames the file doesn't list.
    """
    coco = json.loads(path.read_text())
    img_frame = {img["id"]: frame_number(img["file_name"]) for img in coco["images"]}
    ball_cats = {c["id"] for c in coco.get("categories", []) if str(c.get("name", "")).lower() == "ball"}
    centres: dict[int, tuple[float, float]] = {}
    multi: set[int] = set()
    for a in coco["annotations"]:
        if ball_cats and a.get("category_id") not in ball_cats:
            continue
        bbox = a.get("bbox")
        if not bbox or any(v is None or (isinstance(v, float) and math.isnan(v)) for v in bbox):
            continue
        x, y, w, h = bbox
        f = img_frame.get(a["image_id"])
        if f is not None:
            if f in centres:
                multi.add(f)
            centres[f] = (x + w / 2, y + h / 2)
    for f in multi:
        centres.pop(f, None)
    listed = set(img_frame.values()) - multi
    return centres, listed, max(img_frame.values())


def unknown_frames(centres: dict[int, tuple[float, float]], listed: set[int], n: int, max_gap: int) -> set[int]:
    """Frames whose label can't be trusted: not listed, multiple balls, or
    inside a run of more than max_gap frames without any ball label (in
    PadelTracker100 the men's final is only labelled for its first part, so
    long empty stretches mean "not labelled", not "no ball")."""
    unknown = {f for f in range(n) if f not in listed}
    run: list[int] = []
    for f in range(n + 1):
        if f < n and f not in centres:
            run.append(f)
            continue
        if len(run) > max_gap:
            unknown.update(run)
        run = []
    return unknown


def find_matches(src: Path) -> list[tuple[str, Path, Path]]:
    """(match name, video, ball json) for every *_ball.json with a matching video."""
    videos = [p for p in src.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}]
    out = []
    for ball in sorted(src.rglob("*_ball.json")):
        name = ball.name[: -len("_ball.json")]
        cands = [v for v in videos if v.stem == name] or [
            v for v in videos if v.stem.startswith(name) and "sample" not in v.stem
        ]
        if cands:
            out.append((name, ball, cands[0]))
        else:
            print(f"! no video found for {ball.name}")
    return out


def convert(name: str, ball_json: Path, video: Path, out: Path, clip_frames: int, max_gap: int) -> None:
    centres, listed, last = load_ball_centres(ball_json)
    n_labelled = last + 1
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw, vh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    sx, sy = W / vw, H / vh
    unknown = unknown_frames(centres, listed, n, max_gap)
    print(
        f"{name}: video {int(vw)}x{int(vh)}, {n} frames; labels cover {n_labelled} frames, "
        f"{len(centres)} with ball, {len(unknown)} treated as unlabelled"
    )

    i, clip, rows, writer_dir, written = 0, -1, [], None, 0

    def flush():
        if writer_dir is not None and rows:
            with open(writer_dir / "labels.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "visibility", "x", "y"])
                w.writerows(rows)

    while True:
        if i % clip_frames == 0:
            flush()
            clip += 1
            rows = []
            span = range(i, min(i + clip_frames, n))
            if all(f in unknown for f in span):
                # nothing labelled in this clip: skip decoding-to-disk entirely
                for _ in span:
                    if not cap.grab():
                        break
                i += len(span)
                writer_dir = None
                if i >= n:
                    break
                continue
            writer_dir = out / f"ptk_{name}_{clip:03d}"
            (writer_dir / "frames").mkdir(parents=True, exist_ok=True)
            written += 1
        ok, frame = cap.read()
        if not ok:
            break
        k = i % clip_frames
        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(writer_dir / "frames" / f"{k:06d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        c = centres.get(i)
        if i not in unknown:  # unknown frames get no row, so training skips them
            rows.append([k, 1, f"{c[0] * sx:.2f}", f"{c[1] * sy:.2f}"] if c else [k, 0, -1, -1])
        i += 1
        if i % 3000 == 0:
            print(f"  {name}: {i}/{n} frames")
    flush()
    cap.release()
    if abs(i - n_labelled) > 2:
        print(f"! {name}: video has {i} frames but labels list {n_labelled}. Check they belong together.")
    print(f"{name}: wrote {written} clips (skipped {clip + 1 - written} with no labels)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder the PadelTracker100 zip was extracted to")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--clip-frames", type=int, default=3000, help="frames per clip (3000 = 100 s)")
    ap.add_argument(
        "--max-gap",
        type=int,
        default=300,
        help="longer runs of frames without a ball label are treated as unlabelled (frames)",
    )
    ap.add_argument("--list", action="store_true", help="only list what was found")
    args = ap.parse_args()
    matches = find_matches(args.src)
    if not matches:
        raise SystemExit(f"No <match>_ball.json + video pairs found under {args.src}")
    for name, ball, video in matches:
        print(f"{name}: labels {ball}, video {video}")
    if args.list:
        return
    for name, ball, video in matches:
        if any(args.out.glob(f"ptk_{name}_*/labels.csv")):
            print(f"{name}: already converted, skipping (delete data/ptk_{name}_* to redo)")
            continue
        convert(name, ball, video, args.out, args.clip_frames, args.max_gap)


if __name__ == "__main__":
    main()
