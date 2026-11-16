"""
Label the ball in your own padel footage, for train_tracknetv3.py.

Usage (from the padex folder):

    uv run python label_ball.py match.mp4 --start 1200 --end 1800 --name rally_01
    uv run python label_ball.py match.mp4 --start 1200 --end 1800 --name rally_01 \
        --suggest match_tracking_cache.pkl      # pre-fill with Padex's detections

It writes data/rally_01/frames/000000.jpg ... and data/rally_01/labels.csv
(frame,visibility,x,y), the layout train_tracknetv3.py expects. Label one
continuous stretch per clip (no camera cuts or replays inside a clip).
Re-running with the same --name resumes where you left off.

Controls
    left click     ball is here -> save and go to the next frame
    Enter          accept the suggestion (orange ring) -> next frame
    space / n      no ball visible in this frame -> next frame
    d / a          next / previous frame without changing anything
    w / s          jump 10 frames forward / back
    z              clear this frame's label
    q / Esc        save and quit (also saved every 25 frames)

The small window at the top right is a 4x magnifier around the mouse, for
clicking the centre of a blurred ball. Blue dots are the ball's last labelled
positions, to help you follow it.

Tips
  * Click the centre of the ball. If it's a motion streak, click the middle
    of the streak.
  * Mark "no ball" only when you really can't see it (hidden behind a player,
    out of frame). Partly visible still counts as visible.
  * Pick varied rallies: smashes, lobs, ball near the glass, near players,
    both ends of the court. Hard frames teach the model the most.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np

WIN = "Label ball"
VIEW_W, VIEW_H = 1280, 720
MAG, MAG_HALF = 4, 24  # magnifier zoom and half-size of the source patch (pixels)


def load_labels(path: Path) -> dict[int, tuple[int, float, float]]:
    out: dict[int, tuple[int, float, float]] = {}
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                out[int(row["frame"])] = (int(row["visibility"]), float(row["x"]), float(row["y"]))
    return out


def save_labels(path: Path, labels: dict[int, tuple[int, float, float]]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "visibility", "x", "y"])
        for k in sorted(labels):
            vis, x, y = labels[k]
            w.writerow([k, vis, f"{x:.1f}", f"{y:.1f}"])
    tmp.replace(path)


def load_suggestions(cache: Path | None, start: int) -> dict[int, tuple[float, float]]:
    """Padex's raw ball detections, re-indexed to the clip (frame - start)."""
    if cache is None:
        return {}
    with open(cache, "rb") as f:  # your own Padex cache
        tracking = pickle.load(f)
    return {
        bf.frame_id - start: ((bf.bbox.x1 + bf.bbox.x2) / 2, (bf.bbox.y1 + bf.bbox.y2) / 2)
        for bf in tracking.ball_frames if bf.bbox is not None
    }


def extract_frames(video: Path, start: int, end: int, out: Path) -> int:
    """Copy frames [start, end) of the video into out/ as JPEGs; returns count."""
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.jpg"))
    if len(existing) == end - start:
        return len(existing)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    n = 0
    for i in range(end - start):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out / f"{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    if n == 0:
        raise SystemExit("No frames read; check --start/--end against the video length.")
    return n


class Labeler:
    def __init__(self, frames_dir: Path, labels_path: Path, suggestions: dict[int, tuple[float, float]]):
        self.files = sorted(frames_dir.glob("*.jpg"))
        self.labels_path = labels_path
        self.labels = load_labels(labels_path)
        self.suggest = suggestions
        self.i = next((k for k in range(len(self.files)) if k not in self.labels), len(self.files) - 1)
        self.mouse = (0, 0)
        self.click: tuple[int, int] | None = None
        self.scale = 1.0
        self.frame: np.ndarray | None = None
        self.changes = 0

    # -- events --------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param) -> None:
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def set_label(self, vis: int, x: float = -1.0, y: float = -1.0) -> None:
        self.labels[self.i] = (vis, x, y)
        self.changes += 1
        if self.changes % 25 == 0:
            save_labels(self.labels_path, self.labels)

    def move(self, d: int) -> None:
        self.i = int(np.clip(self.i + d, 0, len(self.files) - 1))

    # -- drawing -------------------------------------------------------
    def load(self) -> None:
        self.frame = cv2.imread(str(self.files[self.i]))
        h, w = self.frame.shape[:2]
        self.scale = min(VIEW_W / w, VIEW_H / h, 1.0)

    def render(self) -> np.ndarray:
        f = self.frame
        view = cv2.resize(f, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA) if self.scale < 1 else f.copy()
        sc = self.scale
        # trail of previous labels
        for k in range(max(0, self.i - 5), self.i):
            lab = self.labels.get(k)
            if lab and lab[0]:
                cv2.circle(view, (int(lab[1] * sc), int(lab[2] * sc)), 3, (255, 140, 40), -1)
        sug = self.suggest.get(self.i)
        if sug:
            cv2.circle(view, (int(sug[0] * sc), int(sug[1] * sc)), 10, (0, 150, 255), 1, cv2.LINE_AA)
        lab = self.labels.get(self.i)
        if lab and lab[0]:
            cv2.drawMarker(view, (int(lab[1] * sc), int(lab[2] * sc)), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
        status = "no ball" if lab and not lab[0] else ("ball" if lab else "unlabelled")
        done = len(self.labels)
        txt = (f"frame {self.i + 1}/{len(self.files)}  [{status}]  labelled {done}  |  click=ball  Enter=accept "
               f"suggestion  space=no ball  a/d=prev/next  w/s=+-10  z=clear  q=quit")
        cv2.rectangle(view, (0, 0), (view.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(view, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        # magnifier from the full-resolution frame
        mx, my = int(self.mouse[0] / sc), int(self.mouse[1] / sc)
        H, W = f.shape[:2]
        x0, y0 = int(np.clip(mx - MAG_HALF, 0, W - 2 * MAG_HALF)), int(np.clip(my - MAG_HALF, 0, H - 2 * MAG_HALF))
        patch = cv2.resize(f[y0:y0 + 2 * MAG_HALF, x0:x0 + 2 * MAG_HALF], None, fx=MAG, fy=MAG,
                           interpolation=cv2.INTER_NEAREST)
        c = (mx - x0) * MAG, (my - y0) * MAG
        cv2.drawMarker(patch, (int(c[0]), int(c[1])), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        ph, pw = patch.shape[:2]
        if ph < view.shape[0] - 30 and pw < view.shape[1]:
            view[30:30 + ph, view.shape[1] - pw:] = patch
            cv2.rectangle(view, (view.shape[1] - pw, 30), (view.shape[1] - 1, 30 + ph), (255, 255, 255), 1)
        return view

    # -- loop ------------------------------------------------------------
    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.load()
        shown = self.i
        while True:
            if shown != self.i:
                self.load()
                shown = self.i
            cv2.imshow(WIN, self.render())
            key = cv2.waitKey(20) & 0xFF
            if self.click is not None:
                x, y = self.click[0] / self.scale, self.click[1] / self.scale
                self.click = None
                self.set_label(1, x, y)
                self.move(1)
            elif key in (ord("q"), 27):
                break
            elif key == 13:  # Enter
                sug = self.suggest.get(self.i)
                if sug:
                    self.set_label(1, *sug)
                    self.move(1)
            elif key in (ord(" "), ord("n")):
                self.set_label(0)
                self.move(1)
            elif key == ord("d"):
                self.move(1)
            elif key == ord("a"):
                self.move(-1)
            elif key == ord("w"):
                self.move(10)
            elif key == ord("s"):
                self.move(-10)
            elif key == ord("z"):
                self.labels.pop(self.i, None)
                self.changes += 1
        cv2.destroyAllWindows()
        save_labels(self.labels_path, self.labels)
        vis = sum(1 for v in self.labels.values() if v[0])
        print(f"Saved {len(self.labels)} labelled frames ({vis} with ball) to {self.labels_path}")
        missing = len(self.files) - len(self.labels)
        if missing:
            print(f"{missing} frames still unlabelled; training skips them (and the frames around them). "
                  f"Run the same command again to continue.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--start", type=int, required=True, help="first video frame of the clip")
    ap.add_argument("--end", type=int, required=True, help="frame after the last one (exclusive)")
    ap.add_argument("--name", required=True, help="clip name, e.g. rally_01")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--suggest", type=Path, help="Padex tracking cache to pre-fill ball positions")
    args = ap.parse_args()
    if args.end <= args.start:
        raise SystemExit("--end must be greater than --start")

    clip = args.data / args.name
    n = extract_frames(args.video, args.start, args.end, clip / "frames")
    print(f"{clip}: {n} frames")
    Labeler(clip / "frames", clip / "labels.csv", load_suggestions(args.suggest, args.start)).run()


if __name__ == "__main__":
    main()
