"""
Command line for Padex.

    padex calibrate VIDEO             click the court keypoints (once per camera position)
    padex run VIDEO                   everything: tracking, tracked video, report, 3D replay
    padex report CACHE                the match report again, from a tracking cache
    padex replay CACHE                the 3D replay again, from a tracking cache

`report` and `replay` read <video>_tracking_cache.pkl, which `run` leaves next
to the video; they need no GPU and do not touch the video.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _existing(path: Path, what: str) -> Path:
    if not path.exists():
        sys.exit(f"{what} not found: {path}")
    return path


def cmd_calibrate(args: argparse.Namespace) -> None:
    from padex.calibration import interactive_calibrate

    if interactive_calibrate(video_path=_existing(args.video, "Video"), save_path=args.output) is None:
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    from padex.pipeline import Padex

    try:
        padex = Padex(
            _existing(args.video, "Video"),
            calibration=args.calibration,
            ball_model_path=args.ball_model,
            cache_tracking=not args.no_cache,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))
    written = padex.run(
        out_dir=args.out,
        video=not args.no_video,
        fit_3d=not args.no_3d,
        redo_video=args.redo_video,
    )
    for what, path in written.items():
        print(f"{what + ':':8s}{path}")


def _analyse(args: argparse.Namespace, **kwargs):
    from padex.analysis.match import analyse_cache

    return analyse_cache(_existing(args.cache, "Tracking cache (made by `padex run`)"), **kwargs)


def cmd_report(args: argparse.Namespace) -> None:
    from padex.analysis.identities import StableIdConfig
    from padex.report.match_report import print_summary, write_match_report

    analysis = _analyse(
        args,
        id_config=StableIdConfig(max_gap_fill=args.gap_fill),
        use_ball=not args.no_ball,
        fit_3d=not args.no_3d,
    )
    try:
        result = write_match_report(analysis, args.out)
    except ValueError as exc:
        sys.exit(str(exc))
    print_summary(result)
    print(f"\nReport: {args.out / 'report.html'}")


def cmd_replay(args: argparse.Namespace) -> None:
    from padex.report.scene3d import write_scene

    analysis = _analyse(args, refit=args.refit, fps=args.fps)
    # the match video shown next to the 3D: <name>.mp4 next to <name>_tracking_cache.pkl
    video = args.video or args.cache.with_name(args.cache.name.replace("_tracking_cache.pkl", ".mp4"))
    try:
        out = write_scene(analysis, args.out, video=video, fragment=args.fragment, also_json=args.json)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"wrote {out} - open it in a browser")


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="padex", description="Padex: padel video analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="Interactive court calibration")
    p.add_argument("video", type=Path)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="where to save it (default: <video>_calibration.json next to the video)",
    )
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("run", help="Tracking, tracked video, match report and 3D replay for a video")
    p.add_argument("video", type=Path)
    p.add_argument(
        "-c",
        "--calibration",
        type=Path,
        default=None,
        help="court calibration JSON (default: <video>_calibration.json next to the video)",
    )
    p.add_argument(
        "--ball-model", type=Path, default=None, help="TrackNetV3 weights (default: checkpoints/padel_tracknetv3.pt)"
    )
    p.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: output/<video name>/)")
    p.add_argument("--no-cache", action="store_true", help="neither reuse nor save the tracking")
    p.add_argument("--no-video", action="store_true", help="skip the tracked video")
    p.add_argument("--redo-video", action="store_true", help="write the tracked video even if it is up to date")
    p.add_argument(
        "--no-3d",
        action="store_true",
        help="skip the 3D ball fit (faster; no ball speed, rougher shot data, no 3D replay)",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="Match report from a tracking cache")
    p.add_argument("cache", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("output/analysis"))
    p.add_argument(
        "--gap-fill", type=float, default=1.5, help="longest gap in a player's track to interpolate, seconds"
    )
    p.add_argument("--no-ball", action="store_true", help="skip everything that needs the ball")
    p.add_argument("--no-3d", action="store_true", help="skip the 3D ball fit (faster; rougher shot data)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("replay", help="3D replay from a tracking cache")
    p.add_argument("cache", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("output/scene3d.html"))
    p.add_argument(
        "--video",
        type=Path,
        default=None,
        help="match video shown next to the 3D (default: <name>.mp4 next to the cache)",
    )
    p.add_argument("--fps", type=float, default=None, help="video frame rate (default: from the cache)")
    p.add_argument("--refit", action="store_true", help="redo the 3D ball fit even if a saved one exists")
    p.add_argument("--json", action="store_true", help="also write the scene data as .json")
    p.add_argument("--fragment", action="store_true", help="write the page without <html>/<head>")
    p.set_defaults(func=cmd_replay)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
