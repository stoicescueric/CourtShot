"""python -m courtshot.worker run | setup | check"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from courtshot.worker.config import add_arguments, settings_from
from courtshot.worker.host import Check, download_models, run_checks

COMMANDS = {
    "run": "check this Mac, then claim and run jobs until SIGTERM or Ctrl-C",
    "setup": "download the YOLO weights and SigLIP, then check this Mac",
    "check": "check ffmpeg, ffprobe, PyTorch with MPS and the ball model; no downloads",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m courtshot.worker", description="CourtShot Mac worker")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in COMMANDS.items():
        add_arguments(commands.add_parser(name, help=help_text))
    args = parser.parse_args(argv)
    settings = settings_from(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "run" and not (settings.server_url and settings.token):
        parser.error("set COURTSHOT_SERVER_URL and COURTSHOT_WORKER_TOKEN (or --server-url and --token)")
    results = (download_models() if args.command == "setup" else []) + run_checks(settings)
    _print(results)
    if not all(check.ok for check in results):
        print("Not ready: fix the items marked FAIL.", file=sys.stderr)
        return 1
    if args.command != "run":
        return 0

    from courtshot.worker.runner import Runner, WorkRootBusy

    runner = Runner(settings)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: runner.stop())
    print(f"worker {runner.version} polling {settings.server_url}; work folder {settings.work_root}", flush=True)
    try:
        runner.serve()
    except WorkRootBusy as busy:
        print(f"{busy}; stop it first, or give this one its own --work-root", file=sys.stderr)
        return 1
    return 0


def _print(results: list[Check]) -> None:
    for check in results:
        print(f"{'ok  ' if check.ok else 'FAIL'}  {check.name}: {check.detail}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
