"""
CourtShot: padel match video -> player tracks, ball flights, shots and a match report.

    from courtshot import CourtShot

    CourtShot("videos/match.mp4").run()        # or, on the command line: courtshot run videos/match.mp4

See courtshot.pipeline for the steps, courtshot.cli for the commands.
"""

__version__ = "0.1.0"

from courtshot.calibration import interactive_calibrate
from courtshot.pipeline import CourtShot

__all__ = ["CourtShot", "interactive_calibrate"]
