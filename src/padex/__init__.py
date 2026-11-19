"""
Padex: padel match video -> player tracks, ball flights, shots and a match report.

    from padex import Padex

    Padex("videos/match.mp4").run()        # or, on the command line: padex run videos/match.mp4

See padex.pipeline for the steps, padex.cli for the commands.
"""

__version__ = "0.1.0"

from padex.calibration import interactive_calibrate
from padex.pipeline import Padex

__all__ = ["Padex", "interactive_calibrate"]
