# Padex

Padex turns a padel match video from one fixed camera into player tracks, a 3D
ball path, a list of classified shots, a match report and a 3D replay.

> **padex** = padel + x (analytics/extraction)

## Setup

```bash
git clone https://github.com/stoicescueric/padex.git
cd padex
uv sync --group dev
```

On Windows this installs the CUDA build of PyTorch; a GPU makes tracking many
times faster. The YOLO weights are downloaded to `~/.padex/weights/` on first
use. The fine-tuned ball model is in `checkpoints/padel_tracknetv3.pt`.

## Use

Run everything from the repository folder.

```bash
# 1. Once per camera position: click the court keypoints.
#    Saves videos/match_calibration.json next to the video.
uv run python -m padex calibrate videos/match.mp4

# 2. Tracking, tracked video, report and 3D replay.
uv run python -m padex run videos/match.mp4
```

Results go to `output/match/`:

| File | What it is |
| --- | --- |
| `match_tracked.mp4` | The video with players, ball, ball speed, bounces and shot types drawn on it |
| `report.html` | The match report: movement, heatmaps, shots, rallies |
| `scene3d.html` | 3D replay of court, players and ball |
| `shots.csv`, `ball_events.csv`, `positions.csv`, `summary.json` | The data behind the report |

`bash run_match.sh videos/match.mp4` does the same and keeps a log in `logs/`.

Tracking is the slow step. Its result is saved next to the video as
`videos/match_tracking_cache.pkl` and reused on later runs, as long as the video, the
calibration, the ball model and the settings are the same. The report and the
replay can be made again from that cache alone, without the video or a GPU:

```bash
uv run python -m padex report videos/match_tracking_cache.pkl --out output/match
uv run python -m padex replay videos/match_tracking_cache.pkl --out output/match/scene3d.html
```

`uv run python -m padex <command> --help` lists the options of each command.
Where Windows lets it run, the shorter `uv run padex ...` is the same thing.

From Python:

```python
from padex import Padex

p = Padex("videos/match.mp4")
p.run()                    # everything, or:
analysis = p.analyse()     # stable player IDs, ball events, 3D ball flights
```

## How it works

```
video ─▶ tracking ─────────────▶ analysis ─────────────────────▶ outputs
         players: YOLO+ByteTrack  identities: 4 stable players    tracked video
         pose:    YOLO-Pose       ball events: hits and bounces   match report
         ball:    TrackNetV3      ball flight: 3D physics fit     3D replay
                                  shots: type, speed, direction
                                  movement: distance, zones
```

1. **Court calibration** maps image pixels to court metres (a homography from up to 12 keypoints).
2. **Tracking** finds the people, their pose and the ball in every frame.
3. **Identities** folds the tracker's short-lived IDs into four players, P_1 to P_4, by motion, and checks them against what the players look like.
4. **Ball events** cleans the ball track and finds hits, floor bounces and wall bounces where the ball's smooth curve in the image breaks.
5. **Ball flight** fits a physics simulation to every flight between two hits, which gives the ball's height, speed and landing spot from one camera.
6. **Shots** names each hit (serve, lob, volley, bandeja/víbora, smash, bajada, chiquita, drive) with rules on measured features.
7. **Movement** measures distance, speed, sprints, net and back time and team formations.

Each module's docstring explains its step and lists its thresholds.

## Layout

```
src/padex/
  pipeline.py          Padex: track -> analyse -> outputs
  cli.py               the padex command (calibrate, run, report, replay)
  court.py             court model and homography
  calibration.py       interactive court calibration
  weights.py           download of the YOLO weights
  schemas/tracking.py  PlayerFrame, BallFrame, CourtCalibration
  io/video.py          video reader and writer
  tracking/            video -> raw tracking (the GPU part)
    player.py            people, tracker IDs, pose
    ball.py              TrackNetV3 ball detector
    pipeline.py          runs both over a video
    cache.py             the tracking cache and its key
  analysis/            raw tracking -> players, shots, statistics
    identities.py        four stable player IDs
    appearance.py        what each player looks like (SigLIP)
    ball_events.py       hits and bounces in 2D
    ball_flight.py       3D ball flights
    swing.py             what the hitter's body did (pose)
    shots.py             shot types and rally endings
    movement.py          movement and formations
    match.py             all of the above for one tracking cache
  report/              match report and 3D replay (with their HTML templates)
  viz/                 drawing on video frames
videos/                your match videos, each with its calibration JSON and caches
tools/                 labelling and training the ball model, checks
tests/
```

## The ball model

The ball detector is TrackNetV3 fine-tuned on padel footage. To improve it on
your own videos:

```bash
uv run python tools/find_gaps.py videos/match.mp4      # where the ball is missed; prints the labelling commands
uv run python tools/label_ball.py videos/match.mp4 --start 1200 --end 1350 --name match_01 --suggest videos/match_tracking_cache.pkl
bash tools/retrain_own.sh match                 # fine-tune on your clips, writes a new checkpoint
BALL_MODEL=checkpoints/padel_tracknetv3_own.pt bash run_match.sh videos/match.mp4
```

`tools/train_padel.sh` is the first fine-tuning, from the original badminton
weights on PadelTracker100. `tools/check_calibration.py` shows which clicked
keypoint hurts a calibration; `tools/audio_hits.py` finds impact sounds in the
audio track (an experiment, not used by the pipeline).

## Settings

| Environment variable | Effect |
| --- | --- |
| `PADEX_TRACKER` | Player tracker config: `padex_bytetrack.yaml` (default) or `padex_botsort.yaml`, or a file of your own |
| `PADEX_FP16=0` | Full precision on the GPU (default: half precision on CUDA) |
| `PADEX_POSE_CLOSEUPS=1` | Pose from close-ups of the players only: faster, slightly worse for the near players |
| `PADEX_THREADS=0` | Run the models one after another on one thread |
| `PADEX_WORKERS` | Processes for the 3D ball fit (default: half the logical CPUs) |

## Development

```bash
uv run pytest
uv run ruff check src tests tools
uv run ruff format src tests tools
```

The tests need no video, no model weights and no GPU.

## Credits and license

Based on [Padex](https://github.com/rondo-labs/Padex) by Xingnan Zhu, of which
the tracking layer, the calibration tool and the frame drawing remain. The ball
model architecture is [TrackNetV3](https://github.com/qaz812345/TrackNetV3).

GPL v3, see [LICENSE](LICENSE).
