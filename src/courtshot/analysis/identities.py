"""
Stable player IDs for CourtShot.

ByteTrack IDs break whenever a player is occluded for a while or two players
cross. Padel has structure ByteTrack doesn't know about: there are exactly
four players, two per side of the net, and they move smoothly. This module
uses that to map the tracker's IDs onto four fixed identities, P_1..P_4,
with teams T_1 (P_1, P_2) and T_2 (P_3, P_4).

How it works, per frame:
  1. Predict where each of the 4 players should be (last position + velocity).
  2. Build a cost matrix between detections and players:
       distance to the prediction
       - a bonus if the detection carries the tracker ID that player had last
         (full when two players stand or move together, where only
         appearance can tell them apart; zero when they pass each other
         quickly, where ByteTrack tends to swap and motion is the better judge)
       + a penalty if the detection is on the other side of the net
     Detections farther than the player could have run since it was last
     seen are ruled out (gate grows with time lost).
  3. Solve the assignment with the Hungarian algorithm. Extra detections
     (umpire, ball kid, reflections) are dropped.

Identities are seeded from a frame where two players stand on each side of
the net, then propagated forwards and backwards from that frame, so the
start of the video is covered too. Teams stay attached to the people, so
after an end change T_1 is simply on the other half.

    stable, report = stabilize_tracking(raw_tracking, appearance=appearance)

Position and motion cannot tell who is who once two players have traded
places unseen, so afterwards the result is checked against what the players
look like (courtshot.analysis.appearance) and identity swaps between players who look
different are undone. See StablePlayerIds._fix_swaps.

This step is cheap and works on the raw tracking cache, so its settings
(StableIdConfig) can be changed without tracking the video again.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from courtshot.schemas.tracking import PlayerFrame
from courtshot.tracking.pipeline import TrackingResult

logger = logging.getLogger(__name__)

NET_Y = 10.0
SLOT_NAMES = ("P_1", "P_2", "P_3", "P_4")
SLOT_TEAMS = ("T_1", "T_1", "T_2", "T_2")
_INF = 1e6


@dataclass
class StableIdConfig:
    """Tuning knobs. Distances in metres, speeds in m/s, times in seconds."""

    max_speed: float = 8.0  # fastest plausible player movement
    base_gate: float = 0.75  # allowed jump even with dt ~ 0 (detection noise)
    max_gate: float = 6.0  # cap on the gate after long occlusions
    id_bonus: float = 0.4  # trust in ByteTrack ID when motion cannot separate players (keep <= 0.5)
    id_bonus_speed: float = 1.5  # ...fading to zero at this relative speed to the nearest player
    cross_net_penalty: float = 3.0  # cost of matching a player to the other half
    side_switch_time: float = 1.0  # time on the other half before a side change counts
    velocity_window: float = 0.4  # fit velocity over this much recent history
    velocity_timeout: float = 0.5  # stop extrapolating after this long unseen
    init_search_frames: int | None = None  # None = search whole video for a 2+2 frame
    max_gap_fill: float = 1.5  # interpolate missing detections up to this long (0 = off)
    # Appearance check (only with appearance vectors, see courtshot.analysis.appearance): what
    # each track looks like is compared with who the motion step says it is.
    look_window: float = 1.0  # a track's look at a moment = average over +- this long
    look_seed: float = 3.0  # reference looks come from this long around the seed frame
    look_evidence: float = 3.0  # two tracks trade labels once the looks have favoured that by
    # this much in total (cosine-distance gain x seconds)...
    look_drift: float = 0.15  # ...counting only gain above this much per second, so that
    # look-alike partners never add up to a switch
    look_learn: float = 0.03  # how fast the first-pass reference looks follow a player (per sample)
    look_rounds: int = 5  # the labelling must confirm itself within this many passes
    look_gap: float = 0.3  # a track unseen this long "reappears" (a likely switch point)


@dataclass
class _Slot:
    name: str
    team: str
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_t: float = 0.0
    track_id: int | None = None
    side: int = -1  # committed half: -1 near (y < 10), +1 far
    other_side_since: float | None = None
    history: list[tuple[float, np.ndarray]] = field(default_factory=list)

    def predict(self, t: float, cfg: StableIdConfig) -> np.ndarray:
        dt = abs(t - self.last_t)
        if dt > cfg.velocity_timeout:
            return self.pos.copy()
        return self.pos + self.vel * (t - self.last_t)

    def gate(self, t: float, cfg: StableIdConfig) -> float:
        dt = abs(t - self.last_t)
        return min(cfg.max_gate, cfg.base_gate + cfg.max_speed * dt)


def _tracker_id(player_id: str) -> int | None:
    """'P_007' -> 7; untracked detections ('P_det_000') -> None."""
    m = re.fullmatch(r"P_(\d+)", player_id)
    return int(m.group(1)) if m else None


def _side(y: float) -> int:
    """-1 = near half (y < 10), +1 = far half."""
    return -1 if y < NET_Y else 1


class StablePlayerIds:
    """Maps raw ByteTrack player IDs onto 4 persistent identities."""

    def __init__(self, config: StableIdConfig | None = None) -> None:
        self.cfg = config or StableIdConfig()
        self.report: dict[str, object] = {}
        # For tuning the appearance check: (direction, time_s, track a, track b,
        # gain, evidence) at every step of the last apply().
        self.look_trace: list[tuple[int, float, int, int, float, float]] = []

    # -- public -----------------------------------------------------------

    def apply(self, player_frames: list[PlayerFrame], appearance=None) -> list[PlayerFrame]:
        """appearance: an Appearance (courtshot.analysis.appearance) for these frames, or None to
        go by position and motion only."""
        by_frame: dict[int, list[PlayerFrame]] = defaultdict(list)
        for pf in player_frames:
            if pf.position is not None:
                by_frame[pf.frame_id].append(pf)
        frame_ids = sorted(by_frame)
        if not frame_ids:
            logger.warning("No player positions (missing calibration?) — IDs unchanged")
            return player_frames

        init_idx = self._find_init_frame(frame_ids, by_frame)
        if init_idx is None:
            logger.warning("Never saw 4 players at once — IDs unchanged")
            return player_frames

        init_dets = self._pick_four(by_frame[frame_ids[init_idx]])
        assignments: dict[tuple[int, int], int] = {}  # (frame_id, idx in frame) -> slot

        # Forward from the seed frame, then backward to cover the start.
        for order in (frame_ids[init_idx:], frame_ids[init_idx::-1]):
            slots = self._seed_slots(init_dets)
            for fid in order:
                for det_idx, slot_idx in self._assign(by_frame[fid], slots).items():
                    assignments[(fid, det_idx)] = slot_idx

        swaps: list[dict[str, object]] = []
        if appearance is not None:
            swaps = self._fix_swaps(frame_ids, by_frame, assignments, appearance, init_idx)

        out: list[PlayerFrame] = []
        dropped = 0
        ids_per_slot: dict[str, Counter[str]] = {n: Counter() for n in SLOT_NAMES}
        frames_per_slot: Counter[str] = Counter()
        for fid in frame_ids:
            for det_idx, pf in enumerate(by_frame[fid]):
                slot_idx = assignments.get((fid, det_idx))
                if slot_idx is None:
                    dropped += 1
                    continue
                name = SLOT_NAMES[slot_idx]
                ids_per_slot[name][pf.player_id] += 1
                frames_per_slot[name] += 1
                out.append(pf.model_copy(update={"player_id": name, "team_id": SLOT_TEAMS[slot_idx]}))

        filled = self._fill_gaps(out) if self.cfg.max_gap_fill > 0 else []
        out.extend(filled)
        out.sort(key=lambda pf: (pf.frame_id, pf.player_id))

        self.report = {
            "frames": len(frame_ids),
            "interpolated_frames": len(filled),
            "seed_frame": frame_ids[init_idx],
            "dropped_detections": dropped,
            "appearance_used": appearance is not None,
            "swaps_fixed": swaps,
            "frames_per_player": dict(frames_per_slot),
            "tracker_ids_merged": {n: dict(c.most_common()) for n, c in ids_per_slot.items()},
        }
        logger.info(
            "Stable IDs: %d raw tracker IDs -> 4 players, %d extra detections dropped",
            len({pf.player_id for pf in player_frames}),
            dropped,
        )
        return out

    # -- appearance check -------------------------------------------------

    def _fix_swaps(self, frame_ids, by_frame, assignments, appearance, init_idx) -> list[dict[str, object]]:
        """Undo identity swaps the motion step made, using what the players look like.

        The motion step keeps four consistent tracks, but when two players
        cross, or one comes back after being hidden, it can attach a track to
        the wrong person from then on. Here each track's look (averaged over
        a moment) is compared with the four reference looks taken around the
        seed frame. For every pair of tracks, the evidence that they carry
        each other's labels is added up over time; once it is strong enough
        the two labels are traded, from the point where the evidence began:
        more exactly, where one of the two reappeared after being unseen or,
        failing that, where they were closest together.

        A look changes with pose and with where on court the player stands,
        and for players in similar kit that can outweigh who they are for
        many seconds. So a corrected labelling is only kept if it confirms
        itself: the reference looks are rebuilt from the whole video under
        the new labels and the check is run again, until the labelling comes
        out the same twice. A real swap makes the references cleaner and is
        found again; a false one mixes two people into each reference and
        disappears. If no labelling confirms itself, the motion step's
        answer stands.

        Rewrites ``assignments`` in place; returns the switches made.
        """
        cfg = self.cfg
        self.look_trace = []
        n_slots = len(SLOT_NAMES)
        n = len(frame_ids)
        fps = appearance.fps
        step = max(1, int(round(fps / 5)))
        dt = step / fps
        grid = list(range(0, n, step))  # indices into frame_ids
        seed_g = min(init_idx // step, len(grid) - 1)

        vec: dict[tuple[int, int], np.ndarray] = {}  # (index into frame_ids, track) -> look
        pos: dict[tuple[int, int], np.ndarray] = {}
        index = {fid: k for k, fid in enumerate(frame_ids)}
        for (fid, det_idx), slot in assignments.items():
            pf = by_frame[fid][det_idx]
            k = index[fid]
            pos[(k, slot)] = np.array([pf.position.x, pf.position.y])
            if k % step == 0:
                v = appearance.vector(fid, pf.player_id)
                if v is not None:
                    vec[(k, slot)] = v

        def unit(v: np.ndarray) -> np.ndarray:
            return v / max(float(np.linalg.norm(v)), 1e-9)

        def look(g: int, slot: int, half: int, at_least: int) -> np.ndarray | None:
            vs = [
                vec[(grid[j], slot)]
                for j in range(max(0, g - half), min(len(grid), g + half + 1))
                if (grid[j], slot) in vec
            ]
            return unit(np.mean(vs, axis=0)) if len(vs) >= at_least else None

        half = max(1, int(round(cfg.look_window * fps / step)))
        seed_half = max(1, int(round(cfg.look_seed * fps / step)))
        ref = [look(seed_g, slot, seed_half, 3) for slot in range(n_slots)]
        if any(r is None for r in ref):
            logger.warning("Appearance: too few crops around the seed frame; identities by motion only")
            return []

        pairs = [(a, b) for a in range(n_slots) for b in range(a + 1, n_slots)]
        identity = tuple(range(n_slots))
        looks_at = [[look(g, slot, half, 2) for slot in range(n_slots)] for g in range(len(grid))]

        def label_tracks(ref, learn: float):
            """Labels per grid point that fit these reference looks, starting
            from the motion step's labels at the seed and going both ways."""
            label_at: list[tuple[int, ...]] = [identity] * len(grid)
            switches: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []  # (grid point, before, after), by time
            for direction in (1, -1):
                refs = [r.copy() for r in ref]
                current = list(identity)
                evidence = dict.fromkeys(pairs, 0.0)
                since = dict.fromkeys(pairs, seed_g)
                g = seed_g
                while 0 <= g < len(grid):
                    looks = looks_at[g]
                    dist = [None if v is None else [1.0 - float(v @ r) for r in refs] for v in looks]
                    switched = False
                    for a, b in pairs:
                        if dist[a] is None or dist[b] is None:
                            continue
                        ia, ib = current[a], current[b]
                        # how much better the two tracks fit each other's labels than their own
                        gain = (dist[a][ia] + dist[b][ib]) - (dist[a][ib] + dist[b][ia])
                        if evidence[(a, b)] <= 0.0:
                            since[(a, b)] = g
                        evidence[(a, b)] = max(0.0, evidence[(a, b)] + (gain - cfg.look_drift) * dt)
                        self.look_trace.append((direction, g * dt, a, b, gain, evidence[(a, b)]))
                        if evidence[(a, b)] >= cfg.look_evidence:
                            before = tuple(current)
                            current[a], current[b] = ib, ia
                            after = tuple(current)
                            g0 = since[(a, b)]
                            lo, hi = sorted((g0, g))
                            for j in range(lo, hi + 1):
                                label_at[j] = after
                            switches.append((g0, before, after) if direction > 0 else (g0 + 1, after, before))
                            evidence = dict.fromkeys(pairs, 0.0)
                            switched = True
                            break
                    if learn > 0 and not switched and not any(e > 0.0 for e in evidence.values()):
                        # nothing speaks against the labels: let the references follow the players
                        for slot, v in enumerate(looks):
                            if v is not None:
                                i = current[slot]
                                refs[i] = unit(refs[i] * (1 - learn) + v * learn)
                    label_at[g] = tuple(current)
                    g += direction
            return label_at, switches

        def whole_video_refs(label_at):
            """Each player's look over the whole video under these labels."""
            sums = [np.zeros_like(ref[0]) for _ in range(n_slots)]
            for (k, slot), v in vec.items():
                sums[label_at[k // step][slot]] += v
            return [unit(v) for v in sums]

        # First pass: references from around the seed frame, following the
        # players. Then confirm with references from the whole video, also
        # when the first pass found nothing: those references are steadier
        # and show swaps the first pass only nearly had enough evidence for.
        label_at, switches = label_tracks(ref, cfg.look_learn)
        confirmed = False
        for _ in range(cfg.look_rounds):
            if confirmed:
                break
            self.look_trace = []
            again, switches_again = label_tracks(whole_video_refs(label_at), 0.0)
            confirmed = again == label_at
            label_at, switches = again, switches_again
        if not confirmed:
            logger.warning("Appearance: no labelling confirmed itself; identities by motion only")
            return []

        per_frame = [label_at[k // step] for k in range(n)]

        # The evidence comes from looks averaged over a window, so the moment
        # it began is only known to within that window. Put the switch where
        # one of the two tracks reappeared after being unseen or, if both were
        # in view throughout, where they were closest together.
        report: list[dict[str, object]] = []
        reach = 2 * half * step
        gap = max(2, int(round(cfg.look_gap * fps)))
        for g, before, after in sorted(switches):
            nominal = grid[min(g, len(grid) - 1)]
            moved = [slot for slot in range(n_slots) if before[slot] != after[slot]]
            lo, hi = max(0, nominal - reach), min(n - 1, nominal + reach)
            cut = None
            longest = 0
            for slot in moved:
                run = 0
                for k in range(lo, hi + 1):
                    if (k, slot) in pos:
                        if run >= gap and run > longest:
                            cut, longest = k, run
                        run = 0
                    else:
                        run += 1
            if cut is None:
                best_d = np.inf
                for k in range(lo, hi + 1):
                    pa, pb = pos.get((k, moved[0])), pos.get((k, moved[1]))
                    if pa is None or pb is None:
                        continue
                    d = float(np.linalg.norm(pa - pb))
                    if d < best_d:
                        cut, best_d = k, d
            if cut is None:
                cut = nominal
            for k in range(min(cut, nominal), max(cut, nominal)):
                per_frame[k] = after if cut < nominal else before
            report.append(
                {
                    "frame": int(frame_ids[cut]),
                    "time_s": round(float(by_frame[frame_ids[cut]][0].timestamp_ms) / 1000.0, 2),
                    "players": sorted(SLOT_NAMES[after[slot]] for slot in moved),
                }
            )

        for key in list(assignments):
            assignments[key] = per_frame[index[key[0]]][assignments[key]]
        if report:
            logger.info("Appearance: corrected %d identity switch(es): %s", len(report), report)
        return report

    # -- gap filling ------------------------------------------------------

    def _fill_gaps(self, frames: list[PlayerFrame]) -> list[PlayerFrame]:
        """Linearly interpolate short gaps in each player's track.

        Filled frames have confidence 0.0 and no pose keypoints, so they can be
        told apart from real detections.
        """
        by_player: dict[str, list[PlayerFrame]] = defaultdict(list)
        for pf in frames:
            by_player[pf.player_id].append(pf)

        filled: list[PlayerFrame] = []
        for track in by_player.values():
            track.sort(key=lambda pf: pf.frame_id)
            for a, b in zip(track, track[1:]):
                n_missing = b.frame_id - a.frame_id - 1
                dt = (b.timestamp_ms - a.timestamp_ms) / 1000.0
                if n_missing <= 0 or dt > self.cfg.max_gap_fill:
                    continue
                for k in range(1, n_missing + 1):
                    w = k / (n_missing + 1)

                    def lerp(u: float, v: float, w: float = w) -> float:
                        return u + (v - u) * w

                    filled.append(
                        a.model_copy(
                            update={
                                "frame_id": a.frame_id + k,
                                "timestamp_ms": lerp(a.timestamp_ms, b.timestamp_ms),
                                "bbox": a.bbox.model_copy(
                                    update={
                                        "x1": lerp(a.bbox.x1, b.bbox.x1),
                                        "y1": lerp(a.bbox.y1, b.bbox.y1),
                                        "x2": lerp(a.bbox.x2, b.bbox.x2),
                                        "y2": lerp(a.bbox.y2, b.bbox.y2),
                                    }
                                ),
                                "position": a.position.model_copy(
                                    update={
                                        "x": lerp(a.position.x, b.position.x),
                                        "y": lerp(a.position.y, b.position.y),
                                    }
                                ),
                                "confidence": 0.0,
                                "keypoints": [],
                            }
                        )
                    )
        return filled

    # -- seeding ----------------------------------------------------------

    def _find_init_frame(self, frame_ids: list[int], by_frame: dict[int, list[PlayerFrame]]) -> int | None:
        """First frame with 2 players clearly on each side; fallback: any 4."""
        limit = self.cfg.init_search_frames or len(frame_ids)
        fallback = None
        for i, fid in enumerate(frame_ids[:limit]):
            dets = by_frame[fid]
            if len(dets) < 4:
                continue
            if fallback is None:
                fallback = i
            ys = [pf.position.y for pf in self._pick_four(dets)]
            if sum(y < NET_Y - 1.0 for y in ys) == 2 and sum(y > NET_Y + 1.0 for y in ys) == 2:
                return i
        return fallback

    @staticmethod
    def _pick_four(dets: list[PlayerFrame]) -> list[PlayerFrame]:
        return sorted(dets, key=lambda pf: pf.confidence, reverse=True)[:4]

    @staticmethod
    def _seed_slots(dets: list[PlayerFrame]) -> list[_Slot]:
        """Near pair -> P_1/P_2 (T_1), far pair -> P_3/P_4 (T_2), each left to right."""
        by_y = sorted(dets, key=lambda pf: pf.position.y)
        near = sorted(by_y[:2], key=lambda pf: pf.position.x)
        far = sorted(by_y[2:], key=lambda pf: pf.position.x)
        slots = []
        for i, pf in enumerate(near + far):
            slots.append(
                _Slot(
                    name=SLOT_NAMES[i],
                    team=SLOT_TEAMS[i],
                    pos=np.array([pf.position.x, pf.position.y]),
                    last_t=pf.timestamp_ms / 1000.0,
                    track_id=_tracker_id(pf.player_id),
                    side=-1 if i < 2 else 1,
                )
            )
        return slots

    # -- per-frame assignment --------------------------------------------

    def _assign(self, dets: list[PlayerFrame], slots: list[_Slot]) -> dict[int, int]:
        cfg = self.cfg
        t = dets[0].timestamp_ms / 1000.0
        cost = np.full((len(dets), len(slots)), _INF)
        preds = [s.predict(t, cfg) for s in slots]

        # How much to trust ByteTrack's ID for each player. When two players
        # pass each other (high relative speed) their motion separates them
        # and ByteTrack is the one likely to swap IDs. When they stand or move
        # together (low relative speed) motion can't separate them, and
        # ByteTrack's appearance-based ID is the better judge.
        id_weight = []
        for j, s in enumerate(slots):
            k = min((k for k in range(len(slots)) if k != j), key=lambda k: np.linalg.norm(preds[k] - preds[j]))
            rel_speed = float(np.linalg.norm(s.vel - slots[k].vel))
            id_weight.append(max(0.0, 1.0 - rel_speed / cfg.id_bonus_speed))

        for i, pf in enumerate(dets):
            p = np.array([pf.position.x, pf.position.y])
            tid = _tracker_id(pf.player_id)
            for j, s in enumerate(slots):
                dist = float(np.linalg.norm(p - preds[j]))
                if dist > s.gate(t, cfg):
                    continue
                c = dist
                if tid is not None and tid == s.track_id:
                    c -= cfg.id_bonus * id_weight[j]
                if _side(p[1]) != s.side:
                    c += cfg.cross_net_penalty
                cost[i, j] = c

        rows, cols = linear_sum_assignment(cost)
        result: dict[int, int] = {}
        for i, j in zip(rows, cols):
            if cost[i, j] >= _INF:
                continue
            result[i] = j
            p = np.array([dets[i].position.x, dets[i].position.y])
            # Only adopt the tracker's ID from a normal, close match. A match
            # made with a widened gate (player was lost) could be a stranger.
            confident = float(np.linalg.norm(p - preds[j])) <= cfg.base_gate
            self._update(slots[j], dets[i], t, adopt_id=confident)
        return result

    def _update(self, slot: _Slot, pf: PlayerFrame, t: float, adopt_id: bool) -> None:
        p = np.array([pf.position.x, pf.position.y])
        if abs(t - slot.last_t) > self.cfg.velocity_timeout:
            slot.history.clear()  # stale motion, start over
        slot.history.append((t, p))
        # Keep only the recent window; frame-to-frame differences are mostly
        # foot-point jitter, so velocity is a least-squares fit over the window.
        while abs(slot.history[0][0] - t) > self.cfg.velocity_window:
            slot.history.pop(0)
        if len(slot.history) >= 3:
            ts = np.array([h[0] for h in slot.history])
            ps = np.array([h[1] for h in slot.history])
            tc = ts - ts.mean()
            v = (tc[:, None] * (ps - ps.mean(axis=0))).sum(axis=0) / (tc**2).sum()
            speed = float(np.linalg.norm(v))
            if speed > self.cfg.max_speed:
                v *= self.cfg.max_speed / speed
            slot.vel = v
            # Smoothed position at time t from the same fit.
            p = ps.mean(axis=0) + v * (t - ts.mean())
        else:
            slot.vel = np.zeros(2)
        slot.pos = p
        slot.last_t = t
        # Commit a side change (end change) only after a sustained stay.
        if _side(p[1]) == slot.side:
            slot.other_side_since = None
        elif slot.other_side_since is None:
            slot.other_side_since = t
        elif abs(t - slot.other_side_since) >= self.cfg.side_switch_time:
            slot.side = _side(p[1])
            slot.other_side_since = None
        tid = _tracker_id(pf.player_id)
        if adopt_id and tid is not None:
            slot.track_id = tid


def stabilize_tracking(
    tracking: TrackingResult, config: StableIdConfig | None = None, appearance=None
) -> tuple[TrackingResult, dict[str, object]]:
    """Return a copy of a TrackingResult with player IDs mapped to P_1..P_4.

    appearance: optional Appearance (courtshot.analysis.appearance) of these frames; with it,
    identity swaps are corrected by what the players look like.
    """
    mapper = StablePlayerIds(config)
    frames = mapper.apply(tracking.player_frames, appearance)
    return (
        TrackingResult(
            player_frames=frames,
            ball_frames=tracking.ball_frames,
            calibration=tracking.calibration,
        ),
        mapper.report,
    )


def frame_rate(tracking: TrackingResult, default: float = 30.0) -> float:
    """Frames per second of the tracked video: the median over all frames of
    frame-number step / time step (one frame's timestamp can be off)."""
    stamps = {pf.frame_id: pf.timestamp_ms / 1000.0 for pf in tracking.player_frames}
    stamps.update({bf.frame_id: bf.timestamp_ms / 1000.0 for bf in tracking.ball_frames})
    fids = np.array(sorted(stamps))
    if len(fids) < 2:
        return default
    secs = np.array([stamps[f] for f in fids])
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.diff(fids) / np.diff(secs)
    rates = rates[np.isfinite(rates) & (rates > 0)]
    return float(np.median(rates)) if len(rates) else default


def cache_appearance(tracking_cache, video=None, tracking=None):
    """Appearance vectors saved for a tracking cache (made now if a video is
    given and there are none), or None when they are not available. With
    None, stabilize_tracking goes by position and motion only."""
    try:
        from courtshot.analysis.appearance import appearance_for

        return appearance_for(tracking_cache, video=video, tracking=tracking)
    except ImportError as exc:  # the image model's library (transformers) is not installed
        logger.warning("Appearance check off (%s); identities by position and motion only", exc)
        return None
