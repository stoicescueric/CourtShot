"""
Shot-level match statistics: what kind of shot each hit was, how fast, how
high, where it went and at whom, and how each rally ended.

Works on what the earlier steps already produce:

    identities     who is where, their boxes in the image and their pose keypoints
    ball_events    hits, bounces, shots, rallies (2D)
    ball_flight    the fitted 3D flight after each hit (optional)

Shot types are decided by rules on a handful of measured features, not by a
trained model:

    serve              first hit of a rally, from the service line, below the head
    smash              an overhead, fast
    bandeja / vibora   an overhead, not fast (the two differ in spin and swing
                       path, which a single camera's ball track doesn't show,
                       so they are one class here)
    bajada             off the back wall, contact high, hit hard from the back
    lob                the ball goes high (apex or time in the air), or the
                       opponents answer it with an overhead a second or more
                       later
    volley             taken before the bounce, near the net
    chiquita           soft and short from the back or the transition zone
    drive              any other shot played after the bounce
    other              not enough seen to tell

When the bounce before a hit was not seen either way, where the hitter stands
decides: close to the net it is counted as a volley, at the back as a drive,
in between as "other".

Every threshold is in ShotRules. The features and their sources:

    overhead         the hitter's pose around the hit (swing.py): an arm
                     above the shoulder is an overhead, a body seen throughout
                     with no arm up is not. Only when the pose can't say
                     (no keypoints for that player then) does the ball decide:
                     at or above the top of the hitter's box.
    contact height   where the ball is within the hitter's box in the image
                     (feet = 0, top of the head = 1) x the player's height.
                     Ball and hitter are at the same distance from the camera
                     at contact, so this needs no 3D fit. The hit is often
                     found a few frames late, with the ball already elsewhere
                     in the box, so the pose's overhead / not overhead bounds it.
    speed            launch speed of the fitted 3D flight when the fit is good,
                     else the average horizontal speed from hitter to landing
    apex             highest point of the fitted 3D flight (good fits only)
    before the hit   floor / wall bounces the 2D step found since the last hit

Each shot also says how much of that was seen (confidence): "high" when the
pose settled overhead or not, the bounce before the hit was seen and the 3D
flight fits; one missing is "medium"; less, or type "other", is "low".

Units: metres, seconds, km/h. Court: x 0..10 across, y 0..20 along, net at y = 10.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from courtshot.analysis.swing import PoseTracks

NET_Y = 10.0
SERVICE_LINE = 6.95  # metres from the net
PLAYER_HEIGHT = 1.75  # metres; turns "height within the box" into metres

SHOT_TYPES = ("serve", "drive", "lob", "volley", "bandeja_vibora", "smash", "bajada", "chiquita", "other")
SHOT_LABELS = {
    "serve": "Serve",
    "drive": "Drive",
    "lob": "Lob",
    "volley": "Volley",
    "bandeja_vibora": "Bandeja / víbora",
    "smash": "Smash",
    "bajada": "Bajada",
    "chiquita": "Chiquita",
    "other": "Other",
}
ATTACKING = {"volley", "bandeja_vibora", "smash", "bajada"}


@dataclass
class ShotRules:
    serve_depth: tuple[float, float] = (6.0, 8.5)  # server's distance from the net
    serve_max_height: float = 0.8  # contact below this share of the hitter's box
    serve_min_time: float = 1.5  # a first hit sooner after the video starts is mid-rally
    overhead_height: float = 1.0  # no pose: contact at or above the top of the hitter's box
    not_overhead_height: float = 0.95  # pose says not an overhead: contact no higher than this
    smash_speed: float = 85.0  # km/h, overheads at least this fast
    bajada_height: float = 0.75
    bajada_depth: float = 5.0
    bajada_speed: float = 50.0
    lob_apex: float = 4.0  # metres
    lob_air_time: float = 1.7  # seconds from hit to the first floor bounce
    lob_answer_time: float = 1.0  # an overhead answer sooner than this was not to a lob
    volley_depth: float = 6.5  # volleys are played inside this distance from the net
    unseen_volley_depth: float = 4.5  # bounce before the hit unknown: closer than this = volley...
    unseen_drive_depth: float = 5.5  # ...farther than this = drive
    chiquita_depth: float = 4.0  # played from at least this far back...
    chiquita_speed: float = 50.0  # ...slower than this...
    chiquita_landing: float = 5.0  # ...landing within this distance of the net
    middle_half_width: float = 1.25  # landing this close to the centre line = down the middle
    net_zone: float = 4.0  # contact closer to the net than this = at the net
    back_zone: float = 7.0  # farther than this = at the back


@dataclass
class ShotInfo:
    index: int
    time_s: float
    rally: int
    player: str
    team: str
    type: str
    speed_kmh: float | None
    speed_source: str | None  # "3d" (launch speed) or "2d" (average to landing)
    contact_height_m: float | None
    contact: tuple[float, float] | None  # hitter's floor position
    depth_m: float | None  # hitter's distance from the net
    zone: str | None  # "net", "middle", "back"
    landing: tuple[float, float] | None
    landing_source: str | None  # "2d" (bounce seen) or "3d" (fitted flight)
    apex_m: float | None
    air_time_s: float | None
    before: str | None  # "serve", "volley", "bounce", "wall" or None
    direction: str | None  # "cross", "line", "middle"
    target: str | None  # the opponent who played the ball next
    wall_after: bool
    attacking: bool
    overhead: bool = False  # an arm above the shoulder (pose), else the ball above the head
    confidence: str = "low"  # "high", "medium", "low": how much of the shot was seen


def _depth(pos) -> float | None:
    return None if pos is None else abs(pos[1] - NET_Y)


def classify(f: dict, r: ShotRules) -> str:
    """Shot type from the features of one hit (see the module docstring)."""
    h, speed, depth = f.get("height_rel"), f.get("speed_kmh"), f.get("depth")
    before = f.get("before")
    overhead, apex = f.get("overhead"), f.get("apex")
    if overhead is None:
        overhead = h is not None and h >= r.overhead_height
    if overhead and not f.get("overhead_by_pose") and apex is not None and apex >= r.lob_apex:
        # Nothing hit from above the head climbs like a lob. With no pose to
        # say otherwise, this was a low ball found late, already on its way up.
        overhead = False
    if (
        f.get("first_in_rally")
        and depth is not None
        and r.serve_depth[0] <= depth <= r.serve_depth[1]
        and (h is None or h < r.serve_max_height)
        and f.get("time_s", 0.0) >= r.serve_min_time
        and before in (None, "serve")
    ):
        return "serve"
    if (
        before == "wall"
        and h is not None
        and h >= r.bajada_height
        and depth is not None
        and depth >= r.bajada_depth
        and speed is not None
        and speed >= r.bajada_speed
    ):
        return "bajada"
    if overhead:
        if speed is not None and speed >= r.smash_speed:
            return "smash"
        return "bandeja_vibora"
    air = f.get("air_time")
    if (
        (apex is not None and apex >= r.lob_apex)
        or (air is not None and air >= r.lob_air_time)
        or f.get("answered_overhead")
    ):
        return "lob"
    if before == "volley" and (depth is None or depth <= r.volley_depth):
        return "volley"
    if before in ("bounce", "wall"):
        land = f.get("landing_depth")
        if (
            depth is not None
            and depth >= r.chiquita_depth
            and speed is not None
            and speed < r.chiquita_speed
            and land is not None
            and land <= r.chiquita_landing
        ):
            return "chiquita"
        return "drive"
    if depth is not None and before is None:
        if depth <= r.unseen_volley_depth:
            return "volley"
        if depth >= r.unseen_drive_depth:
            return "drive"
    return "other"


def _confidence(kind: str, before, by_pose: bool, good3d: bool) -> str:
    """How much of a shot was seen: the pose settled overhead or not, the
    bounce before the hit was seen, the 3D flight fits."""
    if kind == "other":
        return "low"
    return ("low", "low", "medium", "high")[(before is not None) + by_pose + good3d]


def _direction(contact, landing, r: ShotRules) -> str | None:
    """Cross-court, down the line, or down the middle, from where the hitter
    stood and where the ball went."""
    if contact is None or landing is None:
        return None
    if abs(landing[0] - 5.0) <= r.middle_half_width:
        return "middle"
    return "line" if (contact[0] < 5.0) == (landing[0] < 5.0) else "cross"


def build_shots(
    ball,
    stable,
    teams: dict[str, str],
    ball3d=None,
    fps: float = 25.0,
    rules: ShotRules | None = None,
    poses: PoseTracks | None = None,
) -> list[ShotInfo]:
    """One ShotInfo per shot of the 2D ball result.

    ball:   ball_events.BallResult
    stable: tracking with stable player IDs (for the hitters' boxes and poses)
    ball3d: ball_flight.Ball3DResult or None
    poses:  the players' keypoints, if they are already at hand (otherwise
            they are read from ``stable``)
    """
    r = rules or ShotRules()
    boxes = {(pf.frame_id, pf.player_id): pf.bbox for pf in stable.player_frames}
    positions = {
        (pf.frame_id, pf.player_id): (pf.position.x, pf.position.y)
        for pf in stable.player_frames
        if pf.position is not None
    }
    hits = [e for e in ball.events if e.kind == "hit"]
    flights = sorted(ball3d.flights, key=lambda fl: fl.hit_time_s) if ball3d is not None else []
    others = [e for e in ball.events if e.kind != "hit"]

    def flight_for(t: float, player: str):
        best = None
        for fl in flights:
            near = fl.hitter == player and abs(fl.hit_time_s - t) <= 0.06
            if near and (best is None or abs(fl.hit_time_s - t) < abs(best.hit_time_s - t)):
                best = fl
        return best

    def box_near(frame: int, player: str):
        for d in (0, -1, 1, -2, 2):
            b = boxes.get((frame + d, player))
            if b is not None:
                return b
        return None

    def height_in_box(hit, player: str) -> float | None:
        """Where the ball is within the hitter's box: 0 = feet, 1 = top of the head."""
        b = box_near(int(round(hit.frame)), player)
        if b is None or b.y2 <= b.y1:
            return None
        return float((b.y2 - hit.px[1]) / (b.y2 - b.y1))

    heights = [height_in_box(hit, sh.player) for sh, hit in zip(ball.shots, hits)]
    pose = poses or PoseTracks(stable)
    swings = [pose.swing(sh.player, hit.frame, fps) for sh, hit in zip(ball.shots, hits)]
    # Overhead or not: the body's word where it has one, else the ball's place in the box.
    by_pose = [sw is not None and sw.overhead is not None for sw in swings]
    overheads = [
        bool(sw.overhead) if seen else (h is not None and h >= r.overhead_height)
        for sw, seen, h in zip(swings, by_pose, heights)
    ]
    fits = [flight_for(sh.time_s, sh.player) for sh in ball.shots]
    for i, (h, fl) in enumerate(zip(heights, fits)):
        if overheads[i] and not by_pose[i] and fl is not None and fl.ok and fl.max_height >= r.lob_apex:
            # Nothing hit from above the head climbs like a lob. With no pose
            # to say otherwise, this was a low ball found late, on its way up.
            overheads[i] = False
        if h is not None and by_pose[i]:
            heights[i] = max(h, r.overhead_height) if overheads[i] else min(h, r.not_overhead_height)

    out: list[ShotInfo] = []
    for i, (sh, _hit) in enumerate(zip(ball.shots, hits)):
        prev = ball.shots[i - 1] if i > 0 and ball.shots[i - 1].rally == sh.rally else None
        nxt = ball.shots[i + 1] if i + 1 < len(ball.shots) and ball.shots[i + 1].rally == sh.rally else None
        height_rel = heights[i]
        answered_overhead = (
            nxt is not None
            and teams.get(nxt.player) != teams.get(sh.player)
            and overheads[i + 1]
            and nxt.time_s - sh.time_s >= r.lob_answer_time
        )

        # what the ball did between the previous hit and this one
        if prev is None:
            before = "serve"
        else:
            kinds = [e.kind for e in others if prev.time_s < e.time_s < sh.time_s]
            if "ground" in kinds:
                before = "wall" if "wall" in kinds[kinds.index("ground") :] else "bounce"
            elif prev.volleyed_by_next is True:
                before = "volley"
            else:
                before = None

        fl = fits[i]
        good3d = fl is not None and fl.ok
        speed, speed_src = None, None
        if good3d:
            speed, speed_src = fl.launch_speed * 3.6, "3d"
        elif sh.speed_ms is not None:
            speed, speed_src = sh.speed_ms * 3.6, "2d"
        apex = float(fl.max_height) if good3d else None

        landing, landing_src = (tuple(sh.landing), "2d") if sh.landing else (None, None)
        if landing is None and good3d and fl.landing is not None:
            p = fl.landing[2]
            landing, landing_src = (float(p[0]), float(p[1])), "3d"
        contact = tuple(sh.hitter_pos) if sh.hitter_pos else None
        depth = _depth(contact)
        # a landing on the hitter's own half is not where the shot was aimed
        if landing is not None and contact is not None and (landing[1] < NET_Y) == (contact[1] < NET_Y):
            landing, landing_src = None, None

        target = nxt.player if nxt is not None and teams.get(nxt.player) != teams.get(sh.player) else None
        aim = landing
        if aim is None and target is not None:
            aim = positions.get((int(round(nxt.time_s * fps)), target)) or (
                tuple(nxt.hitter_pos) if nxt.hitter_pos else None
            )

        kind = classify(
            {
                "first_in_rally": prev is None,
                "time_s": sh.time_s,
                "height_rel": height_rel,
                "overhead": overheads[i],
                "overhead_by_pose": by_pose[i],
                "speed_kmh": speed,
                "depth": depth,
                "before": before,
                "apex": apex,
                "air_time": sh.flight_s,
                "landing_depth": _depth(landing),
                "answered_overhead": answered_overhead,
            },
            r,
        )
        if kind != "serve" and before == "serve":
            before = None  # first hit seen, but not a serve: the rally was already going
        zone = None if depth is None else ("net" if depth < r.net_zone else "back" if depth > r.back_zone else "middle")
        out.append(
            ShotInfo(
                index=i,
                time_s=float(sh.time_s),
                rally=int(sh.rally),
                player=sh.player,
                team=teams.get(sh.player, "?"),
                type=kind,
                speed_kmh=None if speed is None else round(float(speed), 1),
                speed_source=speed_src,
                contact_height_m=None
                if height_rel is None
                else round(float(np.clip(height_rel, 0.0, 1.9)) * PLAYER_HEIGHT, 2),
                contact=contact,
                depth_m=None if depth is None else round(depth, 2),
                zone=zone,
                landing=landing,
                landing_source=landing_src,
                apex_m=None if apex is None else round(apex, 2),
                air_time_s=None if sh.flight_s is None else round(float(sh.flight_s), 2),
                before=before,
                direction=_direction(contact, aim, r),
                target=target,
                wall_after=bool(sh.wall_after),
                attacking=kind in ATTACKING,
                overhead=overheads[i],
                confidence=_confidence(kind, before, by_pose[i], good3d),
            )
        )
    return out


# ---------------------------------------------------------------------------
# how rallies ended
# ---------------------------------------------------------------------------


def rally_endings(
    shots: list[ShotInfo], rallies, ball3d=None, video_end_s: float | None = None, cut_margin_s: float = 1.5
) -> list[dict]:
    """For each rally: who hit last, and a guess at how the point ended.

    The guess reads the fitted 3D flight of the last shot:
        winner   it bounced on the opponents' floor and was not played back
        error    it never reached the opponents' floor: stayed on the hitter's
                 side or in the net, or hit the opponents' wall or fence first
    "cut" means the video ended (or began) mid-rally; "unknown" that the
    last flight was not seen well enough to say.
    """
    flights = list(ball3d.flights) if ball3d is not None else []
    out = []
    for ri, (t0, t1, _n_hits) in enumerate(rallies):
        mine = [s for s in shots if s.rally == ri]
        last = mine[-1] if mine else None
        end = {
            "rally": ri,
            "start_s": float(t0),
            "end_s": float(t1),
            "duration_s": float(t1 - t0),
            "shots": len(mine),
            "last_player": last.player if last else None,
            "last_type": last.type if last else None,
            "outcome": "unknown",
        }
        if last is None:
            out.append(end)
            continue
        if video_end_s is not None and video_end_s - t1 < cut_margin_s:
            end["outcome"] = "cut"
            out.append(end)
            continue
        fl = next((f for f in flights if f.hitter == last.player and abs(f.hit_time_s - last.time_s) <= 0.06), None)
        if fl is not None and fl.ok and last.contact is not None:
            own_near = last.contact[1] < NET_Y
            first = fl.bounces[0] if fl.bounces else None
            if first is not None:
                kind, _, xyz = first
                on_their_side = (xyz[1] >= NET_Y) == own_near
                end["outcome"] = "winner" if (kind == "ground" and on_their_side) else "error"
        out.append(end)
    return out


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 2) if xs else None


def summarise(shots: list[ShotInfo], endings: list[dict] | None = None) -> dict:
    """Totals for one player's or one team's shots."""
    n = len(shots)
    types = Counter(s.type for s in shots)
    dirs = Counter(s.direction for s in shots if s.direction)
    zones = Counter(s.zone for s in shots if s.zone)
    speeds = [s.speed_kmh for s in shots if s.speed_kmh is not None]
    landed = [s for s in shots if s.landing is not None and s.contact is not None]
    deep = [abs(s.landing[1] - NET_Y) > SERVICE_LINE for s in landed]
    players = {s.player for s in shots}
    won = sum(1 for e in endings or [] if e["outcome"] == "winner" and e["last_player"] in players)
    lost = sum(1 for e in endings or [] if e["outcome"] == "error" and e["last_player"] in players)
    return {
        "shots": n,
        "by_type": {t: types.get(t, 0) for t in SHOT_TYPES},
        "attacking_share": round(sum(s.attacking for s in shots) / n, 3) if n else None,
        "avg_speed_kmh": _mean(speeds),
        "max_speed_kmh": round(max(speeds), 1) if speeds else None,
        "avg_contact_height_m": _mean(s.contact_height_m for s in shots),
        "direction": {d: dirs.get(d, 0) for d in ("cross", "middle", "line")},
        "from_zone": {z: zones.get(z, 0) for z in ("net", "middle", "back")},
        "deep_share": round(float(np.mean(deep)), 3) if deep else None,
        "targets": dict(Counter(s.target for s in shots if s.target)),
        "winners": won,
        "errors": lost,
    }


def match_stats(
    ball,
    stable,
    teams: dict[str, str],
    ball3d=None,
    fps: float = 25.0,
    video_end_s: float | None = None,
    rules: ShotRules | None = None,
    poses: PoseTracks | None = None,
) -> dict:
    """Everything in one go: shots, rally endings, per-player and per-team totals."""
    shots = build_shots(ball, stable, teams, ball3d, fps, rules, poses)
    endings = rally_endings(shots, ball.rallies, ball3d, video_end_s)
    players = sorted(teams)
    team_names = sorted(set(teams.values()))
    lengths = [e["shots"] for e in endings if e["shots"] > 0]
    return {
        "shots": [asdict(s) for s in shots],
        "rallies": endings,
        "players": {p: summarise([s for s in shots if s.player == p], endings) for p in players},
        "teams": {t: summarise([s for s in shots if s.team == t], endings) for t in team_names},
        "match": {
            "rallies": len(endings),
            "shots": len(shots),
            "avg_rally_shots": _mean(lengths),
            "avg_rally_s": _mean(e["duration_s"] for e in endings),
            "longest_rally_shots": max(lengths) if lengths else None,
            "longest_rally_s": round(max((e["duration_s"] for e in endings), default=0.0), 1) if endings else None,
            "time_in_play_s": round(sum(e["duration_s"] for e in endings), 1),
        },
    }
