> Extracted from the reference code under `src/courtshot` on 2026-11-20 and checked against the source by a second reader. Line numbers refer to the reference files at commit c2b1767. Nothing in the pipeline was run to produce this.

## A. ReportData contract

Source: `build_report_data` (`src/courtshot/report/match_report.py:98-164`), returned through `clean()` (`:45-55`). The object is the template's only input (`const D`, `report.html:206`).

The only caller is `write_match_report` (`:263-314`). It raises `ValueError` and writes nothing when no player has a stable ID (`movement.py:94-96`). That happens without court calibration and when no frame ever shows four positioned players (`identities.py:143-150`; raw IDs look like `P_007`, `tracking/player.py:434`). The CLI turns it into an exit message (`cli.py:80-83`); a worker must treat it as a failed job.

`clean()` does four things:
- NaN and ±inf become `null`.
- Every float is rounded to 4 decimals, including coordinates, times and `fps`.
- numpy ints and bools become plain JSON values; tuples become arrays.
- Dict keys starting with `_` are dropped. No current key does.

The result is embedded with `json.dumps(separators=(",", ":"), allow_nan=False)` in place of the literal `/*REPORT_DATA*/null`, with `</` escaped to `<\/` (`:172-176`).

```ts
// [unread] = report.html never reads this field.
type PlayerId = "P_1" | "P_2" | "P_3" | "P_4";   // identities.py:56. Fixed at the seed frame: near pair = P_1, P_2; far pair = P_3, P_4;
                                                  // each pair by ascending court x (identities.py:470-473)
type TeamId   = "T_1" | "T_2";                    // identities.py:57. P_1, P_2 = T_1; P_3, P_4 = T_2
type ShotType = "serve" | "drive" | "lob" | "volley" | "bandeja_vibora"
              | "smash" | "bajada" | "chiquita" | "other";   // shots.py:70
type Sec  = number;   // absolute seconds, not relative to t0. See "Time bases" below
type Frac = number;   // share, 0..1

interface ReportData {
  title: string;            // cache file stem with "_tracking_cache" removed (match_report.py:289)
  duration_s: Sec;          // (n - 1) / fps, n = frames from the first to the last frame with a stable-ID player.
                            // 0 for a single frame; the hero's in-play share then prints "–" (0/0) or "Infinity%"
  t0: Sec;                  // time of that first frame (movement base). Every displayed clock is (time - t0)
  fps: number;              // [unread] float: median of frame step / time step, 30.0 with fewer than 2 stamped frames
                            // (identities.py:591-603). analyse_cache(fps=) can override it
  settings: Settings;
  players: PlayerRow[];     // always 4, in PlayerId order
  teams: TeamRow[];         // always 2: [T_1, T_2] (movement.py:22)
  movement_rallies: MovementRally[];   // never null, may be []
  match: MatchTotals | null;           // null iff has_ball = false
  rallies: RallyEnding[] | null;       // null iff has_ball = false. Ball rallies, not movement_rallies. Can be []:
                                       // the template tests truthiness, so [] gives no timeline bands and an empty table
  shots: ShotInfo[];                   // [] when has_ball = false. In time order; shots[i].index = i
  shot_types: ShotType[];              // the 9 types in the order above
  shot_labels: Record<ShotType, string>;   // Serve, Drive, Lob, Volley, "Bandeja / víbora", Smash, Bajada, Chiquita, Other (shots.py:71-81)
  attacking_types: ShotType[];         // ["bajada", "bandeja_vibora", "smash", "volley"] (sorted)
  timeline: Timeline;                  // [unread]
  quality: Quality;
}

interface Settings {               // movement.Settings defaults (movement.py:26-45). write_match_report(settings=) can override; no caller does
  net_zone: number;                // m from the net, 4.0
  back_zone: number;               // m from the net, 7.0
  still_speed: number;             // m/s, 0.5
  sprint_speed: number;            // m/s, 4.0 (the template shows km/h)
  rally_source: "ball" | "movement";  // "ball" iff has_ball, ball_coverage >= 0.4 and at least one ball rally (match_report.py:277-280)
  service_line: number;            // m from the net, 6.95 (shots.py:67)
}

interface PlayerRow {
  id: PlayerId;
  team: TeamId;
  move: PlayerMove;
  shots: ShotSummary | null;       // null iff has_ball = false. With has_ball every player has one, even with 0 shots
  heat: number[][];                // 40 x 40 integer frame counts, 0.25 m cells (match_report.py:38, :102-108).
                                   // heat[row][col]: row = depth bin (0 = at the net, 39 = back wall),
                                   // col = lateral bin (0 = the player's own left when facing the net).
                                   // Own-half view (movement.py:184-195), decided per frame, so it follows a change of ends.
                                   // Counts in-rally frames with a position; positions outside 0..10 m are dropped. Can be all zeros
  heat_seconds: number;            // [unread] s of in-rally time with a position (frames / fps)
}

interface PlayerMove {             // movement.py:260-276. A player who was never seen still has a row
  team: TeamId;                    // [unread] duplicate of PlayerRow.team
  tracked: Frac;                   // frames with a position (detected or gap-filled) / all frames
  interpolated: Frac;              // gap-filled share of the frames with a position
  distance_m: number;              // m. Frames slower than still_speed add nothing
  distance_in_rallies_m: number;   // m
  top_speed_ms: number | null;     // m/s, best 0.25 s average. null if no speed was ever measured
  sprints: number;                 // int. Runs at >= sprint_speed lasting >= 0.3 s
  avg_dist_to_net_m: number | null;   // m, in rallies. null if the player has no in-rally frame
  time_at_net: Frac | null;        // depth < net_zone. null under the same condition
  time_in_middle: Frac | null;     // null under the same condition
  time_at_back: Frac | null;       // depth > back_zone. null under the same condition
  court_coverage: Frac;            // share of the 100 one-metre cells of the own half with >= 0.5 s in rallies
  time_left_side: Frac | null;     // lateral < 5 m in the own-half view. null under the same condition
}

interface TeamRow {
  id: TeamId;
  players: [PlayerId, PlayerId];
  move: TeamMove;
  shots: ShotSummary | null;       // null iff has_ball = false
}

interface TeamMove {               // movement.py:294-306. Shares are over in-rally frames with both partners seen
  players: [PlayerId, PlayerId];   // [unread] duplicate of TeamRow.players
  both_at_net: Frac;               // 0 when there are no such frames (same for the next three)
  both_at_back: Frac;
  split: Frac;                     // [unread] one at the net, one at the back
  in_transition: Frac;             // [unread] everything else
  avg_partner_distance_m: number | null;   // [unread] m. null if no frame has both
  side_by_side: Frac | null;               // [unread] depth gap < 2 m. null if no frame has both
  depth_sync: number | null;               // [unread] Pearson r of the two depths, -1..1. null with <= 10 frames or a constant depth
  net_lost: number;                        // [unread] int. Both-at-net followed by both-at-back within a rally (runs >= 0.5 s)
  net_regained: number;                    // int. The reverse
  median_time_to_regain_s: number | null;  // [unread] s. null if never regained
}

interface MovementRally {          // movement.py:308-310, movement time base
  start_s: Sec;                    // first frame of the rally
  end_s: Sec;                      // last frame, inclusive
  duration_s: number;              // frames / fps = end_s - start_s + 1 / fps
}
// rally_source = "ball": the ball rallies snapped to frames (spans under 2 frames dropped), so they repeat `rallies`.
// rally_source = "movement": found from player speed (at least 3 s long, pauses under 2.5 s merged).

interface ShotSummary {            // shots.py:445-470. Same shape for a player and for a team
  shots: number;                   // int, can be 0
  by_type: Record<ShotType, number>;   // all 9 keys always present
  attacking_share: Frac | null;    // 3 dp. null if shots = 0
  avg_speed_kmh: number | null;    // km/h, 2 dp. Mixes 3D launch speeds and 2D averages. null if no shot has a speed
  max_speed_kmh: number | null;    // km/h, 1 dp. null under the same condition
  avg_contact_height_m: number | null;   // m, 2 dp. null if no shot has a contact height
  direction: { cross: number; middle: number; line: number };   // int counts, all keys always present
  from_zone: { net: number; middle: number; back: number };     // int counts, all keys always present
  deep_share: Frac | null;         // 3 dp. Landings more than service_line from the net, over shots with contact and landing. null if none
  targets: Partial<Record<PlayerId, number>>;   // int counts. Only opponents that were targeted, so it can be {}
  winners: number;                 // int. Rallies with outcome "winner" whose last shot was by this player or team
  errors: number;                  // int. The same for "error"
}

interface ShotInfo {               // shots.py:110-134, filled at :352-380
  index: number;                   // int, 0-based position in shots
  time_s: Sec;                     // ball time base: hit frame / fps (the frame can be fractional)
  rally: number;                   // [unread] int index into rallies. -1 if the hit is outside every ball rally (0.2 s tolerance).
                                   // Such shots still count in match.shots, the summaries, the timeline and the table
  player: PlayerId;                // hitter
  team: TeamId | "?";              // "?" only for a player missing from the team map, which stable IDs rule out
  type: ShotType;
  speed_kmh: number | null;        // km/h, 1 dp. Launch speed of the 3D flight if its fit is ok, else the 2D average
                                   // from the hitter to the landing, else null
  speed_source: "3d" | "2d" | null;   // [unread] null iff speed_kmh is null
  contact_height_m: number | null; // m, 2 dp, 0..3.325: ball height within the hitter's image box (clipped to 0..1.9) x 1.75.
                                   // Where the pose decided: >= 1.75 for an overhead, <= 1.66 otherwise (shots.py:279-280).
                                   // null if the hitter has no box within 2 frames of the hit
  contact: [number, number] | null;   // [x, y] court m, not clamped to the court: the hitter's floor position at the hit frame.
                                      // null if the hitter has no position in that frame
  depth_m: number | null;          // [unread] |contact.y - 10|, 2 dp. null iff contact is null
  zone: "net" | "middle" | "back" | null;   // depth < 4 / 4..7 / > 7 (ShotRules, shots.py:106-107, not Settings). null iff contact is null
  landing: [number, number] | null;   // [x, y] court m. First floor bounce seen in 2D, else that of an ok 3D fit.
                                      // null if neither exists, or if it lies on the hitter's own half
  landing_source: "2d" | "3d" | null; // [unread] null iff landing is null
  apex_m: number | null;           // [unread] m, 2 dp. Only with an ok 3D fit
  air_time_s: number | null;       // [unread] s, 2 dp. Hit to the first floor bounce, from 2D only
  before: "serve" | "volley" | "bounce" | "wall" | null;   // [unread] what the ball did before the hit.
                                                           // "serve" only when type = "serve". null = not seen
  direction: "cross" | "line" | "middle" | null;   // from contact to landing, else to the position of the opponent who played next.
                                                   // "middle" = within 1.25 m of x = 5. null without contact or aim point
  target: PlayerId | null;         // the opponent who hit next in the same rally. null if there is none
  wall_after: boolean;             // [unread] a wall bounce before the next hit
  attacking: boolean;              // type is in attacking_types
  overhead: boolean;               // [unread]
  confidence: "high" | "medium" | "low";   // "low" for type "other". Otherwise three checks (bounce before the hit seen,
                                           // pose decided overhead or not, 3D fit ok): 3 = high, 2 = medium, fewer = low
}

interface RallyEnding {            // shots.py:389-432. One per ball rally (ball arcs less than 4 s apart), including rallies with 0 shots
  rally: number;                   // int, 0-based = position in rallies
  start_s: Sec;                    // ball time base: first ball-track frame of the rally / fps
  end_s: Sec;                      // last ball-track frame / fps
  duration_s: number;              // end_s - start_s
  shots: number;                   // int. ShotInfo with this rally index; can be 0
  last_player: PlayerId | null;    // null iff shots = 0
  last_type: ShotType | null;      // null iff shots = 0
  outcome: "winner" | "error" | "cut" | "unknown";
  // shots = 0: always "unknown" (tested first).
  // "cut": the last player frame is less than 1.5 s after end_s. Only the end of the video is tested;
  //   the docstring's "(or began)" is not implemented.
  // "winner" / "error": needs an ok 3D fit of the last shot with at least one bounce, and that shot's contact.
  //   First bounce on the floor of the opponents' half = "winner"; any other first bounce = "error".
  // Everything else: "unknown".
}

interface MatchTotals {            // shots.py:493-501
  rallies: number;                 // int = rallies.length, including 0-shot rallies
  shots: number;                   // int = shots.length, including shots with rally = -1
  avg_rally_shots: number | null;  // 2 dp, over rallies with >= 1 shot. null if there are none
  avg_rally_s: number | null;      // [unread] s, 2 dp, over all rallies. null if no rallies
  longest_rally_shots: number | null;   // int. null if no rally has a shot
  longest_rally_s: number | null;  // s, 1 dp, over all rallies. null if no rallies. An independent maximum: it can belong
                                   // to another rally than longest_rally_shots, yet the hero tile shows the two together
  time_in_play_s: number;          // s, 1 dp. Sum of the ball-rally durations, 0 if none
}

interface Timeline {               // [unread] always from player movement, even when rally_source = "ball"
  dt: number;                      // s between samples = round(fps / 2) / fps. Sample k is at t0 + k * dt
  activity: Frac[];                // share of a 5 s window in which the fastest player is above 2.5 m/s. Never null
  threshold: Frac;                 // 0.1 (Settings.rally_share): activity at or above it counts as a rally
}

interface Quality {
  ball_coverage: Frac | null;      // frames with a raw ball detection / frames in the ball span. null iff has_ball = false
  ball_kept: Frac | null;          // share of raw detections kept after cleaning. null iff has_ball = false
  has_ball: boolean;               // [unread] the template tests the nulls instead
  has_3d: boolean;                 // a 3D result exists. True even if no single flight fits ok
  tracker_ids_merged: number;      // int. Distinct raw tracker IDs per player, summed over the four
  interpolated_frames: number;     // int. Gap-filled player-frames (one per player per frame); 0 with gap filling off
  appearance_used: boolean;
  swaps_fixed: { frame: number; time_s: Sec; players: PlayerId[] }[];
  // identities.py:381-387. [] when appearance was not used or nothing was corrected. frame = absolute frame number,
  // time_s = that frame's own timestamp (2 dp), players = the IDs involved, sorted. The template reads only length and time_s
}
```

**Time bases.** Three clocks appear in the data, and the template subtracts `t0` from all of them:
- Movement base: `timestamp_ms / 1000` of the first player frame, plus `i / fps` (`movement.py:109-112`). Used by `t0`, `duration_s`, `movement_rallies`, `timeline` and `positions.csv`.
- Ball base: `frame / fps` (`ball_events.py:412-416`, `:427`, `:464`). Used by `shots[].time_s`, `rallies[].start_s` and `end_s`, and `ball_events.csv`.
- `swaps_fixed[].time_s`: the frame's own timestamp (`identities.py:384`).

The first two differ by the constant `timestamp(f0) − f0 / fps`, where `f0` is the first player frame. Timestamps come from OpenCV's `CAP_PROP_POS_MSEC`, read after each frame (`io/video.py:51-56`). The code does not settle whether that constant is zero; it depends on the video container and the OpenCV backend. The "cut" test compares the two bases directly (`match_report.py:285`, `shots.py:419`).

**No ball analysis (`has_ball = false`).** This happens when the tracking has no ball detection, or when the caller passes `use_ball=False` (CLI `--no-ball`). Missing calibration also stops the ball step (`ball_events.py:538-539`), but then there is no report at all.
- `match`, `rallies`, every `players[].shots` and `teams[].shots`, `quality.ball_coverage` and `quality.ball_kept` are `null`.
- `shots` is `[]`, `settings.rally_source` is `"movement"`, and `quality.has_3d` is `false`.

**Ball, but `rally_source = "movement"`** (coverage under 0.4, or no ball rally). `match`, `rallies` and `shots` are still filled. Only the movement statistics and heatmaps switch to rallies found from player movement.

**Ball but no 3D fit (`has_3d = false`).** This happens with `fit_3d=False` (CLI `--no-3d`), or when the fit returns nothing.
- Every `apex_m` is `null`.
- `speed_source` and `landing_source` are never `"3d"`, so all speeds are 2D averages, which read lower.
- `confidence` is never `"high"`.
- Lobs are found only from `air_time_s` (1.7 s or more) or an overhead answer.
- `outcome` is only `"cut"` or `"unknown"`, so all `winners` and `errors` are 0.

The same limits apply to any single shot whose own flight fit is not ok, even with `has_3d = true`.

**Not in ReportData:**
- `summary.json` (`match_report.py:293-313`): `players` and `teams` are dicts keyed by id that hold the `move` objects, `rallies` is `movement_rallies`, and `stats` repeats `match`, `rallies` and the shot summaries. The only new data is the `ball` block: `coverage`, `kept`, the shot count, the raw ball rallies with a `hits` count, and per player and per team (`ball_stats()`, `:63-95`) `volleys`, `volley_share`, `landed`, a `deep_share` over 2D landings without the contact requirement, `median_flight_s` and `median_speed_ms` (m/s). Without ball analysis there is no `ball` or `stats` key.
- `positions.csv`: one row per frame and player with `frame`, `time_s`, raw and smoothed x/y, `speed_ms`, `detected` and `in_rally`. ReportData holds only the heat grids and the totals.
- `ball_events.csv` (only with ball analysis): every hit, floor bounce and wall bounce with time, hitter, court x/y (empty for wall bounces) and image u/v. ReportData has no bounce events.
- `shots.csv` (only with ball analysis): ShotInfo flattened, without `index`. It adds nothing new.

## B. UI inventory of report.html

Line numbers refer to `report.html`.

**Shell (616-649)**
- Header: logo, "CourtShot", "Match report", `title`, a meta line (`dur(duration_s)`; rally and shot counts if `match`), and a pill per team. `document.title` and a static footer are also set.
- Sticky blurred tabs: Overview, Players, Shots, Rallies, "About the numbers", plus an "Appearance" theme button. `go()` shows one page, writes `#id` with `history.replaceState` and scrolls to the top. The hash is read only at load; there is no `hashchange` listener.
- Overview, Rallies and About are built once. Players and Shots rebuild on every visit and filter change from a module-level `state` (457).

**Overview (355-424)**
- Hero tiles, with chip "Rallies from the {rally_source}":
  - "Longest rally" (key colour): `longest_rally_shots` over `dur(longest_rally_s)`. Without `match`: "Rallies" = `movement_rallies.length`.
  - "Time in play": `match.time_in_play_s`, else the summed `movement_rallies[].duration_s`; minutes from 90 s; below, `pct(inPlay/duration_s)`.
  - Only with `match`: "Rallies" (`rallies`, `avg_rally_shots`) and "Shots" (`shots`, `pct(ball_coverage)`).
  - "Distance covered": summed `distance_m`, km from 1000 m.
- "Team comparison" (`cmpRow`, 355-364), T_1 left, T_2 right. Each row has two values around a label and two 4 px bars of `value/max(a,b)` growing outwards from the centre. The leader gets a team-soft pill: higher leads, lower with `lowerLeads`, nobody on a tie or a missing value. Rows:
  - if `teams[0].shots`: `shots`, `attacking_share`, `avg_speed_kmh`, `max_speed_kmh`, "Overheads" (`smash + bandeja_vibora`), `lob`;
  - if the four counts sum above 0: `winners`, `errors` (lower leads);
  - always: `both_at_net`, `both_at_back` (lower leads), team `distance_m`, `net_regained`.
- "Shot types" (if `teams[0].shots`): mirrored bars per `shot_types` entry with a nonzero total, on one scale (the largest single count).
- "Where each player plays": four mini heatmap buttons (2 columns below 760 px) with `pct(time_at_net)`. A click selects the player and opens Players.
- "Match timeline".

**Players (458-497)**
- Segmented picker (default `players[0]`), large "Heatmap", and "Movement".
- Movement lists `tracked`, `distance_m`, `distance_in_rallies_m`, `top_speed_ms` × 3.6, `sprints` (with `sprint_speed` × 3.6), `avg_dist_to_net_m`, `court_coverage`, `time_left_side`. "Time by zone" (quotes `net_zone`, `back_zone`) has three bars whose width is the share itself, because `barList` scales to `max(1, …values)` (264-271).
- `shots` null: an empty card. Otherwise three cards:
  - "Shots": `shots`, `attacking_share`, `avg_speed_kmh`, `max_speed_kmh`, `avg_contact_height_m`, `deep_share`; winners / errors only if their sum is above 0.
  - "Shot types": nonzero `by_type`, descending.
  - "Direction and target": bar groups for `direction`, `targets` (the two opponents) and `from_zone`. Text is the percentage of the group total plus the raw count; bars are relative to the group's largest count (`shareRows`, 275-278).

**Shots (500-532)**
- Filters: a player segmented control and a `<select>` of the types present; a shot must match both. No shots at all: one empty card.
- "Shot map": the selection with `contact` and `landing`; the sub gives "N of M".
- "Fastest shots": top 8 by `speed_kmh`, bars relative to the fastest. The same card holds the "Contact height" strip.
- "All shots": 560 px scroll box, sticky header, row hover, no sorting. Columns: `index + 1`, clock, `player`, `type`, `speed_kmh`, `contact_height_m`, `zone`, `direction`, `target`, `landing` ("x.x, y.y"), `confidence` (Well / Partly / Poorly).

**Rallies (568-585)**
- `rallies` non-null: table of `rally + 1`, start clock, `dur(duration_s)`, `shots`, `last_player`, `last_type`, `outcome` (Winner / Error / "Video ends mid-rally" / "Not clear").
- Movement table (start, end, length): shown when `rallies` is null or `rally_source` is "movement".

**About (588-613)**
- "Data quality" (`ball_coverage`, `ball_kept`, `has_3d`, per player `tracked` and `interpolated`); "Player identities" (`tracker_ids_merged`, `interpolated_frames`, `appearance_used`, `swaps_fixed`); a static definition list that inlines `still_speed`.

**Custom drawings**
- **Heatmap** (canvas, 283-352)
  - The square is the player's own half, 10 × 10 m. `heat[row][col]` is drawn untransposed: row 0 at the top (net), col 0 at the left (the player's left).
  - Backing size is CSS width × min(2, devicePixelRatio). It redraws on resize and on theme toggle, because colours are read from CSS variables at draw time.
  - Background: five horizontal bands alternating `--court` and `--court2`.
  - Data: separable Gaussian blur, σ = n/16 = 2.5 cells, radius 8. Outside cells count 0 but the divisor stays the full kernel sum, so edges fade. `(blurred/peak)^0.8` goes through `RAMPS[theme][team]` (RGBA stops, linear) into a 40 × 40 image, scaled up smoothed. Peak 0: bare court.
  - Lines (`--court-line`, width m = max(1, W/300) backing px): outline inset by m, service line at `service_line/10` of the height, centre line from the top to the service line, net as a 3m-thick top line.
  - Large only: "Net" and "Back wall" tags, and a tooltip with the share of rally time, depth from the net and left or right side. The share is the raw counts in the 4 × 4 cells (1 m) starting 2 up-left of the pointer cell, over the total.
  - Legend: "Less time", a 120 px `--heat1` or `--heat2` gradient, "More time"; a team dot when both teams show.
- **Shot map** (SVG, 533-553)
  - viewBox 316 × 616. `X = 8 + 30x`, `Y = 8 + 30(20 − y)`: the whole court, y = 0 (camera end) at the bottom, x not mirrored.
  - Lines 1.5 wide: outline (rx 10), service lines at y = 10 ± `service_line`, centre line between them. Net 3 wide with a "Net" label at its right end.
  - A shot runs from `contact` to `landing`: a 2-wide team line (opacity .85) over a 4.5-wide `--court` halo and a 12-wide transparent hit line, with a filled mark (r 5.5) at the landing. Hover dims the other shots to .18 and shows the shot tooltip.
- **Height strip** (SVG, 554-565): viewBox 640 × 150. `x = 34 + (time_s − t0)/max(1, duration_s) · 600`, `y = 6 + (1 − min(h, 3.2)/3.2) · 124`. Gridlines at 0 to 3 m and a "head height" line at 1.75 m. Filled marks r 4.5 with the shot tooltip. No time axis.
- **Timeline** (SVG, 426-449)
  - viewBox 1000 × 138. `x = 44 + (t − t0)/max(1, duration_s) · 948`. Four lanes 26 high from y = 8 in `players` order.
  - Bands: `rallies || movement_rallies`, `start_s` to `end_s`, across all lanes, minimum width 1.
  - Labels `clock(elapsed)` every 10 s (30 above 90 s, 60 above 240 s, 120 above 600 s).
  - Shots sit in the hitter's lane: attacking filled r 5, others hollow r 4.2, with the shot tooltip. The legend, shown only with shots, explains filled and hollow.
- **Shot tooltip** (450-454): "Player · Type", the clock, then where present km/h, contact height, direction and target. One fixed element at cursor +14 px that flips at viewport edges; mouse events only.

## C. Gotchas

- **External resources:** only Google Fonts Inter 400/500/600 (7-9), with a system-stack fallback and `font-feature-settings "cv11","ss01"`. The logo is inline SVG with a fixed gradient id `cs-tile`, rendered twice (233-241), so React needs unique ids.
- **CSS variables** (14-35): 22 custom properties for surfaces, lines, text, accent, key number, court, teams and legend gradients. Dark is the default and light the override. `--clay` is defined but unused.
- **Colours:** `--t1` `#d4f542` (light `#5c6b00`), `--t2` `#4d8dff` (light `#2f6fe6`); `--key` equals `--t1`. The canvas ramps are separate hard-coded RGBA stops per theme and team (283-288), to be kept in sync with the `--heat1` and `--heat2` legend gradients by hand.
- **Identity** is colour plus shape. Dots: T_1 round, T_2 square; a team's second player in `players` order is a hollow ring (229-230); team-level dots are solid. SVG marks (`mark()`, 243-246) show the team only: a circle of radius r, or a square of side 1.8r with rx 0.3r. Hollow marks mean "not attacking" and occur only in the timeline.
- **Theme:** `html[data-theme]`; `?theme=dark|light` forces it at load; the toggle is not persisted and `prefers-color-scheme` is ignored.
- **Helpers** (222-227): `has` rejects null, undefined and NaN. `num(v, d, unit)` and `pct` (rounded integer %) print "–" when missing. `kmh` multiplies by 3.6. `clock` gives `m:ss` from rounded, non-negative seconds, without hours. `dur` gives "x.x min" from 90 s, else "N s". Every displayed time is `value − t0`. Speeds print as whole km/h (top speed 1 dp).
- **Missing data:** `has_ball` is never read. The template branches on `match`, `teams[0].shots`, `players[i].shots`, `rallies` and `shots.length`. `has_3d` changes only the About line.
- **Empty states:** "No ball data in this run…" (Players), "No shots were found in this run." (Shots), "No shots found." (a player's shot types), "No speeds for this selection.", "None found." (movement rallies). An all-zero heatmap shows the bare court.
- **Quirks:** the About text says the heat colour runs "green to red", but the ramps are one hue per team. Layout is a 12-column grid with breakpoints at 900, 760, 640 and 520 px.
