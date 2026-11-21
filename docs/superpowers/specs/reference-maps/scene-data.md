> Extracted from the reference code under `src/courtshot` on 2026-11-20 and checked against the source by a second reader. Line numbers refer to the reference files at commit c2b1767. Nothing in the pipeline was run to produce this.

# CourtShot 3D replay: scene data and viewer reference for the port

All file references are relative to `/Users/kerem/Desktop/courtshot/src/courtshot/`. `scene3d.py` means `report/scene3d.py`, and `html` means `report/templates/scene3d.html`.

## A. SceneData contract

`build_scene` (scene3d.py:216-275) returns the object below. `write_scene` (scene3d.py:278-316) adds `video` (299-303), serialises with `json.dumps(scene, separators=(",",":"))` and substitutes the string for the literal `/*SCENE_DATA*/null` in the template (304-305, html:250).

Other behaviour of `write_scene` that the worker has to know:
- `also_json=True` writes the same string to `out.with_suffix(".json")` (314-315). This is the existing JSON export.
- `fragment=True` leaves out the `<!doctype>/<html>/<head>` wrapper (306-312).
- It raises `ValueError` when `analysis.ball` or `analysis.ball3d` is None (294-295). Without a calibration, a ball track and a 3D fit there is no scene at all; the pipeline skips the replay in that case (pipeline.py:286-287).
- `build_scene` itself fails on an empty `player_frames` list (`max()` of an empty sequence, 217).

```ts
/** Integer "no value" marker in `ball` and in every `players[].pose`. Always -32000 (scene3d.py:42); also sent as `none`. */
type NoneMarker = -32000;

/** Viewer frame, metres: x across (-5..5), y up (height), z along (+10 = near baseline, -10 = far baseline). */
type Vec3m = [number, number, number];

interface SceneData {
  /** Frame rate, unrounded float. MatchAnalysis.fps = median(frame-number step / timestamp step) over the tracking
   *  (analysis/identities.py:591-603, fallback 30.0), or the CLI --fps override (cli.py:159, analysis/match.py:76).
   *  May be non-integer. It is the only frame <-> time base: frame = t * fps. */
  fps: number;
  /** Frame count = max(max player frame_id + 1, len(ball_frames)) (scene3d.py:217-218).
   *  Viewer duration DUR = (n - 1) / fps seconds (html:251). */
  n: number;
  camera: BroadcastCamera;
  /** One entry per distinct player_id in the tracking, sorted by id (scene3d.py:105). */
  players: ScenePlayer[];
  /** Dense flat int array, length n*4, frame-major: index = frame*4 + c.
   *  c 0..2 = x, y, z in the viewer frame, integer centimetres = round(100 * T(p)). Height is NOT clamped.
   *  c 3    = 0-based index into `flights` of the flight that owns this frame.
   *  A frame owned by no flight has all four values = none (scene3d.py:222). The viewer tests only c 0 (html:542).
   *  Flights with ok=false ARE included (unlike Ball3DResult.positions).
   *  A flight owns frames ceil(t0*fps) .. floor(t1*fps), clipped to 0..n-1 (scene3d.py:225-227).
   *  When one flight ends on exactly the frame where the next starts, the later flight owns that frame
   *  (loop sorted by hit time, later writes win, scene3d.py:224-232). */
  ball: number[];
  /** Sorted by hit time. Flights covering fewer than 2 frames are dropped before indexing, so indices are dense
   *  (scene3d.py:228-229). NOT one-to-one with `hits`: the fit can split a flight at a missed hit, or produce no
   *  flight for a hit (ball_flight.py:384-385, 515-537, 621-623). */
  flights: Flight[];
  /** Every 2D "hit" event, in time order (scene3d.py:220, 272; events are sorted at analysis/ball_events.py:429). */
  hits: Hit[];
  /** [start_s, end_s] per ball rally, seconds. BallResult.rallies holds (start_s, end_s, hits); the hit count is
   *  dropped (scene3d.py:273). A rally = ball arcs separated by less than rally_gap_s = 4.0 s
   *  (analysis/ball_events.py:67, 455-466). The current viewer never reads this field. */
  rallies: [number, number][];
  none: NoneMarker;
  /** Added by write_scene (scene3d.py:299-303). A path to a video RELATIVE TO THE FOLDER OF THE WRITTEN HTML,
   *  POSIX separators. Set only when a video path was given, its suffix is .mp4/.mov/.m4v/.webm and the file
   *  exists; otherwise null. The viewer assigns it to video.src unchanged (html:607), so any URL would work.
   *  Both current callers pass the ORIGINAL match video, not the tracked one (pipeline.py:287, cli.py:93).
   *  Video time 0 is assumed to be frame 0; there is no offset field. */
  video: string | null;
}

/** The calibrated broadcast camera in the viewer frame. Source: ball_flight.Camera (ball_flight.py:54-86), a pinhole
 *  with square pixels and the principal point at the image centre (ball_flight.py:11-12, 113); R and t map
 *  world -> camera in the OpenCV convention (x right, y down, z forward). */
interface BroadcastCamera {
  pos: Vec3m;     // camera centre, metres: T(cam.centre), 3 dp
  right: Vec3m;   // unit vector Tv(R[0]), 5 dp
  up: Vec3m;      // unit vector Tv(-R[1]), 5 dp
  back: Vec3m;    // unit vector Tv(-R[2]), 5 dp; the camera looks along -back (three.js camera +z)
                  // Tv(v) = [v.x, v.z, -v.y]: the direction form of T, no translation (scene3d.py:255)
  fov: number;    // VERTICAL field of view, degrees = 2*atan((height/2)/focal), 3 dp
  aspect: number; // video width / height, 4 dp. Pixel width and height are not sent.
  // Viewer use: makeBasis(right, up, back) gives the exact orientation (html:674-675); the Broadcast view orbits
  // around the point where the centre ray meets the floor, pos + back * (-pos.y / back.y) (html:657).
}

interface ScenePlayer {
  /** Stable identity. Expected "P_1".."P_4" (analysis/identities.py:56), but scene3d.py does not enforce it.
   *  Flight.hitter and Hit.player refer to it. The viewer's label is id.replace('_', ' '). */
  id: string;
  /** First non-empty team_id among the player's frames, else "?" (scene3d.py:108, 208). Expected "T_1" (P_1, P_2)
   *  or "T_2" (P_3, P_4) (analysis/identities.py:57). The viewer colours only those two; anything else is grey. */
  team: string;
  /** Racket hand. At each of this player's hits the wrist (2D keypoint) nearer the ball pixel gets a vote.
   *  "left" needs strictly more left votes; a tie or no votes gives "right" (scene3d.py:192-202). */
  hand: "left" | "right";
  /** Dense flat int array, length n*13*3. index = (frame*13 + joint)*3 + axis; axis 0 = x, 1 = y (height), 2 = z.
   *  Viewer frame, integer centimetres (scene3d.py:204, 210; decoded at html:468-472).
   *  Joint order (stride 13), scene3d.py:25-39:
   *    0 head, 1 left_shoulder, 2 right_shoulder, 3 left_elbow, 4 right_elbow, 5 left_wrist, 6 right_wrist,
   *    7 left_hip, 8 right_hip, 9 left_knee, 10 right_knee, 11 left_ankle, 12 right_ankle.
   *  Names are the pose model's keypoint names. "head" is the confidence-weighted mean of nose, eyes and ears
   *  (scene3d.py:40, 117-120).
   *  A frame is all-or-nothing: where the player has a floor position all 39 values are filled, otherwise all 39
   *  are none (scene3d.py:173-190). Joints missing in a present frame are filled from the last good pose shape
   *  relative to the floor position, or from a built-in standing pose with the arms down (scene3d.py:156-172).
   *  Height is at least 3 cm (scene3d.py:151). The viewer hides the player if ANY joint of the frame is none.
   *  Geometry: 2D keypoints are lifted onto the vertical plane that stands on the floor position and faces the
   *  camera (scene3d.py:136-152), so every skeleton is flat: right from the broadcast angle, paper-thin from the side. */
  pose: number[];
}

/** One fitted 3D flight: everything between one hit and the next (ball_flight.py:239-251). */
interface Flight {
  t0: number;            // launch (hit) time, s, 3 dp
  t1: number;            // end, s, 3 dp: the next hit, a split point, or the last tracked frame within 3.5 s
                         //   (ball_flight.py:617-624)
  ok: boolean;           // rms_px <= bad_rms_px, where bad_rms_px = 5.0 px * frame_height/720
                         //   (ball_flight.py:287, 451, 694-695). false = "low confidence": still drawn, in the warn colour
  hitter: string | null; // ScenePlayer.id, or null when unknown
  speed_kmh: number;     // fitted launch speed |v0| * 3.6, 1 dp
  max_h: number;         // peak height above the floor, m, 2 dp (60 samples along the flight)
  net_h: number | null;  // height above the FLOOR (m, 2 dp) at the first of 240 samples after the path first crosses
                         //   court y = 10; null if it never crosses (scene3d.py:233-236). Not a clearance: the viewer
                         //   subtracts 0.88. The flight model has no net collision, so it can be below net height.
  rms_px: number;        // reprojection error, px, 1 dp (rms over the best 90% of detections, ball_flight.py:431-433)
  h_sd: number | null;   // 1-sigma height uncertainty mid-flight, m, 2 dp; null when not finite. The viewer never reads it.
  bounces: Bounce[];     // in time order
}

interface Bounce {
  kind: "ground" | "wall"; // only these two (ball_flight.py:220, 233). Which wall is not recorded; the viewer
                           //   infers side or back wall from p (html:786)
  t: number;               // absolute time, s, 3 dp (ball_flight.py:450)
  p: Vec3m;                // viewer frame, METRES (not centimetres), 3 dp
}

interface Hit {
  t: number;               // s, 3 dp
  player: string | null;   // ScenePlayer.id, or null
}
```

**Transform from court to viewer** (`T`, scene3d.py:45-48). A court point `(x, y, z)` in metres, with x across 0..10, y along 0..20 and z up, maps to viewer `(x - 5, z, 10 - y)` in metres. `ball` and `pose` store `round(100 * that)` as integers; camera vectors and bounce positions stay in metres as floats. The net is at viewer z = 0. Court y = 0 is the near baseline (court.py:4) and lands at z = +10; the far baseline is at z = -10. Decode with `v / 100` (html:471, 543). The docstring of `T` says "cm", but `T` returns metres; the callers multiply by 100.

**Not settled by the code:**
- How `frame = t * fps` relates to a video that has a start offset or a variable frame rate. No offset field exists.
- Whether `players` can hold ids other than `P_1`..`P_4`, or fewer than four. scene3d.py takes whatever ids the tracking has.
- Which hit a flight belongs to. Nothing links `flights` and `hits` except `t0` and `hitter`.
- Which wall a `wall` bounce touched.
- The object has no schema version, match id, court dimensions or video pixel size. The viewer hard-codes the 10 × 20 court and the 0.88 m net.

## B. Payload size

`ball` and `pose` are dense: every frame has a slot. With P players (normally 4):

- **Integers:** `4 + 39·P` = 160 per frame. Per minute: `60 · fps · (4 + 39·P)` = **240,000** at 25 fps. For 90 minutes: **21.6 million**.
- **JSON:** 4.1 to 4.9 bytes per integer (1 to 5 characters plus a comma; `none` costs 7), so **1.0 to 1.2 MB per minute and 90 to 110 MB for 90 minutes**, plus a few KB per minute for everything else. This is estimated from synthetic values, because the repository has no scene file to measure. Gzip and heap sizes are unmeasured.
- **Int16 binary:** every value fits (marker −32000, coordinates about ±1000 cm), giving **480 KB per minute, 43 MB for 90 minutes**, and it can be split by time range. Conditions: fewer than 32,768 flights, and pose values clamped, because the lift (scene3d.py:147-150) has no upper bound.

## C. Viewer feature inventory

**Scene** (html:259-546)
- **Court:** ground, apron and a 10×20 floor with a random noise texture; glass and mesh walls with posts (html:371-389); net 0.88 m at the centre, 0.92 m at the posts; one shadow-casting sun; fog.
- **Lines:** only the service lines (z = ±6.95) and the centre line. The calibration keypoints assume 7.0 m from the net (court.py:30-31), 5 cm off.
- **Players:** bone cylinders, joint spheres, torso, neck, team-coloured shirt, a racket on the `hand` wrist, a name sprite above the head drawn over everything.
- **Ball:** radius 3.25 cm with a halo; a 0.45 s trail within the flight; a dashed height line with a floor ring; the current flight's whole path; bounce rings for 0.8 s (pool of 6; floor rings flat, wall rings facing the nearer wall). On low-confidence flights the halo, trail and path turn orange.
- **Interpolation:** linear between frames, only without video, never across two flights.

**Cameras** (html:654-708)
- OrbitControls: drag orbits, wheel zooms, right-drag pans.
- Views: Broadcast (the calibrated camera), Far end, Side, Top (html:660-662) and Follow ball, which moves 6% of the way to the ball per rendered frame (frame-rate dependent).
- View changes tween for 900 ms, or jump under `prefers-reduced-motion`. Any OrbitControls interaction cancels the tween and sets the view to "free", except in Follow.
- Normal Broadcast loses camera roll through OrbitControls. `lockBroadcast` applies the exact `right/up/back` basis, in overlay only.
- `fovFor` widens the fov on viewports narrower than `camera.aspect`, matching `object-fit: contain`.

**Video modes** (html:573-625)
- **3D only** is the start state. **Side by side** puts the video on the right, or on top on portrait viewports.
- **Overlay:** video under the transparent canvas. Court hidden, lines magenta, magenta court edges added (html:346-354), bodies 55% opaque, camera locked. OrbitControls, view buttons and keys 1 to 5 are inert.
- `video` set: it loads at start and the page switches to Side by side on `loadeddata`.
- `video` null, or a load error: a file picker (`URL.createObjectURL`) shows inside the video pane, so only once Side by side or Overlay is selected.

**Clock** (html:577-590, 801-806)
- **Playing, with a loaded video and a mode other than 3D only: the `<video>` is the clock.** `t` is the `mediaTime` from `requestVideoFrameCallback` (`currentTime` as fallback), capped at DUR. The frame drawn is `min(round(t·fps), n−1)`.
- **Paused, stepping or scrubbing: `t` is the master**, and the video is seeked to `(round(t·fps) + 0.3)/fps`.
- **Otherwise the page's own clock runs:** `t += dt·speed`, dt capped at 0.1 s, wrapping to 0 after DUR.
- Speed sets `playbackRate`. The video is muted and loops.

**Playback**
- Starts playing on load. Play/pause button; speeds ¼×, ½×, 1×; clock `m:ss.ss`.
- Timeline: flight spans (ball or warn colour), hit ticks in team colours, playhead, legend. An invisible 0..1000 `range` input on top scrubs, pausing while the pointer is down. One step is 5.4 s at 90 minutes.
- No rally, shot or hit navigation.

**Layer toggles:** Trail, Flight path, Height line, Low-confidence (off hides those flights and their bounces), Names.

**HUD**
- Subtitle: duration, "k of n ball flights reconstructed in 3D", player count.
- Lower third (html:712-735), rewritten when the flight changes: "Shot i of n" (flight index), hitter and team, a pill (Good fit, Low confidence, No 3D), Launch km/h, Peak m, Over net (`net_h − 0.88`), Ball now (every frame), bounce counts and fit error.
- Help text. A `<noscript>` message is the only error handling.

**Keys** (html:646-652, listener on `window`)
- Space toggles play. ← and → step one frame and pause. 1 to 5 select the views.
- Keys are ignored on an `<input>` target, except Space. The scrub input keeps focus after a click, so the arrows then move 1/1000 of the duration and 1 to 5 do nothing.
- Nothing in the 3D scene is clickable.

**Theme and layout:** dark only, CSS variables (html:9-24). At 720 px or narrower the button groups scroll, the note, help and subtitle hide, and the transport wraps. Safe-area insets are respected.

## D. Port notes

**Dependencies:** three 0.160.0 with its `OrbitControls` addon, from jsDelivr by import map (html:240-248); Barlow and Barlow Condensed from Google Fonts (html:3-5). Use npm and self-hosted fonts. Name sprites bake the font into a canvas at creation, so await `document.fonts.ready` first.

**React components (DOM and CSS):** top bar, lower third, help, transport (play, timeline, scrub, speed, clock), video pane with picker. React renders the `<video>` and passes it by ref.

**One imperative module (all three.js):** `createReplay(host, videoEl, data)` returning setters, `seek` and `dispose`.

**Global state:** `layers`, `t`, `playing`, `speed`, `view`, `curFlight` (html:549-550); `vidReady`, `vidMode`, `autoSrc`, `wantMode`, `vfTime` (575); `wasPlaying` (638); `tween` (665); `last` (800).

**Loop** (html:801-818), every animation frame, paused or not: clock, tween, `update()` (html:737-798, ending with DOM writes), `controls.update()` unless overlay, render.

**Lifecycle care**
- The page owns the document: `position: fixed` layout, `html, body` rules, mode classes on `document.body` (html:612-613), `getElementById`, colours read once from `:root` (html:253-256). Scope these to the component.
- No per-frame `setState`. Publish `t` and the ball height by ref or subscription, and emit `onFlightChange`.
- Nothing is torn down today. On unmount: cancel the rAF and the self-renewing video-frame callback (html:587-590), disconnect the ResizeObserver, and remove the `keydown`, `loadeddata` and `error` listeners.
- GPU: dispose controls, every geometry, material and canvas texture (per-player materials and labels, cloned wall panels, one line per flight), then the renderer.
- `<video>`: revoke blob URLs (html:605); `play()` rejections are swallowed (html:584).
- Init must survive StrictMode's double mount. Scope the key handler to the replay.
- Call `resize()` after layout changes; it re-locks the overlay camera.

**Match-length traps**
- `flightLines` (html:520-531) scans all frames per flight and creates a line and a material per flight; every frame then loops over all of them (html:774) and over all bounces (html:778). Build only the current flight's path.
- The timeline has one DOM node per flight and hit, and spans have a 0.2% minimum width (11 s at 90 minutes), so the lane smears. Zoom it by rally.

**Video**
- Today `video` is the original match video. Overlay on a tracked video would draw the annotations twice, so decide which file the replay gets.
- Sync assumes `time = frame / fps` from frame 0: constant frame rate, no start offset.
