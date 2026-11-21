/** Integer "no value" marker in `ball` and in every `players[].pose`. Always -32000. */
export type NoneMarker = -32000;

/** Viewer frame, metres: x across (-5..5), y up (height), z along (+10 = near baseline, -10 = far baseline). */
export type Vec3m = [number, number, number];

export interface SceneData {
  schema_version: number;
  /** Frame rate, unrounded float; the only frame <-> time base (frame = t * fps). */
  fps: number;
  /** Frame count. Viewer duration = (n - 1) / fps seconds. */
  n: number;
  camera: BroadcastCamera;
  /** One entry per distinct player id, sorted by id. */
  players: ScenePlayer[];
  /** Flat int array, length n*4, frame-major: c 0..2 = x, y, z in integer cm; c 3 = flight index. All four = none when no flight owns the frame. */
  ball: number[];
  flights: Flight[];
  hits: Hit[];
  /** [start_s, end_s] per ball rally, seconds. */
  rallies: [number, number][];
  none: NoneMarker;
  /** Video URL or path; null when there is none. */
  video: string | null;
}

export interface BroadcastCamera {
  pos: Vec3m;
  right: Vec3m;
  up: Vec3m;
  back: Vec3m;
  /** Vertical field of view, degrees. */
  fov: number;
  /** Video width / height. */
  aspect: number;
}

export interface ScenePlayer {
  id: string;
  /** Expected "T_1" or "T_2"; anything else is "?". */
  team: string;
  hand: 'left' | 'right';
  /** Flat int array, length n*13*3: index = (frame*13 + joint)*3 + axis, integer cm, viewer frame. 13 joints:
   *  head, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
   *  left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle. */
  pose: number[];
}

export interface Flight {
  t0: number;
  t1: number;
  ok: boolean;
  hitter: string | null;
  speed_kmh: number;
  max_h: number;
  net_h: number | null;
  rms_px: number;
  h_sd: number | null;
  bounces: Bounce[];
}

export interface Bounce {
  kind: 'ground' | 'wall';
  t: number;
  /** Viewer frame, metres. */
  p: Vec3m;
}

export interface Hit {
  t: number;
  player: string | null;
}
