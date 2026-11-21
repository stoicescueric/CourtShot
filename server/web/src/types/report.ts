export type PlayerId = 'P_1' | 'P_2' | 'P_3' | 'P_4';
export type TeamId = 'T_1' | 'T_2';
export type ShotType =
  | 'serve' | 'drive' | 'lob' | 'volley' | 'bandeja_vibora'
  | 'smash' | 'bajada' | 'chiquita' | 'other';
/** Absolute seconds, not relative to t0. */
export type Sec = number;
/** Share, 0..1. */
export type Frac = number;

export interface ReportData {
  schema_version: number;
  pipeline_version: string;
  title: string;
  duration_s: Sec;
  t0: Sec;
  fps: number;
  settings: Settings;
  /** Always 4, in PlayerId order. */
  players: PlayerRow[];
  /** Always 2: [T_1, T_2]. */
  teams: TeamRow[];
  movement_rallies: MovementRally[];
  /** null iff has_ball = false. */
  match: MatchTotals | null;
  /** null iff has_ball = false. Ball rallies; can be []. */
  rallies: RallyEnding[] | null;
  /** [] when has_ball = false. In time order; shots[i].index = i. */
  shots: ShotInfo[];
  shot_types: ShotType[];
  shot_labels: Record<ShotType, string>;
  attacking_types: ShotType[];
  timeline: Timeline;
  quality: Quality;
}

export interface Settings {
  net_zone: number;
  back_zone: number;
  still_speed: number;
  sprint_speed: number;
  rally_source: 'ball' | 'movement';
  service_line: number;
}

export interface PlayerRow {
  id: PlayerId;
  team: TeamId;
  move: PlayerMove;
  /** null iff has_ball = false. */
  shots: ShotSummary | null;
  /** 40 x 40 integer frame counts; heat[row][col], row 0 = at the net, col 0 = the player's own left. */
  heat: number[][];
  heat_seconds: number;
}

export interface PlayerMove {
  team: TeamId;
  tracked: Frac;
  interpolated: Frac;
  distance_m: number;
  distance_in_rallies_m: number;
  top_speed_ms: number | null;
  sprints: number;
  avg_dist_to_net_m: number | null;
  time_at_net: Frac | null;
  time_in_middle: Frac | null;
  time_at_back: Frac | null;
  court_coverage: Frac;
  time_left_side: Frac | null;
}

export interface TeamRow {
  id: TeamId;
  players: [PlayerId, PlayerId];
  move: TeamMove;
  /** null iff has_ball = false. */
  shots: ShotSummary | null;
}

export interface TeamMove {
  players: [PlayerId, PlayerId];
  both_at_net: Frac;
  both_at_back: Frac;
  split: Frac;
  in_transition: Frac;
  avg_partner_distance_m: number | null;
  side_by_side: Frac | null;
  depth_sync: number | null;
  net_lost: number;
  net_regained: number;
  median_time_to_regain_s: number | null;
}

export interface MovementRally {
  start_s: Sec;
  end_s: Sec;
  duration_s: number;
}

export interface ShotSummary {
  shots: number;
  by_type: Record<ShotType, number>;
  attacking_share: Frac | null;
  avg_speed_kmh: number | null;
  max_speed_kmh: number | null;
  avg_contact_height_m: number | null;
  direction: { cross: number; middle: number; line: number };
  from_zone: { net: number; middle: number; back: number };
  deep_share: Frac | null;
  /** Only opponents that were targeted; can be {}. */
  targets: Partial<Record<PlayerId, number>>;
  winners: number;
  errors: number;
}

export interface ShotInfo {
  index: number;
  time_s: Sec;
  /** Index into rallies; -1 if the hit is outside every ball rally. */
  rally: number;
  player: PlayerId;
  team: TeamId | '?';
  type: ShotType;
  speed_kmh: number | null;
  speed_source: '3d' | '2d' | null;
  contact_height_m: number | null;
  /** [x, y] court m, not clamped. */
  contact: [number, number] | null;
  depth_m: number | null;
  zone: 'net' | 'middle' | 'back' | null;
  /** [x, y] court m. */
  landing: [number, number] | null;
  landing_source: '2d' | '3d' | null;
  apex_m: number | null;
  air_time_s: number | null;
  before: 'serve' | 'volley' | 'bounce' | 'wall' | null;
  direction: 'cross' | 'line' | 'middle' | null;
  target: PlayerId | null;
  wall_after: boolean;
  attacking: boolean;
  overhead: boolean;
  confidence: 'high' | 'medium' | 'low';
}

export interface RallyEnding {
  rally: number;
  start_s: Sec;
  end_s: Sec;
  duration_s: number;
  shots: number;
  last_player: PlayerId | null;
  last_type: ShotType | null;
  outcome: 'winner' | 'error' | 'cut' | 'unknown';
}

export interface MatchTotals {
  rallies: number;
  shots: number;
  avg_rally_shots: number | null;
  avg_rally_s: number | null;
  longest_rally_shots: number | null;
  longest_rally_s: number | null;
  time_in_play_s: number;
}

export interface Timeline {
  dt: number;
  activity: Frac[];
  threshold: Frac;
}

export interface SwapFixed {
  frame: number;
  time_s: Sec;
  players: PlayerId[];
}

export interface Quality {
  ball_coverage: Frac | null;
  ball_kept: Frac | null;
  has_ball: boolean;
  has_3d: boolean;
  tracker_ids_merged: number;
  interpolated_frames: number;
  appearance_used: boolean;
  swaps_fixed: SwapFixed[];
}
