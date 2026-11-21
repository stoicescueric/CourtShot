/**
 * Synthetic ReportData for tests and development: four players, 40 shots over six rallies and
 * structured heat grids, built deterministically and consistent with the contract in
 * reference-maps/report-data.md (summaries are derived from the shots, as the pipeline does).
 */
import type {
  MovementRally,
  PlayerId,
  PlayerMove,
  PlayerRow,
  RallyEnding,
  ReportData,
  ShotInfo,
  ShotSummary,
  ShotType,
  TeamId,
  TeamRow,
} from '../types/report';
import finalFRaw from './final_f.report.json?raw';

const FPS = 30;
const T0 = 12;
const DURATION = 150;
const SERVICE_LINE = 6.95;
const IDS: PlayerId[] = ['P_1', 'P_2', 'P_3', 'P_4'];
const TEAM_OF: Record<PlayerId, TeamId> = { P_1: 'T_1', P_2: 'T_1', P_3: 'T_2', P_4: 'T_2' };
const SHOT_TYPES: ShotType[] = ['serve', 'drive', 'lob', 'volley', 'bandeja_vibora', 'smash', 'bajada', 'chiquita', 'other'];
const SHOT_LABELS: Record<ShotType, string> = {
  serve: 'Serve',
  drive: 'Drive',
  lob: 'Lob',
  volley: 'Volley',
  bandeja_vibora: 'Bandeja / víbora',
  smash: 'Smash',
  bajada: 'Bajada',
  chiquita: 'Chiquita',
  other: 'Other',
};
const ATTACKING: ShotType[] = ['bajada', 'bandeja_vibora', 'smash', 'volley'];
const AT_NET: ShotType[] = ['volley', 'bandeja_vibora', 'smash'];
const RALLY_TYPES: ShotType[] = ['drive', 'lob', 'bandeja_vibora', 'volley', 'drive', 'smash', 'chiquita', 'bajada', 'volley', 'other', 'drive'];
const SPEED: Record<ShotType, number | null> = {
  serve: 62, drive: 66, lob: 41, volley: 54, bandeja_vibora: 71, smash: 96, bajada: 82, chiquita: 31, other: null,
};
const HEIGHT: Record<ShotType, number | null> = {
  serve: 0.95, drive: 1.02, lob: 0.84, volley: 1.31, bandeja_vibora: 2.18, smash: 2.46, bajada: 1.42, chiquita: 0.63, other: null,
};

/** [start_s, end_s, shots, outcome] of each ball rally, absolute seconds. */
type RallySpan = [number, number, number, RallyEnding['outcome']];
const SPANS: RallySpan[] = [
  [14.2, 29.8, 7, 'winner'],
  [37.5, 51.0, 6, 'error'],
  [59.0, 84.4, 9, 'unknown'],
  [92.1, 99.6, 3, 'winner'],
  [107.3, 134.9, 8, 'error'],
  [139.0, 161.5, 7, 'cut'],
];

const round = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d;
const mean = (vs: number[]) => (vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null);

function hitter(rally: number, k: number): PlayerId {
  const team = (rally + k) % 2 === 0 ? 'T_1' : 'T_2';
  if (team === 'T_1') return k % 3 === 1 ? 'P_2' : 'P_1';
  return (k + rally) % 3 === 0 ? 'P_4' : 'P_3';
}

function makeShots(spans: RallySpan[], threeD: boolean): ShotInfo[] {
  const shots: ShotInfo[] = [];
  spans.forEach(([start, end, n], rally) => {
    for (let k = 0; k < n; k++) {
      const i = shots.length;
      const player = hitter(rally, k);
      const team = TEAM_OF[player];
      const type: ShotType = k === 0 ? 'serve' : RALLY_TYPES[(i + rally) % RALLY_TYPES.length];
      const near = team === 'T_1';
      const depth = AT_NET.includes(type) ? 2 + ((i * 7) % 15) / 10 : type === 'serve' ? 9.6 : 5.5 + ((i * 3) % 40) / 10;
      const cx = round(1.5 + ((i * 37 + rally * 11) % 70) / 10, 2);
      const contact: [number, number] = [cx, round(near ? 10 - depth : 10 + depth, 2)];
      const reach = type === 'chiquita' ? 2.5 + (i % 3) * 0.5 : type === 'lob' || type === 'drive' ? 7.2 + (i % 4) * 0.4 : 4 + (i % 5) * 0.8;
      const lx = round(1 + ((i * 53 + 7) % 80) / 10, 2);
      const landing: [number, number] | null = i % 7 === 3 ? null : [lx, round(near ? 10 + reach : 10 - reach, 2)];
      const base = SPEED[type];
      const speed = base === null ? null : round((base + ((i * 13) % 11)) * (threeD ? 1 : 0.8), 1);
      const next = k + 1 < n ? hitter(rally, k + 1) : null;
      const direction = !landing ? null : Math.abs(lx - 5) < 1.25 ? 'middle' : cx < 5 !== lx < 5 ? 'cross' : 'line';
      const checks = 1 + (k % 2) + (threeD && i % 3 !== 0 ? 1 : 0);
      shots.push({
        index: i,
        time_s: round(start + 0.8 + (k * (end - start - 1.6)) / Math.max(1, n - 1), 2),
        rally,
        player,
        team,
        type,
        speed_kmh: speed,
        speed_source: speed === null ? null : threeD ? '3d' : '2d',
        contact_height_m: HEIGHT[type] === null ? null : round(HEIGHT[type] + (i % 4) * 0.05, 2),
        contact,
        depth_m: round(depth, 2),
        zone: depth < 4 ? 'net' : depth <= 7 ? 'middle' : 'back',
        landing,
        landing_source: landing ? (threeD && i % 5 === 0 ? '3d' : '2d') : null,
        apex_m: threeD ? round(2.4 + (i % 6) * 0.6, 2) : null,
        air_time_s: landing ? round(0.6 + (i % 5) * 0.3, 2) : null,
        before: type === 'serve' ? 'serve' : type === 'volley' ? 'volley' : type === 'bajada' ? 'wall' : 'bounce',
        direction,
        target: next,
        wall_after: type === 'lob' || type === 'drive',
        attacking: ATTACKING.includes(type),
        overhead: type === 'smash' || type === 'bandeja_vibora',
        confidence: type === 'other' ? 'low' : checks >= 3 ? 'high' : checks === 2 ? 'medium' : 'low',
      });
    }
  });
  return shots;
}

function makeRallies(spans: RallySpan[], shots: ShotInfo[], threeD: boolean): RallyEnding[] {
  return spans.map(([start, end, , outcome], rally) => {
    const own = shots.filter((s) => s.rally === rally);
    const last = own[own.length - 1];
    return {
      rally,
      start_s: start,
      end_s: end,
      duration_s: round(end - start, 4),
      shots: own.length,
      last_player: last ? last.player : null,
      last_type: last ? last.type : null,
      // Without 3D fits nothing can be called a winner or an error.
      outcome: !last ? 'unknown' : !threeD && (outcome === 'winner' || outcome === 'error') ? 'unknown' : outcome,
    };
  });
}

function summarise(shots: ShotInfo[], rallies: RallyEnding[], who: (p: PlayerId) => boolean): ShotSummary {
  const own = shots.filter((s) => who(s.player));
  const speeds = own.flatMap((s) => (s.speed_kmh === null ? [] : [s.speed_kmh]));
  const heights = own.flatMap((s) => (s.contact_height_m === null ? [] : [s.contact_height_m]));
  const landedY = own.flatMap((s) => (s.contact && s.landing ? [s.landing[1]] : []));
  const avgSpeed = mean(speeds);
  const avgHeight = mean(heights);
  const by_type = Object.fromEntries(SHOT_TYPES.map((t) => [t, own.filter((s) => s.type === t).length])) as Record<ShotType, number>;
  const targets: Partial<Record<PlayerId, number>> = {};
  for (const s of own) if (s.target) targets[s.target] = (targets[s.target] ?? 0) + 1;
  const ended = (o: RallyEnding['outcome']) => rallies.filter((r) => r.outcome === o && r.last_player && who(r.last_player)).length;
  return {
    shots: own.length,
    by_type,
    attacking_share: own.length ? round(own.filter((s) => s.attacking).length / own.length, 3) : null,
    avg_speed_kmh: avgSpeed === null ? null : round(avgSpeed, 2),
    max_speed_kmh: speeds.length ? round(Math.max(...speeds), 1) : null,
    avg_contact_height_m: avgHeight === null ? null : round(avgHeight, 2),
    direction: {
      cross: own.filter((s) => s.direction === 'cross').length,
      middle: own.filter((s) => s.direction === 'middle').length,
      line: own.filter((s) => s.direction === 'line').length,
    },
    from_zone: {
      net: own.filter((s) => s.zone === 'net').length,
      middle: own.filter((s) => s.zone === 'middle').length,
      back: own.filter((s) => s.zone === 'back').length,
    },
    deep_share: landedY.length ? round(landedY.filter((y) => Math.abs(y - 10) > SERVICE_LINE).length / landedY.length, 3) : null,
    targets,
    winners: ended('winner'),
    errors: ended('error'),
  };
}

/** A 40 x 40 grid of frame counts made of Gaussian blobs [row, col, spread, peak]. */
function heatGrid(blobs: [number, number, number, number][]): number[][] {
  return Array.from({ length: 40 }, (_, r) =>
    Array.from({ length: 40 }, (_, c) =>
      Math.round(blobs.reduce((a, [br, bc, s, w]) => a + w * Math.exp(-((r - br) ** 2 + (c - bc) ** 2) / (2 * s * s)), 0)),
    ),
  );
}

const HEAT: Record<PlayerId, number[][]> = {
  P_1: heatGrid([[6, 12, 3, 40], [32, 14, 4, 25]]),
  P_2: heatGrid([[7, 28, 3, 35], [30, 27, 5, 20]]),
  P_3: heatGrid([[8, 11, 4, 30], [34, 10, 3, 30]]),
  P_4: heatGrid([[5, 26, 3, 45], [26, 30, 5, 10]]),
};

const MOVE: Record<PlayerId, Omit<PlayerMove, 'team'>> = {
  P_1: { tracked: 0.97, interpolated: 0.04, distance_m: 412.6, distance_in_rallies_m: 301.2, top_speed_ms: 5.42, sprints: 9, avg_dist_to_net_m: 4.8, time_at_net: 0.46, time_in_middle: 0.21, time_at_back: 0.33, court_coverage: 0.52, time_left_side: 0.71 },
  P_2: { tracked: 0.95, interpolated: 0.06, distance_m: 388.1, distance_in_rallies_m: 279.5, top_speed_ms: 5.11, sprints: 7, avg_dist_to_net_m: 5.1, time_at_net: 0.41, time_in_middle: 0.24, time_at_back: 0.35, court_coverage: 0.48, time_left_side: 0.22 },
  P_3: { tracked: 0.92, interpolated: 0.09, distance_m: 436.9, distance_in_rallies_m: 322.8, top_speed_ms: 5.87, sprints: 11, avg_dist_to_net_m: 5.9, time_at_net: 0.33, time_in_middle: 0.19, time_at_back: 0.48, court_coverage: 0.57, time_left_side: 0.76 },
  P_4: { tracked: 0.9, interpolated: 0.11, distance_m: 351.4, distance_in_rallies_m: 250.3, top_speed_ms: null, sprints: 0, avg_dist_to_net_m: 3.9, time_at_net: 0.58, time_in_middle: 0.27, time_at_back: 0.15, court_coverage: 0.39, time_left_side: 0.18 },
};

const TEAM_MOVE = {
  T_1: { both_at_net: 0.31, both_at_back: 0.22, split: 0.27, in_transition: 0.2, avg_partner_distance_m: 3.4, side_by_side: 0.64, depth_sync: 0.71, net_lost: 4, net_regained: 3, median_time_to_regain_s: 4.6 },
  T_2: { both_at_net: 0.24, both_at_back: 0.3, split: 0.29, in_transition: 0.17, avg_partner_distance_m: 3.9, side_by_side: 0.55, depth_sync: 0.58, net_lost: 3, net_regained: 4, median_time_to_regain_s: 5.2 },
};

/** Rallies as the movement step snaps them: start and end on frames, duration = frames / fps. */
const toMovement = (spans: [number, number][]): MovementRally[] =>
  spans.map(([a, b]) => ({ start_s: a, end_s: b, duration_s: round(b - a + 1 / FPS, 4) }));

const MOVEMENT_FOUND: [number, number][] = SPANS.map(([a, b]) => [round(a - 0.9, 2), round(b + 1.4, 2)]);

function build(spans: RallySpan[], threeD: boolean): ReportData {
  const shots = makeShots(spans, threeD);
  const rallies = makeRallies(spans, shots, threeD);
  const withShots = rallies.filter((r) => r.shots > 0);
  const ballRallies = rallies.length > 0;
  const players: PlayerRow[] = IDS.map((id) => ({
    id,
    team: TEAM_OF[id],
    move: { team: TEAM_OF[id], ...MOVE[id] },
    shots: summarise(shots, rallies, (p) => p === id),
    heat: HEAT[id],
    heat_seconds: round(HEAT[id].flat().reduce((a, b) => a + b, 0) / FPS, 4),
  }));
  const teams: TeamRow[] = (['T_1', 'T_2'] as const).map((t) => {
    const pair = IDS.filter((id) => TEAM_OF[id] === t) as [PlayerId, PlayerId];
    return { id: t, players: pair, move: { players: pair, ...TEAM_MOVE[t] }, shots: summarise(shots, rallies, (p) => TEAM_OF[p] === t) };
  });
  const durations = rallies.map((r) => r.duration_s);
  const activity = Array.from({ length: DURATION * 2 + 1 }, (_, k) => {
    const t = T0 + k * 0.5;
    return spans.some(([a, b]) => t >= a && t <= b) ? round(0.35 + ((k * 7) % 30) / 100, 4) : round(((k * 3) % 8) / 100, 4);
  });
  return {
    schema_version: 1,
    pipeline_version: '0.1.0',
    title: 'final_f',
    duration_s: DURATION,
    t0: T0,
    fps: FPS,
    settings: { net_zone: 4, back_zone: 7, still_speed: 0.5, sprint_speed: 4, rally_source: ballRallies ? 'ball' : 'movement', service_line: SERVICE_LINE },
    players,
    teams,
    movement_rallies: ballRallies ? toMovement(spans.map(([a, b]) => [a, b])) : toMovement(MOVEMENT_FOUND),
    match: {
      rallies: rallies.length,
      shots: shots.length,
      avg_rally_shots: withShots.length ? round(mean(withShots.map((r) => r.shots)) ?? 0, 2) : null,
      avg_rally_s: rallies.length ? round(mean(durations) ?? 0, 2) : null,
      longest_rally_shots: withShots.length ? Math.max(...withShots.map((r) => r.shots)) : null,
      longest_rally_s: rallies.length ? round(Math.max(...durations), 1) : null,
      time_in_play_s: round(durations.reduce((a, b) => a + b, 0), 1),
    },
    rallies,
    shots,
    shot_types: SHOT_TYPES,
    shot_labels: SHOT_LABELS,
    attacking_types: ATTACKING,
    timeline: { dt: 0.5, activity, threshold: 0.1 },
    quality: {
      ball_coverage: ballRallies ? 0.82 : 0.12,
      ball_kept: ballRallies ? 0.91 : 0.64,
      has_ball: true,
      has_3d: threeD,
      tracker_ids_merged: 11,
      interpolated_frames: 214,
      appearance_used: true,
      swaps_fixed: [{ frame: 2163, time_s: 84.1, players: ['P_1', 'P_2'] }],
    },
  };
}

/** Ball analysis with 3D fits: 40 shots over 6 rallies. */
export const reportFull: ReportData = build(SPANS, true);

/** Ball analysis without 3D fits: 2D speeds only, no "high" confidence, no winners or errors. */
export const reportNo3d: ReportData = build(SPANS, false);

/** Ball analysis that found no ball rally (and so no shot): rallies from player movement. */
export const reportEmptyRallies: ReportData = build([], true);

/** No ball analysis (has_ball false): match, rallies and shot summaries null, shots []. */
export const reportNoBall: ReportData = {
  ...reportFull,
  settings: { ...reportFull.settings, rally_source: 'movement' },
  players: reportFull.players.map((p) => ({ ...p, shots: null })),
  teams: reportFull.teams.map((t) => ({ ...t, shots: null })),
  movement_rallies: toMovement(MOVEMENT_FOUND),
  match: null,
  rallies: null,
  shots: [],
  quality: { ...reportFull.quality, ball_coverage: null, ball_kept: null, has_ball: false, has_3d: false },
};

/** report.json of the end-to-end run on videos/final_f.mp4 (90 s, one ball rally, 62 shots), as the worker wrote it. */
export const reportFinalF = JSON.parse(finalFRaw) as ReportData;
