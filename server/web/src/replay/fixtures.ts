import type { Bounce, Flight, Hit, SceneData, ScenePlayer, Vec3m } from '../types/scene';
import finalFRaw from './final_f.scene.json?raw';

const FPS = 25;
const N = 251; // 10 s
const NONE = -32000;
const NJ = 13;

/** Standing pose relative to the floor point, metres: [across, height]. Joint order as ScenePlayer.pose. */
const STANDING: [number, number][] = [
  [0, 1.72], [-0.2, 1.45], [0.2, 1.45], [-0.28, 1.15], [0.28, 1.15], [-0.3, 0.88], [0.3, 0.88],
  [-0.14, 0.95], [0.14, 0.95], [-0.15, 0.5], [0.15, 0.5], [-0.16, 0.08], [0.16, 0.08],
];

const cm = (m: number) => Math.round(m * 100);

/** One player swaying around a home spot; absent (all none) on frames where `present` is false. */
function player(id: string, team: string, hand: 'left' | 'right', home: [number, number], present: (f: number) => boolean): ScenePlayer {
  const pose = new Array<number>(N * NJ * 3).fill(NONE);
  for (let f = 0; f < N; f++) {
    if (!present(f)) continue;
    const s = f / FPS;
    const x = home[0] + 0.8 * Math.sin(s * 0.9 + home[0]);
    const z = home[1] + 0.5 * Math.cos(s * 0.7);
    STANDING.forEach(([dx, h], j) => {
      const k = (f * NJ + j) * 3;
      pose[k] = cm(x + dx);
      pose[k + 1] = cm(h);
      pose[k + 2] = cm(z);
    });
  }
  return { id, team, hand, pose };
}

/** A flight through keypoints [t, x, y, z] (viewer frame, metres), arcing `lift` metres between them. */
interface Leg {
  to: [number, number, number, number];
  lift: number;
  bounce?: Bounce['kind'];
}

function flight(start: [number, number, number, number], legs: Leg[], hitter: string | null, ok: boolean, speed: number): {
  flight: Flight;
  at: (t: number) => Vec3m;
} {
  const at = (t: number): Vec3m => {
    let from = start;
    for (const leg of legs) {
      if (t <= leg.to[0]) {
        const u = (t - from[0]) / (leg.to[0] - from[0]);
        return [
          from[1] + (leg.to[1] - from[1]) * u,
          from[2] + (leg.to[2] - from[2]) * u + 4 * leg.lift * u * (1 - u),
          from[3] + (leg.to[3] - from[3]) * u,
        ];
      }
      from = leg.to;
    }
    return [from[1], from[2], from[3]];
  };
  const t0 = start[0];
  const t1 = legs[legs.length - 1].to[0];
  const nearStart = start[3] > 0;
  let maxH = 0;
  let netH: number | null = null;
  for (let i = 0; i <= 240; i++) {
    const p = at(t0 + ((t1 - t0) * i) / 240);
    maxH = Math.max(maxH, p[1]);
    if (netH === null && p[2] > 0 !== nearStart) netH = Math.round(p[1] * 100) / 100;
  }
  const bounces = legs
    .filter((l) => l.bounce)
    .map((l): Bounce => ({ kind: l.bounce as Bounce['kind'], t: l.to[0], p: [l.to[1], l.to[2], l.to[3]] }));
  return {
    flight: {
      t0, t1, ok, hitter, speed_kmh: speed, max_h: Math.round(maxH * 100) / 100, net_h: netH,
      rms_px: ok ? 2.1 : 7.4, h_sd: ok ? 0.12 : null, bounces,
    },
    at,
  };
}

/**
 * A small synthetic scene: 10 s at 25 fps, four players (P_4 enters at frame 12), three flights in two
 * rallies with ground and wall bounces; the second flight is low confidence.
 */
export function makeSceneFixture(): SceneData {
  const players = [
    player('P_1', 'T_1', 'right', [-2.5, 7], () => true),
    player('P_2', 'T_1', 'left', [2.5, 7.5], () => true),
    player('P_3', 'T_2', 'right', [-2.5, -7], () => true),
    player('P_4', 'T_2', 'right', [2.5, -7.5], (f) => f >= 12),
  ];
  const fits = [
    flight([0.5, -2.5, 1.0, 7], [
      { to: [1.6, 1.0, 0, -6], lift: 1.6, bounce: 'ground' },
      { to: [2.0, 1.8, 1.2, -10], lift: 0.5, bounce: 'wall' },
      { to: [2.6, 2.5, 1.0, -7.5], lift: 0.6 },
    ], 'P_1', true, 74.3),
    flight([2.6, 2.5, 1.0, -7.5], [
      { to: [3.7, -1.5, 0, 5], lift: 2.4, bounce: 'ground' },
      { to: [4.4, -2.5, 1.0, 7.2], lift: 0.8 },
    ], 'P_4', false, 61.8),
    flight([6.0, 2.5, 1.0, 7.5], [
      { to: [7.2, -1.5, 0, -6.5], lift: 1.2, bounce: 'ground' },
      { to: [7.8, -5, 1.0, -8], lift: 0.7, bounce: 'wall' },
      { to: [8.5, -3, 0.4, -9], lift: 0.3 },
    ], 'P_2', true, 82.6),
  ];
  // frames ceil(t0*fps)..floor(t1*fps) belong to the flight; a later flight wins a shared frame
  const ball = new Array<number>(N * 4).fill(NONE);
  fits.forEach(({ flight: fl, at }, i) => {
    for (let f = Math.ceil(fl.t0 * FPS); f <= Math.min(N - 1, Math.floor(fl.t1 * FPS)); f++) {
      const p = at(f / FPS);
      ball.splice(f * 4, 4, cm(p[0]), cm(p[1]), cm(p[2]), i);
    }
  });
  const hits: Hit[] = [
    { t: 0.5, player: 'P_1' },
    { t: 2.6, player: 'P_4' },
    { t: 4.4, player: null },
    { t: 6.0, player: 'P_2' },
  ];
  return {
    schema_version: 1,
    fps: FPS,
    n: N,
    camera: broadcastCamera([0, 6.5, 19], [0, 0, 1], 30, 16 / 9),
    players,
    ball,
    flights: fits.map((f) => f.flight),
    hits,
    rallies: [[0.5, 4.4], [6.0, 8.5]],
    none: NONE,
    video: null,
  };
}

/** A camera at `pos` looking down the court (towards -z) at `target`, no roll; back = -forward. */
function broadcastCamera(pos: Vec3m, target: Vec3m, fov: number, aspect: number): SceneData['camera'] {
  const f = target.map((v, i) => v - pos[i]);
  const len = Math.hypot(...f);
  const back: Vec3m = [-f[0] / len, -f[1] / len, -f[2] / len];
  const right: Vec3m = [1, 0, 0];
  const up: Vec3m = [0, back[2], -back[1]]; // back x right
  return { pos, right, up, back, fov, aspect };
}

/** scene.json of the end-to-end run on videos/final_f.mp4 (2252 frames, 71 flights, 64 fitted), as the worker wrote it. */
export const sceneFinalF = (): SceneData => JSON.parse(finalFRaw) as SceneData;
