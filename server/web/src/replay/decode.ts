import type { Bounce, Flight, SceneData } from '../types/scene';

/** Joints per player per frame (scene-data.md, ScenePlayer.pose). */
export const NJ = 13;
/** The "no value" marker in the decoded Int16 arrays. */
export const NONE = -32000;
const INT16_MAX = 32767;

export interface XYZ {
  x: number;
  y: number;
  z: number;
}

export interface DecodedScene {
  fps: number;
  n: number;
  /** Viewer duration, seconds: (n - 1) / fps. */
  dur: number;
  /** n*4: x, y, z in cm, then the owning flight index; all NONE when no flight owns the frame. */
  ball: Int16Array;
  /** One per scene.players entry, n*13*3, cm. */
  poses: Int16Array[];
}

/** A bounce with the flight it belongs to, for the ring pool. */
export interface TimedBounce extends Bounce {
  ok: boolean;
}

/**
 * Converts a JSON int array to Int16. The payload's marker becomes NONE; every other value is clamped
 * so it can neither wrap around nor collide with the marker (pose heights have no upper bound).
 */
export function toInt16(values: ArrayLike<number>, none: number): Int16Array {
  const out = new Int16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    out[i] = v === none ? NONE : Math.max(NONE + 1, Math.min(INT16_MAX, Math.round(v)));
  }
  return out;
}

const decoded = new WeakMap<SceneData, DecodedScene>();

/** Typed-array view of a scene, built once per scene object. */
export function decodeScene(scene: SceneData): DecodedScene {
  let d = decoded.get(scene);
  if (!d) {
    d = {
      fps: scene.fps,
      n: scene.n,
      dur: Math.max(0, (scene.n - 1) / scene.fps),
      ball: toInt16(scene.ball, scene.none),
      poses: scene.players.map((p) => toInt16(p.pose, scene.none)),
    };
    decoded.set(scene, d);
  }
  return d;
}

/** Flight index that owns a whole frame, or -1. */
export function flightAtFrame(d: DecodedScene, frame: number): number {
  if (frame < 0 || frame >= d.n) return -1;
  const v = d.ball[frame * 4 + 3];
  return v === NONE ? -1 : v;
}

/**
 * Ball position at a (fractional) frame into `out`, metres. Interpolates linearly to the next frame,
 * never across two flights. Returns the flight index, or -1 when no flight owns the frame.
 */
export function ballAt(d: DecodedScene, fr: number, out: XYZ): number {
  const f0 = Math.floor(fr);
  if (f0 < 0 || f0 >= d.n) return -1;
  const B = d.ball;
  const k0 = f0 * 4;
  if (B[k0] === NONE) return -1;
  out.x = B[k0] / 100;
  out.y = B[k0 + 1] / 100;
  out.z = B[k0 + 2] / 100;
  const a = fr - f0;
  const k1 = Math.min(f0 + 1, d.n - 1) * 4;
  if (a > 0 && B[k1] !== NONE && B[k1 + 3] === B[k0 + 3]) {
    out.x += (B[k1] / 100 - out.x) * a;
    out.y += (B[k1 + 1] / 100 - out.y) * a;
    out.z += (B[k1 + 2] / 100 - out.z) * a;
  }
  return B[k0 + 3];
}

/**
 * The 13 joints of a player at a (fractional) frame into `out`, metres. False when the frame has no pose
 * (the viewer hides the player). A joint missing at the next frame is not interpolated.
 */
export function poseAt(d: DecodedScene, player: number, fr: number, out: XYZ[]): boolean {
  const f0 = Math.floor(fr);
  if (f0 < 0 || f0 >= d.n) return false;
  const P = d.poses[player];
  const a = fr - f0;
  const f1 = Math.min(f0 + 1, d.n - 1);
  for (let j = 0; j < NJ; j++) {
    const k = (f0 * NJ + j) * 3;
    if (P[k] === NONE) return false;
    const o = out[j];
    o.x = P[k] / 100;
    o.y = P[k + 1] / 100;
    o.z = P[k + 2] / 100;
    const k1 = (f1 * NJ + j) * 3;
    if (a > 0 && P[k1] !== NONE) {
      o.x += (P[k1] / 100 - o.x) * a;
      o.y += (P[k1 + 1] / 100 - o.y) * a;
      o.z += (P[k1 + 2] / 100 - o.z) * a;
    }
  }
  return true;
}

/**
 * Flat x, y, z points (metres) of the frames owned by one flight. Scans only the flight's own frame span
 * (ceil(t0*fps)..floor(t1*fps), one frame of slack each side), never the whole match.
 */
export function flightPoints(d: DecodedScene, index: number, flight: Flight): number[] {
  const pts: number[] = [];
  const from = Math.max(0, Math.floor(flight.t0 * d.fps) - 1);
  const to = Math.min(d.n - 1, Math.ceil(flight.t1 * d.fps) + 1);
  for (let f = from; f <= to; f++) {
    const k = f * 4;
    if (d.ball[k] !== NONE && d.ball[k + 3] === index) pts.push(d.ball[k] / 100, d.ball[k + 1] / 100, d.ball[k + 2] / 100);
  }
  return pts;
}

/** Every bounce of every flight, sorted by time. */
export function sortedBounces(flights: Flight[]): TimedBounce[] {
  return flights.flatMap((fl) => fl.bounces.map((b) => ({ ...b, ok: fl.ok }))).sort((a, b) => a.t - b.t);
}

/** Index range [lo, hi) of the sorted bounces with t - maxAge <= b.t <= t (binary search). */
export function bounceRange(bounces: TimedBounce[], t: number, maxAge: number): [number, number] {
  let lo = 0;
  let hi = bounces.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (bounces[mid].t < t - maxAge) lo = mid + 1;
    else hi = mid;
  }
  let end = lo;
  while (end < bounces.length && bounces[end].t <= t) end++;
  return [lo, end];
}
