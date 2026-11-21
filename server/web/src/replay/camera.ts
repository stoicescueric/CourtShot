import { MathUtils, Vector3 } from 'three';
import type { BroadcastCamera } from '../types/scene';

export const VIEW_NAMES = ['broadcast', 'far', 'side', 'top', 'follow'] as const;
export type ViewName = (typeof VIEW_NAMES)[number];
export type FixedView = Exclude<ViewName, 'follow'>;

export interface ViewSpec {
  pos: Vector3;
  target: Vector3;
  /** Vertical field of view, degrees. */
  fov: number;
  /** Widen the fov on viewports narrower than the video, like object-fit: contain. */
  match: boolean;
}

export const TWEEN_MS = 900;
export const FOLLOW_FOV = 45;
const FOLLOW_DISTANCE = 9;
const FOLLOW_MIN_HEIGHT = 2.5;

/** Where the broadcast camera's centre ray meets the floor; the court centre if it never does. */
export function floorTarget(cam: BroadcastCamera): Vector3 {
  const pos = new Vector3(...cam.pos);
  const back = new Vector3(...cam.back);
  // the camera looks along -back, so it looks down when back.y > 0
  if (back.y < 1e-3 || pos.y <= 0) return new Vector3(0, 0, 0);
  return pos.addScaledVector(back, -pos.y / back.y);
}

export function fixedViews(cam: BroadcastCamera): Record<FixedView, ViewSpec> {
  const centre = new Vector3(0, 0.6, 0);
  return {
    broadcast: { pos: new Vector3(...cam.pos), target: floorTarget(cam), fov: cam.fov, match: true },
    far: { pos: new Vector3(0, 7.5, -27), target: centre.clone(), fov: 34, match: false },
    side: { pos: new Vector3(21, 7, 0.5), target: centre.clone(), fov: 38, match: false },
    top: { pos: new Vector3(0, 34, 0.01), target: new Vector3(0, 0, 0), fov: 40, match: false },
  };
}

export function fovFor(view: ViewSpec, aspect: number, videoAspect: number): number {
  if (!view.match || !(aspect < videoAspect)) return view.fov;
  const half = Math.tan(MathUtils.degToRad(view.fov) / 2) * (videoAspect / aspect);
  return MathUtils.radToDeg(2 * Math.atan(half));
}

/** The follow view's goal: keep the orbit direction, 9 m from the ball, at least 2.5 m up. */
export function followGoal(cameraPos: Vector3, orbitTarget: Vector3, ball: Vector3 | null): { pos: Vector3; target: Vector3 } {
  const target = ball ? ball.clone() : new Vector3(0, 1, 0);
  const off = cameraPos.clone().sub(orbitTarget).setLength(FOLLOW_DISTANCE);
  if (off.y < FOLLOW_MIN_HEIGHT) off.y = FOLLOW_MIN_HEIGHT;
  return { pos: target.clone().add(off), target };
}

export function easeInOutCubic(k: number): number {
  return k < 0.5 ? 4 * k * k * k : 1 - (-2 * k + 2) ** 3 / 2;
}

export interface Tween {
  p0: Vector3;
  t0: Vector3;
  f0: number;
  p1: Vector3;
  t1: Vector3;
  f1: number;
  /** performance.now() at the start. */
  start: number;
}

/** Writes the tweened camera position and orbit target; returns the fov and whether the tween is over. */
export function tweenAt(tw: Tween, now: number, pos: Vector3, target: Vector3): { fov: number; done: boolean } {
  const k = MathUtils.clamp((now - tw.start) / TWEEN_MS, 0, 1);
  const e = easeInOutCubic(k);
  pos.lerpVectors(tw.p0, tw.p1, e);
  target.lerpVectors(tw.t0, tw.t1, e);
  return { fov: tw.f0 + (tw.f1 - tw.f0) * e, done: k >= 1 };
}
