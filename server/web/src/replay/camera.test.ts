import { MathUtils, Vector3 } from 'three';
import { TWEEN_MS, easeInOutCubic, fixedViews, floorTarget, followGoal, fovFor, tweenAt, type Tween } from './camera';
import { makeSceneFixture } from './fixtures';

describe('broadcast view', () => {
  const cam = makeSceneFixture().camera;

  it('orbits around the point where the camera centre ray meets the floor', () => {
    const target = floorTarget(cam);
    expect(target.y).toBeCloseTo(0);
    const ray = target.clone().sub(new Vector3(...cam.pos)).normalize();
    const forward = new Vector3(...cam.back).negate();
    expect(ray.distanceTo(forward)).toBeLessThan(1e-9);
  });

  it('falls back to the court centre when the camera does not look down', () => {
    expect(floorTarget({ ...cam, back: [0, -0.2, 0.98] }).toArray()).toEqual([0, 0, 0]);
  });

  it('is the calibrated camera; the other views are fixed', () => {
    const views = fixedViews(cam);
    expect(views.broadcast.pos.toArray()).toEqual(cam.pos);
    expect(views.broadcast.fov).toBe(cam.fov);
    expect(views.broadcast.match).toBe(true);
    expect(views.top.pos.toArray()).toEqual([0, 34, 0.01]);
  });
});

describe('fovFor', () => {
  const view = { pos: new Vector3(), target: new Vector3(), fov: 30, match: true };

  it('keeps the fov on a viewport at least as wide as the video', () => {
    expect(fovFor(view, 2, 16 / 9)).toBe(30);
    expect(fovFor({ ...view, match: false }, 0.5, 16 / 9)).toBe(30);
  });

  it('widens the fov on a narrower viewport so the whole frame fits, like object-fit: contain', () => {
    const fov = fovFor(view, 1, 16 / 9);
    const half = Math.tan(MathUtils.degToRad(fov) / 2);
    expect(half).toBeCloseTo(Math.tan(MathUtils.degToRad(15)) * (16 / 9));
  });
});

describe('view tweens', () => {
  it('ease in and out over 0..1', () => {
    expect(easeInOutCubic(0)).toBe(0);
    expect(easeInOutCubic(0.5)).toBe(0.5);
    expect(easeInOutCubic(1)).toBe(1);
    expect(easeInOutCubic(0.25)).toBeLessThan(0.25);
    expect(easeInOutCubic(0.75)).toBeGreaterThan(0.75);
  });

  const tw: Tween = {
    p0: new Vector3(0, 0, 0), t0: new Vector3(0, 0, 0), f0: 30,
    p1: new Vector3(10, 20, 30), t1: new Vector3(2, 0, 0), f1: 50, start: 1000,
  };

  it('start at the old camera, reach the new one after 900 ms, and clamp outside that', () => {
    const pos = new Vector3();
    const target = new Vector3();
    expect(tweenAt(tw, 900, pos, target)).toEqual({ fov: 30, done: false });
    expect(pos.toArray()).toEqual([0, 0, 0]);
    expect(tweenAt(tw, 1000 + TWEEN_MS / 2, pos, target)).toEqual({ fov: 40, done: false });
    expect(pos.toArray()).toEqual([5, 10, 15]);
    expect(target.toArray()).toEqual([1, 0, 0]);
    expect(tweenAt(tw, 1000 + TWEEN_MS + 50, pos, target)).toEqual({ fov: 50, done: true });
    expect(pos.toArray()).toEqual([10, 20, 30]);
  });
});

describe('follow view', () => {
  it('keeps the orbit direction 9 m from the ball', () => {
    const goal = followGoal(new Vector3(0, 10, 20), new Vector3(0, 0, 0), new Vector3(1, 1, 1));
    expect(goal.target.toArray()).toEqual([1, 1, 1]);
    expect(goal.pos.distanceTo(goal.target)).toBeCloseTo(9);
    expect(goal.pos.clone().sub(goal.target).normalize().distanceTo(new Vector3(0, 10, 20).normalize())).toBeLessThan(1e-9);
  });

  it('stays at least 2.5 m above the ball and aims mid-court without a ball', () => {
    const goal = followGoal(new Vector3(0, 0.1, 20), new Vector3(0, 0, 0), null);
    expect(goal.target.toArray()).toEqual([0, 1, 0]);
    expect(goal.pos.y - goal.target.y).toBe(2.5);
  });
});
