import type { SceneData } from '../types/scene';
import {
  NJ, NONE, ballAt, bounceRange, decodeScene, flightAtFrame, flightPoints, poseAt, sortedBounces, toInt16, type XYZ,
} from './decode';
import { makeSceneFixture } from './fixtures';

const xyz = (): XYZ => ({ x: 0, y: 0, z: 0 });
const joints = () => Array.from({ length: NJ }, xyz);

/** n = 4 at 10 fps: frames 0-1 flight 0, frame 2 flight 1, frame 3 none. One player present on frames 0-1. */
function tinyScene(): SceneData {
  const base = makeSceneFixture();
  const pose = new Array<number>(4 * NJ * 3).fill(-32000);
  for (let j = 0; j < NJ; j++) {
    pose.splice(j * 3, 3, 0, 100 + j, 0);
    pose.splice((NJ + j) * 3, 3, 100, 200 + j, -100);
  }
  return {
    ...base,
    fps: 10,
    n: 4,
    ball: [100, 200, 300, 0, 200, 400, 500, 0, 0, 0, 0, 1, -32000, -32000, -32000, -32000],
    players: [{ id: 'P_1', team: 'T_1', hand: 'right', pose }],
    flights: base.flights.slice(0, 2),
  };
}

describe('toInt16', () => {
  it('turns the payload marker into NONE and clamps everything else into int16 without hitting the marker', () => {
    const out = toInt16([12, -32000, 40000, -40000, 1.6, -31999.6], -32000);
    expect(out).toBeInstanceOf(Int16Array);
    expect(Array.from(out)).toEqual([12, NONE, 32767, NONE + 1, 2, NONE + 1]);
  });

  it('uses the marker the payload declares', () => {
    expect(Array.from(toInt16([-1, 5], -1))).toEqual([NONE, 5]);
  });
});

describe('decodeScene', () => {
  it('converts the ball and every pose to Int16Array once per scene object', () => {
    const scene = makeSceneFixture();
    const d = decodeScene(scene);
    expect(decodeScene(scene)).toBe(d);
    expect(decodeScene(makeSceneFixture())).not.toBe(d);
    expect(d.ball).toBeInstanceOf(Int16Array);
    expect(d.ball).toHaveLength(scene.n * 4);
    expect(d.poses).toHaveLength(4);
    for (const p of d.poses) {
      expect(p).toBeInstanceOf(Int16Array);
      expect(p).toHaveLength(scene.n * NJ * 3);
    }
    expect(d.dur).toBeCloseTo(10);
  });
});

describe('frame and flight lookup', () => {
  const d = decodeScene(tinyScene());

  it('reads the flight that owns a frame', () => {
    expect([0, 1, 2, 3].map((f) => flightAtFrame(d, f))).toEqual([0, 0, 1, -1]);
    expect(flightAtFrame(d, -1)).toBe(-1);
    expect(flightAtFrame(d, 4)).toBe(-1);
  });

  it('decodes the ball in metres and interpolates within a flight only', () => {
    const out = xyz();
    expect(ballAt(d, 0, out)).toBe(0);
    expect(out).toEqual({ x: 1, y: 2, z: 3 });
    expect(ballAt(d, 0.5, out)).toBe(0);
    expect(out).toEqual({ x: 1.5, y: 3, z: 4 });
    expect(ballAt(d, 1.5, out)).toBe(0); // the next frame is another flight: no blending
    expect(out).toEqual({ x: 2, y: 4, z: 5 });
    expect(ballAt(d, 3, out)).toBe(-1);
    expect(ballAt(d, -0.5, out)).toBe(-1);
    expect(ballAt(d, 4, out)).toBe(-1);
  });

  it('decodes poses, interpolates them and hides a frame with no pose', () => {
    const out = joints();
    expect(poseAt(d, 0, 0.5, out)).toBe(true);
    expect(out[0]).toEqual({ x: 0.5, y: 1.5, z: -0.5 });
    expect(out[12].y).toBeCloseTo(1.62);
    expect(poseAt(d, 0, 2, out)).toBe(false);
    expect(poseAt(d, 0, 1.5, out)).toBe(true); // next frame missing: holds frame 1
    expect(out[0]).toEqual({ x: 1, y: 2, z: -1 });
  });

  it('lists the points of one flight from its own frames', () => {
    expect(flightPoints(d, 0, { ...tinyScene().flights[0], t0: 0, t1: 0.1 })).toEqual([1, 2, 3, 2, 4, 5]);
  });
});

describe('the fixture follows the ownership rules of the contract', () => {
  const scene = makeSceneFixture();
  const d = decodeScene(scene);

  it('gives a shared frame to the later flight and leaves gaps empty', () => {
    expect(flightAtFrame(d, 12)).toBe(-1); // before ceil(0.5 * 25)
    expect(flightAtFrame(d, 13)).toBe(0);
    expect(flightAtFrame(d, 64)).toBe(0);
    expect(flightAtFrame(d, 65)).toBe(1); // 2.6 s: end of flight 0 and start of flight 1
    expect(flightAtFrame(d, 111)).toBe(-1);
    expect(flightAtFrame(d, 150)).toBe(2);
  });

  it('builds the path of a flight from its frames only', () => {
    const pts = flightPoints(d, 0, scene.flights[0]);
    expect(pts.length / 3).toBe(64 - 13 + 1);
  });

  it('has a player who is absent at the start', () => {
    expect(poseAt(d, 3, 0, joints())).toBe(false);
    expect(poseAt(d, 3, 12, joints())).toBe(true);
  });
});

describe('bounces', () => {
  const bounces = sortedBounces(makeSceneFixture().flights);

  it('are flattened in time order with their flight quality', () => {
    expect(bounces.map((b) => b.t)).toEqual([1.6, 2.0, 3.7, 7.2, 7.8]);
    expect(bounces.map((b) => b.ok)).toEqual([true, true, false, true, true]);
  });

  it('are found by time window without a scan', () => {
    expect(bounceRange(bounces, 2.1, 0.8)).toEqual([0, 2]);
    expect(bounceRange(bounces, 1.5, 0.8)).toEqual([0, 0]);
    expect(bounceRange(bounces, 8.5, 0.8)).toEqual([4, 5]);
    expect(bounceRange(bounces, 9.5, 0.8)).toEqual([5, 5]);
  });
});
