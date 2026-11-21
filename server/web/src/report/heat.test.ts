import { blur, ramp, RAMPS, smoothHeat, squareCount } from './heat';

const grid = (n: number, f: (r: number, c: number) => number) => Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => f(r, c)));
const impulse = (n: number, at: number) => grid(n, (r, c) => (r === at && c === at ? 1 : 0));
const sum = (g: number[][]) => g.flat().reduce((a, b) => a + b, 0);

describe('smoothHeat', () => {
  test('is a Gaussian with sigma = n/16 (2.5 cells for the 40 x 40 grid)', () => {
    const out = smoothHeat(impulse(40, 20));
    const g = (d: number) => Math.exp(-0.5 * (d / 2.5) ** 2);
    expect(out[20][21] / out[20][20]).toBeCloseTo(g(1), 12);
    expect(out[20][23] / out[20][20]).toBeCloseTo(g(3), 12);
    expect(out[23][24] / out[20][20]).toBeCloseTo(g(3) * g(4), 12);
    expect(out).toEqual(blur(impulse(40, 20), 2.5));
  });

  test('has a radius of 8 cells', () => {
    const out = smoothHeat(impulse(40, 20));
    expect(out[20][28]).toBeGreaterThan(0);
    expect(out[20][12]).toBeGreaterThan(0);
    expect(out[20][29]).toBe(0);
    expect(out[11][20]).toBe(0);
    expect(out[28][28]).toBeGreaterThan(0);
  });

  test('keeps the mass of a spot away from the edges, symmetric', () => {
    const out = smoothHeat(impulse(40, 20));
    expect(sum(out)).toBeCloseTo(1, 12);
    expect(out[20][15]).toBeCloseTo(out[20][25], 15);
    expect(out[15][20]).toBeCloseTo(out[20][15], 15);
  });

  test('fades at the edges: outside cells count 0, the divisor stays the full kernel sum', () => {
    const out = smoothHeat(grid(40, () => 1));
    const k = Array.from({ length: 17 }, (_, i) => Math.exp(-0.5 * ((i - 8) / 2.5) ** 2));
    const inside = k.slice(8).reduce((a, b) => a + b, 0) / k.reduce((a, b) => a + b, 0);
    expect(out[20][20]).toBeCloseTo(1, 12);
    expect(out[0][20]).toBeCloseTo(inside, 12);
    expect(out[0][0]).toBeCloseTo(inside * inside, 12);
    expect(out[0][0]).toBeLessThan(out[0][20]);
  });

  test('scales the kernel with the grid: a 16-cell grid has sigma 1 and radius 3', () => {
    const out = smoothHeat(impulse(16, 8));
    expect(out[8][9] / out[8][8]).toBeCloseTo(Math.exp(-0.5), 12);
    expect(out[8][11]).toBeGreaterThan(0);
    expect(out[8][12]).toBe(0);
  });

  test('leaves an all-zero grid at zero', () => {
    expect(sum(smoothHeat(grid(40, () => 0)))).toBe(0);
  });
});

describe('squareCount', () => {
  test('sums the 4 x 4 cells starting 2 up and left of the pointer cell', () => {
    const one = grid(40, (r, c) => (r === 10 && c === 10 ? 5 : 0));
    expect(squareCount(one, 9, 9)).toBe(5);
    expect(squareCount(one, 12, 12)).toBe(5);
    expect(squareCount(one, 8, 8)).toBe(0);
    expect(squareCount(one, 13, 13)).toBe(0);
    expect(squareCount(one, 12, 8)).toBe(0);
  });

  test('counts cells outside the grid as 0', () => {
    const ones = grid(40, () => 1);
    expect(squareCount(ones, 20, 20)).toBe(16);
    expect(squareCount(ones, 0, 0)).toBe(4);
    expect(squareCount(ones, 39, 39)).toBe(9);
  });
});

test('ramp interpolates linearly between stops and clamps', () => {
  const stops = RAMPS.dark.T_1;
  expect(ramp(stops, 0)).toEqual([92, 107, 0, 0]);
  expect(ramp(stops, -1)).toEqual([92, 107, 0, 0]);
  expect(ramp(stops, 1)).toEqual([251, 255, 224, 250]);
  expect(ramp(stops, 2)).toEqual([251, 255, 224, 250]);
  expect(ramp(stops, 0.07)).toEqual([96, 112.5, 2, 75]);
});
