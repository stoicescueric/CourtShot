import type { Theme } from '../theme';
import type { TeamId } from '../types/report';

export type Rgba = [number, number, number, number];
export type Ramp = [number, Rgba][];

/**
 * One hue per team, from barely there to the busiest spot. Dark theme: deep -> bright -> near white;
 * light theme the other way round. Keep in sync with --heat1 and --heat2 in styles/theme.css.
 */
export const RAMPS: Record<Theme, Record<TeamId, Ramp>> = {
  dark: {
    T_1: [[0, [92, 107, 0, 0]], [0.14, [100, 118, 4, 150]], [0.55, [212, 245, 66, 228]], [1, [251, 255, 224, 250]]],
    T_2: [[0, [36, 72, 160, 0]], [0.14, [40, 84, 186, 160]], [0.55, [77, 141, 255, 230]], [1, [227, 236, 255, 250]]],
  },
  light: {
    T_1: [[0, [212, 245, 66, 0]], [0.14, [205, 238, 60, 150]], [0.6, [147, 170, 20, 225]], [1, [92, 107, 0, 248]]],
    T_2: [[0, [160, 196, 255, 0]], [0.14, [150, 190, 255, 160]], [0.6, [77, 141, 255, 228]], [1, [23, 70, 180, 248]]],
  },
};

/** Linear interpolation between the stops; t is clamped to 0..1. */
export function ramp(stops: Ramp, t: number): Rgba {
  const u = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    const [b, cb] = stops[i];
    if (u <= b) {
      const [a, ca] = stops[i - 1];
      const k = (u - a) / (b - a);
      return ca.map((v, j) => v + (cb[j] - v) * k) as Rgba;
    }
  }
  return stops[stops.length - 1][1];
}

/**
 * Separable Gaussian blur with radius ceil(3 sigma). Cells outside the grid count as 0 while the
 * divisor stays the full kernel sum, so the edges fade.
 */
export function blur(grid: number[][], sigma: number): number[][] {
  const r = Math.ceil(3 * sigma);
  const k = Array.from({ length: 2 * r + 1 }, (_, i) => Math.exp(-0.5 * ((i - r) / sigma) ** 2));
  const ks = k.reduce((a, b) => a + b, 0);
  const pass = (src: number[][], horizontal: boolean) =>
    src.map((row, y) =>
      row.map((_, x) => {
        let s = 0;
        for (let i = -r; i <= r; i++) {
          const v = horizontal ? row[x + i] : src[y + i]?.[x];
          if (v !== undefined) s += v * k[i + r];
        }
        return s / ks;
      }),
    );
  return pass(pass(grid, true), false);
}

/** The report's smoothing: sigma = n/16 (2.5 cells, radius 8, for the 40 x 40 grid). */
export const smoothHeat = (heat: number[][]): number[][] => blur(heat, heat.length / 16);

/**
 * Raw counts in the 1 m square under the pointer cell (cx, cy): the 2·half x 2·half cells starting
 * `half` up and left of it, half = round(n / 20). Cells outside the grid count 0.
 */
export function squareCount(heat: number[][], cx: number, cy: number): number {
  const half = Math.round(heat.length / 20);
  let s = 0;
  for (let y = cy - half; y < cy + half; y++) {
    for (let x = cx - half; x < cx + half; x++) s += heat[y]?.[x] ?? 0;
  }
  return s;
}
