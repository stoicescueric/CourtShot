/** A stretch of the clip shown by the timeline, [start_s, end_s]. */
export type TimeWindow = [number, number];

export const SCRUB_MAX = 1000;

/**
 * Splits 0..dur into one window per rally, cut halfway through each gap between rallies, so every
 * rally fills most of the lane however long the clip is. No rallies: one window for the whole clip.
 */
export function rallyWindows(rallies: [number, number][], dur: number): TimeWindow[] {
  const sorted = rallies
    .map(([s, e]): TimeWindow => [Math.max(0, Math.min(s, dur)), Math.max(0, Math.min(e, dur))])
    .sort((a, b) => a[0] - b[0]);
  if (sorted.length === 0 || dur <= 0) return [[0, Math.max(0, dur)]];
  const cuts = [0];
  for (let i = 1; i < sorted.length; i++) {
    const mid = (sorted[i - 1][1] + sorted[i][0]) / 2;
    cuts.push(Math.min(dur, Math.max(cuts[cuts.length - 1], mid)));
  }
  cuts.push(dur);
  return cuts.slice(1).map((end, i): TimeWindow => [cuts[i], end]);
}

/** The window holding t: the last one that starts at or before t. */
export function windowIndexAt(t: number, windows: TimeWindow[]): number {
  let i = 0;
  while (i + 1 < windows.length && windows[i + 1][0] <= t) i++;
  return i;
}

/** Position of t in a window, percent, clamped to 0..100. */
export function percentIn(t: number, [w0, w1]: TimeWindow): number {
  if (w1 <= w0) return 0;
  return Math.min(100, Math.max(0, ((t - w0) / (w1 - w0)) * 100));
}

/** Left and width (percent) of [t0, t1] clipped to a window; null when it lies outside. */
export function spanBox(t0: number, t1: number, w: TimeWindow): { left: number; width: number } | null {
  if (t1 < w[0] || t0 > w[1]) return null;
  const left = percentIn(t0, w);
  return { left, width: percentIn(t1, w) - left };
}

export function timeFromScrub(value: number, [w0, w1]: TimeWindow): number {
  return w0 + (value / SCRUB_MAX) * (w1 - w0);
}

export function scrubValue(t: number, w: TimeWindow): number {
  return Math.round((percentIn(t, w) / 100) * SCRUB_MAX);
}
