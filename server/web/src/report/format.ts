/** Number formatting of the report. Missing values (null, undefined, NaN, ±Infinity) print "–". */

type Maybe = number | null | undefined;

export const DASH = '–';

/** The reference also accepted ±Infinity, which printed "Infinity%" for a zero-length video. */
export const has = (v: Maybe): v is number => typeof v === 'number' && Number.isFinite(v);

export const num = (v: Maybe, digits = 0, unit = ''): string => (has(v) ? v.toFixed(digits) + unit : DASH);

/** A share 0..1 as a rounded whole percentage. */
export const pct = (v: Maybe): string => (has(v) ? `${Math.round(100 * v)}%` : DASH);

/** m/s to km/h. */
export const kmh = (ms: Maybe): number | null => (has(ms) ? ms * 3.6 : null);

/** m:ss from rounded, non-negative seconds, without hours. */
export function clock(s: Maybe): string {
  if (!has(s)) return DASH;
  const t = Math.max(0, Math.round(s));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
}

/** "x.x min" from 90 s, else "N s". */
export function dur(s: Maybe): string {
  if (!has(s)) return DASH;
  return s >= 90 ? `${(s / 60).toFixed(1)} min` : `${Math.round(s)} s`;
}
