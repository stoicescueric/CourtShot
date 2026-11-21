import { clock, dur, has, kmh, num, pct } from './format';

test('has accepts finite numbers only', () => {
  expect(has(0)).toBe(true);
  expect(has(-2.5)).toBe(true);
  for (const v of [null, undefined, NaN, Infinity, -Infinity]) expect(has(v)).toBe(false);
});

test('pct rounds a share to a whole percentage', () => {
  expect(pct(0.456)).toBe('46%');
  expect(pct(0)).toBe('0%');
  expect(pct(1)).toBe('100%');
  for (const v of [null, undefined, NaN, Infinity]) expect(pct(v)).toBe('–');
});

test('num prints the digits and unit, or a dash', () => {
  expect(num(3.14159, 2, ' m')).toBe('3.14 m');
  expect(num(5)).toBe('5');
  expect(num(0, 1, ' km/h')).toBe('0.0 km/h');
  expect(num(null, 1, ' m')).toBe('–');
  expect(num(NaN)).toBe('–');
  expect(num(undefined)).toBe('–');
});

test('kmh converts m/s and keeps missing values missing', () => {
  expect(kmh(10)).toBeCloseTo(36);
  expect(kmh(null)).toBeNull();
  expect(kmh(NaN)).toBeNull();
});

test('clock gives m:ss from rounded, non-negative seconds, without hours', () => {
  expect(clock(0)).toBe('0:00');
  expect(clock(65.4)).toBe('1:05');
  expect(clock(59.6)).toBe('1:00');
  expect(clock(-3)).toBe('0:00');
  expect(clock(3725)).toBe('62:05');
  expect(clock(null)).toBe('–');
  expect(clock(NaN)).toBe('–');
});

test('dur switches to minutes from 90 s', () => {
  expect(dur(0)).toBe('0 s');
  expect(dur(89.4)).toBe('89 s');
  expect(dur(90)).toBe('1.5 min');
  expect(dur(754)).toBe('12.6 min');
  expect(dur(null)).toBe('–');
  expect(dur(NaN)).toBe('–');
});
