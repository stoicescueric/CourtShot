import { bytes, clock, duration, percent, relativeTime } from './format';

test('clock is m:ss of rounded, non-negative seconds', () => {
  expect(clock(0)).toBe('0:00');
  expect(clock(89.6)).toBe('1:30');
  expect(clock(492)).toBe('8:12');
  expect(clock(-3)).toBe('0:00');
  expect(clock(null)).toBe('–');
});

test('durations', () => {
  expect(duration(45.4)).toBe('45 s');
  expect(duration(270)).toBe('4.5 min');
  expect(duration(3900)).toBe('1 h 05 min');
});

test('bytes in decimal units', () => {
  expect(bytes(820)).toBe('820 B');
  expect(bytes(4_200_000)).toBe('4.20 MB');
  expect(bytes(43_000_000)).toBe('43.0 MB');
  expect(bytes(1_350_000_000)).toBe('1.35 GB');
});

test('percent', () => {
  expect(percent(0.615)).toBe('62%');
  expect(percent(null)).toBe('–');
});

test('relative times', () => {
  const now = Date.parse('2026-11-20T21:00:00Z');
  expect(relativeTime('2026-11-20T20:59:58Z', now)).toBe('just now');
  expect(relativeTime('2026-11-20T20:59:20Z', now)).toBe('40 s ago');
  expect(relativeTime('2026-11-20T20:55:00Z', now)).toBe('5 min ago');
  expect(relativeTime('2026-11-20T18:00:00Z', now)).toBe('3 h ago');
  expect(relativeTime('2026-11-19T20:00:00Z', now)).toBe('1 day ago');
  expect(relativeTime('2026-11-01T20:00:00Z', now)).toBe('1 Nov');
  expect(relativeTime(null, now)).toBe('never');
});
