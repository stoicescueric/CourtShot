/** `m:ss` from rounded, non-negative seconds; "–" when missing. */
export function clock(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '–';
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/** "45 s", "4.5 min", "1 h 05 min". */
export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '–';
  if (seconds < 90) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`;
  const minutes = Math.round(seconds / 60);
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, '0')} min`;
}

/** Decimal units, as macOS shows file sizes: "820 B", "4.2 MB", "1.35 GB". */
export function bytes(n: number): string {
  if (n < 1000) return `${n} B`;
  const units = ['kB', 'MB', 'GB', 'TB'];
  let value = n / 1000;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value.toFixed(value < 10 ? 2 : value < 100 ? 1 : 0)} ${units[unit]}`;
}

/** Share 0..1 as a whole percentage. */
export function percent(share: number | null | undefined): string {
  if (share == null || !Number.isFinite(share)) return '–';
  return `${Math.round(share * 100)}%`;
}

/** "20 Nov". */
export function shortDate(iso: string): string {
  return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short' }).format(new Date(iso));
}

/** "just now", "40 s ago", "5 min ago", "3 h ago", "2 days ago", then the date. */
export function relativeTime(iso: string | null, now: number = Date.now()): string {
  if (!iso) return 'never';
  const seconds = Math.round((now - new Date(iso).getTime()) / 1000);
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${seconds} s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  if (seconds < 7 * 86400) {
    const days = Math.floor(seconds / 86400);
    return `${days} day${days === 1 ? '' : 's'} ago`;
  }
  return shortDate(iso);
}
