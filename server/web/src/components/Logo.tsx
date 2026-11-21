import { useId } from 'react';

const DOTS: [number, number, number, number][] = [
  [32.96, 62.93, 2.2, 0.3],
  [41.82, 45.34, 2.5, 0.45],
  [50.59, 35.21, 2.8, 0.6],
  [59.28, 32.56, 3.1, 0.75],
  [67.88, 37.38, 3.4, 0.88],
  [76.38, 49.66, 3.7, 1],
];

interface LogoProps {
  size?: number;
  withWordmark?: boolean;
}

/** The CourtShot mark: dark tile, white arc of the ball path, Volt ball. */
export function Logo({ size = 32, withWordmark = false }: LogoProps) {
  const uid = useId();
  const tile = `${uid}-tile`;
  const arc = `${uid}-arc`;
  const shade = `${uid}-shade`;
  const ballClip = `${uid}-ball`;
  const mark = (
    <svg width={size} height={size} viewBox="0 0 120 120" role="img" aria-label="CourtShot" style={{ flex: 'none' }}>
      <defs>
        <linearGradient id={tile} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#2C2C2E" />
          <stop offset="1" stopColor="#000000" />
        </linearGradient>
        <linearGradient id={arc} gradientUnits="userSpaceOnUse" x1="24" y1="0" x2="82" y2="0">
          <stop offset="0" stopColor="#FFFFFF" stopOpacity="0.08" />
          <stop offset="1" stopColor="#FFFFFF" stopOpacity="0.7" />
        </linearGradient>
        <radialGradient id={shade} cx="0.35" cy="0.3" r="0.8">
          <stop offset="0" stopColor="#FFFFFF" stopOpacity="0.35" />
          <stop offset="0.55" stopColor="#FFFFFF" stopOpacity="0" />
          <stop offset="1" stopColor="#000000" stopOpacity="0.28" />
        </radialGradient>
        <clipPath id={ballClip}>
          <circle cx="85" cy="70" r="9" />
        </clipPath>
      </defs>
      <rect x="0.5" y="0.5" width="119" height="119" rx="27" fill={`url(#${tile})`} stroke="#FFFFFF" strokeOpacity="0.12" />
      <path d="M24 88 Q54 -8 82 62" fill="none" stroke={`url(#${arc})`} strokeWidth="2.5" strokeLinecap="round" />
      {DOTS.map(([cx, cy, r, o]) => (
        <circle key={cx} cx={cx} cy={cy} r={r} fill="#FFFFFF" fillOpacity={o} />
      ))}
      <circle cx="85" cy="70" r="9" fill="#D4F542" />
      <g clipPath={`url(#${ballClip})`} fill="none" stroke="#FFFFFF" strokeWidth="1.5" strokeLinecap="round">
        <path d="M78.5 61 C 83.5 65, 83.5 75, 78.5 79" />
        <path d="M91.5 61 C 86.5 65, 86.5 75, 91.5 79" />
      </g>
      <circle cx="85" cy="70" r="9" fill={`url(#${shade})`} />
    </svg>
  );
  if (!withWordmark) return mark;
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: size * 0.3 }}>
      {mark}
      <span style={{ fontWeight: 600, fontSize: size * 0.56, letterSpacing: '-0.035em', color: 'var(--text)' }}>
        CourtShot
      </span>
    </span>
  );
}
