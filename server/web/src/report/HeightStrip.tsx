import type { ShotInfo } from '../types/report';
import { useReport } from './context';
import { has } from './format';
import { Mark } from './marks';
import { shotTip, useTip } from './Tooltip';

const W = 640;
const H = 150;
const LEFT = 34;
const BOTTOM = 20;
const TOP_M = 3.2;
const HEAD_M = 1.75;

/** Contact height of each shot over the match; no time axis. */
export function HeightStrip({ shots }: { shots: ShotInfo[] }) {
  const report = useReport();
  const tip = useTip();
  const span = Math.max(1, report.duration_s);
  const x = (t: number) => LEFT + ((t - report.t0) / span) * (W - LEFT - 6);
  const y = (v: number) => 6 + (1 - Math.min(v, TOP_M) / TOP_M) * (H - BOTTOM - 6);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Contact height of each shot over time">
      {[0, 1, 2, 3].map((v) => (
        <g key={v}>
          <line x1={LEFT} x2={W - 6} y1={y(v)} y2={y(v)} stroke="var(--line)" />
          <text x={LEFT - 6} y={y(v) + 4} textAnchor="end">
            {v} m
          </text>
        </g>
      ))}
      <line x1={LEFT} x2={W - 6} y1={y(HEAD_M)} y2={y(HEAD_M)} stroke="var(--muted)" strokeWidth={1} />
      <text className="halo" x={W - 6} y={y(HEAD_M) - 5} textAnchor="end">
        head height
      </text>
      {shots.map((s) =>
        has(s.contact_height_m) ? (
          <g key={s.index} className="shot" {...tip(() => shotTip(report, s))}>
            <circle cx={x(s.time_s)} cy={y(s.contact_height_m)} r={10} fill="transparent" />
            <Mark x={x(s.time_s)} y={y(s.contact_height_m)} r={4.5} team={s.team} filled />
          </g>
        ) : null,
      )}
    </svg>
  );
}
