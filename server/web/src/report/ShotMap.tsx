import { useState } from 'react';
import type { ShotInfo } from '../types/report';
import { useReport } from './context';
import { Mark } from './marks';
import { shotTip, useTip } from './Tooltip';

export type DrawnShot = ShotInfo & { contact: [number, number]; landing: [number, number] };

export const isDrawn = (s: ShotInfo): s is DrawnShot => s.contact !== null && s.landing !== null;

const S = 30;
const PAD = 8;
const X = (x: number) => PAD + x * S;
/** Court drawn upright: y = 0 (the camera end) at the bottom; x is not mirrored. */
const Y = (y: number) => PAD + (20 - y) * S;

/** The whole court with a line from each shot's contact to its landing; hover dims the others. */
export function ShotMap({ shots }: { shots: DrawnShot[] }) {
  const report = useReport();
  const tip = useTip();
  const [hover, setHover] = useState<number | null>(null);
  const sv = report.settings.service_line;
  return (
    <svg className="chart" viewBox={`0 0 ${10 * S + 2 * PAD} ${20 * S + 2 * PAD}`} role="img" aria-label="Shot map" style={{ maxHeight: 640 }}>
      <rect className="court-bg" x={PAD} y={PAD} width={10 * S} height={20 * S} rx={10} />
      <rect className="court-ln" x={PAD} y={PAD} width={10 * S} height={20 * S} rx={10} strokeWidth={1.5} />
      {[10 - sv, 10 + sv].map((y) => (
        <line key={y} className="court-ln" x1={X(0)} x2={X(10)} y1={Y(y)} y2={Y(y)} strokeWidth={1.5} />
      ))}
      <line className="court-ln" x1={X(5)} x2={X(5)} y1={Y(10 - sv)} y2={Y(10 + sv)} strokeWidth={1.5} />
      <line className="court-ln" x1={X(0)} x2={X(10)} y1={Y(10)} y2={Y(10)} strokeWidth={3} />
      <text className="court-tag" x={X(10) - 6} y={Y(10) - 7} textAnchor="end">
        Net
      </text>
      {shots.map((s) => {
        const [x1, y1] = s.contact;
        const [x2, y2] = s.landing;
        const ends = { x1: X(x1), y1: Y(y1), x2: X(x2), y2: Y(y2) };
        const { onMouseMove, onMouseLeave } = tip(() => shotTip(report, s));
        return (
          <g
            key={s.index}
            className="shot"
            style={{ opacity: hover === null || hover === s.index ? 1 : 0.18 }}
            onMouseEnter={() => setHover(s.index)}
            onMouseMove={onMouseMove}
            onMouseLeave={() => {
              setHover(null);
              onMouseLeave();
            }}
          >
            <line {...ends} stroke="transparent" strokeWidth={12} />
            <line {...ends} stroke="var(--court)" strokeWidth={4.5} strokeLinecap="round" />
            <line {...ends} stroke={s.team === 'T_1' ? 'var(--t1)' : 'var(--t2)'} strokeWidth={2} strokeLinecap="round" opacity={0.85} />
            <Mark x={X(x2)} y={Y(y2)} r={5.5} team={s.team} filled ring="var(--court)" />
          </g>
        );
      })}
    </svg>
  );
}
