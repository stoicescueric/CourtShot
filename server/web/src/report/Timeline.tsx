import { useReport, shotLabel } from './context';
import { clock } from './format';
import { Mark } from './marks';
import { TeamLegend } from './parts';
import { shotTip, useTip } from './Tooltip';

const W = 1000;
const LANE = 26;
const TOP = 8;
const LEFT = 44;
const H = TOP + LANE * 4 + 26;

/** Seconds between time labels for a video of `span` seconds. */
const tickStep = (span: number): number => (span > 600 ? 120 : span > 240 ? 60 : span > 90 ? 30 : 10);

/** Rally bands across all lanes and every shot in its hitter's lane. */
export function Timeline() {
  const report = useReport();
  const tip = useTip();
  const { players, shots, t0 } = report;
  const span = Math.max(1, report.duration_s);
  const x = (t: number) => LEFT + ((t - t0) / span) * (W - LEFT - 8);
  // The reference used `rallies || movement_rallies`, so an empty ball-rally list drew no bands at all.
  const bands = report.rallies?.length ? report.rallies : report.movement_rallies;
  const ticks: number[] = [];
  for (let t = 0, step = tickStep(span); t <= span; t += step) ticks.push(t);
  const laneY = (i: number) => TOP + LANE * i + LANE / 2;
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Shots over time by player">
        {bands.map((r, i) => (
          <rect key={i} x={x(r.start_s)} y={TOP} width={Math.max(1, x(r.end_s) - x(r.start_s))} height={LANE * 4} fill="var(--raise)" rx={4} />
        ))}
        {players.map((p, i) => (
          <g key={p.id}>
            <line x1={LEFT} x2={W - 8} y1={laneY(i)} y2={laneY(i)} stroke="var(--line)" />
            <text className="lane" x={0} y={laneY(i) + 4}>
              {p.id}
            </text>
          </g>
        ))}
        {ticks.map((t) => (
          <text key={t} x={x(t0 + t)} y={H - 4} textAnchor="middle">
            {clock(t)}
          </text>
        ))}
        {shots.map((s) => {
          const cx = x(s.time_s);
          const cy = laneY(players.findIndex((p) => p.id === s.player));
          return (
            <g key={s.index} className="shot" {...tip(() => shotTip(report, s))}>
              <circle cx={cx} cy={cy} r={11} fill="transparent" />
              <Mark x={cx} y={cy} r={s.attacking ? 5 : 4.2} team={s.team} filled={s.attacking} />
            </g>
          );
        })}
      </svg>
      {shots.length > 0 && (
        <TeamLegend>
          <span>Filled: attacking shot ({report.attacking_types.map((t) => shotLabel(report, t)).join(', ').toLowerCase()})</span>
          <span>Hollow: other shot</span>
        </TeamLegend>
      )}
    </div>
  );
}
