import type { TeamId } from '../types/report';
import { useReport } from './context';

/** Team identity dot: T_1 round, T_2 square; solid. */
export const TeamDot = ({ team }: { team: TeamId | '?' }) => <i className={`dot ${team}`} />;

/** Player identity dot: the team's dot, a hollow ring for the team's second player in `players` order. */
export function Dot({ id }: { id: string }) {
  const { players } = useReport();
  const p = players.find((q) => q.id === id);
  if (!p) return <i className="dot" />;
  const second = players.filter((q) => q.team === p.team).indexOf(p) === 1;
  return <i className={`dot ${p.team}${second ? ' ring' : ''}`} />;
}

export const Who = ({ id }: { id: string }) => (
  <span className="who">
    <Dot id={id} />
    {id}
  </span>
);

/** An SVG data mark in a team's colour and shape (T_1 round, T_2 square), filled or hollow. */
export function Mark(props: { x: number; y: number; r: number; team: TeamId | '?'; filled: boolean; ring?: string }) {
  const { x, y, r, team, filled, ring = 'var(--card)' } = props;
  const c = team === 'T_1' ? 'var(--t1)' : 'var(--t2)';
  const paint = { fill: filled ? c : ring, stroke: filled ? ring : c, strokeWidth: filled ? 1.5 : 2 };
  return team === 'T_1' ? (
    <circle cx={x} cy={y} r={r} {...paint} />
  ) : (
    <rect x={x - r * 0.9} y={y - r * 0.9} width={r * 1.8} height={r * 1.8} rx={r * 0.3} {...paint} />
  );
}
