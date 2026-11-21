import { Fragment, type ReactNode } from 'react';
import type { TeamId, TeamRow } from '../types/report';
import { useReport } from './context';
import { TeamDot } from './marks';

export function Card(props: { className?: string; title?: string; sub?: string; children?: ReactNode }) {
  return (
    <div className={props.className ? `card ${props.className}` : 'card'}>
      {props.title && <h2>{props.title}</h2>}
      {props.sub && <p className="sub">{props.sub}</p>}
      {props.children}
    </div>
  );
}

export const Empty = ({ children }: { children: ReactNode }) => <p className="empty">{children}</p>;

export interface BarRow {
  label: string;
  value: number;
  text?: string;
  note?: string;
}

/** Label, bar and number per row; bars relative to max(1, largest value). */
export function BarList({ rows, team }: { rows: BarRow[]; team: TeamId }) {
  const max = Math.max(1, ...rows.map((r) => r.value || 0));
  return (
    <div className="bl">
      {rows.map((r) => (
        <Fragment key={r.label}>
          <div className="k">{r.label}</div>
          <div className="track">
            <i className={`fill ${team}`} style={{ width: `${(100 * (r.value || 0)) / max}%` }} />
          </div>
          <div className="n">
            {r.text ?? String(r.value)}
            {r.note && <small>{r.note}</small>}
          </div>
        </Fragment>
      ))}
    </div>
  );
}

export type StatRow = [label: string, value: string, note?: string | false] | false;

export function StatList({ rows }: { rows: StatRow[] }) {
  return (
    <div className="sl">
      {rows.map(
        (row) =>
          row && (
            <Fragment key={row[0]}>
              <div className="k">{row[0]}</div>
              <div className="n">
                {row[1]}
                {row[2] && <small> {row[2]}</small>}
              </div>
            </Fragment>
          ),
      )}
    </div>
  );
}

/** Rows of a count group: the share of the group total as text, the raw count as note. */
export function shareRows<K extends string>(counts: Partial<Record<K, number>>, order: K[], name: (k: K) => string): BarRow[] {
  const total = order.reduce((a, k) => a + (counts[k] || 0), 0);
  return order.map((k) => {
    const v = counts[k] || 0;
    return { label: name(k), value: v, text: total ? `${Math.round((100 * v) / total)}%` : '–', note: String(v) };
  });
}

/** The team's dot and its two players. */
export const TeamName = ({ team }: { team: TeamRow }) => (
  <span className="who">
    <TeamDot team={team.id} />
    {team.players.join(' · ')}
  </span>
);

/** A chart legend: each team's dot and players, then the chart's own items. */
export function TeamLegend({ children }: { children: ReactNode }) {
  const { teams } = useReport();
  return (
    <div className="legend">
      {teams.map((t) => (
        <TeamName key={t.id} team={t} />
      ))}
      {children}
    </div>
  );
}

/** "Less time ▬ More time" per team; with both teams each scale gets the team's dot. */
export function HeatScale({ teams }: { teams: TeamId[] }) {
  return (
    <div className="scales">
      {teams.map((t) => (
        <div key={t} className="scale">
          {teams.length > 1 && <TeamDot team={t} />}
          Less time
          <i className={`grad ${t}`} />
          More time
        </div>
      ))}
    </div>
  );
}
