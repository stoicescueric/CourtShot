import { Fragment } from 'react';
import { Logo } from '../components/Logo';
import type { ShotSummary, TeamRow } from '../types/report';
import { shotLabel, useReport } from './context';
import { dur, has, num, pct } from './format';
import { Heatmap } from './Heatmap';
import { Dot } from './marks';
import { Card, HeatScale, TeamName } from './parts';
import { Timeline } from './Timeline';

export function Overview({ onOpenPlayer }: { onOpenPlayer: (player: string) => void }) {
  const report = useReport();
  const [a, b] = report.teams;
  return (
    <>
      <Hero />
      <div className="grid">
        <TeamComparison />
        {a.shots && b.shots && <ShotTypes a={a.shots} b={b.shots} />}
        <Card
          title="Where each player plays"
          sub="Share of rally time, seen from behind each player: their own half, net at the top. Select a player for the details."
        >
          <div className="minis">
            {report.players.map((p) => (
              <button key={p.id} type="button" className="mini" onClick={() => onOpenPlayer(p.id)}>
                <span className="name">
                  <Dot id={p.id} />
                  {p.id}
                  <small>{pct(p.move.time_at_net)} at net</small>
                </span>
                <Heatmap player={p} mini />
              </button>
            ))}
          </div>
          <HeatScale teams={report.teams.map((t) => t.id)} />
        </Card>
        <Card title="Match timeline" sub={report.shots.length ? 'Every shot, by player. Shaded bands are rallies.' : 'Shaded bands are rallies.'}>
          <Timeline />
        </Card>
      </div>
    </>
  );
}

function Stat(props: { label: string; value: string; unit?: string; sub?: string; highlight?: boolean }) {
  return (
    <div className={props.highlight ? 'stat key' : 'stat'}>
      <div className="k">{props.label}</div>
      <div className="v">
        {props.value}
        {props.unit && <small>{props.unit}</small>}
      </div>
      {props.sub && <div className="s">{props.sub}</div>}
    </div>
  );
}

/** Headline numbers: the ones the tracking is surest of; single-shot speeds are not among them. */
function Hero() {
  const { match: m, movement_rallies: mv, players, duration_s, settings, quality } = useReport();
  const inPlay = m ? m.time_in_play_s : mv.reduce((a, r) => a + r.duration_s, 0);
  const metres = players.reduce((a, p) => a + (p.move.distance_m || 0), 0);
  const mins = inPlay >= 90;
  const km = metres >= 1000;
  return (
    <div className="card hero">
      <div className="hero-head">
        <h2>
          <Logo size={30} />
          Match summary
        </h2>
        <span className="chip">Rallies from the {settings.rally_source}</span>
      </div>
      <div className="stats">
        {m ? (
          // longest_rally_s is its own maximum and can belong to another rally, so it is labelled as such.
          <Stat label="Longest rally" value={String(m.longest_rally_shots ?? '–')} unit="shots" sub={`longest lasted ${dur(m.longest_rally_s)}`} highlight />
        ) : (
          <Stat label="Rallies" value={String(mv.length)} highlight />
        )}
        <Stat
          label="Time in play"
          value={mins ? (inPlay / 60).toFixed(1) : String(Math.round(inPlay))}
          unit={mins ? 'min' : 's'}
          sub={`${pct(inPlay / duration_s)} of the video`}
        />
        {m && <Stat label="Rallies" value={String(m.rallies)} sub={`${num(m.avg_rally_shots, m.rallies > 1 ? 1 : 0)} shots on average`} />}
        <Stat label="Distance covered" value={km ? (metres / 1000).toFixed(1) : String(Math.round(metres))} unit={km ? 'km' : 'm'} sub="all four players" />
        {m && <Stat label="Shots" value={String(m.shots)} sub={`ball seen in ${pct(quality.ball_coverage)} of frames`} />}
      </div>
    </div>
  );
}

type Value = number | null;

interface CmpSpec {
  label: string;
  a: Value;
  b: Value;
  fmt: (v: Value) => string;
  lowerLeads?: boolean;
}

/** Two values around a label, bars growing outwards; the leader's value gets the team pill. */
function CmpRow({ label, a, b, fmt, lowerLeads = false }: CmpSpec) {
  const max = Math.max(a || 0, b || 0, 1e-9);
  const lead = !has(a) || !has(b) || a === b ? null : a > b !== lowerLeads ? 0 : 1;
  return (
    <div className="cmp">
      <div className="vals">
        <div className={lead === 0 ? 'v T_1 lead' : 'v T_1'}>
          <span>{fmt(a)}</span>
        </div>
        <div className="lab">{label}</div>
        <div className={lead === 1 ? 'v r T_2 lead' : 'v r T_2'}>
          <span>{fmt(b)}</span>
        </div>
      </div>
      <div className="bars">
        <div>
          <i style={{ width: `${(100 * (a || 0)) / max}%` }} />
        </div>
        <div>
          <i style={{ width: `${(100 * (b || 0)) / max}%` }} />
        </div>
      </div>
    </div>
  );
}

function CmpHead() {
  const [a, b] = useReport().teams;
  return (
    <div className="cmp-head">
      <div>
        <TeamName team={a} />
      </div>
      <div />
      <div>
        <TeamName team={b} />
      </div>
    </div>
  );
}

const count = (v: Value) => num(v);
const speed = (v: Value) => num(v, 0, ' km/h');
const overheads = (s: ShotSummary) => (s.by_type.smash || 0) + (s.by_type.bandeja_vibora || 0);

function TeamComparison() {
  const { teams, players } = useReport();
  const [A, B] = teams;
  const distance = (t: TeamRow) => t.players.reduce((s, id) => s + (players.find((p) => p.id === id)?.move.distance_m || 0), 0);
  const rows: CmpSpec[] = [];
  if (A.shots && B.shots) {
    const [a, b] = [A.shots, B.shots];
    rows.push(
      { label: 'Shots', a: a.shots, b: b.shots, fmt: count },
      { label: 'Attacking shots', a: a.attacking_share, b: b.attacking_share, fmt: pct },
      { label: 'Average shot speed', a: a.avg_speed_kmh, b: b.avg_speed_kmh, fmt: speed },
      { label: 'Fastest shot', a: a.max_speed_kmh, b: b.max_speed_kmh, fmt: speed },
      { label: 'Overheads', a: overheads(a), b: overheads(b), fmt: count },
      { label: 'Lobs', a: a.by_type.lob || 0, b: b.by_type.lob || 0, fmt: count },
    );
    if (a.winners + a.errors + b.winners + b.errors > 0) {
      rows.push(
        { label: 'Winners', a: a.winners, b: b.winners, fmt: count },
        { label: 'Errors', a: a.errors, b: b.errors, fmt: count, lowerLeads: true },
      );
    }
  }
  rows.push(
    { label: 'Both players at the net', a: A.move.both_at_net, b: B.move.both_at_net, fmt: pct },
    { label: 'Both players at the back', a: A.move.both_at_back, b: B.move.both_at_back, fmt: pct, lowerLeads: true },
    { label: 'Distance covered', a: distance(A), b: distance(B), fmt: (v) => num(v, 0, ' m') },
    { label: 'Net regained', a: A.move.net_regained, b: B.move.net_regained, fmt: count },
  );
  return (
    <Card className="c6" title="Team comparison" sub="Shots from the ball track; positions during rallies only.">
      <CmpHead />
      {rows.map((r) => (
        <CmpRow key={r.label} {...r} />
      ))}
    </Card>
  );
}

/** Mirrored bars per type with a nonzero total, on one scale: the largest single count. */
function ShotTypes({ a, b }: { a: ShotSummary; b: ShotSummary }) {
  const report = useReport();
  const order = report.shot_types.filter((t) => (a.by_type[t] || 0) + (b.by_type[t] || 0) > 0);
  const max = Math.max(1, ...order.map((t) => Math.max(a.by_type[t] || 0, b.by_type[t] || 0)));
  return (
    <Card className="c6" title="Shot types" sub="How many of each shot the two teams played. Types are estimated by rules (see About).">
      <CmpHead />
      <div className="mir">
        {order.map((t) => {
          const [x, y] = [a.by_type[t] || 0, b.by_type[t] || 0];
          return (
            <Fragment key={t}>
              <div className="n">{x}</div>
              <div className="track l">
                <i style={{ width: `${(100 * x) / max}%` }} />
              </div>
              <div className="lab">{shotLabel(report, t)}</div>
              <div className="track r">
                <i style={{ width: `${(100 * y) / max}%` }} />
              </div>
              <div className="n r">{y}</div>
            </Fragment>
          );
        })}
      </div>
    </Card>
  );
}
