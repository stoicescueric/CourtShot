import { Fragment } from 'react';
import type { ShotInfo } from '../types/report';
import { shotLabel, useReport } from './context';
import { clock, DASH, has, num } from './format';
import { HeightStrip } from './HeightStrip';
import { Dot, Who } from './marks';
import { Card, Empty, TeamLegend } from './parts';
import { isDrawn, ShotMap } from './ShotMap';

/** 'all' or a player id / a shot type. */
export interface ShotFilter {
  player: string;
  type: string;
}

export const ALL_SHOTS: ShotFilter = { player: 'all', type: 'all' };

const ZONE = { net: 'Net', middle: 'Mid-court', back: 'Back' };
const DIRECTION = { cross: 'Cross-court', line: 'Down the line', middle: 'Middle' };
const SEEN = { high: 'Well', medium: 'Partly', low: 'Poorly' };
const COLUMNS = ['#', 'Time', 'Player', 'Type', 'Speed', 'Contact', 'From', 'Direction', 'To', 'Landing', 'Seen'];

type Timed = ShotInfo & { speed_kmh: number };
const hasSpeed = (s: ShotInfo): s is Timed => has(s.speed_kmh);

export function Shots({ filter, onFilter }: { filter: ShotFilter; onFilter: (f: ShotFilter) => void }) {
  const report = useReport();
  const { shots, players, t0 } = report;
  if (!shots.length) {
    return (
      <div className="grid">
        <Card title="Shots">
          <Empty>No shots were found in this run.</Empty>
        </Card>
      </div>
    );
  }
  const typesHere = report.shot_types.filter((t) => shots.some((s) => s.type === t));
  const sel = shots.filter(
    (s) => (filter.player === 'all' || s.player === filter.player) && (filter.type === 'all' || s.type === filter.type),
  );
  const drawn = sel.filter(isDrawn);
  const fastest = sel.filter(hasSpeed).sort((a, b) => b.speed_kmh - a.speed_kmh).slice(0, 8);
  const top = fastest[0]?.speed_kmh || 1;
  return (
    <>
      <div className="filters">
        <div className="seg" role="group" aria-label="Player">
          {[{ id: 'all', text: 'All players' }, ...players.map((p) => ({ id: p.id, text: p.id }))].map(({ id, text }) => (
            <button key={id} type="button" aria-pressed={filter.player === id} onClick={() => onFilter({ ...filter, player: id })}>
              {id !== 'all' && <Dot id={id} />}
              {text}
            </button>
          ))}
        </div>
        <select aria-label="Shot type" value={filter.type} onChange={(ev) => onFilter({ ...filter, type: ev.target.value })}>
          <option value="all">All shot types</option>
          {typesHere.map((t) => (
            <option key={t} value={t}>
              {shotLabel(report, t)}
            </option>
          ))}
        </select>
      </div>
      <div className="grid">
        <Card
          className="c5"
          title="Shot map"
          sub={`${drawn.length} of ${sel.length} selected shots have a landing spot. Line: from the hitter to where the ball landed.`}
        >
          <ShotMap shots={drawn} />
          <TeamLegend>
            <span>Mark: landing spot</span>
          </TeamLegend>
        </Card>
        <Card className="c7" title="Fastest shots" sub="Launch speed from the 3D fit where it is good, else the average speed to the landing spot.">
          {fastest.length ? (
            <div className="bl">
              {fastest.map((s) => (
                <Fragment key={s.index}>
                  <div className="k">
                    <Who id={s.player} /> {shotLabel(report, s.type)}
                  </div>
                  <div className="track">
                    <i className={`fill ${s.team}`} style={{ width: `${(100 * s.speed_kmh) / top}%` }} />
                  </div>
                  <div className="n">
                    {Math.round(s.speed_kmh)} km/h<small>{clock(s.time_s - t0)}</small>
                  </div>
                </Fragment>
              ))}
            </div>
          ) : (
            <Empty>No speeds for this selection.</Empty>
          )}
          <h2 style={{ marginTop: 20 }}>Contact height</h2>
          <p className="sub">How high the ball was when each shot was hit, by time. Above 1.75 m is over the head.</p>
          <HeightStrip shots={sel} />
        </Card>
        <Card title="All shots" sub={`${sel.length} shots`}>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  {COLUMNS.map((c, i) => (
                    <th key={c} className={i === 2 || i === 3 || i > 5 ? 'l' : undefined}>
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sel.map((s) => (
                  <tr key={s.index}>
                    <td>{s.index + 1}</td>
                    <td>{clock(s.time_s - t0)}</td>
                    <td className="l">
                      <Who id={s.player} />
                    </td>
                    <td className="l">{shotLabel(report, s.type)}</td>
                    <td>{has(s.speed_kmh) ? `${Math.round(s.speed_kmh)} km/h` : DASH}</td>
                    <td>{num(s.contact_height_m, 1, ' m')}</td>
                    <td className="l">{s.zone ? ZONE[s.zone] : DASH}</td>
                    <td className="l">{s.direction ? DIRECTION[s.direction] : DASH}</td>
                    <td className="l">{s.target ? <Who id={s.target} /> : DASH}</td>
                    <td className="l">{s.landing ? `${s.landing[0].toFixed(1)}, ${s.landing[1].toFixed(1)}` : DASH}</td>
                    <td className="l">{SEEN[s.confidence] || DASH}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </>
  );
}
