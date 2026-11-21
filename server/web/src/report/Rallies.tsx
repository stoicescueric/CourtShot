import type { RallyEnding } from '../types/report';
import { shotLabel, useReport } from './context';
import { clock, DASH, dur } from './format';
import { Who } from './marks';
import { Card, Empty } from './parts';

const OUTCOME = { winner: 'Winner', error: 'Error', cut: 'Video ends mid-rally', unknown: 'Not clear' };
const BALL_COLUMNS = ['#', 'Start', 'Length', 'Shots', 'Last shot by', 'Last shot', 'Ended with'];
const MOVEMENT_COLUMNS = ['#', 'Start', 'End', 'Length'];

export function Rallies() {
  const { rallies, movement_rallies: mv, settings, t0 } = useReport();
  // With ball rallies in use the movement rallies repeat them: show those only when movement decided.
  const showMovement = !rallies || settings.rally_source === 'movement';
  return (
    <div className="grid">
      {rallies && <BallRallies list={rallies} />}
      {showMovement && (
        <Card
          title={rallies ? 'Rallies used for the position statistics' : 'Rallies'}
          sub="Found from how the players move (someone keeps making fast moves), because the ball was not seen often enough to mark them."
        >
          {mv.length ? (
            <div className="scroll">
              <table>
                <thead>
                  <tr>
                    {MOVEMENT_COLUMNS.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {mv.map((r, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>{clock(r.start_s - t0)}</td>
                      <td>{clock(r.end_s - t0)}</td>
                      <td>{dur(r.duration_s)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>None found.</Empty>
          )}
        </Card>
      )}
    </div>
  );
}

function BallRallies({ list }: { list: RallyEnding[] }) {
  const report = useReport();
  if (!list.length) {
    // The reference drew an empty table here.
    return (
      <Card title="Rallies" sub="From the ball track.">
        <Empty>None found.</Empty>
      </Card>
    );
  }
  const known = list.filter((r) => r.outcome === 'winner' || r.outcome === 'error').length;
  return (
    <Card
      title="Rallies"
      sub={`From the ball track. How a rally ended is read from the last shot's fitted flight: an estimate, clear for ${known} of ${list.length}.`}
    >
      <div className="scroll">
        <table>
          <thead>
            <tr>
              {BALL_COLUMNS.map((c, i) => (
                <th key={c} className={i > 3 ? 'l' : undefined}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {list.map((r) => (
              <tr key={r.rally}>
                <td>{r.rally + 1}</td>
                <td>{clock(r.start_s - report.t0)}</td>
                <td>{dur(r.duration_s)}</td>
                <td>{r.shots}</td>
                <td className="l">{r.last_player ? <Who id={r.last_player} /> : DASH}</td>
                <td className="l">{r.last_type ? shotLabel(report, r.last_type) : DASH}</td>
                <td className="l">{OUTCOME[r.outcome] || DASH}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
