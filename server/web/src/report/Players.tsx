import type { PlayerRow, ShotSummary } from '../types/report';
import { shotLabel, useReport } from './context';
import { has, kmh, num, pct } from './format';
import { Heatmap } from './Heatmap';
import { Dot } from './marks';
import { BarList, Card, Empty, HeatScale, shareRows, StatList } from './parts';

const DIRECTIONS = { cross: 'Cross-court', middle: 'Middle', line: 'Down the line' };
const ZONES = { net: 'Hit at the net', middle: 'Hit mid-court', back: 'Hit at the back' };

/** One player: picker, large heatmap, movement and, with ball analysis, the shot cards. */
export function Players({ player, onOpenPlayer }: { player?: string; onOpenPlayer: (player: string) => void }) {
  const { players, settings } = useReport();
  const p = players.find((q) => q.id === player) ?? players[0];
  const mv = p.move;
  const mate = players.find((q) => q.team === p.team && q.id !== p.id);
  return (
    <>
      <div className="filters">
        <div className="seg" role="group" aria-label="Player">
          {players.map((q) => (
            <button key={q.id} type="button" aria-pressed={q.id === p.id} onClick={() => onOpenPlayer(q.id)}>
              <Dot id={q.id} />
              {q.id}
            </button>
          ))}
        </div>
      </div>
      <div className="grid">
        <Card className="c5" title="Heatmap" sub={`Where ${p.id} stood during rallies: own half, net at the top, ${p.id}'s left on the left.`}>
          <Heatmap player={p} />
          <HeatScale teams={[p.team]} />
        </Card>
        <Card className="c7" title="Movement" sub={`${p.team}, with ${mate ? mate.id : '–'}. Tracked in ${pct(mv.tracked)} of frames.`}>
          <StatList
            rows={[
              ['Distance covered', num(mv.distance_m, 0, ' m'), has(mv.distance_in_rallies_m) && `(${num(mv.distance_in_rallies_m, 0)} m in rallies)`],
              ['Top speed', num(kmh(mv.top_speed_ms), 1, ' km/h')],
              ['Sprints', String(mv.sprints), `over ${num(kmh(settings.sprint_speed), 0)} km/h`],
              ['Average distance from the net', num(mv.avg_dist_to_net_m, 1, ' m')],
              ['Court coverage', pct(mv.court_coverage), 'of their half'],
              ['Time on the left side', pct(mv.time_left_side)],
            ]}
          />
          <h2 style={{ marginTop: 18 }}>Time by zone</h2>
          <p className="sub">
            Net: closer than {settings.net_zone} m. Back: farther than {settings.back_zone} m.
          </p>
          <BarList
            team={p.team}
            rows={[
              { label: 'At the net', value: mv.time_at_net || 0, text: pct(mv.time_at_net) },
              { label: 'In the middle', value: mv.time_in_middle || 0, text: pct(mv.time_in_middle) },
              { label: 'At the back', value: mv.time_at_back || 0, text: pct(mv.time_at_back) },
            ]}
          />
        </Card>
        {p.shots ? (
          <PlayerShots player={p} sh={p.shots} />
        ) : (
          <Card title="Shots">
            <Empty>No ball data in this run, so there are no shot statistics.</Empty>
          </Card>
        )}
      </div>
    </>
  );
}

function PlayerShots({ player, sh }: { player: PlayerRow; sh: ShotSummary }) {
  const report = useReport();
  const team = player.team;
  const typeRows = report.shot_types
    .filter((t) => sh.by_type[t] > 0)
    .sort((a, b) => sh.by_type[b] - sh.by_type[a])
    .map((t) => ({ label: shotLabel(report, t), value: sh.by_type[t] }));
  const opponents = report.players.filter((q) => q.team !== team).map((q) => q.id);
  return (
    <>
      <Card className="c4" title="Shots">
        <StatList
          rows={[
            ['Shots', String(sh.shots)],
            ['Attacking shots', pct(sh.attacking_share)],
            ['Average speed', num(sh.avg_speed_kmh, 0, ' km/h')],
            ['Fastest', num(sh.max_speed_kmh, 0, ' km/h')],
            ['Average contact height', num(sh.avg_contact_height_m, 2, ' m')],
            ['Landing deep', pct(sh.deep_share), 'past the service line'],
            sh.winners + sh.errors > 0 && ['Winners / errors', `${sh.winners} / ${sh.errors}`],
          ]}
        />
      </Card>
      <Card className="c4" title="Shot types" sub="Estimated by rules.">
        {typeRows.length ? <BarList rows={typeRows} team={team} /> : <Empty>No shots found.</Empty>}
      </Card>
      <Card className="c4" title="Direction and target" sub="Where the shots went, and which opponent played the next ball.">
        <BarList rows={shareRows(sh.direction, ['cross', 'middle', 'line'], (k) => DIRECTIONS[k])} team={team} />
        <div style={{ height: 14 }} />
        <BarList rows={shareRows(sh.targets, opponents, (id) => `To ${id}`)} team={team} />
        <div style={{ height: 14 }} />
        <BarList rows={shareRows(sh.from_zone, ['net', 'middle', 'back'], (k) => ZONES[k])} team={team} />
      </Card>
    </>
  );
}
