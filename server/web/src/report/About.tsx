import { useReport } from './context';
import { clock, pct } from './format';
import { Card, StatList } from './parts';

export function About() {
  const { quality: q, players, settings, t0 } = useReport();
  const swaps = q.swaps_fixed.length
    ? `${q.swaps_fixed.length} identity swap(s) were corrected (at ${q.swaps_fixed.map((s) => clock(s.time_s - t0)).join(', ')}).`
    : 'no identity swaps were found.';
  return (
    <div className="grid about">
      <Card className="c6" title="Data quality" sub="Check these first: everything else is only as good as the tracking.">
        <StatList
          rows={[
            ['Frames with the ball detected', pct(q.ball_coverage)],
            ['Ball detections kept after cleaning', pct(q.ball_kept)],
            ['3D ball flights', q.has_3d ? 'fitted' : 'not used'],
            ...players.map((p): [string, string, string] => [`${p.id} tracked`, pct(p.move.tracked), `(${pct(p.move.interpolated)} filled in)`]),
          ]}
        />
      </Card>
      <Card className="c6" title="Player identities">
        <p className="note">
          The four players are followed by position and motion; {q.tracker_ids_merged} raw tracker IDs were merged into them and{' '}
          {q.interpolated_frames} frames filled in across short gaps.{' '}
          {q.appearance_used
            ? `The result was checked against what the players look like: ${swaps} Players in the same kit can't be told apart this way.`
            : 'Player looks were not checked (no appearance data in this run).'}
        </p>
      </Card>
      <Card title="How the numbers are made">
        <dl>
          <dt>Positions</dt>
          <dd>
            Each player's feet mapped through the court calibration (accuracy roughly 0.2–0.4 m). Distance and speed come from smoothed
            positions; standing still (under {settings.still_speed} m/s) adds no distance. Position statistics count rally time only.
          </dd>
          <dt>Heatmaps</dt>
          <dd>
            Share of rally time in each spot of the player's own half, flipped so the net is at the top whichever end they play from. The
            colour, in the player's team colour, runs from faint (little time) to strong (most time), relative to that player's own
            busiest spot.
          </dd>
          <dt>Rallies</dt>
          <dd>From the ball track (first shot to last ball movement) when the ball is seen often enough, otherwise from player movement.</dd>
          <dt>Shots</dt>
          <dd>A shot is a change in the ball's path within a player's reach. Missed ball detections mean missed shots.</dd>
          <dt>Shot types</dt>
          <dd>
            Decided by rules, not a trained model: contact above the head is a smash when fast and a bandeja / víbora otherwise (the two
            are not told apart); a ball that goes high, or that the opponents answer with an overhead, is a lob; taken before the bounce
            near the net is a volley; hard and high off the back wall is a bajada; soft and short from the back is a chiquita; other shots
            after the bounce are drives. Expect mistakes, mostly between neighbouring types.
          </dd>
          <dt>Speed</dt>
          <dd>
            The launch speed of the fitted 3D flight when that fit is good, otherwise the average speed from the hitter to the landing spot
            (which reads lower).
          </dd>
          <dt>Contact height</dt>
          <dd>Where the ball is within the hitter's outline in the image, scaled to a 1.75 m player.</dd>
          <dt>Direction and target</dt>
          <dd>
            From where the hitter stood to where the ball landed (or to the opponent who played it next). Middle = within 1.25 m of the
            centre line.
          </dd>
          <dt>Winners and errors</dt>
          <dd>An estimate from the last shot of each rally. They need whole points on video: a clip that starts or stops mid-rally has none.</dd>
        </dl>
      </Card>
    </div>
  );
}
