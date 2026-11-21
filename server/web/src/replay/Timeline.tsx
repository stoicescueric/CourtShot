import { useMemo, type PointerEvent, type RefObject } from 'react';
import type { SceneData } from '../types/scene';
import { SCRUB_MAX, percentIn, spanBox, timeFromScrub, type TimeWindow } from './layout';

export type ScrubKind = 'rally' | 'clip';

export interface TimelineRefs {
  head: RefObject<HTMLDivElement>;
  scrub: RefObject<HTMLInputElement>;
  clipHead: RefObject<HTMLDivElement>;
  clipScrub: RefObject<HTMLInputElement>;
}

function teamClass(team: string | undefined): string {
  return team === 'T_1' ? ' rp-tick--t1' : team === 'T_2' ? ' rp-tick--t2' : '';
}

/**
 * A thin strip for the whole clip (rallies, the zoomed window, a scrub over everything) above the lane of
 * the current rally window (flight spans, hit ticks, a scrub over the window). Playheads are moved by the
 * parent through the refs, not by re-rendering.
 */
export function Timeline(props: {
  scene: SceneData;
  windows: TimeWindow[];
  win: number;
  dur: number;
  refs: TimelineRefs;
  onScrubStart: (kind: ScrubKind) => void;
  onScrub: (t: number) => void;
  onScrubEnd: () => void;
}): JSX.Element {
  const { scene, windows, win, dur, refs, onScrubStart, onScrub, onScrubEnd } = props;
  const clip: TimeWindow = [0, dur];
  const w = windows[win] ?? clip;
  const teamOf = useMemo(() => new Map(scene.players.map((p) => [p.id, p.team])), [scene]);
  const spans = useMemo(
    () => scene.flights.flatMap((fl, i) => {
      const box = spanBox(fl.t0, fl.t1, w);
      return box ? [{ i, ok: fl.ok, box }] : [];
    }),
    [scene, w],
  );
  const ticks = useMemo(
    () => scene.hits.flatMap((h, i) => (h.t >= w[0] && h.t <= w[1] ? [{ i, left: percentIn(h.t, w), team: h.player ? teamOf.get(h.player) : undefined }] : [])),
    [scene, w, teamOf],
  );

  const start = (kind: ScrubKind) => (e: PointerEvent<HTMLInputElement>) => {
    e.currentTarget.setPointerCapture?.(e.pointerId);
    onScrubStart(kind);
  };
  const scrubProps = {
    type: 'range',
    className: 'rp-scrub',
    min: 0,
    max: SCRUB_MAX,
    step: 1,
    defaultValue: 0,
    onPointerUp: onScrubEnd,
    onPointerCancel: onScrubEnd,
  } as const;

  return (
    <div className="rp-timeline">
      <div className="rp-clip">
        {scene.rallies.map(([s, e], i) => (
          <span key={i} className="rp-rally" style={{ left: `${percentIn(s, clip)}%`, width: `${percentIn(e, clip) - percentIn(s, clip)}%` }} />
        ))}
        <span className="rp-window" style={{ left: `${percentIn(w[0], clip)}%`, width: `${percentIn(w[1], clip) - percentIn(w[0], clip)}%` }} />
        <div ref={refs.clipHead} className="rp-head" />
        <input
          {...scrubProps}
          ref={refs.clipScrub}
          aria-label="Time in the clip"
          onPointerDown={start('clip')}
          onChange={(e) => onScrub(timeFromScrub(Number(e.currentTarget.value), clip))}
        />
      </div>
      <div className="rp-lanes">
        <div className="rp-hit-lane">
          {ticks.map((h) => <i key={h.i} className={`rp-tick${teamClass(h.team)}`} style={{ left: `${h.left}%` }} />)}
        </div>
        <div className="rp-flight-lane">
          {spans.map((s) => (
            <span key={s.i} className={s.ok ? 'rp-span' : 'rp-span rp-span--low'} style={{ left: `${s.box.left}%`, width: `${s.box.width}%` }} />
          ))}
        </div>
        <div ref={refs.head} className="rp-head" />
        <input
          {...scrubProps}
          ref={refs.scrub}
          aria-label={scene.rallies.length ? 'Time in the rally' : 'Time in the clip'}
          onPointerDown={start('rally')}
          onChange={(e) => onScrub(timeFromScrub(Number(e.currentTarget.value), w))}
        />
      </div>
      <div className="rp-legend">
        <span className="rp-legend-where">{scene.rallies.length ? `Rally ${win + 1} of ${windows.length}` : 'Whole clip'}</span>
        <span><b className="rp-key-ball" />3D flight</span>
        <span><b className="rp-key-low" />low confidence</span>
        <span><b className="rp-key-t1" />team 1 hit</span>
        <span><b className="rp-key-t2" />team 2 hit</span>
      </div>
    </div>
  );
}
