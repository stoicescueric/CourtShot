import type { CSSProperties, RefObject } from 'react';
import type { Flight, SceneData } from '../types/scene';

/** Net height at the centre, metres (the scene carries none; the reference viewer hard-codes it). */
const NET_H = 0.88;

export function teamColor(team: string | undefined): string {
  return team === 'T_1' ? 'var(--t1)' : team === 'T_2' ? 'var(--t2)' : 'var(--muted)';
}

function Stat({ label, value, unit }: { label: string; value: string | null; unit: string }) {
  return (
    <div className="rp-stat">
      <div className="rp-v">
        {value ?? '–'}
        {value !== null && <small>{unit}</small>}
      </div>
      <div className="rp-k">{label}</div>
    </div>
  );
}

/** The TV-style panel for the flight on show (html:712-735). "Ball now" is written per frame through the refs. */
export function LowerThird(props: {
  scene: SceneData;
  flight: number;
  heightRef: RefObject<HTMLSpanElement>;
  unitRef: RefObject<HTMLElement>;
}): JSX.Element {
  const { scene, flight, heightRef, unitRef } = props;
  const fl = scene.flights[flight] as Flight | undefined;
  const team = fl?.hitter ? scene.players.find((p) => p.id === fl.hitter)?.team : undefined;
  const groundBounces = fl ? fl.bounces.filter((b) => b.kind === 'ground').length : 0;
  const overNet = fl?.net_h == null ? null : fl.net_h - NET_H;

  return (
    <section className="rp-ui rp-third" style={{ '--team': teamColor(fl ? team : undefined) } as CSSProperties}>
      <div className="rp-third-head" aria-live="polite">
        <div>
          <div className="rp-shot">{fl ? `Shot ${flight + 1} of ${scene.flights.length}` : 'Between flights'}</div>
          <div className="rp-who">
            {!fl
              ? 'Ball not tracked in 3D'
              : fl.hitter
                ? `${fl.hitter.replace('_', ' ')}${team ? ` · ${team.replace('T_', 'Team ')}` : ''}`
                : 'Hitter unknown'}
          </div>
        </div>
        <span className={`rp-pill rp-pill--${!fl ? 'none' : fl.ok ? 'good' : 'low'}`}>
          {!fl ? 'No 3D' : fl.ok ? 'Good fit' : 'Low confidence'}
        </span>
      </div>
      <div className="rp-stats">
        <Stat label="Launch" value={fl ? String(Math.round(fl.speed_kmh)) : null} unit="km/h" />
        <Stat label="Peak" value={fl ? fl.max_h.toFixed(1) : null} unit="m" />
        <Stat label="Over net" value={overNet === null ? null : `${overNet >= 0 ? '+' : ''}${overNet.toFixed(2)}`} unit="m" />
        <div className="rp-stat">
          <div className="rp-v">
            <span ref={heightRef}>–</span>
            <small ref={unitRef} hidden>m</small>
          </div>
          <div className="rp-k">Ball now</div>
        </div>
      </div>
      <p className="rp-note">
        {!fl
          ? 'Missed detections or a hit that could not be fitted.'
          : `${groundBounces} floor ${groundBounces === 1 ? 'bounce' : 'bounces'}, ${fl.bounces.length - groundBounces} wall · ` +
            `fit error ${fl.rms_px} px${fl.ok ? '' : ' · path is a rough guess'}`}
      </p>
    </section>
  );
}
