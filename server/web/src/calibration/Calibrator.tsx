import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { getMatch, listMatches } from '../api/client';
import type { Calibration } from '../api/types';
import { messageOf } from '../auth';
import { useResource } from '../hooks';
import { useConfig } from '../shell';
import { COURT_LINES, COURT_POINTS, POINT_LABELS, POINT_NAMES, isPointName, type PointName, type Pt } from './court';
import { FrameView } from './FrameView';
import { apply, assessCalibration, weight } from './homography';

const LINE_COLOUR = '#d4f542';

interface CalibratorProps {
  /** The match being calibrated: left out of "Use points from". */
  matchId: number;
  videoUrl: string;
  width: number;
  height: number;
  duration: number;
  initialPoints?: Record<string, Pt>;
  onSave: (calibration: Calibration) => Promise<void>;
  /** Shown above the fit, e.g. the upload's progress. */
  aside?: ReactNode;
}

interface Clicks {
  points: Partial<Record<PointName, Pt>>;
  /** Points placed or skipped, in the order it happened: what Undo takes back. */
  done: PointName[];
}

function fromPoints(points: Record<string, Pt> = {}): Clicks {
  const known = Object.entries(points).filter((e): e is [PointName, Pt] => isPointName(e[0]));
  return { points: Object.fromEntries(known), done: POINT_NAMES.filter((n) => points[n]) };
}

/** Guided clicking of the court points on a video frame, the live fit drawn back, and Save. */
export function Calibrator({ matchId, videoUrl, width, height, duration, initialPoints, onSave, aside }: CalibratorProps) {
  const config = useConfig();
  const [clicks, setClicks] = useState<Clicks>(() => fromPoints(initialPoints));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const current = POINT_NAMES.find((n) => !clicks.done.includes(n)) ?? null;
  const fit = useMemo(
    () => assessCalibration(clicks.points, width, height, {
      minPoints: config.min_calibration_points,
      maxErrorM: config.calibration_max_error_m,
    }),
    [clicks.points, width, height, config],
  );

  const place = (point: Pt) =>
    current && setClicks((c) => ({ points: { ...c.points, [current]: point }, done: [...c.done, current] }));
  const skip = () => current && setClicks((c) => ({ ...c, done: [...c.done, current] }));
  const undo = () =>
    setClicks((c) => {
      const last = c.done.at(-1);
      if (!last) return c;
      const points = { ...c.points };
      delete points[last];
      return { points, done: c.done.slice(0, -1) };
    });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      const scale = Math.max(1, width / 960);
      const inverse = fit.inverse;
      if (inverse) {
        ctx.strokeStyle = LINE_COLOUR;
        ctx.lineWidth = 2 * scale;
        for (const [p, q] of COURT_LINES) {
          if (weight(inverse, p) * weight(inverse, q) <= 0) continue; // crosses the horizon: a bad fit
          const [x1, y1] = apply(inverse, p);
          const [x2, y2] = apply(inverse, q);
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
        if (current && weight(inverse, COURT_POINTS[current]) > 0) {
          const [x, y] = apply(inverse, COURT_POINTS[current]);
          ctx.setLineDash([4 * scale, 4 * scale]);
          ctx.beginPath();
          ctx.arc(x, y, 10 * scale, 0, 2 * Math.PI);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      ctx.font = `600 ${12 * scale}px Inter, sans-serif`;
      POINT_NAMES.forEach((name, i) => {
        const p = clicks.points[name];
        if (!p) return;
        ctx.beginPath();
        ctx.arc(p[0], p[1], 4.5 * scale, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.lineWidth = 1.5 * scale;
        ctx.strokeStyle = '#000000';
        ctx.stroke();
        ctx.fillStyle = '#ffffff';
        ctx.strokeText(String(i + 1), p[0] + 7 * scale, p[1] - 7 * scale);
        ctx.fillText(String(i + 1), p[0] + 7 * scale, p[1] - 7 * scale);
      });
    },
    [fit.inverse, clicks.points, current, width],
  );

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await onSave({ points: clicks.points as Record<string, Pt>, frame_width: width, frame_height: height });
    } catch (e) {
      setError(messageOf(e));
      setSaving(false);
    }
  };

  return (
    <div className="calibrator">
      <div className="calibrator-main">
        <FrameView videoUrl={videoUrl} width={width} height={height} duration={duration} draw={draw} onPick={place} />
        <div className="guide" aria-live="polite">
          {current ? (
            <p>
              Click: <strong>{POINT_LABELS[current]}</strong>{' '}
              <span className="muted">({POINT_NAMES.indexOf(current) + 1} of {POINT_NAMES.length})</span>
            </p>
          ) : (
            <p>All points done. Undo to change one.</p>
          )}
          <div className="button-row">
            <button type="button" className="btn" onClick={undo} disabled={clicks.done.length === 0}>Undo</button>
            <button type="button" className="btn" onClick={skip} disabled={!current}>Skip</button>
            <button type="button" className="btn" onClick={() => setClicks({ points: {}, done: [] })}
              disabled={clicks.done.length === 0}>Clear</button>
            <EarlierPoints matchId={matchId} width={width} height={height} onLoad={(points) => setClicks(fromPoints(points))} />
          </div>
        </div>
      </div>
      <aside className="calibrator-side">
        {aside}
        <CourtDiagram points={clicks.points} done={clicks.done} current={current} />
        <div className="card fit" aria-live="polite">
          <p>
            Fit error{' '}
            <strong className="num">{fit.errorM == null ? '–' : `${fit.errorM.toFixed(2)} m`}</strong>
            <span className="muted"> · limit {config.calibration_max_error_m} m · {fit.count} points</span>
          </p>
          {fit.problems.length > 0 && (
            <ul className="problems">{fit.problems.map((p) => <li key={p}>{p}</li>)}</ul>
          )}
        </div>
        {error && <p role="alert" className="error">{error}</p>}
        <button type="button" className="btn btn-primary btn-wide" onClick={save} disabled={fit.problems.length > 0 || saving}>
          {saving ? 'Saving…' : 'Save and queue'}
        </button>
      </aside>
    </div>
  );
}

/** "Use points from": the clicks of an earlier match with the same frame size. */
function EarlierPoints({ matchId, width, height, onLoad }: {
  matchId: number;
  width: number;
  height: number;
  onLoad: (points: Record<string, Pt>) => void;
}) {
  const list = useResource(listMatches, []);
  const [error, setError] = useState<string | null>(null);
  const options = (list.data?.matches ?? []).filter(
    (m) => m.id !== matchId && m.has_calibration && m.frame_width === width && m.frame_height === height,
  );
  if (options.length === 0) return null;

  const load = async (id: number) => {
    setError(null);
    try {
      const calibration = (await getMatch(id)).calibration;
      if (calibration) onLoad(calibration.points);
    } catch (e) {
      setError(messageOf(e));
    }
  };
  return (
    <label className="inline-field">
      <span>Use points from</span>
      <select value="" onChange={(e) => e.target.value && load(Number(e.target.value))}>
        <option value="">earlier match…</option>
        {options.map((m) => <option key={m.id} value={m.id}>{m.title}</option>)}
      </select>
      {error && <span role="alert" className="error">{error}</span>}
    </label>
  );
}

/** The court seen from above, near end at the bottom: which point to click next. */
function CourtDiagram({ points, done, current }: { points: Partial<Record<PointName, Pt>>; done: PointName[]; current: PointName | null }) {
  const X = (x: number) => 10 + x * 8;
  const Y = (y: number) => 10 + (20 - y) * 8;
  return (
    <svg className="court-diagram" viewBox="0 0 100 180" role="img" aria-label={current ? `Next point: ${POINT_LABELS[current]}` : 'All points done'}>
      <rect x={X(0)} y={Y(20)} width={80} height={160} className="cd-court" />
      {COURT_LINES.map(([p, q], i) => <line key={i} x1={X(p[0])} y1={Y(p[1])} x2={X(q[0])} y2={Y(q[1])} className="cd-line" />)}
      {POINT_NAMES.map((n) => {
        const [x, y] = COURT_POINTS[n];
        const state = n === current ? 'current' : points[n] ? 'placed' : done.includes(n) ? 'skipped' : 'todo';
        return <circle key={n} cx={X(x)} cy={Y(y)} r={n === current ? 5 : 3.5} className={`cd-point cd-${state}`} />;
      })}
    </svg>
  );
}
