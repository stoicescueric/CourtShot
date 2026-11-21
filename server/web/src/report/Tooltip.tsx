import { createContext, useContext, useLayoutEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import type { ReportData, ShotInfo } from '../types/report';
import { shotLabel } from './context';
import { clock, has } from './format';

/** Plain text only: payload values never become markup. */
export interface TipContent {
  title: string;
  text: string;
}

interface TipApi {
  show: (ev: MouseEvent, content: TipContent) => void;
  hide: () => void;
}

interface TipState {
  content: TipContent;
  x: number;
  y: number;
  on: boolean;
}

const TipContext = createContext<TipApi | null>(null);

const PAD = 14;
const EDGE = 8;

/** Top-left corner of the tip: cursor + 14 px, flipped to the other side at the viewport edges. */
export function placeTip(
  cursor: { x: number; y: number },
  size: { width: number; height: number },
  viewport: { width: number; height: number },
): { left: number; top: number } {
  let x = cursor.x + PAD;
  let y = cursor.y + PAD;
  if (x + size.width > viewport.width - EDGE) x = cursor.x - size.width - PAD;
  if (y + size.height > viewport.height - EDGE) y = cursor.y - size.height - PAD;
  return { left: Math.max(EDGE, x), top: Math.max(EDGE, y) };
}

/** Holds the report's one shared tooltip, rendered at the end of <body> so no transformed ancestor can offset it. */
export function TipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const api = useMemo<TipApi>(
    () => ({
      show: (ev, content) => setTip({ content, x: ev.clientX, y: ev.clientY, on: true }),
      hide: () => setTip((t) => t && { ...t, on: false }),
    }),
    [],
  );
  return (
    <TipContext.Provider value={api}>
      {children}
      {createPortal(<TipBox tip={tip} />, document.body)}
    </TipContext.Provider>
  );
}

function TipBox({ tip }: { tip: TipState | null }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !tip?.on) return;
    const r = el.getBoundingClientRect();
    const { left, top } = placeTip(tip, r, { width: window.innerWidth, height: window.innerHeight });
    el.style.left = `${left}px`;
    el.style.top = `${top}px`;
  }, [tip]);
  return (
    <div ref={ref} className={tip?.on ? 'cs-report-tip on' : 'cs-report-tip'} role="status">
      {tip && (
        <>
          <b>{tip.content.title}</b>
          {tip.content.text}
        </>
      )}
    </div>
  );
}

export interface TipHandlers {
  onMouseMove: (ev: MouseEvent) => void;
  onMouseLeave: () => void;
}

/** Mouse handlers that show `content(ev)` at the cursor and hide it on leave. */
export function useTip(): (content: (ev: MouseEvent) => TipContent) => TipHandlers {
  const api = useContext(TipContext);
  if (!api) throw new Error('useTip must be used inside TipProvider');
  return (content) => ({ onMouseMove: (ev) => api.show(ev, content(ev)), onMouseLeave: api.hide });
}

const DIRECTION: Record<NonNullable<ShotInfo['direction']>, string> = {
  cross: 'cross-court',
  line: 'down the line',
  middle: 'down the middle',
};

export function shotTip(report: ReportData, s: ShotInfo): TipContent {
  const bits = [
    has(s.speed_kmh) && `${Math.round(s.speed_kmh)} km/h`,
    has(s.contact_height_m) && `contact at ${s.contact_height_m.toFixed(1)} m`,
    s.direction && DIRECTION[s.direction],
    s.target && `to ${s.target}`,
  ].filter(Boolean);
  return {
    title: `${s.player} · ${shotLabel(report, s.type)}`,
    text: clock(s.time_s - report.t0) + (bits.length ? ` · ${bits.join(' · ')}` : ''),
  };
}
