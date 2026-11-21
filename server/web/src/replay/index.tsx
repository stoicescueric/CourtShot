import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import type { SceneData } from '../types/scene';
import { VIEW_NAMES, type ViewName } from './camera';
import { formatClock, stepTime } from './clock';
import { decodeScene } from './decode';
import { INITIAL_STATE, createReplay, type Layers, type ReplayHandle, type ReplayState, type VideoMode } from './engine';
import { LowerThird } from './LowerThird';
import { Timeline, type ScrubKind, type TimelineRefs } from './Timeline';
import { percentIn, rallyWindows, scrubValue, windowIndexAt } from './layout';
import './replay.css';

const VIEW_LABELS: Record<ViewName, string> = {
  broadcast: 'Broadcast', far: 'Far end', side: 'Side', top: 'Top', follow: 'Follow ball',
};
const MODES: [VideoMode, string][] = [['off', '3D only'], ['side', 'Side by side'], ['overlay', 'Overlay']];
const LAYERS: [keyof Layers, string][] = [
  ['trail', 'Trail'], ['path', 'Flight path'], ['guide', 'Height line'], ['low', 'Low-confidence'], ['labels', 'Names'],
];
const SPEEDS: [number, string][] = [[0.25, '¼×'], [0.5, '½×'], [1, '1×']];
const WEBGL_ERROR = 'The 3D replay needs WebGL, which this browser could not start.';

/** The 3D replay of one match. Fills its parent, which must have a size. */
export function Replay(props: { scene: SceneData; videoUrl: string | null }): JSX.Element {
  const { scene, videoUrl } = props;
  // typed arrays at once, once per scene object; the engine reads the same memoised result
  const decoded = useMemo(() => decodeScene(scene), [scene]);
  const dur = decoded.dur;
  const windows = useMemo(() => rallyWindows(scene.rallies, dur), [scene, dur]);

  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const clockRef = useRef<HTMLSpanElement>(null);
  const heightRef = useRef<HTMLSpanElement>(null);
  const unitRef = useRef<HTMLElement>(null);
  const timeline: TimelineRefs = {
    head: useRef<HTMLDivElement>(null),
    scrub: useRef<HTMLInputElement>(null),
    clipHead: useRef<HTMLDivElement>(null),
    clipScrub: useRef<HTMLInputElement>(null),
  };
  const engineRef = useRef<ReplayHandle | null>(null);
  const [state, setState] = useState<ReplayState>(INITIAL_STATE);
  const [win, setWin] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const live = useRef({ state, t: 0, win: 0, windows, dragging: null as ScrubKind | null, wasPlaying: true });
  live.current.state = state;
  live.current.windows = windows;

  /** Per-frame DOM writes: clock, ball height, playheads; React state only when the rally window changes. */
  const paintTime = (t: number, ballY: number | null) => {
    const l = live.current;
    l.t = t;
    if (clockRef.current) clockRef.current.textContent = formatClock(t);
    if (heightRef.current) heightRef.current.textContent = ballY === null ? '–' : ballY.toFixed(2);
    if (unitRef.current) unitRef.current.hidden = ballY === null;
    if (l.dragging !== 'rally') {
      const i = windowIndexAt(t, l.windows);
      if (i !== l.win) {
        l.win = i;
        setWin(i);
      }
    }
    const w = l.windows[l.win] ?? [0, dur];
    const { head, scrub, clipHead, clipScrub } = timeline;
    if (head.current) head.current.style.left = `${percentIn(t, w)}%`;
    if (clipHead.current) clipHead.current.style.left = `${percentIn(t, [0, dur])}%`;
    if (scrub.current && l.dragging !== 'rally') scrub.current.value = String(scrubValue(t, w));
    if (clipScrub.current && l.dragging !== 'clip') clipScrub.current.value = String(scrubValue(t, [0, dur]));
  };
  const paintRef = useRef(paintTime);
  paintRef.current = paintTime;

  useEffect(() => {
    const host = stageRef.current;
    if (!host) return;
    let engine: ReplayHandle;
    try {
      engine = createReplay(host, videoRef.current, scene, { onError: setError });
    } catch {
      setError(WEBGL_ERROR);
      return;
    }
    setError(null);
    live.current.win = 0;
    setWin(0);
    engineRef.current = engine;
    const unsubscribe = engine.subscribe((e) => (e.type === 'time' ? paintRef.current(e.t, e.ballY) : setState(e.state)));
    return () => {
      unsubscribe();
      engine.dispose();
      engineRef.current = null;
    };
  }, [scene]);

  const onScrubStart = (kind: ScrubKind) => {
    live.current.dragging = kind;
    live.current.wasPlaying = live.current.state.playing;
    engineRef.current?.setPlaying(false);
  };
  const onScrubEnd = () => {
    if (live.current.dragging === null) return;
    live.current.dragging = null;
    engineRef.current?.setPlaying(live.current.wasPlaying);
  };

  /** Shortcuts on the replay element only: Space, arrows, 1-5 (html:646-652). */
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const engine = engineRef.current;
    if (!engine || e.altKey || e.ctrlKey || e.metaKey) return;
    const tag = (e.target as HTMLElement).tagName;
    if (e.key === ' ') {
      if (tag === 'BUTTON') return; // the focused button's own Space press wins
      e.preventDefault();
      engine.setPlaying(!live.current.state.playing);
      return;
    }
    if (tag === 'INPUT') return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      engine.setPlaying(false);
      engine.seek(stepTime(live.current.t, decoded.fps, dur, e.key === 'ArrowRight' ? 1 : -1));
      return;
    }
    const view = VIEW_NAMES[Number(e.key) - 1];
    if (e.key.length === 1 && view) engine.setView(view);
  };

  const okFlights = scene.flights.filter((f) => f.ok).length;
  const overlay = state.mode === 'overlay';

  return (
    <div
      className={`cs-replay cs-replay--${state.mode}`}
      tabIndex={0}
      role="region"
      aria-label="3D replay"
      aria-keyshortcuts="Space ArrowLeft ArrowRight 1 2 3 4 5"
      onKeyDown={onKeyDown}
    >
      <div className="rp-video">
        <video ref={videoRef} src={videoUrl ?? undefined} muted playsInline preload="auto" loop />
        {state.videoError && <p className="rp-video-msg">This browser could not play the replay video.</p>}
      </div>
      <div ref={stageRef} className="rp-stage" />

      <div className="rp-ui rp-top">
        <p className="rp-sub">
          {formatClock(dur)} · {okFlights} of {scene.flights.length} ball flights reconstructed in 3D · {scene.players.length} players
        </p>
        <div className="rp-groups">
          {videoUrl && (
            <div className="rp-seg" role="group" aria-label="Video">
              {MODES.map(([m, label]) => (
                <button key={m} type="button" aria-pressed={state.mode === m} onClick={() => engineRef.current?.setMode(m)}>
                  {label}
                </button>
              ))}
            </div>
          )}
          <div className="rp-seg" role="group" aria-label="Camera">
            {VIEW_NAMES.map((v, i) => (
              <button
                key={v}
                type="button"
                title={`Key ${i + 1}`}
                aria-pressed={state.view === v}
                disabled={overlay}
                onClick={() => engineRef.current?.setView(v)}
              >
                {VIEW_LABELS[v]}
              </button>
            ))}
          </div>
          <div className="rp-seg" role="group" aria-label="Layers">
            {LAYERS.map(([k, label]) => (
              <button key={k} type="button" aria-pressed={state.layers[k]} onClick={() => engineRef.current?.setLayers({ [k]: !state.layers[k] })}>
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <LowerThird scene={scene} flight={state.flight} heightRef={heightRef} unitRef={unitRef} />

      <p className="rp-ui rp-help">
        Drag to orbit · scroll to zoom · right-drag to pan
        <br />
        Space play/pause · ← → step a frame · 1–5 cameras
      </p>

      <div className="rp-ui rp-transport">
        <button
          type="button"
          className="rp-play"
          aria-label={state.playing ? 'Pause' : 'Play'}
          onClick={() => engineRef.current?.setPlaying(!state.playing)}
        >
          <svg viewBox="0 0 16 16" aria-hidden="true">
            {state.playing ? (
              <>
                <rect x="3" y="2" width="3.5" height="12" rx="1" />
                <rect x="9.5" y="2" width="3.5" height="12" rx="1" />
              </>
            ) : (
              <path d="M4 2.5v11l9-5.5z" />
            )}
          </svg>
        </button>
        <Timeline
          scene={scene}
          windows={windows}
          win={win}
          dur={dur}
          refs={timeline}
          onScrubStart={onScrubStart}
          onScrub={(t) => engineRef.current?.seek(t)}
          onScrubEnd={onScrubEnd}
        />
        <div className="rp-right">
          <div className="rp-seg rp-speed" role="group" aria-label="Playback speed">
            {SPEEDS.map(([s, label]) => (
              <button key={s} type="button" aria-pressed={state.speed === s} onClick={() => engineRef.current?.setSpeed(s)}>
                {label}
              </button>
            ))}
          </div>
          <span ref={clockRef} className="rp-clock" role="timer">
            {formatClock(0)}
          </span>
        </div>
      </div>

      {error && (
        <p className="rp-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
