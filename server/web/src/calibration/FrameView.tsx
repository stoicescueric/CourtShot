import { useEffect, useRef, useState, type MouseEvent } from 'react';
import { clock } from '../format';
import type { Pt } from './court';

/** How long to wait for a presented frame after a seek before drawing anyway. */
const FRAME_FALLBACK_MS = 200;

interface FrameViewProps {
  videoUrl: string;
  /** Frame size in video pixels: the coordinate system of the clicks and of `draw`. */
  width: number;
  height: number;
  duration: number;
  /** Draws the overlay; called with a cleared canvas in video pixels whenever it changes. */
  draw: (ctx: CanvasRenderingContext2D) => void;
  onPick: (point: Pt) => void;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** A frame of the video chosen with a slider, with an overlay canvas that turns clicks into video pixels. */
export function FrameView({ videoUrl, width, height, duration, draw, onPick }: FrameViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const frameRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [time, setTime] = useState(() => duration * 0.1);
  const [shown, setShown] = useState(false);
  const [failed, setFailed] = useState(false);

  // Draw only once the seek has finished and the new frame has been presented; otherwise the canvas can be blank.
  useEffect(() => {
    const video = videoRef.current!;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let handle: number | undefined;
    const paint = () => {
      clearTimeout(timer);
      if (handle !== undefined) video.cancelVideoFrameCallback?.(handle);
      handle = undefined;
      frameRef.current?.getContext('2d')?.drawImage(video, 0, 0, width, height);
      setShown(true);
    };
    const onSeeked = () => {
      clearTimeout(timer);
      if (video.requestVideoFrameCallback) handle = video.requestVideoFrameCallback(paint);
      timer = setTimeout(paint, FRAME_FALLBACK_MS);
    };
    video.addEventListener('seeked', onSeeked);
    return () => {
      video.removeEventListener('seeked', onSeeked);
      clearTimeout(timer);
      if (handle !== undefined) video.cancelVideoFrameCallback?.(handle);
    };
  }, [width, height]);

  useEffect(() => {
    const video = videoRef.current!;
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) video.currentTime = time;
  }, [time]);

  useEffect(() => {
    const ctx = overlayRef.current?.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    draw(ctx);
  }, [draw, width, height]);

  const pick = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * width) / rect.width;
    const y = ((e.clientY - rect.top) * height) / rect.height;
    onPick([clamp(Math.round(x * 10) / 10, 0, width - 1), clamp(Math.round(y * 10) / 10, 0, height - 1)]);
  };

  return (
    <div className="frame-view">
      <video
        ref={videoRef}
        src={videoUrl}
        className="offscreen-video"
        muted
        playsInline
        preload="auto"
        aria-hidden="true"
        onLoadedMetadata={(e) => (e.currentTarget.currentTime = time)}
        onError={() => setFailed(true)}
      />
      <div className="frame" style={{ aspectRatio: `${width} / ${height}` }}>
        <canvas ref={frameRef} width={width} height={height} aria-hidden="true" />
        <canvas ref={overlayRef} width={width} height={height} className="overlay" onClick={pick}
          aria-label="Video frame: click the court point named below" role="img" />
        {!shown && <p className="frame-note">{failed ? 'The video could not be loaded.' : 'Loading the frame…'}</p>}
      </div>
      <label className="field frame-slider">
        <span>Frame at <span className="num">{clock(time)}</span>: pick one where the court lines are clear</span>
        <input type="range" min={0} max={Math.max(0, duration - 0.1)} step="any" value={time} onChange={(e) => setTime(Number(e.target.value))} />
      </label>
    </div>
  );
}
