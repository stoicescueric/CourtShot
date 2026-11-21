/** Longest step the page's own clock takes in one animation frame, seconds (a hidden tab must not jump). */
export const MAX_DT = 0.1;

/**
 * Who sets the time this animation frame (scene-data.md part C, Clock):
 * - "video": playing, with a loaded video shown (side by side or overlay); the <video> is the clock;
 * - "page": playing otherwise; t advances by dt * speed and wraps to 0 after the end;
 * - "held": paused, stepping or scrubbing; t is the master and the video is seeked to it.
 */
export type ClockSource = 'video' | 'page' | 'held';

export function clockSource(playing: boolean, videoActive: boolean): ClockSource {
  if (!playing) return 'held';
  return videoActive ? 'video' : 'page';
}

export interface ClockInput {
  t: number;
  /** Seconds since the last animation frame. */
  dt: number;
  dur: number;
  speed: number;
  playing: boolean;
  /** A loaded video is shown (mode is not "3D only"). */
  videoActive: boolean;
  videoPaused: boolean;
  /** mediaTime of the last presented video frame (requestVideoFrameCallback), when known. */
  frameTime: number | null;
  currentTime: number;
}

/** The time for this animation frame. */
export function tick(c: ClockInput): number {
  switch (clockSource(c.playing, c.videoActive)) {
    case 'held':
      return c.t;
    case 'video':
      return Math.min(c.dur, c.frameTime != null && !c.videoPaused ? c.frameTime : c.currentTime);
    case 'page': {
      const t = c.t + Math.min(MAX_DT, Math.max(0, c.dt)) * c.speed;
      return t > c.dur ? 0 : t;
    }
  }
}

/** The frame drawn at time t: exactly the video's frame while a video is shown, else fractional. */
export function drawnFrame(t: number, fps: number, n: number, videoActive: boolean): number {
  return videoActive ? Math.min(Math.round(t * fps), n - 1) : Math.min(t * fps, n - 1);
}

/** Where to seek the video for time t: 0.3 of a frame into frame round(t*fps), kept inside the file. */
export function videoSeekTime(t: number, fps: number, duration: number): number {
  return Math.min((Math.round(t * fps) + 0.3) / fps, (duration || 1e9) - 0.01);
}

/** One frame forward (dir 1) or back (dir -1) from t, on the frame grid, inside 0..dur. */
export function stepTime(t: number, fps: number, dur: number, dir: 1 | -1): number {
  return Math.min(dur, Math.max(0, Math.round(t * fps + dir) / fps));
}

/** m:ss.ss */
export function formatClock(s: number): string {
  const cs = Math.round(Math.max(0, s) * 100);
  const rest = ((cs % 6000) / 100).toFixed(2).padStart(5, '0');
  return `${Math.floor(cs / 6000)}:${rest}`;
}
