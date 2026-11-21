import { MAX_DT, clockSource, drawnFrame, formatClock, stepTime, tick, videoSeekTime, type ClockInput } from './clock';

const base: ClockInput = {
  t: 2, dt: 0.04, dur: 10, speed: 1, playing: true,
  videoActive: false, videoPaused: true, frameTime: null, currentTime: 0,
};

describe('which source is the clock', () => {
  it('is the video while playing with a loaded video shown', () => {
    expect(clockSource(true, true)).toBe('video');
  });
  it("is the page's own clock while playing without a video shown", () => {
    expect(clockSource(true, false)).toBe('page');
  });
  it('is the held time while paused, stepping or scrubbing, video or not', () => {
    expect(clockSource(false, true)).toBe('held');
    expect(clockSource(false, false)).toBe('held');
  });
});

describe('tick', () => {
  it('holds t while paused even when the video moves', () => {
    expect(tick({ ...base, playing: false, videoActive: true, videoPaused: false, frameTime: 7, currentTime: 7 })).toBe(2);
  });

  it('follows the presented video frame while the video plays', () => {
    expect(tick({ ...base, videoActive: true, videoPaused: false, frameTime: 3.2, currentTime: 3.25 })).toBe(3.2);
  });

  it('falls back to currentTime without a frame callback or while the video is paused', () => {
    expect(tick({ ...base, videoActive: true, videoPaused: false, frameTime: null, currentTime: 3.25 })).toBe(3.25);
    expect(tick({ ...base, videoActive: true, videoPaused: true, frameTime: 3.2, currentTime: 3.25 })).toBe(3.25);
  });

  it('caps the video clock at the scene duration', () => {
    expect(tick({ ...base, videoActive: true, videoPaused: false, frameTime: 10.04, currentTime: 10.04 })).toBe(10);
  });

  it('advances its own clock by dt times speed, with dt capped', () => {
    expect(tick({ ...base, speed: 0.5 })).toBeCloseTo(2.02);
    expect(tick({ ...base, dt: 3 })).toBeCloseTo(2 + MAX_DT);
    expect(tick({ ...base, dt: -0.5 })).toBe(2);
  });

  it('wraps its own clock to 0 after the end', () => {
    expect(tick({ ...base, t: 9.99, dt: 0.05 })).toBe(0);
  });
});

describe('frames and seeking', () => {
  it('draws exactly the video frame while a video is shown, a fractional frame otherwise', () => {
    expect(drawnFrame(1.03, 25, 251, true)).toBe(26);
    expect(drawnFrame(1.03, 25, 251, false)).toBeCloseTo(25.75);
    expect(drawnFrame(20, 25, 251, false)).toBe(250);
    expect(drawnFrame(20, 25, 251, true)).toBe(250);
  });

  it('seeks the video 0.3 of a frame into the frame of t, inside the file', () => {
    expect(videoSeekTime(1.03, 25, 12)).toBeCloseTo(26.3 / 25);
    expect(videoSeekTime(9.99, 25, 10)).toBeCloseTo(9.99);
    expect(videoSeekTime(1, 25, NaN)).toBeCloseTo(25.3 / 25);
  });

  it('steps one frame on the frame grid, inside the clip', () => {
    expect(stepTime(1.01, 25, 10, 1)).toBeCloseTo(26 / 25);
    expect(stepTime(1.01, 25, 10, -1)).toBeCloseTo(24 / 25);
    expect(stepTime(0, 25, 10, -1)).toBe(0);
    expect(stepTime(10, 25, 10, 1)).toBe(10);
  });

  it('formats m:ss.ss', () => {
    expect(formatClock(0)).toBe('0:00.00');
    expect(formatClock(61.5)).toBe('1:01.50');
    expect(formatClock(59.999)).toBe('1:00.00');
    expect(formatClock(605.25)).toBe('10:05.25');
  });
});
