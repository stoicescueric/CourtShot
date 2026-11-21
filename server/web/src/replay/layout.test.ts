import { SCRUB_MAX, percentIn, rallyWindows, scrubValue, spanBox, timeFromScrub, windowIndexAt } from './layout';

describe('rally windows', () => {
  it('cover the whole clip with one window when there are no rallies', () => {
    expect(rallyWindows([], 10)).toEqual([[0, 10]]);
  });

  it('cut halfway through each gap, from 0 to the end', () => {
    expect(rallyWindows([[0.5, 4.4], [6, 8.5]], 10)).toEqual([[0, 5.2], [5.2, 10]]);
    expect(rallyWindows([[10, 20], [30, 40], [100, 110]], 300)).toEqual([[0, 25], [25, 70], [70, 300]]);
  });

  it('sort the rallies and keep the cuts inside the clip and in order', () => {
    expect(rallyWindows([[6, 8.5], [0.5, 4.4]], 10)).toEqual([[0, 5.2], [5.2, 10]]);
    const w = rallyWindows([[2, 9], [5, 12]], 10); // overlapping, past the end
    expect(w[0][0]).toBe(0);
    expect(w[w.length - 1][1]).toBe(10);
    for (const [a, b] of w) expect(b).toBeGreaterThanOrEqual(a);
  });

  it('give each time the window it lies in; a cut belongs to the later window', () => {
    const w = rallyWindows([[0.5, 4.4], [6, 8.5]], 10);
    expect(windowIndexAt(-1, w)).toBe(0);
    expect(windowIndexAt(5.1, w)).toBe(0);
    expect(windowIndexAt(5.2, w)).toBe(1);
    expect(windowIndexAt(10, w)).toBe(1);
  });
});

describe('lane layout', () => {
  it('places a time in a window as a clamped percentage', () => {
    expect(percentIn(15, [10, 30])).toBe(25);
    expect(percentIn(5, [10, 30])).toBe(0);
    expect(percentIn(40, [10, 30])).toBe(100);
    expect(percentIn(3, [3, 3])).toBe(0);
  });

  it('clips a span to the window and drops it when outside', () => {
    expect(spanBox(12, 14, [10, 30])).toEqual({ left: 10, width: 10 });
    const clipped = spanBox(4, 6, [0, 5]);
    expect(clipped?.left).toBe(80);
    expect(clipped?.width).toBe(20);
    expect(spanBox(31, 33, [10, 30])).toBeNull();
    expect(spanBox(1, 9, [10, 30])).toBeNull();
  });

  it('maps the scrub control to the window and back', () => {
    expect(timeFromScrub(0, [5, 15])).toBe(5);
    expect(timeFromScrub(SCRUB_MAX, [5, 15])).toBe(15);
    expect(timeFromScrub(250, [5, 15])).toBe(7.5);
    expect(scrubValue(7.5, [5, 15])).toBe(250);
    expect(scrubValue(100, [5, 15])).toBe(SCRUB_MAX);
  });
});
