import { COURT_POINTS, POINT_NAMES, type PointName, type Pt } from './court';
import { apply, assessCalibration, fitHomography, invert, rmsError, type Mat3 } from './homography';

// A broadcast-like view of the court in a 1280 x 720 frame: court metres -> pixels, near baseline at the bottom.
const VIEW: Mat3 = [90, 20, 190, 0, -30, 680, 0, 0.07, 1];
const LIMITS = { minPoints: 6, maxErrorM: 0.25 };

function clicks(names: readonly PointName[] = POINT_NAMES, view: Mat3 = VIEW): Partial<Record<PointName, Pt>> {
  return Object.fromEntries(names.map((n) => [n, apply(view, COURT_POINTS[n])]));
}

describe('fitHomography', () => {
  test('is exact on synthetic points and inverts the view', () => {
    const src = POINT_NAMES.map((n) => apply(VIEW, COURT_POINTS[n]));
    const dst = POINT_NAMES.map((n) => COURT_POINTS[n]);
    const fit = fitHomography(src, dst)!;
    expect(fit.invertible).toBe(true);
    expect(rmsError(fit.H, src, dst)).toBeLessThan(1e-9);
    const truth = invert(VIEW)!;
    fit.H.forEach((v, i) => expect(v / fit.H[8]).toBeCloseTo(truth[i] / truth[8], 9));
  });

  test('is exact with four points', () => {
    const four: PointName[] = ['bottom_left', 'bottom_right', 'top_left', 'top_right'];
    const fit = fitHomography(four.map((n) => apply(VIEW, COURT_POINTS[n])), four.map((n) => COURT_POINTS[n]))!;
    const [x, y] = apply(fit.H, apply(VIEW, [5, 10]));
    expect(x).toBeCloseTo(5, 9);
    expect(y).toBeCloseTo(10, 9);
  });

  test('gives the fit and error of cv2.findHomography(src, dst, 0) on noisy clicks', () => {
    // videos/final_f_calibration.json, six of its points moved by up to 30 px; the expected H and RMS error are
    // OpenCV 4.11's (findHomography, method 0). The DLT alone gives 0.279 m here, above the 0.25 m limit.
    const names: PointName[] = ['bottom_left', 'top_right', 'net_left', 'service_near_center', 'service_far_left', 'service_far_center'];
    const src: Pt[] = [
      [227.39579554855578, 962.0328247901242], [1343.6321442366643, 306.9312840382482], [480.38267321489934, 516.9986776031519],
      [948.9429456616847, 756.7530809153715], [558.9470441783731, 374.4465132721803], [956.2592299807941, 368.78225296486113],
    ];
    const dst = names.map((n) => COURT_POINTS[n]);
    const opencv = [
      0.04253179338919256, 0.02692466174556961, -34.12357667909161,
      -0.0014442550747385315, -0.09896008850779497, 94.05837702377322,
      0.0002161811779159133, 0.005828658026722506, 1,
    ];
    const fit = fitHomography(src, dst)!;
    expect(rmsError(fit.H, src, dst)).toBeCloseTo(0.2482940440247067, 6);
    fit.H.forEach((v, i) => expect(Math.abs(v / fit.H[8] / opencv[i] - 1)).toBeLessThan(1e-4));
  });

  test('refuses fewer than four points, coincident points and points on one line', () => {
    const line = (k: number): Pt => [100 + 50 * k, 200 + 20 * k];
    expect(fitHomography([line(0), line(1), line(2)], [[0, 0], [10, 0], [0, 20]])).toBeNull();
    expect(fitHomography([line(0), line(0), line(0), line(0)], [[0, 0], [10, 0], [0, 20], [10, 20]])).toBeNull();
    expect(fitHomography([line(0), line(1), line(2), line(3)], [[0, 0], [10, 0], [0, 20], [10, 20]])).toBeNull();
  });
});

describe('rmsError', () => {
  test('is the root mean square distance in court metres', () => {
    const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const src: Pt[] = [[0, 0], [10, 0], [0, 20], [10, 20]];
    expect(rmsError(identity, src, src.map(([x, y]) => [x + 0.3, y - 0.4]))).toBeCloseTo(0.5, 12);
    expect(rmsError(identity, src, [[0, 0], [10, 0], [0, 20], [10, 22]])).toBeCloseTo(1, 12);
  });
});

describe('assessCalibration', () => {
  test('accepts a clean calibration', () => {
    const a = assessCalibration(clicks(), 1280, 720, LIMITS);
    expect(a.problems).toEqual([]);
    expect(a.errorM).toBeLessThan(1e-9);
    expect(a.inverse).not.toBeNull();
    const [u, v] = apply(a.inverse!, [10, 20]);
    expect(u).toBeCloseTo(apply(VIEW, [10, 20])[0], 6);
    expect(v).toBeCloseTo(apply(VIEW, [10, 20])[1], 6);
  });

  test('reports the error of a misplaced click in metres and refuses one above the limit', () => {
    const points = clicks();
    const [x, y] = points.service_near_center!;
    const slightly = assessCalibration({ ...points, service_near_center: [x + 4, y] }, 1280, 720, LIMITS);
    expect(slightly.errorM).toBeGreaterThan(0.005);
    expect(slightly.errorM).toBeLessThan(LIMITS.maxErrorM);
    expect(slightly.problems).toEqual([]);

    const badly = assessCalibration({ ...points, service_near_center: [x + 150, y] }, 1280, 720, LIMITS);
    expect(badly.errorM).toBeGreaterThan(LIMITS.maxErrorM);
    expect(badly.problems.join(' ')).toMatch(/above the 0.25 m limit/);
  });

  test('detects left and right swapped, although the fit itself is exact', () => {
    const mirror: Mat3 = [-1, 0, 1280, 0, 1, 0, 0, 0, 1];
    const a = assessCalibration(clicks(POINT_NAMES, compose(mirror, VIEW)), 1280, 720, LIMITS);
    expect(a.errorM).toBeLessThan(1e-9);
    expect(a.problems).toEqual([
      'The points look mirrored: “Near baseline, left corner” must be left of “Near baseline, right corner” in the image.',
    ]);
  });

  test('detects near and far swapped', () => {
    const flip: Mat3 = [1, 0, 0, 0, -1, 720, 0, 0, 1];
    const a = assessCalibration(clicks(POINT_NAMES, compose(flip, VIEW)), 1280, 720, LIMITS);
    expect(a.problems).toEqual([
      'The points look mirrored: “Near baseline, left corner” must be lower in the image than “Far baseline, left corner”.',
    ]);
  });

  // The two halves of the court are alike, so these mirrored fits are exact and draw the real lines.
  test('detects near points clicked on the far half, with the net points right', () => {
    const near: PointName[] = ['bottom_left', 'bottom_right', 'net_left', 'net_right', 'service_near_left', 'service_near_center', 'service_near_right'];
    const atFar: Partial<Record<PointName, Pt>> = {
      bottom_left: apply(VIEW, COURT_POINTS.top_left),
      bottom_right: apply(VIEW, COURT_POINTS.top_right),
      service_near_left: apply(VIEW, COURT_POINTS.service_far_left),
      service_near_center: apply(VIEW, COURT_POINTS.service_far_center),
      service_near_right: apply(VIEW, COURT_POINTS.service_far_right),
    };
    const a = assessCalibration({ ...clicks(near), ...atFar }, 1280, 720, LIMITS);
    expect(a.errorM).toBeLessThan(1e-9);
    expect(a.problems).toEqual([
      'The points look mirrored: “Near baseline, left corner” must be lower in the image than “Net, left end”.',
    ]);
  });

  test('detects left points clicked on the right side, with no right points to compare', () => {
    const leftAndCentre: PointName[] = ['bottom_left', 'top_left', 'net_left', 'service_near_left', 'service_near_center', 'service_far_left', 'service_far_center'];
    const atRight: Partial<Record<PointName, Pt>> = {
      bottom_left: apply(VIEW, COURT_POINTS.bottom_right),
      top_left: apply(VIEW, COURT_POINTS.top_right),
      net_left: apply(VIEW, COURT_POINTS.net_right),
      service_near_left: apply(VIEW, COURT_POINTS.service_near_right),
      service_far_left: apply(VIEW, COURT_POINTS.service_far_right),
    };
    const a = assessCalibration({ ...clicks(leftAndCentre), ...atRight }, 1280, 720, LIMITS);
    expect(a.errorM).toBeLessThan(1e-9);
    expect(a.problems).toEqual([
      'The points look mirrored: “Near service line, left end” must be left of “Near service line, centre” in the image.',
    ]);
  });

  test('accepts points on one half only', () => {
    const nearOnly: PointName[] = ['bottom_left', 'bottom_right', 'net_left', 'net_right', 'service_near_left', 'service_near_right'];
    const a = assessCalibration(clicks(nearOnly), 1280, 720, LIMITS);
    expect(a.mirrored).toBeNull();
    expect(a.problems).toEqual([]);
  });

  test('needs the minimum number of points', () => {
    const a = assessCalibration(clicks(['bottom_left', 'bottom_right', 'top_left', 'top_right']), 1280, 720, LIMITS);
    expect(a.errorM).toBeLessThan(1e-9);
    expect(a.problems).toEqual(['Place at least 6 points (4 so far).']);
  });

  test('refuses points outside the frame', () => {
    const a = assessCalibration({ ...clicks(), top_right: [1290, 40] }, 1280, 720, LIMITS);
    expect(a.outside).toEqual(['top_right']);
    expect(a.problems.join(' ')).toMatch(/Outside the frame: Far baseline, right corner/);
  });

  test('refuses degenerate clicks', () => {
    const line = (k: number): Pt => [100 + 50 * k, 600 - 20 * k];
    const names: PointName[] = ['bottom_left', 'bottom_right', 'top_left', 'top_right', 'net_left', 'net_right'];
    const a = assessCalibration(Object.fromEntries(names.map((n, k) => [n, line(k)])), 1280, 720, LIMITS);
    expect(a.H === null || !a.invertible).toBe(true);
    expect(a.inverse).toBeNull();
    expect(a.problems.join(' ')).toMatch(/do not fit a court/);
  });

  test('refuses a fit that maps the clicks onto one court line', () => {
    // Every chosen court point lies on x = 0, the clicks do not: only a singular matrix fits.
    const names: PointName[] = ['bottom_left', 'top_left', 'net_left', 'service_near_left', 'service_far_left'];
    const scattered: Pt[] = [[100, 600], [400, 80], [700, 300], [200, 400], [900, 650]];
    const a = assessCalibration(Object.fromEntries(names.map((n, k) => [n, scattered[k]])), 1280, 720, LIMITS);
    expect(a.H === null || !a.invertible).toBe(true);
    expect(a.problems.join(' ')).toMatch(/do not fit a court/);
  });
});

/** The view followed by an image transform (a mirror or a flip). */
function compose(image: Mat3, view: Mat3): Mat3 {
  const out = new Array<number>(9).fill(0);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) out[r * 3 + c] += image[r * 3 + k] * view[k * 3 + c];
  return out;
}
