/**
 * The browser's court fit: the same least-squares fit over all clicked points and the same checks as the worker.
 *
 * - Fit: normalised DLT (Hartley), pixels -> court metres, h33 = 1, solved by least squares over all points, then
 *   refined by Levenberg-Marquardt on |H(pixel) - court point|, as cv2.findHomography(src, dst, 0) does.
 * - Error: RMS over the clicked points of |H(pixel) - court point|, in metres.
 * - Invertible: |det| of the fit in normalised coordinates above 1e-10 of its Frobenius norm cubed.
 * - Mirror (the worker's pairwise rule): two points with the same court y must run left to right in the image, and of
 *   two points with the same court x the nearer (smaller court y) must be lower in the image.
 * - Inside: 0 <= x <= frame width and 0 <= y <= frame height.
 */
import { COURT_POINTS, POINT_LABELS, POINT_NAMES, type PointName, type Pt } from './court';

/** 3 x 3 matrix, row-major. */
export type Mat3 = number[];

export function apply(H: Mat3, [x, y]: Pt): Pt {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

/** Homogeneous weight of a point: its sign tells on which side of the horizon the point lies. */
export function weight(H: Mat3, [x, y]: Pt): number {
  return H[6] * x + H[7] * y + H[8];
}

function multiply(A: Mat3, B: Mat3): Mat3 {
  const C = new Array<number>(9).fill(0);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      for (let k = 0; k < 3; k++) C[r * 3 + c] += A[r * 3 + k] * B[k * 3 + c];
  return C;
}

function det(H: Mat3): number {
  return H[0] * (H[4] * H[8] - H[5] * H[7]) - H[1] * (H[3] * H[8] - H[5] * H[6]) + H[2] * (H[3] * H[7] - H[4] * H[6]);
}

export function invert(H: Mat3): Mat3 | null {
  const d = det(H);
  if (!Number.isFinite(d) || d === 0) return null;
  const [a, b, c, e, f, g, h, i, j] = H;
  return [f * j - g * i, c * i - b * j, b * g - c * f, g * h - e * j, a * j - c * h, c * e - a * g, e * i - f * h, b * h - a * i, a * f - b * e]
    .map((v) => v / d);
}

/** Moves the centroid to the origin and the mean distance to sqrt(2); null when all points coincide. */
function normaliser(points: Pt[]): Mat3 | null {
  const cx = points.reduce((s, p) => s + p[0], 0) / points.length;
  const cy = points.reduce((s, p) => s + p[1], 0) / points.length;
  const mean = points.reduce((s, p) => s + Math.hypot(p[0] - cx, p[1] - cy), 0) / points.length;
  if (!(mean > 0)) return null;
  const s = Math.SQRT2 / mean;
  return [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1];
}

/** Gaussian elimination with partial pivoting; null when the system is singular. */
function solve(M: number[][], r: number[]): number[] | null {
  const n = r.length;
  const A = M.map((row, i) => [...row, r[i]]);
  const scale = Math.max(...M.flat().map(Math.abs));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(A[row][col]) > Math.abs(A[pivot][col])) pivot = row;
    if (!(Math.abs(A[pivot][col]) > 1e-12 * scale)) return null;
    [A[col], A[pivot]] = [A[pivot], A[col]];
    for (let row = col + 1; row < n; row++) {
      const f = A[row][col] / A[col][col];
      for (let k = col; k <= n; k++) A[row][k] -= f * A[col][k];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let s = A[row][n];
    for (let k = row + 1; k < n; k++) s -= A[row][k] * x[k];
    x[row] = s / A[row][row];
  }
  return x;
}

/** Normal equations (A^T A, A^T b) of the 2n x 8 system A h = b built from the rows of every point. */
function normalEquations(points: Pt[], rows: (p: Pt, i: number) => [number[], number][]): [number[][], number[]] {
  const M = Array.from({ length: 8 }, () => new Array<number>(8).fill(0));
  const r = new Array<number>(8).fill(0);
  points.forEach((p, i) => {
    for (const [row, b] of rows(p, i)) {
      for (let j = 0; j < 8; j++) {
        r[j] += row[j] * b;
        for (let k = 0; k < 8; k++) M[j][k] += row[j] * row[k];
      }
    }
  });
  return [M, r];
}

/** Sum of the squared distances between the mapped source points and their targets. */
function squaredError(H: Mat3, src: Pt[], dst: Pt[]): number {
  return src.reduce((s, p, i) => {
    const [x, y] = apply(H, p);
    return s + (x - dst[i][0]) ** 2 + (y - dst[i][1]) ** 2;
  }, 0);
}

/**
 * Levenberg-Marquardt on the eight entries of H (h33 = 1) minimising squaredError, at most 10 iterations: the
 * refinement cv2.findHomography applies to its DLT, so the browser reaches the worker's fit and error.
 */
function refine(H: Mat3, src: Pt[], dst: Pt[]): Mat3 {
  let h = H.map((v) => v / H[8]);
  let cost = squaredError(h, src, dst);
  let lambda = 1e-3;
  for (let iteration = 0; iteration < 10; iteration++) {
    // Gauss-Newton step on the residuals H(p) - d: J^T J delta = -J^T r.
    const [JtJ, minusJtr] = normalEquations(src, ([x, y], i) => {
      const w = h[6] * x + h[7] * y + 1;
      const [u, v] = apply(h, [x, y]);
      return [
        [[x / w, y / w, 1 / w, 0, 0, 0, (-x * u) / w, (-y * u) / w], dst[i][0] - u],
        [[0, 0, 0, x / w, y / w, 1 / w, (-x * v) / w, (-y * v) / w], dst[i][1] - v],
      ];
    });
    let improved = false;
    for (; lambda < 1e10 && !improved; lambda *= 10) {
      const step = solve(JtJ.map((row, j) => row.map((v, k) => (j === k ? v * (1 + lambda) : v))), minusJtr);
      if (!step) continue;
      const candidate = [...step.map((d, j) => h[j] + d), 1];
      const candidateCost = squaredError(candidate, src, dst);
      if (candidateCost < cost) [h, cost, improved] = [candidate, candidateCost, true];
    }
    if (!improved) break;
    lambda /= 100; // undo the loop's last growth, then relax
  }
  return h;
}

export interface Fit {
  /** Pixels -> court metres. */
  H: Mat3;
  invertible: boolean;
}

/** Least-squares fit over all pairs; null with fewer than 4 pairs or a degenerate layout (e.g. points on one line). */
export function fitHomography(src: Pt[], dst: Pt[]): Fit | null {
  if (src.length < 4 || src.length !== dst.length) return null;
  const Ts = normaliser(src);
  const Td = normaliser(dst);
  if (!Ts || !Td) return null;

  // Similarity transforms leave the least-squares minimiser unchanged, so the refinement runs normalised too.
  const srcN = src.map((p) => apply(Ts, p));
  const dstN = dst.map((p) => apply(Td, p));
  const h = solve(
    ...normalEquations(srcN, ([x, y], i) => {
      const [u, v] = dstN[i];
      return [
        [[x, y, 1, 0, 0, 0, -x * u, -y * u], u],
        [[0, 0, 0, x, y, 1, -x * v, -y * v], v],
      ];
    }),
  );
  if (!h) return null;

  const Hn = refine([...h, 1], srcN, dstN);
  const norm = Math.hypot(...Hn);
  const invertible = Math.abs(det(Hn)) > 1e-10 * norm ** 3;
  return { H: multiply(multiply(invert(Td)!, Hn), Ts), invertible };
}

/** Root mean square of the distances between the mapped source points and their targets (court metres). */
export function rmsError(H: Mat3, src: Pt[], dst: Pt[]): number {
  const rms = Math.sqrt(squaredError(H, src, dst) / src.length);
  return Number.isFinite(rms) ? rms : Infinity;
}

export interface Limits {
  minPoints: number;
  maxErrorM: number;
}

export interface Assessment {
  count: number;
  /** Pixels -> court metres; null with fewer than 4 points or a degenerate layout. */
  H: Mat3 | null;
  /** Court metres -> pixels, for drawing the court back; null when H is null or not invertible. */
  inverse: Mat3 | null;
  /** RMS reprojection error, metres. */
  errorM: number | null;
  invertible: boolean;
  /** The first pair of points in the wrong image order, as a sentence; null when none is. */
  mirrored: string | null;
  outside: PointName[];
  /** Why it cannot be saved; empty when it can. */
  problems: string[];
}

/** The worker's check (court._check_not_mirrored), pair by pair over the clicked points. */
function mirrorProblem(points: Partial<Record<PointName, Pt>>, names: PointName[]): string | null {
  for (const a of names) {
    for (const b of names) {
      const [[cax, cay], [cbx, cby]] = [COURT_POINTS[a], COURT_POINTS[b]];
      const [[ax, ay], [bx, by]] = [points[a]!, points[b]!];
      if (cay === cby && cax < cbx && !(ax < bx))
        return `The points look mirrored: “${POINT_LABELS[a]}” must be left of “${POINT_LABELS[b]}” in the image.`;
      if (cax === cbx && cay < cby && !(ay > by))
        return `The points look mirrored: “${POINT_LABELS[a]}” must be lower in the image than “${POINT_LABELS[b]}”.`;
    }
  }
  return null;
}

export function assessCalibration(
  points: Partial<Record<PointName, Pt>>,
  width: number,
  height: number,
  limits: Limits,
): Assessment {
  const names = POINT_NAMES.filter((n) => points[n]);
  const src = names.map((n) => points[n]!);
  const dst = names.map((n) => COURT_POINTS[n]);

  const fit = fitHomography(src, dst);
  const H = fit?.H ?? null;
  const invertible = fit?.invertible ?? false;
  const inverse = H && invertible ? invert(H) : null;
  const errorM = H ? rmsError(H, src, dst) : null;

  const mirrored = mirrorProblem(points, names);
  const outside = names.filter((n) => {
    const [x, y] = points[n]!;
    return !(x >= 0 && x <= width && y >= 0 && y <= height);
  });

  const problems: string[] = [];
  if (names.length < limits.minPoints) problems.push(`Place at least ${limits.minPoints} points (${names.length} so far).`);
  if (names.length >= 4 && (!H || !invertible))
    problems.push('These points do not fit a court: are several of them on one line?');
  if (errorM != null && invertible && errorM > limits.maxErrorM)
    problems.push(`The fit error ${errorM.toFixed(2)} m is above the ${limits.maxErrorM} m limit. Check the points.`);
  if (mirrored) problems.push(mirrored);
  if (outside.length) problems.push(`Outside the frame: ${outside.map((n) => POINT_LABELS[n]).join(', ')}.`);

  return { count: names.length, H, inverse, errorM, invertible, mirrored, outside, problems };
}
