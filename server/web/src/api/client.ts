import type { ReportData } from '../types/report';
import type { SceneData } from '../types/scene';
import type { Calibration, Config, MatchDetail, Match, Worker } from './types';

/** An answer other than 2xx, or no answer at all (status 0). */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    /** The parsed JSON body, e.g. `{detail, offset}` of an upload 409. */
    readonly body: unknown = null,
    /** Seconds from a `Retry-After` header (429). */
    readonly retryAfter: number | null = null,
  ) {
    super(detail);
    this.name = 'ApiError';
  }
}

/** A cancel by the caller. Checked by name: an abort reason may come from another realm's DOMException. */
export function isAbort(e: unknown): boolean {
  return (e as { name?: unknown } | null)?.name === 'AbortError';
}

/** Where a 401 sends the user; `target` is the page to come back to. The app replaces it with a router navigation. */
let unauthorized: (target: string) => void = (target) =>
  window.location.assign(`/login?next=${encodeURIComponent(target)}`);
let currentTarget: () => string = () => window.location.pathname + window.location.search;

export function setUnauthorizedHandler(handler: (target: string) => void, target: () => string): void {
  unauthorized = handler;
  currentTarget = target;
}

interface RequestOptions {
  json?: unknown;
  body?: Blob;
  signal?: AbortSignal;
  /** false for sign-in and sign-out, where a 401 is an answer, not an expired session. */
  redirectOn401?: boolean;
}

function detailOf(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === 'string') return detail;
  // FastAPI's own validation errors: [{msg, loc, ...}, ...]
  if (Array.isArray(detail)) {
    const messages = detail.map((d) => (d as { msg?: unknown })?.msg).filter((m) => typeof m === 'string');
    if (messages.length) return messages.join('; ');
  }
  return fallback;
}

async function request<T>(method: string, path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  let body: BodyInit | undefined;
  if (options.json !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(options.json);
  } else if (options.body) {
    headers['Content-Type'] = 'application/octet-stream';
    body = options.body;
  }

  let response: Response;
  let text: string;
  try {
    response = await fetch(path, { method, headers, body, credentials: 'same-origin', signal: options.signal });
    text = await response.text();
  } catch (e) {
    if (isAbort(e)) throw e;
    throw new ApiError(0, 'The server could not be reached. Check the connection.');
  }

  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = null;
    }
  }
  if (response.status === 204) return null as T;
  // A 2xx answer always carries JSON: an empty or unreadable one was cut off on the way, so it counts as no answer.
  if (response.ok && parsed === null) throw new ApiError(0, 'The answer from the server was incomplete. Check the connection.');
  if (response.ok) return parsed as T;

  if (response.status === 401 && options.redirectOn401 !== false) unauthorized(currentTarget());
  const retryAfter = Number(response.headers.get('Retry-After'));
  throw new ApiError(
    response.status,
    detailOf(parsed, response.statusText || `Request failed (${response.status})`),
    parsed,
    Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter : null,
  );
}

const enc = encodeURIComponent;

// Session
export const login = (username: string, password: string) =>
  request<{ username: string }>('POST', '/api/auth/login', { json: { username, password }, redirectOn401: false });
export const logout = () => request<void>('POST', '/api/auth/logout', { redirectOn401: false });
export const me = () => request<{ username: string }>('GET', '/api/auth/me');

export const getConfig = () => request<Config>('GET', '/api/config');
export const listWorkers = () => request<{ workers: Worker[] }>('GET', '/api/workers');

// Matches
export interface NewMatch {
  title: string;
  filename: string;
  size_bytes: number;
  fingerprint: string;
  duration_s: number;
  frame_width: number;
  frame_height: number;
}
export const listMatches = () => request<{ matches: Match[]; workers_online: boolean }>('GET', '/api/matches');
export const createMatch = (match: NewMatch) => request<MatchDetail>('POST', '/api/matches', { json: match });
export const getMatch = (id: number) => request<MatchDetail>('GET', `/api/matches/${id}`);
export const renameMatch = (id: number, title: string) =>
  request<MatchDetail>('PATCH', `/api/matches/${id}`, { json: { title } });
export const deleteMatch = (id: number) => request<void>('DELETE', `/api/matches/${id}`);
export const saveCalibration = (id: number, calibration: Calibration) =>
  request<MatchDetail>('PUT', `/api/matches/${id}/calibration`, { json: calibration });
export const runAgain = (id: number) => request<MatchDetail>('POST', `/api/matches/${id}/jobs`);
export const cancelJob = (id: number) => request<MatchDetail>('POST', `/api/matches/${id}/cancel`);

// Upload
export interface UploadOffset {
  offset: number;
  size_bytes: number;
  complete: boolean;
  fingerprint: string;
}
export const getUpload = (id: number, signal?: AbortSignal) =>
  request<UploadOffset>('GET', `/api/matches/${id}/upload`, { signal });
export const patchUpload = (id: number, offset: number, chunk: Blob, signal?: AbortSignal) =>
  request<{ offset: number; complete: boolean }>('PATCH', `/api/matches/${id}/upload?offset=${offset}`, {
    body: chunk,
    signal,
  });

// Files: the browser loads videos and downloads by URL; the JSON payloads are fetched.
export const matchVideoUrl = (id: number) => `/api/matches/${id}/video`;
export const resultUrl = (id: number, jobId: number, name: string) => `/api/matches/${id}/runs/${jobId}/${enc(name)}`;
export const getReport = (id: number, jobId: number) => request<ReportData>('GET', resultUrl(id, jobId, 'report.json'));
export const getScene = (id: number, jobId: number) => request<SceneData>('GET', resultUrl(id, jobId, 'scene.json'));
