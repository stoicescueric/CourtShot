import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import type { Config, MatchDetail, Worker } from '../api/types';
import { App } from '../App';
import { ThemeProvider } from '../theme';

export const json = (status: number, body: unknown) =>
  new Response(body === undefined ? null : JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

/** A route answers with a Response, a JSON body (200), or a function of the request giving either. */
type Answer = unknown;
export type Routes = Record<string, Answer | ((url: URL, init: RequestInit) => Answer)>;

export const CONFIG: Config = {
  min_video_seconds: 10,
  max_video_seconds: 900,
  max_upload_bytes: 8_000_000_000,
  chunk_bytes: 8_388_608,
  calibration_max_error_m: 0.25,
  min_calibration_points: 6,
  extensions: ['mp4', 'mov', 'm4v', 'webm'],
  worker_online_seconds: 30,
};

export const WORKER_IDLE: Worker = { name: 'lab-mac-1', last_seen_at: '2026-11-20T21:14:03Z', online: true, job: null };
export const WORKER_OFF: Worker = { ...WORKER_IDLE, online: false };

/**
 * Replaces fetch with a table of "METHOD /path" -> answer. The session, config and workers routes answer by default.
 * Unknown routes answer 404, so a test sees what it forgot.
 */
export function mockApi(routes: Routes) {
  const table: Routes = {
    'GET /api/auth/me': { username: 'lab' },
    'GET /api/config': CONFIG,
    'GET /api/workers': { workers: [WORKER_IDLE] },
    ...routes,
  };
  const fetchMock = vi.fn(async (input: string, init: RequestInit = {}) => {
    const url = new URL(input, 'http://localhost');
    const key = `${init.method ?? 'GET'} ${url.pathname}`;
    if (!(key in table)) return json(404, { detail: `not mocked: ${key}` });
    const route = table[key];
    const answer = typeof route === 'function' ? await route(url, init) : route;
    return answer instanceof Response ? answer : json(200, answer);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/** The calls made to one route. */
export function callsTo(fetchMock: ReturnType<typeof mockApi>, key: string) {
  return fetchMock.mock.calls.filter(([input, init = {}]) => `${init.method ?? 'GET'} ${new URL(input, 'http://localhost').pathname}` === key);
}

export function renderApp(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </MemoryRouter>,
  );
}

const JOB_BASE = {
  attempt: 1,
  stage: null,
  progress: null,
  error: null,
  worker: null,
  log_tail: null,
  created_at: '2026-11-20T20:00:00Z',
  started_at: null,
  finished_at: null,
};

export function job(fields: Partial<NonNullable<MatchDetail['job']>> & { id: number; status: NonNullable<MatchDetail['job']>['status'] }) {
  return { ...JOB_BASE, ...fields };
}

export function match(fields: Partial<MatchDetail> = {}): MatchDetail {
  return {
    id: 5,
    title: 'final_f',
    original_filename: 'final_f.mp4',
    created_at: '2026-11-20T19:00:00Z',
    size_bytes: 100_000_000,
    duration_s: 90,
    frame_width: 1280,
    frame_height: 720,
    status: 'ready',
    upload: { offset: 100_000_000, complete: true },
    has_calibration: true,
    job: null,
    results: null,
    summary: null,
    fingerprint: 'f'.repeat(64),
    calibration: null,
    ...fields,
  };
}
