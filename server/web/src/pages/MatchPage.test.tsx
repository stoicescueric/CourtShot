import { act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReportData } from '../types/report';
import { callsTo, job, json, match, mockApi, renderApp, WORKER_OFF } from '../test/api';

// The report and the replay are their own components with their own tests; here they only show what they got.
vi.mock('../report', () => ({
  ReportSection: (props: { tab: string; player?: string; report: ReportData; onOpenPlayer: (p: string) => void }) => (
    <section data-testid="report">
      {props.tab} {props.player ?? ''} {props.report.pipeline_version}
      <button type="button" onClick={() => props.onOpenPlayer('P_3')}>open P_3</button>
    </section>
  ),
}));
vi.mock('../replay', () => ({
  Replay: (props: { scene: { n: number }; videoUrl: string | null }) => (
    <div data-testid="replay">{props.scene.n} frames · {props.videoUrl}</div>
  ),
}));
// jsdom has no Web Crypto digest; the fingerprint has its own tests.
vi.mock('../upload/fingerprint', () => ({ fingerprint: async (file: Blob) => (file.size === 20 ? 'right' : 'other') }));

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const REPORT = { schema_version: 1, pipeline_version: '0.3.0+abc1234', quality: { has_ball: true, has_3d: true } };
const ALL_FILES = ['report.json', 'tracked.mp4', 'plain.mp4', 'summary.json', 'positions.csv', 'scene.json', 'shots.csv', 'ball_events.csv'];
const RESULTS = { job_id: 9, files: ALL_FILES, finished_at: '2026-11-20T20:30:00Z' };
const READY = match({ status: 'ready', job: job({ id: 9, status: 'succeeded' }), results: RESULTS });

test('uploading: asks for the file again, checks it, then continues the upload', async () => {
  const fetchMock = mockApi({
    'GET /api/matches/5': match({
      status: 'uploading', has_calibration: false, size_bytes: 20, original_filename: 'clip.mp4', fingerprint: 'right',
      upload: { offset: 8, complete: false },
    }),
    'GET /api/matches/5/upload': { offset: 8, size_bytes: 20, complete: false, fingerprint: 'right' },
    'PATCH /api/matches/5/upload': { offset: 20, complete: true },
  });
  renderApp('/matches/5');
  expect(await screen.findByText(/The upload stopped at/)).toHaveTextContent('40%');
  expect(screen.getByRole('link', { name: 'Mark the court' })).toHaveAttribute('href', '/matches/5/calibrate');

  const input = screen.getByLabelText('Video file');
  await userEvent.upload(input, new File(['x'.repeat(20)], 'other.mp4'));
  expect(await screen.findByRole('alert')).toHaveTextContent('the match was created from "clip.mp4"');

  await userEvent.upload(input, new File(['x'.repeat(19)], 'clip.mp4'));
  expect(await screen.findByRole('alert')).toHaveTextContent('This file is 19 B');

  await userEvent.upload(input, new File(['0123456789abcdefghij'], 'clip.mp4'));
  expect(await screen.findByText('Upload complete')).toBeInTheDocument();
  const patch = callsTo(fetchMock, 'PATCH /api/matches/5/upload');
  expect(patch).toHaveLength(1);
  expect(patch[0][0]).toBe('/api/matches/5/upload?offset=8');
});

test('needs calibration: offers to mark the court', async () => {
  mockApi({ 'GET /api/matches/5': match({ status: 'needs_calibration', has_calibration: false }) });
  renderApp('/matches/5');
  expect(await screen.findByText(/Mark the court to queue the analysis/)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Mark the court' })).toHaveAttribute('href', '/matches/5/calibrate');
  expect(screen.queryByRole('button', { name: 'Run again' })).not.toBeInTheDocument();
});

test('queued with no worker online says so, and can be cancelled', async () => {
  const fetchMock = mockApi({
    'GET /api/matches/5': match({ status: 'queued', job: job({ id: 9, status: 'queued' }) }),
    'GET /api/workers': { workers: [WORKER_OFF] },
    'POST /api/matches/5/cancel': match({ status: 'cancelled', job: job({ id: 9, status: 'cancelled' }) }),
  });
  renderApp('/matches/5');
  expect(await screen.findByText(/No worker online/)).toBeInTheDocument();
  expect(screen.getByText('Queued', { selector: '.badge' })).toBeInTheDocument();
  expect(screen.queryByRole('link', { name: 'Redo calibration' })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Cancel run' }));
  expect(callsTo(fetchMock, 'POST /api/matches/5/cancel')).toHaveLength(1);
});

test('processing: stage, progress and worker, polled every 3 s', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const fetchMock = mockApi({
    'GET /api/matches/5': match({
      status: 'processing',
      job: job({ id: 9, status: 'running', stage: 'ball3d', progress: 0.62, worker: 'lab-mac-1', started_at: '2026-11-20T20:00:00Z' }),
    }),
  });
  renderApp('/matches/5');
  expect(await screen.findByText('lab-mac-1')).toBeInTheDocument();
  expect(screen.getByText('3D fit')).toBeInTheDocument();
  expect(screen.getByText('62%')).toBeInTheDocument();
  expect(callsTo(fetchMock, 'GET /api/matches/5')).toHaveLength(1);
  await act(() => vi.advanceTimersByTimeAsync(3000));
  expect(callsTo(fetchMock, 'GET /api/matches/5')).toHaveLength(2);
});

test('failed: the reason, the log tail, Run again and Redo calibration', async () => {
  const fetchMock = mockApi({
    'GET /api/matches/5': match({
      status: 'failed',
      job: job({ id: 9, status: 'failed', error: 'four players never visible together', log_tail: 'tracking 100%\nidentities: 3 players' }),
    }),
    'POST /api/matches/5/jobs': match({ status: 'queued', job: job({ id: 10, status: 'queued' }) }),
  });
  renderApp('/matches/5');
  expect(await screen.findByRole('alert')).toHaveTextContent('Failed: four players never visible together');
  expect(screen.getByText(/identities: 3 players/)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Redo calibration' })).toHaveAttribute('href', '/matches/5/calibrate');
  await userEvent.click(screen.getByRole('button', { name: 'Run again' }));
  expect(callsTo(fetchMock, 'POST /api/matches/5/jobs')).toHaveLength(1);
});

test('ready: tabs routed by URL, report fetched once for the results job', async () => {
  const fetchMock = mockApi({ 'GET /api/matches/5': READY, 'GET /api/matches/5/runs/9/report.json': REPORT });
  renderApp('/matches/5');
  expect(await screen.findByTestId('report')).toHaveTextContent('overview');
  const tabs = screen.getByRole('navigation', { name: 'Match sections' });
  expect(Array.from(tabs.querySelectorAll('a'), (a) => a.textContent)).toEqual([
    'Overview', 'Players', 'Shots', 'Rallies', 'Video', '3D replay', 'About',
  ]);

  await userEvent.click(screen.getByRole('link', { name: 'Shots' }));
  expect(await screen.findByTestId('report')).toHaveTextContent('shots');
  expect(screen.getByRole('link', { name: 'Shots' })).toHaveAttribute('aria-current', 'page');

  await userEvent.click(screen.getByRole('button', { name: 'open P_3' }));
  expect(await screen.findByTestId('report')).toHaveTextContent('players P_3');

  await userEvent.click(screen.getByRole('link', { name: 'About' }));
  expect(await screen.findByText(/pipeline 0.3.0\+abc1234/)).toBeInTheDocument();
  expect(callsTo(fetchMock, 'GET /api/matches/5/runs/9/report.json')).toHaveLength(1);
  expect(callsTo(fetchMock, 'GET /api/matches/5/runs/9/scene.json')).toHaveLength(0);
});

test('ready with a failed newest job: a banner over the previous results', async () => {
  mockApi({
    'GET /api/matches/5': match({
      status: 'failed', results: RESULTS,
      job: job({ id: 12, status: 'failed', error: 'video too short', log_tail: 'ffprobe: 8.2 s' }),
    }),
    'GET /api/matches/5/runs/9/report.json': REPORT,
  });
  renderApp('/matches/5/overview');
  expect(await screen.findByRole('status')).toHaveTextContent('The last run failed: video too short.');
  expect(screen.getByText('ffprobe: 8.2 s')).toBeInTheDocument();
  expect(await screen.findByTestId('report')).toHaveTextContent('overview');
});

test('ready with a new run processing: a banner with its progress', async () => {
  mockApi({
    'GET /api/matches/5': match({
      status: 'processing', results: RESULTS,
      job: job({ id: 12, status: 'running', stage: 'tracking', progress: 0.25, worker: 'lab-mac-2' }),
    }),
    'GET /api/matches/5/runs/9/report.json': REPORT,
  });
  renderApp('/matches/5');
  expect(await screen.findByRole('status')).toHaveTextContent('A new run is processing on lab-mac-2: tracking25%');
});

test('video tab: the tracked video and the data files of the run', async () => {
  mockApi({
    'GET /api/matches/5': match({ ...READY, results: { ...RESULTS, files: ['report.json', 'tracked.mp4', 'summary.json', 'positions.csv'] } }),
    'GET /api/matches/5/runs/9/report.json': REPORT,
  });
  renderApp('/matches/5/video');
  const links = await screen.findAllByRole('link', { name: /\.(json|csv)$/ });
  expect(links.map((a) => a.getAttribute('href'))).toEqual([
    '/api/matches/5/runs/9/summary.json',
    '/api/matches/5/runs/9/positions.csv',
  ]);
  expect(document.querySelector('video.result-video')).toHaveAttribute('src', '/api/matches/5/runs/9/tracked.mp4');
});

test('3D replay tab: scene.json only when opened, with plain.mp4', async () => {
  const fetchMock = mockApi({
    'GET /api/matches/5': READY,
    'GET /api/matches/5/runs/9/report.json': REPORT,
    'GET /api/matches/5/runs/9/scene.json': { schema_version: 1, n: 2250 },
  });
  renderApp('/matches/5');
  await screen.findByTestId('report');
  expect(callsTo(fetchMock, 'GET /api/matches/5/runs/9/scene.json')).toHaveLength(0);
  await userEvent.click(screen.getByRole('link', { name: '3D replay' }));
  expect(await screen.findByTestId('replay')).toHaveTextContent('2250 frames · /api/matches/5/runs/9/plain.mp4');
  await userEvent.click(screen.getByRole('link', { name: 'Overview' }));
  await userEvent.click(screen.getByRole('link', { name: '3D replay' }));
  await screen.findByTestId('replay');
  expect(callsTo(fetchMock, 'GET /api/matches/5/runs/9/scene.json')).toHaveLength(1);
});

test('3D replay tab without a scene says why', async () => {
  mockApi({
    'GET /api/matches/5': match({ ...READY, results: { ...RESULTS, files: ['report.json', 'tracked.mp4', 'plain.mp4'] } }),
    'GET /api/matches/5/runs/9/report.json': { ...REPORT, quality: { has_ball: false, has_3d: false } },
  });
  renderApp('/matches/5/replay');
  expect(await screen.findByText(/No 3D replay for this run: the ball was not tracked/)).toBeInTheDocument();
});

test('a report from another schema version asks for a new run', async () => {
  mockApi({ 'GET /api/matches/5': READY, 'GET /api/matches/5/runs/9/report.json': { ...REPORT, schema_version: 0 } });
  renderApp('/matches/5');
  expect(await screen.findByText(/Made by an older version/)).toBeInTheDocument();
  expect(screen.queryByTestId('report')).not.toBeInTheDocument();
});

test('rename and delete', async () => {
  const fetchMock = mockApi({
    'GET /api/matches/5': READY,
    'GET /api/matches/5/runs/9/report.json': REPORT,
    'PATCH /api/matches/5': match({ ...READY, title: 'Final, set 1' }),
    'DELETE /api/matches/5': json(204, undefined),
    'GET /api/matches': { matches: [], workers_online: true },
  });
  renderApp('/matches/5');
  const title = await screen.findByLabelText('Match title');
  await userEvent.clear(title);
  await userEvent.type(title, 'Final, set 1{Enter}');
  const [, init] = callsTo(fetchMock, 'PATCH /api/matches/5')[0];
  expect(JSON.parse(init!.body as string)).toEqual({ title: 'Final, set 1' });

  const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true);
  await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
  expect(callsTo(fetchMock, 'DELETE /api/matches/5')).toHaveLength(0);
  await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
  expect(confirm).toHaveBeenCalledTimes(2);
  expect(await screen.findByText('No matches yet.')).toBeInTheDocument();
  expect(callsTo(fetchMock, 'DELETE /api/matches/5')).toHaveLength(1);
});

test('an unknown match', async () => {
  mockApi({ 'GET /api/matches/5': json(404, { detail: 'Not found' }) });
  renderApp('/matches/5');
  expect(await screen.findByText(/This match does not exist/)).toBeInTheDocument();
});
