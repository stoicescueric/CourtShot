import { act, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Match } from '../api/types';
import { callsTo, job, match, mockApi, renderApp } from '../test/api';

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const READY = match({
  id: 1, title: 'final_f', status: 'ready', job: job({ id: 11, status: 'succeeded' }),
  results: { job_id: 11, files: ['report.json'], finished_at: '2026-11-20T20:30:00Z' },
  summary: { duration_s: 90, rallies: 6, shots: 41, time_in_play_s: 50 },
});
const PROCESSING = match({
  id: 2, title: 'club_tue', status: 'processing', duration_s: 492,
  job: job({ id: 12, status: 'running', stage: 'tracking', progress: 0.62, worker: 'lab-mac-1' }),
});
const FAILED = match({
  id: 3, title: 'court2_am', status: 'failed', job: job({ id: 13, status: 'failed', error: 'four players never visible together' }),
});
const QUEUED = match({ id: 4, title: 'court3_pm', status: 'queued', job: job({ id: 14, status: 'queued' }) });
const UPLOADING = match({
  id: 6, title: 'phone_clip', status: 'uploading', has_calibration: false, upload: { offset: 43_000_000, complete: false },
});
const NEEDS_CALIBRATION = match({ id: 7, title: 'court4', status: 'needs_calibration', has_calibration: false });

function row(title: string) {
  return within(screen.getByRole('link', { name: title }).closest('tr')!);
}

function list(matches: Match[], workersOnline = true) {
  return mockApi({ 'GET /api/matches': { matches, workers_online: workersOnline } });
}

test('shows every state as in the wireframe', async () => {
  list([READY, PROCESSING, FAILED, QUEUED, UPLOADING, NEEDS_CALIBRATION], false);
  renderApp('/');
  await screen.findByRole('link', { name: 'final_f' });

  expect(row('final_f').getByText('Ready')).toBeInTheDocument();
  expect(row('final_f').getByText('1:30')).toBeInTheDocument();
  expect(row('final_f').getByText('6')).toBeInTheDocument();
  expect(row('final_f').getByText('41')).toBeInTheDocument();

  expect(row('club_tue').getByText('Processing')).toBeInTheDocument();
  expect(row('club_tue').getByText('tracking')).toBeInTheDocument();
  expect(row('club_tue').getByText('62%')).toBeInTheDocument();
  expect(row('club_tue').getByText('8:12')).toBeInTheDocument();

  expect(row('court2_am').getByText(/four players never visible together/)).toBeInTheDocument();
  expect(row('court2_am').getByRole('button', { name: 'Run again' })).toBeInTheDocument();

  expect(row('court3_pm').getByText('Queued')).toBeInTheDocument();
  expect(row('court3_pm').getByText('no worker online')).toBeInTheDocument();

  expect(row('phone_clip').getByText('43%')).toBeInTheDocument();
  expect(row('phone_clip').getByText(/paused, open to resume/)).toBeInTheDocument();

  expect(row('court4').getByRole('link', { name: 'Mark the court' })).toHaveAttribute('href', '/matches/7/calibrate');
});

test('a queued match does not warn while a worker is online', async () => {
  list([QUEUED], true);
  renderApp('/');
  await screen.findByRole('link', { name: 'court3_pm' });
  expect(screen.queryByText('no worker online')).not.toBeInTheDocument();
});

test('Run again on a failed match queues it and refreshes the list', async () => {
  const fetchMock = mockApi({
    'GET /api/matches': { matches: [FAILED], workers_online: true },
    'POST /api/matches/3/jobs': match({ ...FAILED, status: 'queued' }),
  });
  renderApp('/');
  await userEvent.click(await screen.findByRole('button', { name: 'Run again' }));
  expect(callsTo(fetchMock, 'POST /api/matches/3/jobs')).toHaveLength(1);
  await vi.waitFor(() => expect(callsTo(fetchMock, 'GET /api/matches')).toHaveLength(2));
});

test('polls every 3 s while a match is unfinished, and not otherwise', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const fetchMock = list([READY, PROCESSING]);
  renderApp('/');
  await screen.findByRole('link', { name: 'club_tue' });
  expect(callsTo(fetchMock, 'GET /api/matches')).toHaveLength(1);
  await act(() => vi.advanceTimersByTimeAsync(3000));
  expect(callsTo(fetchMock, 'GET /api/matches')).toHaveLength(2);

  fetchMock.mockClear();
  list([READY]);
  await act(() => vi.advanceTimersByTimeAsync(3000)); // this poll sees only finished matches
  const settled = callsTo(vi.mocked(fetch), 'GET /api/matches').length;
  await act(() => vi.advanceTimersByTimeAsync(9000));
  expect(callsTo(vi.mocked(fetch), 'GET /api/matches')).toHaveLength(settled);
});

test('payload text is shown as text, never as markup', async () => {
  list([match({ ...READY, title: '<img src=x onerror="alert(1)">' })]);
  renderApp('/');
  expect(await screen.findByRole('link', { name: '<img src=x onerror="alert(1)">' })).toBeInTheDocument();
  expect(document.querySelector('img')).toBeNull();
});

test('no matches yet', async () => {
  list([]);
  renderApp('/');
  expect(await screen.findByText('No matches yet.')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Upload a video' })).toHaveAttribute('href', '/matches/new');
});

test('the header lists the workers', async () => {
  mockApi({
    'GET /api/matches': { matches: [], workers_online: true },
    'GET /api/workers': {
      workers: [
        { name: 'lab-mac-1', last_seen_at: null, online: true, job: null },
        {
          name: 'lab-mac-2', last_seen_at: null, online: true,
          job: { id: 12, match_id: 2, match_title: 'club_tue', stage: 'tracking', progress: 0.6 },
        },
      ],
    },
  });
  renderApp('/');
  const line = await screen.findByText(/Workers:/);
  await vi.waitFor(() => expect(line.parentElement).toHaveTextContent('lab-mac-1 idle · lab-mac-2 busy (club_tue, tracking)'));
});
