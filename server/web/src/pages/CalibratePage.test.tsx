import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { COURT_POINTS, POINT_NAMES } from '../calibration/court';
import { apply, type Mat3 } from '../calibration/homography';
import { callsTo, job, match, mockApi, renderApp } from '../test/api';

// A 1280 x 720 frame shown 640 px wide: every click counts double in video pixels.
const VIEW: Mat3 = [90, 20, 190, 0, -30, 680, 0, 0.07, 1];
const SCALE = 2;

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  vi.spyOn(HTMLCanvasElement.prototype, 'getBoundingClientRect').mockReturnValue(
    { left: 10, top: 20, width: 1280 / SCALE, height: 720 / SCALE, right: 650, bottom: 380, x: 10, y: 20, toJSON: () => ({}) },
  );
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function clickCourt(name: (typeof POINT_NAMES)[number]) {
  const [x, y] = apply(VIEW, COURT_POINTS[name]);
  const overlay = screen.getByRole('img', { name: /Video frame/ });
  fireEvent.click(overlay, { clientX: 10 + x / SCALE, clientY: 20 + y / SCALE });
}

test('guided clicks in video pixels, undo, skip, clear and save', async () => {
  const fetchMock = mockApi({
    'GET /api/matches/5': match({ status: 'needs_calibration', has_calibration: false }),
    'GET /api/matches': { matches: [], workers_online: true },
    'PUT /api/matches/5/calibration': match({ status: 'queued', job: job({ id: 9, status: 'queued' }) }),
  });
  renderApp('/matches/5/calibrate');
  expect(await screen.findByText('Near baseline, left corner')).toBeInTheDocument();
  expect(document.querySelector('video')).toHaveAttribute('src', '/api/matches/5/video');
  const save = screen.getByRole('button', { name: 'Save and queue' });

  clickCourt('bottom_left');
  expect(screen.getByText('Near baseline, right corner')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
  expect(screen.getByText('Near baseline, left corner')).toBeInTheDocument();

  for (const name of ['bottom_left', 'bottom_right', 'top_left', 'top_right', 'net_left'] as const) clickCourt(name);
  expect(screen.getByText('Place at least 6 points (5 so far).')).toBeInTheDocument();
  expect(save).toBeDisabled();

  await userEvent.click(screen.getByRole('button', { name: 'Skip' })); // net_right
  expect(screen.getByText('Near service line, left end')).toBeInTheDocument();
  clickCourt('service_near_left');
  expect(screen.getByText('0.00 m')).toBeInTheDocument();
  expect(save).toBeEnabled();

  await userEvent.click(save);
  const [, init] = callsTo(fetchMock, 'PUT /api/matches/5/calibration')[0];
  const body = JSON.parse(init!.body as string);
  expect(body.frame_width).toBe(1280);
  expect(body.frame_height).toBe(720);
  expect(Object.keys(body.points)).toEqual(['bottom_left', 'bottom_right', 'top_left', 'top_right', 'net_left', 'service_near_left']);
  const [x, y] = apply(VIEW, COURT_POINTS.top_right);
  expect(body.points.top_right[0]).toBeCloseTo(x, 0);
  expect(body.points.top_right[1]).toBeCloseTo(y, 0);
});

test('a mirrored calibration cannot be saved', async () => {
  mockApi({
    'GET /api/matches/5': match({ status: 'needs_calibration', has_calibration: false }),
    'GET /api/matches': { matches: [], workers_online: true },
  });
  renderApp('/matches/5/calibrate');
  await screen.findByText('Near baseline, left corner');
  // Left and right clicked the wrong way round.
  for (const name of ['bottom_right', 'bottom_left', 'top_right', 'top_left', 'net_right', 'net_left'] as const) clickCourt(name);
  expect(screen.getByText(/The points look mirrored/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Save and queue' })).toBeDisabled();
  await userEvent.click(screen.getByRole('button', { name: 'Clear' }));
  expect(screen.getByText('Near baseline, left corner')).toBeInTheDocument();
});

test('points from an earlier match with the same frame size', async () => {
  const earlier = Object.fromEntries(POINT_NAMES.map((n) => [n, apply(VIEW, COURT_POINTS[n])]));
  mockApi({
    'GET /api/matches/5': match({ status: 'needs_calibration', has_calibration: false }),
    'GET /api/matches': {
      matches: [
        match({ id: 3, title: 'court2_am' }),
        match({ id: 4, title: 'other_camera', frame_width: 1920, frame_height: 1080 }),
        match({ id: 8, title: 'not_marked', has_calibration: false }),
      ],
      workers_online: true,
    },
    'GET /api/matches/3': match({ id: 3, calibration: { points: earlier, frame_width: 1280, frame_height: 720 } }),
  });
  renderApp('/matches/5/calibrate');
  const select = await screen.findByLabelText('Use points from');
  expect(Array.from((select as HTMLSelectElement).options, (o) => o.text)).toEqual(['earlier match…', 'court2_am']);
  await userEvent.selectOptions(select, '3');
  expect(await screen.findByText('All points done. Undo to change one.')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Save and queue' })).toBeEnabled();
});

test('refused while a run is active', async () => {
  mockApi({ 'GET /api/matches/5': match({ status: 'processing', job: job({ id: 9, status: 'running' }) }) });
  renderApp('/matches/5/calibrate');
  expect(await screen.findByRole('alert')).toHaveTextContent('A run of this match is active');
});
