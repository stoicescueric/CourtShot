import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { callsTo, json, mockApi, renderApp } from '../test/api';

afterEach(() => vi.unstubAllGlobals());

async function signIn(password = 'secret') {
  await userEvent.type(await screen.findByLabelText('Username'), 'lab');
  await userEvent.type(screen.getByLabelText('Password'), password);
  await userEvent.click(screen.getByRole('button', { name: 'Sign in' }));
}

test('signs in and goes to the page asked for', async () => {
  const fetchMock = mockApi({
    'POST /api/auth/login': { username: 'lab' },
    'GET /api/matches': { matches: [], workers_online: true },
  });
  renderApp('/login?next=%2F');
  await signIn();
  expect(await screen.findByRole('heading', { name: 'Matches' })).toBeInTheDocument();
  const [, init] = callsTo(fetchMock, 'POST /api/auth/login')[0];
  expect(JSON.parse(init!.body as string)).toEqual({ username: 'lab', password: 'secret' });
});

test('a wrong password is said so, and nothing else happens', async () => {
  mockApi({ 'POST /api/auth/login': json(401, { detail: 'Invalid credentials' }) });
  renderApp('/login');
  await signIn('wrong');
  expect(await screen.findByRole('alert')).toHaveTextContent('Wrong username or password.');
  expect(screen.getByRole('heading', { name: 'Sign in' })).toBeInTheDocument();
});

test('throttling shows the wait', async () => {
  mockApi({
    'POST /api/auth/login': () =>
      new Response(JSON.stringify({ detail: 'Too many attempts' }), { status: 429, headers: { 'Retry-After': '20' } }),
  });
  renderApp('/login');
  await signIn();
  expect(await screen.findByRole('alert')).toHaveTextContent('Try again in 20 s.');
});

test('a protected page without a session leads to sign-in and back', async () => {
  let signedIn = false;
  mockApi({
    'GET /api/auth/me': () => (signedIn ? { username: 'lab' } : json(401, { detail: 'Not signed in' })),
    'POST /api/auth/login': () => {
      signedIn = true;
      return { username: 'lab' };
    },
    'GET /api/matches': { matches: [], workers_online: true },
  });
  renderApp('/matches/new');
  expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeInTheDocument();
  await signIn();
  expect(await screen.findByRole('heading', { name: 'New match' })).toBeInTheDocument();
});

test('a next link to another site is ignored', async () => {
  mockApi({
    'POST /api/auth/login': { username: 'lab' },
    'GET /api/matches': { matches: [], workers_online: true },
  });
  renderApp('/login?next=%2F%2Fevil.example%2F');
  await signIn();
  expect(await screen.findByRole('heading', { name: 'Matches' })).toBeInTheDocument();
});
