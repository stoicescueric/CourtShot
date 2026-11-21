import { ApiError, getMatch, login, logout, patchUpload, setUnauthorizedHandler } from './client';

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', ...headers } });

let fetchMock: ReturnType<typeof vi.fn>;
let redirected: string[];

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  redirected = [];
  setUnauthorizedHandler((target) => redirected.push(target), () => '/matches/7/shots?player=P_2');
});
afterEach(() => vi.unstubAllGlobals());

test('sends same-origin credentials and JSON, and parses the answer', async () => {
  fetchMock.mockResolvedValue(json(200, { username: 'lab' }));
  await expect(login('lab', 'secret')).resolves.toEqual({ username: 'lab' });
  const [url, init] = fetchMock.mock.calls[0];
  expect(url).toBe('/api/auth/login');
  expect(init.method).toBe('POST');
  expect(init.credentials).toBe('same-origin');
  expect(init.headers['Content-Type']).toBe('application/json');
  expect(JSON.parse(init.body)).toEqual({ username: 'lab', password: 'secret' });
});

test('an error answer becomes an ApiError with status, detail and body', async () => {
  fetchMock.mockResolvedValue(json(409, { detail: 'offset beyond the current size', offset: 4096 }));
  const error = await patchUpload(3, 8192, new Blob(['x'])).catch((e: unknown) => e);
  expect(error).toBeInstanceOf(ApiError);
  expect(error).toMatchObject({ status: 409, detail: 'offset beyond the current size', body: { offset: 4096 } });
  const [url, init] = fetchMock.mock.calls[0];
  expect(url).toBe('/api/matches/3/upload?offset=8192');
  expect(init.method).toBe('PATCH');
  expect(init.headers['Content-Type']).toBe('application/octet-stream');
});

test('FastAPI validation lists and bodies without detail still give a message', async () => {
  fetchMock.mockResolvedValueOnce(json(422, { detail: [{ msg: 'field required' }, { msg: 'too long' }] }));
  await expect(getMatch(1)).rejects.toMatchObject({ status: 422, detail: 'field required; too long' });
  fetchMock.mockResolvedValueOnce(new Response('<html>Bad gateway</html>', { status: 502, statusText: 'Bad Gateway' }));
  await expect(getMatch(1)).rejects.toMatchObject({ status: 502, detail: 'Bad Gateway' });
});

test('a 429 carries Retry-After', async () => {
  fetchMock.mockResolvedValue(json(429, { detail: 'Too many attempts' }, { 'Retry-After': '30' }));
  await expect(login('lab', 'x')).rejects.toMatchObject({ status: 429, retryAfter: 30 });
});

test('no answer at all is an ApiError with status 0', async () => {
  fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));
  await expect(getMatch(1)).rejects.toMatchObject({ status: 0 });
});

test('a 401 redirects to sign-in, keeping the page to come back to', async () => {
  fetchMock.mockResolvedValue(json(401, { detail: 'Not signed in' }));
  await expect(getMatch(7)).rejects.toMatchObject({ status: 401 });
  expect(redirected).toEqual(['/matches/7/shots?player=P_2']);
});

test('a 401 from sign-in or sign-out is an answer, not a redirect', async () => {
  fetchMock.mockImplementation(async () => json(401, { detail: 'Wrong username or password' }));
  await expect(login('lab', 'wrong')).rejects.toMatchObject({ status: 401, detail: 'Wrong username or password' });
  await expect(logout()).rejects.toMatchObject({ status: 401 });
  expect(redirected).toEqual([]);
});

test('an empty 204 answer resolves', async () => {
  fetchMock.mockResolvedValue(new Response(null, { status: 204 }));
  await expect(logout()).resolves.toBeNull();
});

test('a 2xx answer cut off before its JSON is complete is a network error, not null', async () => {
  fetchMock.mockResolvedValueOnce(new Response('{"id": 1, "tit', { status: 200 }));
  await expect(getMatch(1)).rejects.toMatchObject({ status: 0 });
  fetchMock.mockResolvedValueOnce(new Response('', { status: 200 }));
  await expect(getMatch(1)).rejects.toMatchObject({ status: 0 });
});

test('a connection dropped while the body loads is a network error; a cancel stays a cancel', async () => {
  const failing = (error: Error) =>
    new Response(new ReadableStream({ start: (controller) => controller.error(error) }), { status: 200 });
  fetchMock.mockResolvedValueOnce(failing(new TypeError('network error')));
  await expect(getMatch(1)).rejects.toMatchObject({ status: 0 });
  fetchMock.mockResolvedValueOnce(failing(new DOMException('cancelled', 'AbortError')));
  await expect(getMatch(1)).rejects.toMatchObject({ name: 'AbortError' });
});
