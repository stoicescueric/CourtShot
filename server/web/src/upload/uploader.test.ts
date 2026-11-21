import { Uploader, type UploadProgress } from './uploader';

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

const FILE = new Blob(['0123456789abcdefghij']); // 20 bytes

interface Sent {
  offset: number;
  bytes: string;
}

type Fault = (server: { lose: (size: number) => void }) => Response | Promise<never> | undefined;

/**
 * A fake server: GET answers its offset, PATCH writes at an offset up to its size and refuses one beyond.
 * A fault, by call number, answers instead of the server; one that returns nothing lets the server answer.
 */
function server(startOffset = 0, faults: Record<number, Fault> = {}) {
  let size = startOffset;
  let call = 0;
  const sent: Sent[] = [];
  const control = { lose: (n: number) => (size = n) };
  const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
    call += 1;
    const faulty = faults[call]?.(control);
    if (faulty) return faulty;
    if (init.method === 'GET') return json(200, { offset: size, size_bytes: 20, complete: size === 20, fingerprint: 'f' });
    const offset = Number(new URL(url, 'http://x').searchParams.get('offset'));
    if (offset > size) return json(409, { detail: 'offset beyond the current size', offset: size });
    const bytes = await (init.body as Blob).text();
    sent.push({ offset, bytes });
    size = Math.max(size, offset + bytes.length);
    return json(200, { offset: size, complete: size === 20 });
  });
  vi.stubGlobal('fetch', fetchMock);
  return { sent, fetchMock };
}

function run(chunkBytes: number, wait = vi.fn(async (_ms: number, _signal: AbortSignal) => {})) {
  const events: UploadProgress[] = [];
  const uploader = new Uploader(5, FILE, chunkBytes, (p) => events.push(p), wait);
  return { uploader, events, wait, done: uploader.run() };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

test('sends sequential chunks from the server offset until complete', async () => {
  const { sent } = server();
  const { events, done } = run(8);
  await done;
  expect(sent).toEqual([
    { offset: 0, bytes: '01234567' },
    { offset: 8, bytes: '89abcdef' },
    { offset: 16, bytes: 'ghij' },
  ]);
  expect(events.map((e) => [e.status, e.offset])).toEqual([
    ['uploading', 0],
    ['uploading', 8],
    ['uploading', 16],
    ['complete', 20],
  ]);
});

test('starts where the server is after a reload', async () => {
  const { sent } = server(12);
  await run(8).done;
  expect(sent.map((s) => s.offset)).toEqual([12]);
  expect(sent[0].bytes).toBe('cdefghij');
});

test('an upload the server already has completes at once', async () => {
  const { sent } = server(20);
  const { events, done } = run(8);
  await done;
  expect(sent).toEqual([]);
  expect(events.at(-1)).toMatchObject({ status: 'complete', offset: 20 });
});

test('resumes from the server offset on 409', async () => {
  // The server lost the second chunk: the third (offset 16) is refused and the second is sent again.
  const { sent, fetchMock } = server(0, { 4: ({ lose }) => void lose(8) });
  await run(8).done;
  expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
    '/api/matches/5/upload',
    '/api/matches/5/upload?offset=0',
    '/api/matches/5/upload?offset=8',
    '/api/matches/5/upload?offset=16',
    '/api/matches/5/upload?offset=8',
    '/api/matches/5/upload?offset=16',
  ]);
  expect(sent.map((s) => s.offset)).toEqual([0, 8, 8, 16]);
});

test('retries network errors with growing pauses, then carries on', async () => {
  const down = () => Promise.reject(new TypeError('Failed to fetch'));
  const { sent } = server(0, { 3: down, 4: down, 5: () => json(503, { detail: 'busy' }) });
  const { events, wait, done } = run(8);
  await done;
  expect(wait.mock.calls.map(([ms]) => ms)).toEqual([1000, 2000, 4000]);
  expect(events.filter((e) => e.status === 'retrying')).toHaveLength(3);
  expect(sent.map((s) => s.offset)).toEqual([0, 8, 16]);
  expect(events.at(-1)).toMatchObject({ status: 'complete', offset: 20, error: null });
});

test('stops on a refusal that a retry cannot fix', async () => {
  server(0, { 2: () => json(413, { detail: 'chunk too large' }) });
  const { events, wait, done } = run(8);
  await done;
  expect(wait).not.toHaveBeenCalled();
  expect(events.at(-1)).toMatchObject({ status: 'failed', error: 'chunk too large' });
});

test('can be cancelled while waiting to retry', async () => {
  server(0, { 2: () => Promise.reject(new TypeError('Failed to fetch')) });
  let uploader!: Uploader;
  const wait = vi.fn(
    (_ms: number, signal: AbortSignal) =>
      new Promise<void>((_, reject) => {
        signal.addEventListener('abort', () => reject(new DOMException('cancelled', 'AbortError')));
        uploader.cancel();
      }),
  );
  const events: UploadProgress[] = [];
  uploader = new Uploader(5, FILE, 8, (p) => events.push(p), wait);
  await uploader.run();
  expect(events.at(-1)?.status).toBe('cancelled');
});

/** A request that never answers: it ends only when its signal aborts, as fetch does. */
const stalled = (init: RequestInit) =>
  new Promise<never>((_, reject) => {
    const signal = init.signal!;
    if (signal.aborted) reject(signal.reason);
    signal.addEventListener('abort', () => reject(signal.reason));
  });

test('a request that stalls without an error times out and is retried', async () => {
  // The platform's timer is replaced by one the test fires.
  const timeouts: { ms: number; fire: () => void }[] = [];
  vi.spyOn(AbortSignal, 'timeout').mockImplementation((ms) => {
    const controller = new AbortController();
    timeouts.push({ ms, fire: () => controller.abort(new DOMException('timed out', 'TimeoutError')) });
    return controller.signal;
  });
  let stall = true;
  const { sent, fetchMock } = server();
  const answer = fetchMock.getMockImplementation()!;
  fetchMock.mockImplementation(async (url: string, init: RequestInit) => {
    if (init.method === 'PATCH' && stall) {
      stall = false;
      return stalled(init);
    }
    return answer(url, init);
  });
  const { events, wait, done } = run(8);
  await vi.waitFor(() => expect(timeouts).toHaveLength(2));
  // GET: a minute; the 8-byte chunk: a minute plus its time at 64 kB/s, rounded up to a whole millisecond.
  expect(timeouts.map((t) => t.ms)).toEqual([60_000, 60_001]);
  timeouts[1].fire();
  await done;
  expect(wait).toHaveBeenCalledTimes(1);
  expect(events.find((e) => e.status === 'retrying')?.error).toMatch(/could not be reached/);
  expect(sent.map((s) => s.offset)).toEqual([0, 8, 16]);
  expect(events.at(-1)).toMatchObject({ status: 'complete', offset: 20 });
});

test('cancelling aborts the request in flight', async () => {
  const { fetchMock } = server();
  fetchMock.mockImplementation(async (_url: string, init: RequestInit) => stalled(init));
  const { uploader, events, done } = run(8);
  await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());
  uploader.cancel();
  await done;
  expect(events.at(-1)?.status).toBe('cancelled');
});
