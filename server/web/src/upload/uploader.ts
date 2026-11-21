import { ApiError, getUpload, isAbort, patchUpload } from '../api/client';

export type UploadStatus = 'uploading' | 'retrying' | 'complete' | 'failed' | 'cancelled';

export interface UploadProgress {
  status: UploadStatus;
  /** Bytes the server has. */
  offset: number;
  size: number;
  /** Why it is retrying or failed. */
  error: string | null;
}

/** Pauses between retries, in seconds; the last one repeats until the connection comes back. */
const RETRY_PAUSES_S = [1, 2, 4, 8, 15, 30];

/** No answer (0), a timeout, throttling or a proxy or server hiccup: worth sending again. */
const RETRY_STATUSES = new Set([0, 408, 429, 500, 502, 503, 504]);

/**
 * A request still unanswered after a minute plus the chunk's time at 64 kB/s (0.5 Mbit/s) is given up and retried,
 * so a connection that stalls without an error does not hang the upload.
 */
const REQUEST_TIMEOUT_MS = 60_000;
const SLOWEST_BYTES_PER_MS = 64;

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const id = setTimeout(resolve, ms);
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(id);
        reject(new DOMException('Upload cancelled', 'AbortError'));
      },
      { once: true },
    );
  });
}

/**
 * Sends a video in sequential chunks of `chunkBytes`, starting from the server's offset.
 * A 409 means the server holds fewer bytes than assumed: continue from its offset.
 * Network errors are retried with growing pauses for as long as it takes; other refusals stop it.
 */
export class Uploader {
  private readonly controller = new AbortController();
  private progress: UploadProgress;

  constructor(
    private readonly matchId: number,
    private readonly file: Blob,
    private readonly chunkBytes: number,
    private readonly onProgress: (progress: UploadProgress) => void,
    private readonly wait: (ms: number, signal: AbortSignal) => Promise<void> = sleep,
  ) {
    this.progress = { status: 'uploading', offset: 0, size: file.size, error: null };
  }

  cancel(): void {
    this.controller.abort();
  }

  /** Resolves when the upload ends: complete, failed or cancelled (see the last progress event). */
  async run(): Promise<void> {
    const signal = this.controller.signal;
    let failures = 0;
    let offset: number | null = null;
    try {
      for (;;) {
        try {
          if (offset == null) {
            const server = await getUpload(this.matchId, this.deadline(0));
            if (server.complete) return this.emit({ status: 'complete', offset: server.size_bytes, error: null });
            offset = server.offset;
            this.emit({ status: 'uploading', offset, error: null });
          }
          const chunk = this.file.slice(offset, Math.min(offset + this.chunkBytes, this.file.size));
          const answer = await patchUpload(this.matchId, offset, chunk, this.deadline(chunk.size));
          failures = 0;
          offset = answer.offset;
          if (answer.complete) return this.emit({ status: 'complete', offset, error: null });
          this.emit({ status: 'uploading', offset, error: null });
        } catch (e) {
          if (e instanceof ApiError && e.status === 409) {
            const serverOffset = (e.body as { offset?: unknown } | null)?.offset;
            offset = typeof serverOffset === 'number' ? serverOffset : null; // null: ask for it
            continue;
          }
          if (!(e instanceof ApiError) || !RETRY_STATUSES.has(e.status)) throw e;
          const pause = Math.max(RETRY_PAUSES_S[Math.min(failures, RETRY_PAUSES_S.length - 1)], e.retryAfter ?? 0);
          failures += 1;
          this.emit({ status: 'retrying', error: `${e.detail} Retrying in ${pause} s.` });
          await this.wait(pause * 1000, signal);
        }
      }
    } catch (e) {
      if (isAbort(e)) return this.emit({ status: 'cancelled', error: null });
      this.emit({ status: 'failed', error: e instanceof ApiError ? e.detail : String(e) });
    }
  }

  /** The cancel signal, also aborted (with a TimeoutError, which the client turns into status 0) after the timeout. */
  private deadline(bytes: number): AbortSignal {
    const timeout = AbortSignal.timeout(REQUEST_TIMEOUT_MS + Math.ceil(bytes / SLOWEST_BYTES_PER_MS));
    return AbortSignal.any([this.controller.signal, timeout]);
  }

  private emit(change: Partial<UploadProgress>): void {
    this.progress = { ...this.progress, ...change };
    this.onProgress(this.progress);
  }
}
