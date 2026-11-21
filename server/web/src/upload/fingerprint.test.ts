// @vitest-environment node
// Node's Blob and Web Crypto are the browser's APIs; jsdom has no crypto.subtle.
import { fingerprint } from './fingerprint';

const MB = 1024 * 1024;

/** Bytes i % 251; the expected digests were computed independently with Python's hashlib. */
function file(n: number): Blob {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = i % 251;
  return new Blob([out]);
}

test('hashes the whole file below two megabytes', async () => {
  await expect(fingerprint(file(2 * MB - 1))).resolves.toBe(
    'efce3efcbba94299b27237589f84eac495e51e03a1d600024d465c9bbf253740',
  );
});

test('at exactly two megabytes the two halves are the whole file', async () => {
  await expect(fingerprint(file(2 * MB))).resolves.toBe(
    '1e075c8d478ad21844e33e830a695ef03a4d2488b69ee275bd8947618bb1be1e',
  );
});

test('hashes the first megabyte followed by the last megabyte', async () => {
  await expect(fingerprint(file(5 * MB + 123))).resolves.toBe(
    '288428ec81aa20afc45a440035841b1bb07caa325712428e868d36296b4ec27b',
  );
});

test('an empty file', async () => {
  await expect(fingerprint(new Blob([]))).resolves.toBe(
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
  );
});
