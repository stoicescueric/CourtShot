const MB = 1024 * 1024;

/**
 * Lowercase hex SHA-256 of the file's first megabyte followed by its last megabyte,
 * or of the whole file below two megabytes (API.md). A megabyte is 1,048,576 bytes.
 */
export async function fingerprint(file: Blob): Promise<string> {
  const parts = file.size < 2 * MB ? [file] : [file.slice(0, MB), file.slice(file.size - MB)];
  const data = new Uint8Array(await new Blob(parts).arrayBuffer());
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}
