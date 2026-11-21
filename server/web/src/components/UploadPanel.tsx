import { useState } from 'react';
import type { MatchDetail } from '../api/types';
import { messageOf } from '../auth';
import { bytes, percent } from '../format';
import { useConfig } from '../shell';
import { fingerprint } from '../upload/fingerprint';
import { isUploading, useUploads } from '../upload/UploadsContext';
import { uploadShare } from './status';
import { UploadProgress } from './UploadProgress';

/** The upload of a match's video: its progress, Retry after a failure, or resuming by picking the file again. */
export function UploadPanel({ match }: { match: MatchDetail }) {
  const config = useConfig();
  const { uploads, start } = useUploads();
  const entry = uploads[match.id];
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (entry && (isUploading(entry) || entry.progress.status === 'complete')) return <UploadProgress progress={entry.progress} />;

  const resume = async (file: File) => {
    setError(null);
    if (file.name !== match.original_filename)
      return setError(`This file is "${file.name}"; the match was created from "${match.original_filename}".`);
    if (file.size !== match.size_bytes)
      return setError(`This file is ${bytes(file.size)}; the one the match was created from is ${bytes(match.size_bytes)}.`);
    setChecking(true);
    try {
      if ((await fingerprint(file)) !== match.fingerprint)
        return setError('This file has the same name and size, but different content than the one the match was created from.');
      start(match.id, file, config.chunk_bytes);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="upload-panel">
      {entry ? (
        <>
          <UploadProgress progress={entry.progress} />
          <button type="button" className="btn" onClick={() => start(match.id, entry.file, config.chunk_bytes)}>Retry upload</button>
        </>
      ) : (
        <>
          <p>
            The upload stopped at <span className="num">{percent(uploadShare(match))}</span>{' '}
            <span className="muted">({bytes(match.upload.offset)} of {bytes(match.size_bytes)})</span>. Choose the same file
            again to continue: <strong>{match.original_filename}</strong>.
          </p>
          <label className="field">
            <span>Video file</span>
            <input type="file" disabled={checking} onChange={(e) => e.target.files?.[0] && resume(e.target.files[0])} />
          </label>
          {checking && <p className="muted" aria-live="polite">Checking the file…</p>}
        </>
      )}
      {error && <p role="alert" className="error">{error}</p>}
    </div>
  );
}
