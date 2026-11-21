import { bytes, percent } from '../format';
import type { UploadProgress as Progress } from '../upload/uploader';

const STATUS_TEXT: Record<Progress['status'], string> = {
  uploading: 'Uploading',
  retrying: 'Connection lost',
  complete: 'Upload complete',
  failed: 'Upload stopped',
  cancelled: 'Upload cancelled',
};

export function UploadProgress({ progress }: { progress: Progress }) {
  const share = progress.size > 0 ? progress.offset / progress.size : 1;
  return (
    <div className="upload-progress">
      <progress max={progress.size || 1} value={progress.offset} aria-label="Upload progress" />
      <p aria-live="polite">
        <strong>{STATUS_TEXT[progress.status]}</strong> <span className="num">{percent(share)}</span>{' '}
        <span className="muted">
          {bytes(progress.offset)} of {bytes(progress.size)}
        </span>
      </p>
      {progress.error && (
        <p className={progress.status === 'failed' ? 'error' : 'warn'} role={progress.status === 'failed' ? 'alert' : undefined}>
          {progress.error}
        </p>
      )}
    </div>
  );
}
