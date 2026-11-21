import { useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { cancelJob, deleteMatch, renameMatch, runAgain } from '../../api/client';
import type { MatchDetail } from '../../api/types';
import { messageOf } from '../../auth';
import { StatusBadge, jobActive } from '../../components/status';
import { clock, shortDate } from '../../format';
import { useUploads } from '../../upload/UploadsContext';

/** Editable title, status and the actions: Run again, Redo calibration, Cancel, Delete. */
export function MatchHeader({ match, onChanged }: { match: MatchDetail; onChanged: () => void }) {
  const navigate = useNavigate();
  const { cancel: cancelUpload } = useUploads();
  // null while not editing, so a poll does not overwrite what is being typed.
  const [draft, setDraft] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const act = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
      onChanged();
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  };

  const rename = (e?: FormEvent) => {
    e?.preventDefault();
    const title = draft?.trim();
    setDraft(null);
    if (title && title !== match.title) void act(() => renameMatch(match.id, title));
  };

  const remove = async () => {
    if (!window.confirm(`Delete "${match.title}" with its video and all results? This cannot be undone.`)) return;
    setBusy(true);
    try {
      cancelUpload(match.id);
      await deleteMatch(match.id);
      navigate('/');
    } catch (e) {
      setError(messageOf(e));
      setBusy(false);
    }
  };

  const active = jobActive(match.job);
  const canRun = !active && match.upload.complete && match.has_calibration;

  return (
    <header className="match-head">
      <form className="title-form" onSubmit={rename}>
        <label className="sr-only" htmlFor="match-title">Match title</label>
        <input id="match-title" className="title-input" value={draft ?? match.title} maxLength={200}
          onFocus={() => setDraft(match.title)} onChange={(e) => setDraft(e.target.value)} onBlur={() => rename()} />
      </form>
      <div className="match-meta">
        <StatusBadge status={match.status} />
        <span className="muted">
          {match.original_filename} · {clock(match.duration_s)} · {match.frame_width}×{match.frame_height} · uploaded{' '}
          {shortDate(match.created_at)}
        </span>
      </div>
      <div className="button-row">
        {canRun && <button type="button" className="btn" disabled={busy} onClick={() => act(() => runAgain(match.id))}>Run again</button>}
        {!active && match.has_calibration && (
          <Link to={`/matches/${match.id}/calibrate`} className="btn">Redo calibration</Link>
        )}
        {active && <button type="button" className="btn" disabled={busy} onClick={() => act(() => cancelJob(match.id))}>Cancel run</button>}
        <button type="button" className="btn btn-danger" disabled={busy} onClick={remove}>Delete</button>
      </div>
      {error && <p role="alert" className="error">{error}</p>}
    </header>
  );
}
