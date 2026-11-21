import { useState } from 'react';
import { Link } from 'react-router-dom';
import { listMatches, runAgain } from '../api/client';
import type { Match } from '../api/types';
import { messageOf } from '../auth';
import { JobProgress, NoWorker, POLL_MS, STATUS_LABEL, UNFINISHED, uploadShare } from '../components/status';
import { clock, percent, shortDate } from '../format';
import { useInterval, useResource } from '../hooks';
import { isUploading, useUploads } from '../upload/UploadsContext';

export function MatchesPage() {
  const list = useResource(listMatches, []);
  const matches = list.data?.matches;
  const workersOnline = list.data?.workers_online ?? true;
  useInterval(list.reload, matches?.some((m) => UNFINISHED.has(m.status)) ? POLL_MS : null);
  const [actionError, setActionError] = useState<string | null>(null);

  const rerun = async (id: number) => {
    setActionError(null);
    try {
      await runAgain(id);
    } catch (e) {
      setActionError(messageOf(e));
    }
    list.reload();
  };

  return (
    <main className="page">
      <div className="page-head">
        <h1>Matches</h1>
      </div>
      {actionError && <p role="alert" className="error">{actionError}</p>}
      {!matches ? (
        list.error ? <p role="alert" className="error">{messageOf(list.error)}</p> : <p className="muted">Loading…</p>
      ) : matches.length === 0 ? (
        <div className="card empty">
          <p>No matches yet.</p>
          <Link to="/matches/new" className="btn btn-primary">Upload a video</Link>
        </div>
      ) : (
        <div className="table-wrap card">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Title</th>
                <th scope="col">Uploaded</th>
                <th scope="col" className="num">Length</th>
                <th scope="col">Status</th>
                <th scope="col" className="num">Rallies</th>
                <th scope="col" className="num">Shots</th>
              </tr>
            </thead>
            <tbody>
              {matches.map((m) => (
                <tr key={m.id}>
                  <td className="title-cell"><Link to={`/matches/${m.id}`}>{m.title}</Link></td>
                  <td>{shortDate(m.created_at)}</td>
                  <td className="num">{clock(m.duration_s)}</td>
                  <td><StatusCell match={m} workersOnline={workersOnline} onRunAgain={() => rerun(m.id)} /></td>
                  <td className="num">{m.summary?.rallies ?? '–'}</td>
                  <td className="num">{m.summary?.shots ?? '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}

function StatusCell({ match, workersOnline, onRunAgain }: { match: Match; workersOnline: boolean; onRunAgain: () => void }) {
  const { uploads } = useUploads();
  const label = <strong className={`status status-${match.status}`}>{STATUS_LABEL[match.status]}</strong>;
  switch (match.status) {
    case 'uploading':
      return (
        <span className="status-line">
          {label} <span className="num">{percent(uploadShare(match))}</span>
          {!isUploading(uploads[match.id]) && <span className="muted">· paused, open to resume</span>}
        </span>
      );
    case 'needs_calibration':
      return (
        <span className="status-line">
          {label} <Link to={`/matches/${match.id}/calibrate`}>Mark the court</Link>
        </span>
      );
    case 'queued':
      return <span className="status-line">{label}{!workersOnline && <> · <NoWorker /></>}</span>;
    case 'processing':
      return <span className="status-line">{label} · {match.job && <JobProgress job={match.job} />}</span>;
    case 'failed':
      return (
        <span className="status-line">
          {label}
          {match.job?.error && <span className="truncate" title={match.job.error}> · {match.job.error}</span>}
          <button type="button" className="btn btn-small" onClick={onRunAgain}>Run again</button>
        </span>
      );
    default:
      return label;
  }
}
