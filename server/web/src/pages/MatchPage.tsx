import { Link, useParams } from 'react-router-dom';
import { ApiError, getMatch } from '../api/client';
import type { MatchDetail } from '../api/types';
import { messageOf } from '../auth';
import { POLL_MS, UNFINISHED } from '../components/status';
import { UploadPanel } from '../components/UploadPanel';
import { useInterval, useResource } from '../hooks';
import { useWorkers } from '../shell';
import { JobDetails } from './match/JobDetails';
import { MatchHeader } from './match/MatchHeader';
import { ResultsView } from './match/ResultsView';

export function MatchPage() {
  const params = useParams();
  const id = Number(params.id);
  const match = useResource(() => getMatch(id), [id]);
  const m = match.data;
  useInterval(match.reload, m && UNFINISHED.has(m.status) ? POLL_MS : null);
  const workers = useWorkers();
  // Until the workers list arrives nobody is said to be missing.
  const workersOnline = workers == null || workers.some((w) => w.online);

  if (!m) {
    const gone = match.error instanceof ApiError && match.error.status === 404;
    return (
      <main className="page">
        {gone || !Number.isInteger(id) ? (
          <p className="card notice">This match does not exist. <Link to="/">All matches</Link></p>
        ) : match.error ? (
          <p role="alert" className="error">{messageOf(match.error)}</p>
        ) : (
          <p className="muted">Loading…</p>
        )}
      </main>
    );
  }

  return (
    <main className="page">
      <MatchHeader match={m} onChanged={match.reload} />
      {m.results ? (
        <ResultsView key={m.results.job_id} match={m} tab={params.tab} workersOnline={workersOnline} />
      ) : (
        <section className="card state">
          <MatchState match={m} workersOnline={workersOnline} />
        </section>
      )}
    </main>
  );
}

function MatchState({ match, workersOnline }: { match: MatchDetail; workersOnline: boolean }) {
  const calibrate = `/matches/${match.id}/calibrate`;
  if (match.status === 'uploading') {
    return (
      <>
        <UploadPanel match={match} />
        {!match.has_calibration && <p><Link to={calibrate}>Mark the court</Link> while the video uploads.</p>}
      </>
    );
  }
  if (match.status === 'needs_calibration') {
    return (
      <>
        <p>The video is uploaded. Mark the court to queue the analysis.</p>
        <Link to={calibrate} className="btn btn-primary">Mark the court</Link>
      </>
    );
  }
  return match.job ? <JobDetails job={match.job} workersOnline={workersOnline} /> : null;
}
