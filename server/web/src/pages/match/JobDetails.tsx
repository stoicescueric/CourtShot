import type { MatchDetail } from '../../api/types';
import { JobProgress } from '../../components/status';
import { relativeTime } from '../../format';

type DetailJob = NonNullable<MatchDetail['job']>;

export function NoWorkerNote() {
  return <p className="warn">No worker online: the match waits until a lab Mac runs its worker.</p>;
}

export function LogTail({ log }: { log: string | null }) {
  if (!log) return null;
  return (
    <details className="log" open>
      <summary>Log</summary>
      <pre>{log}</pre>
    </details>
  );
}

/** The newest job when there are no results to show: stage, progress, worker, error and log. */
export function JobDetails({ job, workersOnline }: { job: DetailJob; workersOnline: boolean }) {
  switch (job.status) {
    case 'queued':
      return (
        <div>
          <p>Queued {relativeTime(job.created_at)}.</p>
          {!workersOnline && <NoWorkerNote />}
        </div>
      );
    case 'running':
      return (
        <div aria-live="polite">
          <p>
            Processing on <strong>{job.worker ?? 'a worker'}</strong>
            <span className="muted">, started {relativeTime(job.started_at)}</span>
          </p>
          <JobProgress job={job} />
        </div>
      );
    case 'failed':
      return (
        <div>
          <p className="error" role="alert">Failed: {job.error ?? 'no reason given'}</p>
          <LogTail log={job.log_tail} />
        </div>
      );
    case 'cancelled':
      return <p>Cancelled {relativeTime(job.finished_at)}.</p>;
    default:
      return null;
  }
}
