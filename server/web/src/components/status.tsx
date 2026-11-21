import type { Job, Match, MatchStatus } from '../api/types';
import { percent } from '../format';

/** Statuses during which the match page and the list poll. */
export const UNFINISHED: ReadonlySet<MatchStatus> = new Set(['uploading', 'queued', 'processing']);

export const POLL_MS = 3000;

export const STATUS_LABEL: Record<MatchStatus, string> = {
  uploading: 'Uploading',
  needs_calibration: 'Needs calibration',
  queued: 'Queued',
  processing: 'Processing',
  ready: 'Ready',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

const STAGE_LABEL: Record<string, string> = {
  download: 'downloading',
  prepare: 'preparing',
  tracking: 'tracking',
  appearance: 'appearance',
  analysis: 'analysis',
  ball3d: '3D fit',
  render: 'rendering',
  upload: 'delivering',
};

export const stageLabel = (stage: string | null) => (stage ? (STAGE_LABEL[stage] ?? stage) : 'starting');

export const jobActive = (job: Job | null) => job?.status === 'queued' || job?.status === 'running';

export function StatusBadge({ status }: { status: MatchStatus }) {
  return <span className={`badge badge-${status}`}>{STATUS_LABEL[status]}</span>;
}

/** Stage, bar and percentage of a running job. */
export function JobProgress({ job }: { job: Job }) {
  return (
    <span className="job-progress">
      <span>{stageLabel(job.stage)}</span>
      <progress max={1} value={job.progress ?? 0} aria-label="Progress" />
      <span className="num">{percent(job.progress ?? 0)}</span>
    </span>
  );
}

export function NoWorker() {
  return <span className="warn">no worker online</span>;
}

export function uploadShare(match: Match): number {
  return match.size_bytes > 0 ? match.upload.offset / match.size_bytes : 0;
}
