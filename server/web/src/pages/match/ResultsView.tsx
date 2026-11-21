import { Suspense, lazy, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { getReport, getScene, resultUrl } from '../../api/client';
import type { MatchDetail } from '../../api/types';
import { messageOf } from '../../auth';
import { JobProgress } from '../../components/status';
import { shortDate } from '../../format';
import { useResource } from '../../hooks';
import { ReportSection, type ReportTab } from '../../report';
import type { ReportData } from '../../types/report';
import { LogTail, NoWorkerNote } from './JobDetails';

const TABS = [
  ['overview', 'Overview'],
  ['players', 'Players'],
  ['shots', 'Shots'],
  ['rallies', 'Rallies'],
  ['video', 'Video'],
  ['replay', '3D replay'],
  ['about', 'About'],
] as const;
type Tab = (typeof TABS)[number][0];

// three.js is most of the bundle: loaded only when a replay is shown.
const Replay = lazy(() => import('../../replay').then((m) => ({ default: m.Replay })));

const SCHEMA_VERSION = 1;
const DOWNLOADS = ['summary.json', 'positions.csv', 'shots.csv', 'ball_events.csv'];

const OlderVersion = () => (
  <p className="card notice">Made by an older version of the pipeline, which this dashboard cannot show. Run again.</p>
);

/** The current results of a match: one per results job id (the parent keys it by that id). */
export function ResultsView({ match, tab, workersOnline }: { match: MatchDetail; tab: string | undefined; workersOnline: boolean }) {
  const results = match.results!;
  const { id } = match;
  const jobId = results.job_id;
  const current: Tab = TABS.find(([slug]) => slug === tab)?.[0] ?? 'overview';
  const [params] = useSearchParams();
  const navigate = useNavigate();

  const report = useResource(() => getReport(id, jobId), [id, jobId]);
  // scene.json is large: fetched the first time the 3D tab opens, then kept while this run is on show.
  const hasScene = results.files.includes('scene.json');
  const [sceneOpened, setSceneOpened] = useState(current === 'replay');
  if (current === 'replay' && !sceneOpened) setSceneOpened(true);
  const scene = useResource(
    () => (sceneOpened && hasScene ? getScene(id, jobId) : Promise.resolve(null)),
    [id, jobId, sceneOpened, hasScene],
  );

  let body: ReactNode;
  if (current === 'video') {
    body = <VideoTab matchId={id} jobId={jobId} files={results.files} />;
  } else if (!report.data) {
    body = report.error ? <p role="alert" className="error">{messageOf(report.error)}</p> : <p className="muted">Loading the report…</p>;
  } else if (report.data.schema_version !== SCHEMA_VERSION) {
    body = <OlderVersion />;
  } else if (current === 'replay') {
    body = !hasScene ? (
      <p className="card notice">No 3D replay for this run: {noSceneReason(report.data)}.</p>
    ) : !scene.data ? (
      scene.error ? <p role="alert" className="error">{messageOf(scene.error)}</p> : <p className="muted">Loading the 3D scene…</p>
    ) : scene.data.schema_version !== SCHEMA_VERSION ? (
      <OlderVersion />
    ) : (
      <FillViewport>
        <Suspense fallback={<p className="muted">Loading the 3D viewer…</p>}>
          <Replay scene={scene.data} videoUrl={results.files.includes('plain.mp4') ? resultUrl(id, jobId, 'plain.mp4') : null} />
        </Suspense>
      </FillViewport>
    );
  } else {
    body = (
      <>
        <ReportSection
          report={report.data}
          tab={current as ReportTab}
          player={params.get('player') ?? undefined}
          onOpenPlayer={(player) => navigate(`/matches/${id}/players?player=${encodeURIComponent(player)}`)}
        />
        {current === 'about' && (
          <p className="muted run-info">
            Run {jobId}, finished {shortDate(results.finished_at)} · pipeline {report.data.pipeline_version}
          </p>
        )}
      </>
    );
  }

  return (
    <>
      <NewestJobBanner match={match} workersOnline={workersOnline} />
      <nav className="tabs" aria-label="Match sections">
        {TABS.map(([slug, label]) => (
          <Link key={slug} to={`/matches/${id}/${slug}`} aria-current={slug === current ? 'page' : undefined}>{label}</Link>
        ))}
      </nav>
      <div className="tab-body">{body}</div>
    </>
  );
}

function noSceneReason(report: ReportData): string {
  if (!report.quality.has_ball) return 'the ball was not tracked';
  if (!report.quality.has_3d) return "the ball's flights could not be fitted in 3D";
  return 'the run produced no 3D scene';
}

/** Shown when the newest job is not the one whose results are on show. */
function NewestJobBanner({ match, workersOnline }: { match: MatchDetail; workersOnline: boolean }) {
  const job = match.job;
  if (!job || job.id === match.results?.job_id) return null;
  const shown = ' The results below are from the previous run.';
  return (
    <div className="banner" role="status">
      {job.status === 'queued' && (
        <>
          <p>A new run is queued.{shown}</p>
          {!workersOnline && <NoWorkerNote />}
        </>
      )}
      {job.status === 'running' && (
        <p>A new run is processing on {job.worker ?? 'a worker'}: <JobProgress job={job} /></p>
      )}
      {job.status === 'failed' && (
        <>
          <p>The last run failed: {job.error ?? 'no reason given'}.{shown}</p>
          <LogTail log={job.log_tail} />
        </>
      )}
      {job.status === 'cancelled' && <p>The last run was cancelled.{shown}</p>}
    </div>
  );
}

function VideoTab({ matchId, jobId, files }: { matchId: number; jobId: number; files: string[] }) {
  return (
    <div className="video-tab">
      <video className="result-video" controls preload="metadata" src={resultUrl(matchId, jobId, 'tracked.mp4')} />
      <div className="card downloads">
        <h2>Data files</h2>
        <ul>
          {DOWNLOADS.filter((name) => files.includes(name)).map((name) => (
            <li key={name}><a href={resultUrl(matchId, jobId, name)} download={name}>{name}</a></li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/** A box from its own top to the bottom of the viewport, for the 3D replay. */
function FillViewport({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [top, setTop] = useState(0);
  useLayoutEffect(() => {
    const measure = () => setTop((ref.current?.getBoundingClientRect().top ?? 0) + window.scrollY);
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);
  return <div ref={ref} className="replay-fill" style={{ height: `calc(100dvh - ${top}px - 16px)` }}>{children}</div>;
}
