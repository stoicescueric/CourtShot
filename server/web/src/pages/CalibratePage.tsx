import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { getMatch, matchVideoUrl, saveCalibration } from '../api/client';
import { messageOf } from '../auth';
import { Calibrator } from '../calibration/Calibrator';
import { jobActive } from '../components/status';
import { useResource } from '../hooks';
import { useUploads } from '../upload/UploadsContext';

/** An object URL for a file this tab still holds; revoked when it changes or the page closes. */
function useObjectUrl(file: Blob | undefined): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file) return;
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => {
      URL.revokeObjectURL(u);
      setUrl(null);
    };
  }, [file]);
  return url;
}

/** Marks the court again on the uploaded video (or on the local file while it is still uploading). */
export function CalibratePage() {
  const id = Number(useParams().id);
  const navigate = useNavigate();
  const match = useResource(() => getMatch(id), [id]);
  const localUrl = useObjectUrl(useUploads().uploads[id]?.file);
  const m = match.data;

  if (!m) {
    return (
      <main className="page">
        {match.error ? <p role="alert" className="error">{messageOf(match.error)}</p> : <p className="muted">Loading…</p>}
      </main>
    );
  }
  const videoUrl = m.upload.complete ? matchVideoUrl(id) : localUrl;

  return (
    <main className="page">
      <div className="page-head">
        <h1>Mark the court <span className="muted">· {m.title}</span></h1>
        <Link to={`/matches/${id}`} className="btn btn-quiet">Back to the match</Link>
      </div>
      {jobActive(m.job) ? (
        <p role="alert" className="warn">A run of this match is active. Cancel it on the match page before marking the court again.</p>
      ) : !videoUrl ? (
        <p role="alert" className="warn">
          The video is still uploading and this tab no longer has the file. Resume the upload on the{' '}
          <Link to={`/matches/${id}`}>match page</Link> first.
        </p>
      ) : (
        <Calibrator
          matchId={id}
          videoUrl={videoUrl}
          width={m.frame_width}
          height={m.frame_height}
          duration={m.duration_s}
          initialPoints={m.calibration?.points}
          onSave={async (calibration) => {
            await saveCalibration(id, calibration);
            navigate(`/matches/${id}`);
          }}
        />
      )}
    </main>
  );
}
