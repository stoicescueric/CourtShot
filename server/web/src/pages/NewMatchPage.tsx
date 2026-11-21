import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createMatch, saveCalibration } from '../api/client';
import type { MatchDetail } from '../api/types';
import { messageOf } from '../auth';
import { Calibrator } from '../calibration/Calibrator';
import { probeVideo } from '../calibration/probeVideo';
import { UploadPanel } from '../components/UploadPanel';
import { bytes, clock } from '../format';
import { useConfig } from '../shell';
import { fingerprint } from '../upload/fingerprint';
import { useUploads } from '../upload/UploadsContext';

const extensionOf = (name: string) => (name.includes('.') ? name.slice(name.lastIndexOf('.') + 1).toLowerCase() : '');
const stemOf = (name: string) => (name.includes('.') ? name.slice(0, name.lastIndexOf('.')) : name) || name;

export function NewMatchPage() {
  const config = useConfig();
  const { start } = useUploads();
  const navigate = useNavigate();
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<{ match: MatchDetail; url: string } | null>(null);

  useEffect(() => (created ? () => URL.revokeObjectURL(created.url) : undefined), [created]);

  const choose = async (file: File) => {
    setError(null);
    const ext = extensionOf(file.name);
    if (!config.extensions.includes(ext))
      return setError(`Choose a video file: ${config.extensions.map((e) => `.${e}`).join(', ')}.`);
    if (file.size > config.max_upload_bytes)
      return setError(`The file is ${bytes(file.size)}; the limit is ${bytes(config.max_upload_bytes)}.`);

    const url = URL.createObjectURL(file);
    try {
      setBusy('Checking that this browser can decode the video…');
      const video = await probeVideo(url);
      if (video.duration < config.min_video_seconds || video.duration > config.max_video_seconds)
        throw new Error(
          `The video is ${clock(video.duration)} long; it must be between ${clock(config.min_video_seconds)} ` +
            `and ${clock(config.max_video_seconds)}.`,
        );
      setBusy('Reading the file…');
      const print = await fingerprint(file);
      setBusy('Creating the match…');
      const match = await createMatch({
        title: stemOf(file.name).slice(0, 200), // API.md: a title has at most 200 characters
        filename: file.name,
        size_bytes: file.size,
        fingerprint: print,
        duration_s: video.duration,
        frame_width: video.width,
        frame_height: video.height,
      });
      start(match.id, file, config.chunk_bytes);
      setCreated({ match, url });
    } catch (e) {
      URL.revokeObjectURL(url);
      setError(messageOf(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <main className="page">
      <ol className="steps" aria-label="Steps">
        <li aria-current={!created ? 'step' : undefined}>1 Choose video</li>
        <li aria-current={created ? 'step' : undefined}>2 Mark the court</li>
        <li>3 Save</li>
      </ol>
      {!created ? (
        <div className="card choose">
          <h1>New match</h1>
          <p className="muted">
            A clip of {clock(config.min_video_seconds)} to {clock(config.max_video_seconds)}, up to{' '}
            {bytes(config.max_upload_bytes)}. The upload runs while you mark the court.
          </p>
          <label className="field">
            <span>Video file</span>
            <input type="file" accept={config.extensions.map((e) => `.${e}`).join(',')} disabled={busy != null}
              onChange={(e) => e.target.files?.[0] && choose(e.target.files[0])} />
          </label>
          {busy && <p aria-live="polite" className="muted">{busy}</p>}
          {error && <p role="alert" className="error">{error}</p>}
        </div>
      ) : (
        <>
          <h1 className="page-title">{created.match.title}</h1>
          <Calibrator
            matchId={created.match.id}
            videoUrl={created.url}
            width={created.match.frame_width}
            height={created.match.frame_height}
            duration={created.match.duration_s}
            aside={<div className="card"><UploadPanel match={created.match} /></div>}
            onSave={async (calibration) => {
              await saveCalibration(created.match.id, calibration);
              navigate(`/matches/${created.match.id}`);
            }}
          />
        </>
      )}
    </main>
  );
}
