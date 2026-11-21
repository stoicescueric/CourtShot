export type Calibration = {
  /** Point name -> [x, y] in video pixels; at least 6. */
  points: Record<string, [number, number]>;
  frame_width: number;
  frame_height: number;
};

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export type Job = {
  id: number;
  status: JobStatus;
  attempt: number;
  stage: string | null;
  /** 0..1, overall. */
  progress: number | null;
  error: string | null;
  worker: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

export type Results = {
  /** The newest succeeded job. */
  job_id: number;
  files: string[];
  finished_at: string;
};

export type MatchStatus =
  | 'uploading' | 'needs_calibration' | 'queued' | 'processing' | 'ready' | 'failed' | 'cancelled';

export type Match = {
  id: number;
  title: string;
  original_filename: string;
  created_at: string;
  size_bytes: number;
  duration_s: number;
  frame_width: number;
  frame_height: number;
  status: MatchStatus;
  upload: { offset: number; complete: boolean };
  has_calibration: boolean;
  job: Job | null;
  results: Results | null;
  summary: {
    duration_s: number | null;
    rallies: number | null;
    shots: number | null;
    time_in_play_s: number | null;
  } | null;
};

export type MatchDetail = Match & {
  fingerprint: string;
  calibration: Calibration | null;
  job: (Job & { log_tail: string | null }) | null;
};

export type Config = {
  min_video_seconds: number;
  max_video_seconds: number;
  max_upload_bytes: number;
  chunk_bytes: number;
  calibration_max_error_m: number;
  min_calibration_points: number;
  extensions: string[];
  worker_online_seconds: number;
};

export type Worker = {
  name: string;
  last_seen_at: string | null;
  online: boolean;
  job: {
    id: number;
    match_id: number;
    match_title: string;
    stage: string | null;
    progress: number | null;
  } | null;
};
