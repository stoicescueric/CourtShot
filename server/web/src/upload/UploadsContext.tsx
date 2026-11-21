import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Uploader, type UploadProgress } from './uploader';

export interface UploadEntry {
  file: File;
  progress: UploadProgress;
}

interface UploadsValue {
  uploads: Record<number, UploadEntry>;
  /** Starts (or restarts after a failure) the upload of a match's video; a running one is left alone. */
  start: (matchId: number, file: File, chunkBytes: number) => void;
  cancel: (matchId: number) => void;
}

const UploadsContext = createContext<UploadsValue | null>(null);

const active = (p: UploadProgress) => p.status === 'uploading' || p.status === 'retrying';

/** Holds the running uploads above the routes, so an upload keeps going while the user moves around the app. */
export function UploadsProvider({ children }: { children: ReactNode }) {
  const [uploads, setUploads] = useState<Record<number, UploadEntry>>({});
  const running = useRef(new Map<number, Uploader>());

  const start = useCallback((matchId: number, file: File, chunkBytes: number) => {
    if (running.current.has(matchId)) return;
    const uploader = new Uploader(matchId, file, chunkBytes, (progress) =>
      setUploads((all) => ({ ...all, [matchId]: { file, progress } })),
    );
    running.current.set(matchId, uploader);
    setUploads((all) => ({
      ...all,
      [matchId]: { file, progress: { status: 'uploading', offset: 0, size: file.size, error: null } },
    }));
    void uploader.run().finally(() => running.current.delete(matchId));
  }, []);

  const cancel = useCallback((matchId: number) => running.current.get(matchId)?.cancel(), []);

  const anyActive = Object.values(uploads).some((u) => active(u.progress));
  useEffect(() => {
    if (!anyActive) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [anyActive]);

  const value = useMemo(() => ({ uploads, start, cancel }), [uploads, start, cancel]);
  return <UploadsContext.Provider value={value}>{children}</UploadsContext.Provider>;
}

export function useUploads(): UploadsValue {
  const value = useContext(UploadsContext);
  if (!value) throw new Error('useUploads must be used inside UploadsProvider');
  return value;
}

export function isUploading(entry: UploadEntry | undefined): boolean {
  return entry != null && active(entry.progress);
}
