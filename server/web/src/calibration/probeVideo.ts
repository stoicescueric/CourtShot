export interface VideoFacts {
  duration: number;
  width: number;
  height: number;
}

export const CANNOT_DECODE =
  'This browser cannot decode this video, so the court cannot be marked on it. ' +
  'HEVC videos need Safari or a Chrome with hardware HEVC support; converting the video to H.264 works everywhere.';

/** Loads a video's first frame in a detached element: length and decoded frame size, or why it cannot be read. */
export function probeVideo(url: string, timeoutMs = 20000): Promise<VideoFacts> {
  const video = document.createElement('video');
  video.muted = true;
  video.preload = 'auto';
  return new Promise<VideoFacts>((resolve, reject) => {
    const finish = (result: VideoFacts | Error) => {
      clearTimeout(timer);
      video.onloadeddata = video.onerror = null;
      video.removeAttribute('src');
      video.load();
      if (result instanceof Error) reject(result);
      else resolve(result);
    };
    const timer = setTimeout(() => finish(new Error(CANNOT_DECODE)), timeoutMs);
    video.onerror = () => finish(new Error(CANNOT_DECODE));
    video.onloadeddata = () => {
      const { duration, videoWidth: width, videoHeight: height } = video;
      if (!width || !height) finish(new Error(CANNOT_DECODE));
      else if (!Number.isFinite(duration)) finish(new Error('The browser cannot tell how long this video is.'));
      else finish({ duration, width, height });
    };
    video.src = url;
  });
}
