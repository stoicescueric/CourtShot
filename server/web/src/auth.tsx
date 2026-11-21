import type { ReactNode } from 'react';
import { ApiError, me } from './api/client';
import { useResource } from './hooks';

/** Renders its children only for a signed-in user. A 401 from /api/auth/me sends the client to /login. */
export function RequireAuth({ children }: { children: ReactNode }) {
  const session = useResource(me, []);
  if (session.data) return <>{children}</>;
  if (session.error && !(session.error instanceof ApiError && session.error.status === 401)) {
    return (
      <main className="page narrow">
        <p role="alert" className="error">{messageOf(session.error)}</p>
        <button type="button" className="btn" onClick={session.reload}>Try again</button>
      </main>
    );
  }
  return <main className="page narrow" aria-busy="true"><p className="muted">Loading…</p></main>;
}

/** A user-facing sentence for any failure. */
export function messageOf(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error) return error.message;
  return String(error);
}
