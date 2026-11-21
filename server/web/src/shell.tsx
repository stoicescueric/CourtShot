import { createContext, useContext } from 'react';
import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom';
import { getConfig, listWorkers, logout } from './api/client';
import type { Config, Worker } from './api/types';
import { messageOf } from './auth';
import { Logo } from './components/Logo';
import { stageLabel } from './components/status';
import { relativeTime } from './format';
import { useInterval, useResource } from './hooks';
import { useTheme } from './theme';

const WORKERS_POLL_MS = 5000;

const ConfigContext = createContext<Config | null>(null);
const WorkersContext = createContext<Worker[] | null>(null);

/** The server's limits; available on every page inside the shell. */
export function useConfig(): Config {
  const config = useContext(ConfigContext);
  if (!config) throw new Error('useConfig must be used inside AppShell');
  return config;
}

/** The registered workers, refreshed every few seconds; null until the first answer. */
export function useWorkers(): Worker[] | null {
  return useContext(WorkersContext);
}

/** Header, workers line and the signed-in pages. */
export function AppShell() {
  const config = useResource(getConfig, []);
  const workers = useResource(listWorkers, []);
  useInterval(workers.reload, WORKERS_POLL_MS);
  const workerList = workers.data?.workers ?? null;

  return (
    <WorkersContext.Provider value={workerList}>
      <Header />
      <WorkersLine workers={workerList} />
      {config.data ? (
        <ConfigContext.Provider value={config.data}>
          <Outlet />
        </ConfigContext.Provider>
      ) : config.error ? (
        <main className="page narrow">
          <p role="alert" className="error">The settings could not be loaded: {messageOf(config.error)}</p>
          <button type="button" className="btn" onClick={config.reload}>Try again</button>
        </main>
      ) : (
        <main className="page narrow" aria-busy="true"><p className="muted">Loading…</p></main>
      )}
    </WorkersContext.Provider>
  );
}

function Header() {
  const { theme, toggle } = useTheme();
  const navigate = useNavigate();
  const signOut = async () => {
    await logout().catch(() => undefined); // signed out either way: the session is gone or was already
    navigate('/login');
  };
  return (
    <header className="topbar">
      <Link to="/" className="brand" aria-label="CourtShot, all matches">
        <Logo size={30} withWordmark />
      </Link>
      <nav aria-label="Main" className="topnav">
        <NavLink to="/" end>Matches</NavLink>
        <NavLink to="/matches/new" className="btn btn-primary">New match</NavLink>
      </nav>
      <div className="topbar-end">
        <button type="button" className="btn btn-quiet" onClick={toggle} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>
          {theme === 'dark' ? 'Light' : 'Dark'}
        </button>
        <button type="button" className="btn btn-quiet" onClick={signOut}>Sign out</button>
      </div>
    </header>
  );
}

function WorkersLine({ workers }: { workers: Worker[] | null }) {
  if (!workers) return <p className="workers-line muted">Workers: …</p>;
  return (
    <p className="workers-line" aria-live="polite">
      <span className="muted">Workers: </span>
      {workers.length === 0 && <span className="warn">none registered</span>}
      {workers.map((w, i) => (
        <span key={w.name}>
          {i > 0 && <span className="muted"> · </span>}
          <span className={`dot ${w.online ? (w.job ? 'dot-busy' : 'dot-idle') : 'dot-off'}`} aria-hidden="true" />
          {w.name}{' '}
          {!w.online ? (
            <span className="muted">offline, seen {relativeTime(w.last_seen_at)}</span>
          ) : w.job ? (
            <span className="muted">
              busy (<Link to={`/matches/${w.job.match_id}`}>{w.job.match_title}</Link>, {stageLabel(w.job.stage)})
            </span>
          ) : (
            <span className="muted">idle</span>
          )}
        </span>
      ))}
    </p>
  );
}
