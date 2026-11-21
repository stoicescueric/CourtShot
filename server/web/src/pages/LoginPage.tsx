import { useState, type FormEvent } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { ApiError, login } from '../api/client';
import { messageOf } from '../auth';
import { Logo } from '../components/Logo';

/** Only a path on this site, so the sign-in page cannot send anyone elsewhere. */
function safeNext(next: string | null): string {
  return next && next.startsWith('/') && !next.startsWith('//') && !next.startsWith('/\\') ? next : '/';
}

function loginMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 401) return 'Wrong username or password.';
  if (error instanceof ApiError && error.status === 429)
    return error.retryAfter ? `Too many attempts. Try again in ${error.retryAfter} s.` : 'Too many attempts. Try again later.';
  return messageOf(error);
}

export function LoginPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(username, password);
      navigate(safeNext(params.get('next')), { replace: true });
    } catch (err) {
      setError(loginMessage(err));
      setBusy(false);
    }
  };

  return (
    <main className="login">
      <form className="card login-card" onSubmit={submit}>
        <Logo size={40} withWordmark />
        <h1>Sign in</h1>
        <label className="field">
          <span>Username</span>
          <input name="username" autoComplete="username" required value={username} onChange={(e) => setUsername(e.target.value)} />
        </label>
        <label className="field">
          <span>Password</span>
          <input name="password" type="password" autoComplete="current-password" required value={password}
            onChange={(e) => setPassword(e.target.value)} />
        </label>
        {error && <p role="alert" className="error">{error}</p>}
        <button type="submit" className="btn btn-primary" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'}</button>
      </form>
    </main>
  );
}
