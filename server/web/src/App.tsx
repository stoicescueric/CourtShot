import { useEffect, useRef } from 'react';
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { setUnauthorizedHandler } from './api/client';
import { RequireAuth } from './auth';
import { CalibratePage } from './pages/CalibratePage';
import { LoginPage } from './pages/LoginPage';
import { MatchPage } from './pages/MatchPage';
import { MatchesPage } from './pages/MatchesPage';
import { NewMatchPage } from './pages/NewMatchPage';
import { AppShell } from './shell';
import { UploadsProvider } from './upload/UploadsContext';

export function App() {
  const navigate = useNavigate();
  const location = useLocation();
  const here = useRef(location);
  here.current = location;

  // A 401 anywhere: to the sign-in page inside the app, so running uploads survive. Later 401s of the same
  // expiry find the user there already and must not wrap the target in a second `next`.
  useEffect(
    () =>
      setUnauthorizedHandler(
        (target) => {
          if (here.current.pathname !== '/login') navigate(`/login?next=${encodeURIComponent(target)}`, { replace: true });
        },
        () => here.current.pathname + here.current.search,
      ),
    [navigate],
  );

  return (
    <UploadsProvider>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<RequireAuth><AppShell /></RequireAuth>}>
          <Route index element={<MatchesPage />} />
          <Route path="matches/new" element={<NewMatchPage />} />
          <Route path="matches/:id/calibrate" element={<CalibratePage />} />
          <Route path="matches/:id/:tab?" element={<MatchPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </UploadsProvider>
  );
}
