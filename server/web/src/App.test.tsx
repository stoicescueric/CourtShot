import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { App } from './App';
import { ThemeProvider } from './theme';

test('renders the login page at /login', () => {
  render(
    <MemoryRouter initialEntries={['/login']}>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </MemoryRouter>,
  );
  expect(screen.getByRole('heading', { name: 'Sign in' })).toBeInTheDocument();
  expect(document.documentElement.dataset.theme).toBe('dark');
});
