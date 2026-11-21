import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeProvider } from '../theme';
import type { ReportData } from '../types/report';
import { reportEmptyRallies, reportFinalF, reportFull, reportNo3d, reportNoBall } from './fixtures';
import { ReportSection, type ReportTab } from './index';
import { placeTip } from './Tooltip';

beforeAll(() => {
  // jsdom has neither layout nor ResizeObserver; the canvas is never painted here.
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      disconnect() {}
    },
  );
});

afterAll(() => vi.unstubAllGlobals());

function show(report: ReportData, tab: ReportTab, player?: string) {
  const onOpenPlayer = vi.fn();
  const ui = (t: ReportTab) => (
    <ThemeProvider>
      <ReportSection report={report} tab={t} player={player} onOpenPlayer={onOpenPlayer} />
    </ThemeProvider>
  );
  const view = render(ui(tab));
  return { ...view, onOpenPlayer, switchTo: (t: ReportTab) => view.rerender(ui(t)) };
}

const card = (title: string) => screen.getByRole('heading', { name: title, level: 2 }).closest('.card') as HTMLElement;
const bodyRows = (el: HTMLElement) => within(el).getAllByRole('row').slice(1);

const FIXTURES: [string, ReportData][] = [
  ['full', reportFull],
  ['no ball', reportNoBall],
  ['no 3D', reportNo3d],
  ['empty rallies', reportEmptyRallies],
  ['real run (final_f)', reportFinalF],
];
const TABS: ReportTab[] = ['overview', 'players', 'shots', 'rallies', 'about'];

describe.each(FIXTURES)('%s fixture', (_, report) => {
  test.each(TABS)('renders the %s tab', (tab) => {
    const { container } = show(report, tab);
    expect(container.querySelector('.cs-report')?.children.length).toBeGreaterThan(0);
    expect(screen.getAllByRole('heading', { level: 2 }).length).toBeGreaterThan(0);
  });
});

describe('fixtures', () => {
  test('match the contract shape', () => {
    expect(reportFull.shots).toHaveLength(40);
    expect(reportFull.rallies).toHaveLength(6);
    expect(reportFull.shots.every((s, i) => s.index === i)).toBe(true);
    expect(reportFull.players.map((p) => p.heat.length)).toEqual([40, 40, 40, 40]);
    expect(reportNoBall.quality.has_ball).toBe(false);
    expect(reportNo3d.shots.some((s) => s.confidence === 'high')).toBe(false);
    expect(reportNo3d.rallies?.some((r) => r.outcome === 'winner' || r.outcome === 'error')).toBe(false);
  });
});

test('every tab carries the analysed span of the report, with rally and shot counts when the ball was analysed', () => {
  const { switchTo } = show(reportFull, 'overview');
  const counts = `${reportFull.match!.rallies} rallies · ${reportFull.match!.shots} shots`;
  expect(screen.getByText(`2.5 min analysed · ${counts}`)).toBeInTheDocument();
  switchTo('shots');
  expect(screen.getByText(`2.5 min analysed · ${counts}`)).toBeInTheDocument();
});

test('the real run renders its own numbers', () => {
  const { switchTo } = show(reportFinalF, 'overview');
  expect(screen.getByText('1.5 min analysed · 1 rally · 62 shots')).toBeInTheDocument();
  switchTo('shots');
  expect(bodyRows(card('All shots'))).toHaveLength(62);
  switchTo('rallies');
  expect(bodyRows(card('Rallies'))).toHaveLength(1);
  expect(screen.getByText('Video ends mid-rally')).toBeInTheDocument();
});

test('without ball analysis the meta line is the analysed span alone', () => {
  show(reportNoBall, 'overview');
  expect(screen.getByText('2.5 min analysed')).toBeInTheDocument();
});

describe('Overview', () => {
  test('hero, comparison and shot types with ball analysis', () => {
    show(reportFull, 'overview');
    expect(screen.getByText('Rallies from the ball')).toBeInTheDocument();
    expect(screen.getByText('Longest rally')).toBeInTheDocument();
    expect(screen.getByText('longest lasted 28 s')).toBeInTheDocument();
    expect(screen.getByText('ball seen in 82% of frames')).toBeInTheDocument();
    const cmp = card('Team comparison');
    for (const row of ['Shots', 'Attacking shots', 'Overheads', 'Lobs', 'Winners', 'Errors', 'Net regained']) {
      expect(within(cmp).getByText(row)).toBeInTheDocument();
    }
    expect(within(card('Shot types')).getByText('Bandeja / víbora')).toBeInTheDocument();
    expect(within(card('Match timeline')).getByText(/^Filled: attacking shot \(bajada, bandeja \/ víbora, smash, volley\)$/)).toBeInTheDocument();
  });

  test('without ball analysis: movement rallies, no shot rows', () => {
    show(reportNoBall, 'overview');
    expect(screen.getByText('Rallies from the movement')).toBeInTheDocument();
    expect(screen.queryByText('Longest rally')).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Shot types' })).not.toBeInTheDocument();
    expect(within(card('Team comparison')).queryByText('Attacking shots')).not.toBeInTheDocument();
    expect(within(card('Team comparison')).getByText('Both players at the net')).toBeInTheDocument();
    expect(within(card('Match timeline')).getByText('Shaded bands are rallies.')).toBeInTheDocument();
  });

  test('without 3D there are no winners or errors to compare', () => {
    show(reportNo3d, 'overview');
    expect(within(card('Team comparison')).queryByText('Winners')).not.toBeInTheDocument();
  });

  test('a zero-length video prints a dash, not Infinity%', () => {
    show({ ...reportFull, duration_s: 0 }, 'overview');
    expect(screen.queryByText(/Infinity/)).not.toBeInTheDocument();
    expect(screen.getByText('– of the video')).toBeInTheDocument();
  });

  test('a mini heatmap opens that player', async () => {
    const { onOpenPlayer } = show(reportFull, 'overview');
    await userEvent.click(within(card('Where each player plays')).getByRole('button', { name: /^P_3/ }));
    expect(onOpenPlayer).toHaveBeenCalledWith('P_3');
  });

  test('a timeline shot shows its tooltip as text', () => {
    const { container } = show(reportFull, 'overview');
    const tip = document.querySelector('.cs-report-tip') as HTMLElement;
    const shot = container.querySelector('svg[aria-label="Shots over time by player"] .shot') as Element;
    fireEvent.mouseMove(shot, { clientX: 40, clientY: 40 });
    expect(tip).toHaveClass('on');
    expect(tip).toHaveTextContent(/^P_1 · Serve0:03 · 62 km\/h · contact at \d\.\d m/);
    fireEvent.mouseLeave(shot);
    expect(tip).not.toHaveClass('on');
  });
});

describe('Players', () => {
  test('defaults to the first player and the picker opens another', async () => {
    const { onOpenPlayer } = show(reportFull, 'players');
    const picker = screen.getByRole('group', { name: 'Player' });
    expect(within(picker).getByRole('button', { name: 'P_1' })).toHaveAttribute('aria-pressed', 'true');
    await userEvent.click(within(picker).getByRole('button', { name: 'P_2' }));
    expect(onOpenPlayer).toHaveBeenCalledWith('P_2');
  });

  test('shows the requested player with movement and shot cards', () => {
    show(reportFull, 'players', 'P_3');
    expect(screen.getByText("Where P_3 stood during rallies: own half, net at the top, P_3's left on the left.")).toBeInTheDocument();
    expect(screen.getByText('T_2, with P_4. Tracked in 92% of frames.')).toBeInTheDocument();
    expect(within(card('Movement')).getByText('21.1 km/h')).toBeInTheDocument();
    expect(within(card('Direction and target')).getByText('To P_1')).toBeInTheDocument();
    expect(within(card('Shots')).getByText('Winners / errors')).toBeInTheDocument();
  });

  test('a missing top speed prints a dash', () => {
    show(reportFull, 'players', 'P_4');
    expect(within(card('Movement')).getByText('Top speed').nextElementSibling).toHaveTextContent('–');
  });

  test('without ball analysis the shot cards give way to one empty card', () => {
    show(reportNoBall, 'players');
    expect(screen.getByText('No ball data in this run, so there are no shot statistics.')).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Direction and target' })).not.toBeInTheDocument();
  });

  test('a player without shots', () => {
    show(reportEmptyRallies, 'players');
    expect(within(card('Shot types')).getByText('No shots found.')).toBeInTheDocument();
  });
});

describe('Shots', () => {
  test('lists every shot by default', () => {
    show(reportFull, 'shots');
    expect(bodyRows(card('All shots'))).toHaveLength(40);
    const drawn = reportFull.shots.filter((s) => s.contact && s.landing).length;
    expect(screen.getByText(`${drawn} of 40 selected shots have a landing spot. Line: from the hitter to where the ball landed.`)).toBeInTheDocument();
  });

  test('the player and type filters combine', async () => {
    const { switchTo } = show(reportFull, 'shots');
    await userEvent.click(within(screen.getByRole('group', { name: 'Player' })).getByRole('button', { name: 'P_3' }));
    const p3 = reportFull.shots.filter((s) => s.player === 'P_3');
    let rows = bodyRows(card('All shots'));
    expect(rows).toHaveLength(p3.length);
    expect(rows.every((r) => within(r).getAllByRole('cell')[2].textContent === 'P_3')).toBe(true);

    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Shot type' }), 'drive');
    const p3Drives = p3.filter((s) => s.type === 'drive');
    expect(p3Drives.length).toBeGreaterThan(0);
    rows = bodyRows(card('All shots'));
    expect(rows).toHaveLength(p3Drives.length);
    expect(rows.every((r) => within(r).getAllByRole('cell')[3].textContent === 'Drive')).toBe(true);

    switchTo('overview');
    switchTo('shots');
    expect(bodyRows(card('All shots'))).toHaveLength(p3Drives.length);
  });

  test('only the types present are offered', () => {
    show(reportFull, 'shots');
    const options = within(screen.getByRole('combobox', { name: 'Shot type' })).getAllByRole('option').map((o) => o.textContent);
    const present = reportFull.shot_types.filter((t) => reportFull.shots.some((s) => s.type === t));
    expect(options).toEqual(['All shot types', ...present.map((t) => reportFull.shot_labels[t])]);
  });

  test('a selection without speeds', async () => {
    show(reportFull, 'shots');
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Shot type' }), 'other');
    expect(screen.getByText('No speeds for this selection.')).toBeInTheDocument();
  });

  test('no shots at all', () => {
    show(reportNoBall, 'shots');
    expect(screen.getByText('No shots were found in this run.')).toBeInTheDocument();
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
  });
});

describe('Rallies', () => {
  test('ball rallies with their endings; movement rallies hidden when the ball decided', () => {
    show(reportFull, 'rallies');
    const rows = bodyRows(card('Rallies'));
    expect(rows).toHaveLength(6);
    for (const text of ['Winner', 'Error', 'Not clear', 'Video ends mid-rally']) expect(screen.getAllByText(text).length).toBeGreaterThan(0);
    expect(screen.queryByText('Rallies used for the position statistics')).not.toBeInTheDocument();
  });

  test('without ball analysis: the movement rallies', () => {
    show(reportNoBall, 'rallies');
    expect(bodyRows(card('Rallies'))).toHaveLength(6);
    expect(screen.queryByText('Ended with')).not.toBeInTheDocument();
  });

  test('no ball rally: an empty state, then the movement rallies', () => {
    show(reportEmptyRallies, 'rallies');
    expect(within(card('Rallies')).getByText('None found.')).toBeInTheDocument();
    expect(bodyRows(card('Rallies used for the position statistics'))).toHaveLength(6);
  });

  test('no movement rally either', () => {
    show({ ...reportNoBall, movement_rallies: [] }, 'rallies');
    expect(within(card('Rallies')).getByText('None found.')).toBeInTheDocument();
  });
});

describe('About', () => {
  test('data quality and identities', () => {
    show(reportNo3d, 'about');
    expect(within(card('Data quality')).getByText('not used')).toBeInTheDocument();
    expect(screen.getByText(/1 identity swap\(s\) were corrected \(at 1:12\)/)).toBeInTheDocument();
  });

  test('describes the heat colours as they are drawn', () => {
    show(reportFull, 'about');
    expect(screen.queryByText(/green/)).not.toBeInTheDocument();
    expect(screen.getByText(/runs from faint \(little time\) to strong \(most time\)/)).toBeInTheDocument();
  });

  test('without ball analysis the ball lines are dashes', () => {
    show(reportNoBall, 'about');
    expect(within(card('Data quality')).getByText('Frames with the ball detected').nextElementSibling).toHaveTextContent('–');
  });
});

test('payload markup renders as text', () => {
  const evil = '<img src=x onerror="alert(1)">';
  const report: ReportData = { ...reportFull, shot_labels: { ...reportFull.shot_labels, drive: evil } };
  const { container } = show(report, 'shots');
  expect(container.querySelector('img')).toBeNull();
  expect(screen.getAllByText(evil).length).toBeGreaterThan(0);
});

describe('placeTip', () => {
  const viewport = { width: 1000, height: 800 };
  const size = { width: 200, height: 60 };

  test('sits 14 px below and right of the cursor', () => {
    expect(placeTip({ x: 100, y: 100 }, size, viewport)).toEqual({ left: 114, top: 114 });
  });

  test('flips at the right and bottom edges', () => {
    expect(placeTip({ x: 900, y: 780 }, size, viewport)).toEqual({ left: 686, top: 706 });
  });

  test('never leaves the viewport on the left or top', () => {
    expect(placeTip({ x: 100, y: 30 }, { width: 950, height: 790 }, viewport)).toEqual({ left: 8, top: 8 });
  });
});
