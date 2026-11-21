import { useState } from 'react';
import type { ReportData } from '../types/report';
import { About } from './About';
import { ReportContext } from './context';
import { Overview } from './Overview';
import { Players } from './Players';
import { Rallies } from './Rallies';
import './report.css';
import { dur } from './format';
import { ALL_SHOTS, Shots, type ShotFilter } from './Shots';
import { TipProvider } from './Tooltip';

export type ReportTab = 'overview' | 'players' | 'shots' | 'rallies' | 'about';

/**
 * One tab of the match report, ported from the reference report.html. The host page owns the tab
 * bar, the header and the selected player; the shot filters live here and survive tab switches.
 */
export function ReportSection(props: {
  report: ReportData;
  tab: ReportTab;
  player?: string;
  onOpenPlayer: (player: string) => void;
}): JSX.Element {
  const { report, tab, player, onOpenPlayer } = props;
  const [filter, setFilter] = useState<ShotFilter>(ALL_SHOTS);
  return (
    <ReportContext.Provider value={report}>
      <TipProvider>
        <div className="cs-report">
          <p className="muted">{metaLine(report)}</p>
          {tab === 'overview' && <Overview onOpenPlayer={onOpenPlayer} />}
          {tab === 'players' && <Players player={player} onOpenPlayer={onOpenPlayer} />}
          {tab === 'shots' && <Shots filter={filter} onFilter={setFilter} />}
          {tab === 'rallies' && <Rallies />}
          {tab === 'about' && <About />}
        </div>
      </TipProvider>
    </ReportContext.Provider>
  );
}

/** The reference header's meta line: the analysed span, and the rally and shot counts when the ball was analysed. */
function metaLine({ duration_s, match }: ReportData): string {
  const parts = [`${dur(duration_s)} analysed`];
  if (match) parts.push(`${match.rallies} ${match.rallies === 1 ? 'rally' : 'rallies'}`, `${match.shots} shots`);
  return parts.join(' · ');
}
