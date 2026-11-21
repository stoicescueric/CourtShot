import { createContext, useContext } from 'react';
import type { ReportData, ShotType } from '../types/report';

export const ReportContext = createContext<ReportData | null>(null);

export function useReport(): ReportData {
  const report = useContext(ReportContext);
  if (!report) throw new Error('useReport must be used inside ReportSection');
  return report;
}

export const shotLabel = (report: ReportData, type: ShotType): string => report.shot_labels[type] || type;
