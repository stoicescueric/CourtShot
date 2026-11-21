import { useEffect, useMemo, useRef, type MouseEvent } from 'react';
import { useTheme } from '../theme';
import type { PlayerRow } from '../types/report';
import { useReport } from './context';
import { RAMPS, ramp, smoothHeat, squareCount, type Ramp } from './heat';
import { useTip, type TipContent } from './Tooltip';

interface Field {
  smooth: number[][];
  peak: number;
}

/**
 * A player's own half, 10 x 10 m, net at the top and the player's left on the left. `heat[row][col]`
 * is drawn untransposed. Redraws on resize and on theme change; colours come from CSS variables.
 */
export function Heatmap({ player, mini = false }: { player: PlayerRow; mini?: boolean }) {
  const { settings } = useReport();
  const { theme } = useTheme();
  const tip = useTip();
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const field = useMemo<Field>(() => {
    const smooth = smoothHeat(player.heat);
    return { smooth, peak: smooth.flat().reduce((a, b) => Math.max(a, b), 0) };
  }, [player.heat]);
  const total = useMemo(() => player.heat.flat().reduce((a, b) => a + b, 0), [player.heat]);

  useEffect(() => {
    const b = box.current;
    const c = canvas.current;
    if (!b || !c) return;
    const stops = RAMPS[theme][player.team];
    // The observer's first callback comes after every effect of this commit, the theme provider's
    // data-theme update included, so the CSS variables read while painting belong to `theme`.
    const observer = new ResizeObserver(() => paint(b, c, field, stops, settings.service_line));
    observer.observe(b);
    return () => observer.disconnect();
  }, [theme, field, player.team, settings.service_line]);

  const heatTip = (ev: MouseEvent): TipContent => {
    const r = ev.currentTarget.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width;
    const fy = (ev.clientY - r.top) / r.height;
    const n = player.heat.length;
    const s = squareCount(player.heat, Math.floor(fx * n), Math.floor(fy * n));
    return {
      title: `${total ? ((100 * s) / total).toFixed(1) : 0}% of rally time`,
      text: `within this 1 m square · ${(fy * 10).toFixed(1)} m from the net, ${fx * 10 < 5 ? 'left' : 'right'} side`,
    };
  };

  return (
    <div ref={box} className="heat" {...(mini ? {} : tip(heatTip))}>
      <canvas ref={canvas} role="img" aria-label={`Where ${player.id} stood during rallies`} />
      {!mini && <span className="tag net">Net</span>}
      {!mini && <span className="tag back">Back wall</span>}
    </div>
  );
}

function paint(box: HTMLElement, canvas: HTMLCanvasElement, field: Field, stops: Ramp, serviceLine: number) {
  const W = Math.round((box.clientWidth || 300) * Math.min(2, window.devicePixelRatio || 1));
  canvas.width = canvas.height = W;
  const g = canvas.getContext('2d');
  if (!g) return;
  const css = getComputedStyle(box);
  // court: faint bands, like a mown pitch
  for (let i = 0; i < 5; i++) {
    g.fillStyle = css.getPropertyValue(i % 2 ? '--court2' : '--court');
    g.fillRect(0, (i * W) / 5, W, W / 5 + 1);
  }
  if (field.peak > 0) drawField(g, W, field, stops);
  // lines: net (top), walls, service line, centre line from the net to the service line
  const m = Math.max(1, W / 300);
  const sy = (serviceLine / 10) * W;
  g.strokeStyle = css.getPropertyValue('--court-line');
  g.lineWidth = m;
  g.strokeRect(m, m, W - 2 * m, W - 2 * m);
  g.beginPath();
  g.moveTo(m, sy);
  g.lineTo(W - m, sy);
  g.moveTo(W / 2, m);
  g.lineTo(W / 2, sy);
  g.stroke();
  g.lineWidth = m * 3;
  g.beginPath();
  g.moveTo(0, m * 1.5);
  g.lineTo(W, m * 1.5);
  g.stroke();
}

/** (blurred / peak)^0.8 through the team's ramp into a grid-sized image, scaled up smoothed. */
function drawField(g: CanvasRenderingContext2D, W: number, { smooth, peak }: Field, stops: Ramp) {
  const rows = smooth.length;
  const cols = smooth[0].length;
  const off = document.createElement('canvas');
  off.width = cols;
  off.height = rows;
  const og = off.getContext('2d');
  if (!og) return;
  const img = og.createImageData(cols, rows);
  smooth.forEach((row, y) => row.forEach((v, x) => img.data.set(ramp(stops, (v / peak) ** 0.8), 4 * (y * cols + x))));
  og.putImageData(img, 0, 0);
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = 'high';
  g.drawImage(off, 0, 0, W, W);
}
