import { StrictMode } from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import type { SceneData } from '../types/scene';
import { INITIAL_STATE, type ReplayEvent, type ReplayHandle, type ReplayListener, type ReplayState } from './engine';
import { makeSceneFixture, sceneFinalF } from './fixtures';
import { Replay } from './index';

const { createReplay } = vi.hoisted(() => ({ createReplay: vi.fn() }));
vi.mock('./engine', async (importOriginal) => ({ ...(await importOriginal<typeof import('./engine')>()), createReplay }));

type FakeHandle = { [K in keyof ReplayHandle]: ReturnType<typeof vi.fn> } & { emit: (e: ReplayEvent) => void; host: HTMLElement; video: HTMLVideoElement | null };
let handles: FakeHandle[] = [];

/** A stand-in engine that publishes its state like the real one. */
function fakeEngine(host: HTMLElement, video: HTMLVideoElement | null): FakeHandle {
  let listener: ReplayListener | null = null;
  let state: ReplayState = { ...INITIAL_STATE };
  const set = (patch: Partial<ReplayState>) => {
    state = { ...state, ...patch };
    listener?.({ type: 'state', state });
  };
  return {
    host,
    video,
    setView: vi.fn(),
    setLayers: vi.fn(),
    setSpeed: vi.fn(),
    setMode: vi.fn(),
    seek: vi.fn(),
    dispose: vi.fn(),
    setPlaying: vi.fn((playing: boolean) => set({ playing })),
    subscribe: vi.fn((l: ReplayListener) => {
      listener = l;
      l({ type: 'state', state });
      return () => {
        listener = null;
      };
    }),
    emit: (e) => act(() => listener?.(e)),
  };
}

beforeEach(() => {
  handles = [];
  createReplay.mockImplementation((host: HTMLElement, video: HTMLVideoElement | null) => {
    const h = fakeEngine(host, video);
    handles.push(h);
    return h;
  });
});

const engine = () => handles[handles.length - 1];
const region = () => screen.getByRole('region', { name: '3D replay' });
const state = (patch: Partial<ReplayState>): ReplayEvent => ({ type: 'state', state: { ...INITIAL_STATE, ...patch } });

describe('Replay', () => {
  it('renders the controls, without video modes when there is no video', () => {
    const { container } = render(<Replay scene={makeSceneFixture()} videoUrl={null} />);
    expect(screen.queryByRole('button', { name: 'Side by side' })).toBeNull();
    for (const name of ['Broadcast', 'Far end', 'Side', 'Top', 'Follow ball', 'Trail', 'Names', '¼×', 'Pause']) {
      expect(screen.getByRole('button', { name })).toBeInTheDocument();
    }
    expect(screen.getByText(/2 of 3 ball flights reconstructed in 3D · 4 players/)).toBeInTheDocument();
    expect(container.querySelector('.cs-replay')).toHaveAttribute('tabindex', '0');
    expect(container.querySelector('video')).not.toHaveAttribute('src');
  });

  it('offers the video modes and plays the given video', () => {
    const { container } = render(<Replay scene={makeSceneFixture()} videoUrl="/api/matches/1/runs/2/plain.mp4" />);
    fireEvent.click(screen.getByRole('button', { name: 'Overlay' }));
    expect(engine().setMode).toHaveBeenCalledWith('overlay');
    const video = container.querySelector('video');
    expect(video).toHaveAttribute('src', '/api/matches/1/runs/2/plain.mp4');
    expect(engine().video).toBe(video);
  });

  it('survives StrictMode: the first engine is disposed, the second lives until unmount', () => {
    const { unmount } = render(
      <StrictMode>
        <Replay scene={makeSceneFixture()} videoUrl={null} />
      </StrictMode>,
    );
    expect(handles).toHaveLength(2);
    expect(handles[0].dispose).toHaveBeenCalledOnce();
    expect(handles[1].dispose).not.toHaveBeenCalled();
    expect(handles[1].host).toBe(handles[0].host);
    unmount();
    expect(handles[1].dispose).toHaveBeenCalledOnce();
  });

  it('recreates the engine only when the scene object changes', () => {
    const scene = makeSceneFixture();
    const { rerender } = render(<Replay scene={scene} videoUrl={null} />);
    rerender(<Replay scene={scene} videoUrl="/v.mp4" />);
    expect(handles).toHaveLength(1);
    rerender(<Replay scene={makeSceneFixture()} videoUrl="/v.mp4" />);
    expect(handles).toHaveLength(2);
    expect(handles[0].dispose).toHaveBeenCalledOnce();
  });

  it('sends the buttons to the engine', () => {
    render(<Replay scene={makeSceneFixture()} videoUrl={null} />);
    fireEvent.click(screen.getByRole('button', { name: 'Top' }));
    expect(engine().setView).toHaveBeenCalledWith('top');
    fireEvent.click(screen.getByRole('button', { name: 'Trail' }));
    expect(engine().setLayers).toHaveBeenCalledWith({ trail: false });
    fireEvent.click(screen.getByRole('button', { name: '½×' }));
    expect(engine().setSpeed).toHaveBeenCalledWith(0.5);
    fireEvent.click(screen.getByRole('button', { name: 'Pause' }));
    expect(engine().setPlaying).toHaveBeenCalledWith(false);
    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
  });

  it('handles the shortcuts on the replay element', () => {
    render(<Replay scene={makeSceneFixture()} videoUrl={null} />);
    fireEvent.keyDown(region(), { key: '4' });
    expect(engine().setView).toHaveBeenCalledWith('top');
    fireEvent.keyDown(region(), { key: '1', ctrlKey: true });
    expect(engine().setView).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(region(), { key: ' ' });
    expect(engine().setPlaying).toHaveBeenLastCalledWith(false);
    engine().emit({ type: 'time', t: 1, ballY: null });
    fireEvent.keyDown(region(), { key: 'ArrowRight' });
    expect(engine().seek).toHaveBeenLastCalledWith(26 / 25);
    fireEvent.keyDown(region(), { key: 'ArrowLeft' });
    expect(engine().seek).toHaveBeenLastCalledWith(24 / 25);
    const calls = engine().setPlaying.mock.calls.length;
    fireEvent.keyDown(screen.getByRole('button', { name: 'Trail' }), { key: ' ' });
    expect(engine().setPlaying).toHaveBeenCalledTimes(calls); // a focused button keeps its own Space
    fireEvent.keyDown(document.body, { key: '2' });
    expect(engine().setView).toHaveBeenCalledTimes(1); // not on window
  });

  it('writes the clock and the ball height without re-rendering, and the flight panel on flight changes', () => {
    const { container } = render(<Replay scene={makeSceneFixture()} videoUrl={null} />);
    expect(screen.getByText('Between flights')).toBeInTheDocument();
    engine().emit({ type: 'time', t: 1.5, ballY: 1.234 });
    expect(container.querySelector('.rp-clock')).toHaveTextContent('0:01.50');
    expect(screen.getByText('1.23')).toBeInTheDocument();
    engine().emit(state({ flight: 0 }));
    expect(screen.getByText('Shot 1 of 3')).toBeInTheDocument();
    expect(screen.getByText('P 1 · Team 1')).toBeInTheDocument();
    expect(screen.getByText('Good fit')).toBeInTheDocument();
    expect(screen.getByText('74')).toBeInTheDocument();
    engine().emit(state({ flight: 1 }));
    expect(screen.getByText('Low confidence')).toBeInTheDocument();
    expect(screen.getByText(/path is a rough guess/)).toBeInTheDocument();
  });

  it('renders the scene of a real run', () => {
    const { container } = render(<Replay scene={sceneFinalF()} videoUrl="/api/matches/1/runs/2/plain.mp4" />);
    expect(screen.getByText(/1:30\.04 · 64 of 71 ball flights reconstructed in 3D · 4 players/)).toBeInTheDocument();
    expect(screen.getByText('Rally 1 of 1')).toBeInTheDocument();
    expect(container.querySelectorAll('.rp-hit-lane .rp-tick')).toHaveLength(62);
    engine().emit(state({ flight: 0 }));
    expect(screen.getByText('Shot 1 of 71')).toBeInTheDocument();
    expect(screen.getByText('P 3 · Team 2')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
  });

  it('zooms the timeline to the rally around the playhead', () => {
    const { container } = render(<Replay scene={makeSceneFixture()} videoUrl={null} />);
    expect(screen.getByText('Rally 1 of 2')).toBeInTheDocument();
    expect(container.querySelectorAll('.rp-flight-lane .rp-span')).toHaveLength(2);
    expect(container.querySelectorAll('.rp-hit-lane .rp-tick')).toHaveLength(3);
    engine().emit({ type: 'time', t: 7, ballY: null });
    expect(screen.getByText('Rally 2 of 2')).toBeInTheDocument();
    expect(container.querySelectorAll('.rp-flight-lane .rp-span')).toHaveLength(1);
    expect(container.querySelector<HTMLElement>('.rp-lanes .rp-head')?.style.left).toBe(`${((7 - 5.2) / 4.8) * 100}%`);
  });

  it('pauses while scrubbing, seeks inside the rally window, then resumes', () => {
    render(<Replay scene={makeSceneFixture()} videoUrl={null} />);
    const scrub = screen.getByRole('slider', { name: 'Time in the rally' });
    fireEvent.pointerDown(scrub);
    expect(engine().setPlaying).toHaveBeenLastCalledWith(false);
    fireEvent.change(scrub, { target: { value: '500' } });
    expect(engine().seek).toHaveBeenLastCalledWith(2.6);
    fireEvent.pointerUp(scrub);
    expect(engine().setPlaying).toHaveBeenLastCalledWith(true);
    fireEvent.change(screen.getByRole('slider', { name: 'Time in the clip' }), { target: { value: '1000' } });
    expect(engine().seek).toHaveBeenLastCalledWith(10);
  });

  it('renders payload text as text, never as markup', () => {
    const scene: SceneData = makeSceneFixture();
    const evil = '<img src=x onerror="alert(1)">';
    scene.players[0].id = evil;
    scene.flights[0].hitter = evil;
    const { container } = render(<Replay scene={scene} videoUrl={null} />);
    engine().emit(state({ flight: 0 }));
    expect(within(container).getByText(`${evil.replace('_', ' ')} · Team 1`)).toBeInTheDocument();
    expect(container.querySelector('img')).toBeNull();
  });

  it('says so when WebGL cannot start', () => {
    createReplay.mockImplementation(() => {
      throw new Error('no WebGL');
    });
    render(<Replay scene={makeSceneFixture()} videoUrl={null} />);
    expect(screen.getByRole('alert')).toHaveTextContent('needs WebGL');
  });
});
