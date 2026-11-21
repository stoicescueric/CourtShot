import * as THREE from 'three';
import { createReplay, type ReplayEvent, type ReplayState } from './engine';
import { makeSceneFixture, sceneFinalF } from './fixtures';

// jsdom has no WebGL: everything but the renderer is real three.js
const renderers = vi.hoisted(() => [] as { dispose: () => void; forceContextLoss: () => void; domElement: HTMLCanvasElement }[]);
vi.mock('three', async (importOriginal) => {
  const actual = await importOriginal<typeof import('three')>();
  class WebGLRenderer {
    domElement = document.createElement('canvas');
    shadowMap = { enabled: false, type: 0 };
    toneMapping = 0;
    toneMappingExposure = 1;
    setPixelRatio = vi.fn();
    setSize = vi.fn();
    render = vi.fn();
    dispose = vi.fn();
    forceContextLoss = vi.fn();
    constructor() {
      renderers.push(this);
    }
  }
  return { ...actual, WebGLRenderer };
});

const frames = new Map<number, FrameRequestCallback>();
let frameId = 0;
let clock = 0;
const observers: { observe: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }[] = [];

/** Runs n animation frames, 40 ms apart. */
function runFrames(n: number) {
  for (let i = 0; i < n; i++) {
    clock += 40;
    const pending = [...frames.values()];
    frames.clear();
    pending.forEach((cb) => cb(clock));
  }
}
const flush = () => new Promise((r) => setTimeout(r, 0));

const fake2d = new Proxy({}, {
  get: (_t, key) => (key === 'createRadialGradient' ? () => ({ addColorStop() {} }) : () => {}),
  set: () => true,
});

beforeEach(() => {
  frames.clear();
  observers.length = 0;
  renderers.length = 0;
  clock = performance.now();
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    frames.set(++frameId, cb);
    return frameId;
  });
  vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id));
  vi.stubGlobal('ResizeObserver', class {
    observe = vi.fn();
    disconnect = vi.fn();
    constructor() {
      observers.push(this);
    }
  });
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(fake2d as never);
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function setup() {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const video = document.createElement('video');
  video.requestVideoFrameCallback = vi.fn(() => 42);
  video.cancelVideoFrameCallback = vi.fn();
  const engine = createReplay(host, video, makeSceneFixture(), { onError: vi.fn() });
  const states: ReplayState[] = [];
  const times: number[] = [];
  engine.subscribe((e: ReplayEvent) => (e.type === 'state' ? states.push(e.state) : times.push(e.t)));
  return { host, video, engine, states, times, last: () => states[states.length - 1] };
}

/** Records the ids of every geometry, material and texture disposed. */
function trackDisposal() {
  const ids = { geometry: new Set<number>(), material: new Set<number>(), texture: new Set<number>() };
  vi.spyOn(THREE.BufferGeometry.prototype, 'dispose').mockImplementation(function (this: THREE.BufferGeometry) {
    ids.geometry.add(this.id);
  });
  vi.spyOn(THREE.Material.prototype, 'dispose').mockImplementation(function (this: THREE.Material) {
    ids.material.add(this.id);
  });
  vi.spyOn(THREE.Texture.prototype, 'dispose').mockImplementation(function (this: THREE.Texture) {
    ids.texture.add(this.id);
  });
  return ids;
}

/** Ids are sequential per kind: a probe before and after brackets everything created in between. */
function probe() {
  return { geometry: new THREE.BufferGeometry().id, material: new THREE.MeshBasicMaterial().id, texture: new THREE.Texture().id };
}

function expectAllDisposed(before: ReturnType<typeof probe>, after: ReturnType<typeof probe>, disposed: ReturnType<typeof trackDisposal>) {
  for (const kind of ['geometry', 'material', 'texture'] as const) {
    const created = Array.from({ length: after[kind] - before[kind] - 1 }, (_, i) => before[kind] + 1 + i);
    expect(created.length).toBeGreaterThan(5);
    expect(created.filter((id) => !disposed[kind].has(id))).toEqual([]);
  }
}

describe('createReplay', () => {
  beforeAll(() => {
    new THREE.Sprite(new THREE.SpriteMaterial()); // three's one shared sprite geometry is not the replay's
  });

  it('dispose() releases every geometry, material and texture, the renderer, timers, observers and listeners', async () => {
    const disposed = trackDisposal();
    const before = probe();
    const added: [string, EventListenerOrEventListenerObject | null][] = [];
    const removed: [string, EventListenerOrEventListenerObject | null][] = [];
    vi.spyOn(HTMLMediaElement.prototype, 'addEventListener').mockImplementation(function (type, fn) {
      added.push([type, fn]);
    });
    vi.spyOn(HTMLMediaElement.prototype, 'removeEventListener').mockImplementation(function (type, fn) {
      removed.push([type, fn]);
    });
    const { host, video, engine } = setup();
    await flush(); // name tags are baked once the font is ready
    for (const t of [1.0, 3.0, 7.3, 2.0]) {
      engine.seek(t); // through every flight, so path lines are built and replaced
      runFrames(2);
    }
    engine.setMode('overlay');
    runFrames(1);
    expect(host.querySelector('canvas')).not.toBeNull();
    expect(frames.size).toBe(1);

    engine.dispose();
    const after = probe();

    expectAllDisposed(before, after, disposed);
    expect(renderers[0].dispose).toHaveBeenCalled();
    expect(renderers[0].forceContextLoss).toHaveBeenCalled();
    expect(host.querySelector('canvas')).toBeNull();
    expect(frames.size).toBe(0);
    expect(observers[0].disconnect).toHaveBeenCalled();
    expect(video.cancelVideoFrameCallback).toHaveBeenCalledWith(42);
    expect(added.length).toBeGreaterThan(0);
    expect(removed).toEqual(added);
  });

  it('creates no name tags after an early dispose (StrictMode unmounts before the font is ready)', async () => {
    const disposed = trackDisposal();
    const before = probe();
    const { engine } = setup();
    engine.dispose();
    await flush();
    expectAllDisposed(before, probe(), disposed);
  });

  it('survives a dispose and a second create on the same host', () => {
    const first = setup();
    first.engine.dispose();
    const host = first.host;
    const engine = createReplay(host, null, makeSceneFixture(), { onError: vi.fn() });
    expect(host.querySelectorAll('canvas')).toHaveLength(1);
    runFrames(1);
    engine.dispose();
    expect(host.querySelectorAll('canvas')).toHaveLength(0);
  });

  it("runs its own clock without a video and publishes time and flight changes", () => {
    const { engine, last, times } = setup();
    expect(last()).toMatchObject({ playing: true, mode: 'off', view: 'broadcast', flight: -1 });
    runFrames(5);
    expect(times[times.length - 1]).toBeCloseTo(0.2, 1);
    engine.setPlaying(false);
    engine.seek(1.0);
    runFrames(3);
    expect(times[times.length - 1]).toBe(1.0);
    expect(last()).toMatchObject({ playing: false, flight: 0 });
    engine.seek(3.0);
    runFrames(1);
    expect(last().flight).toBe(1);
    engine.setLayers({ low: false }); // flight 1 is low confidence: hidden
    runFrames(1);
    expect(last().flight).toBe(-1);
    engine.seek(5.0);
    runFrames(1);
    expect(last().flight).toBe(-1);
    engine.dispose();
  });

  it('plays the scene of a real run through every flight, then disposes it all', async () => {
    const disposed = trackDisposal();
    const before = probe();
    const host = document.createElement('div');
    document.body.appendChild(host);
    const scene = sceneFinalF();
    const engine = createReplay(host, null, scene, { onError: vi.fn() });
    const flights = new Set<number>();
    engine.subscribe((e: ReplayEvent) => e.type === 'state' && flights.add(e.state.flight));
    await flush();
    for (const f of scene.flights) {
      engine.seek((f.t0 + f.t1) / 2);
      runFrames(2);
    }
    expect(flights.size).toBeGreaterThan(60);
    engine.dispose();
    expectAllDisposed(before, probe(), disposed);
  });

  it('locks the broadcast camera in the overlay and ignores other views there', () => {
    const { engine, last } = setup();
    engine.setView('top');
    expect(last().view).toBe('top');
    engine.setMode('overlay');
    expect(last()).toMatchObject({ mode: 'overlay', view: 'broadcast' });
    engine.setView('side');
    expect(last().view).toBe('broadcast');
    engine.setMode('off');
    expect(last().mode).toBe('off');
    engine.dispose();
  });

  it('refuses video modes without a video element', () => {
    const host = document.createElement('div');
    const engine = createReplay(host, null, makeSceneFixture(), { onError: vi.fn() });
    let mode = '';
    engine.subscribe((e) => {
      if (e.type === 'state') mode = e.state.mode;
    });
    engine.setMode('side');
    expect(mode).toBe('off');
    engine.dispose();
  });

  it('reports a lost GPU context and stops drawing', () => {
    const host = document.createElement('div');
    const onError = vi.fn();
    const engine = createReplay(host, null, makeSceneFixture(), { onError });
    renderers[0].domElement.dispatchEvent(new Event('webglcontextlost'));
    expect(onError).toHaveBeenCalledOnce();
    expect(frames.size).toBe(0);
    engine.dispose();
  });
});
