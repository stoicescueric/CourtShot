/**
 * All three.js code of the 3D replay: a port of the reference scene3d.html (scene, loop, cameras, video sync)
 * behind a small imperative interface. React renders the controls from `subscribe`.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Flight, SceneData } from '../types/scene';
import { FOLLOW_FOV, fixedViews, followGoal, fovFor, tweenAt, type Tween, type ViewName } from './camera';
import { drawnFrame, tick, videoSeekTime } from './clock';
import { NJ, ballAt, bounceRange, decodeScene, flightPoints, poseAt, sortedBounces } from './decode';

export type VideoMode = 'off' | 'side' | 'overlay';

export interface Layers {
  trail: boolean;
  path: boolean;
  guide: boolean;
  low: boolean;
  labels: boolean;
}

export interface ReplayState {
  playing: boolean;
  speed: number;
  /** "free" after the user orbits away from a named view. */
  view: ViewName | 'free';
  mode: VideoMode;
  layers: Layers;
  /** Index into scene.flights of the flight on show; -1 between flights or when it is hidden. */
  flight: number;
  videoReady: boolean;
  videoError: boolean;
}

export const INITIAL_STATE: ReplayState = {
  playing: true,
  speed: 1,
  view: 'broadcast',
  mode: 'off',
  layers: { trail: true, path: true, guide: true, low: true, labels: true },
  flight: -1,
  videoReady: false,
  videoError: false,
};

export type ReplayEvent =
  | { type: 'time'; t: number; ballY: number | null }
  | { type: 'state'; state: ReplayState };
export type ReplayListener = (event: ReplayEvent) => void;

export interface ReplayCallbacks {
  /** The 3D view stopped for good; the message is for the user. */
  onError(message: string): void;
}

export interface ReplayHandle {
  setView(view: ViewName): void;
  setLayers(layers: Partial<Layers>): void;
  setSpeed(speed: number): void;
  setPlaying(playing: boolean): void;
  setMode(mode: VideoMode): void;
  seek(t: number): void;
  dispose(): void;
  /** Sends the current state at once, then every state change and every new time. Returns the unsubscribe. */
  subscribe(listener: ReplayListener): () => void;
}

interface Disposable {
  dispose(): void;
}
type Own = <T extends Disposable>(x: T) => T;

interface Palette {
  t1: THREE.Color;
  t2: THREE.Color;
  other: THREE.Color;
  ball: THREE.Color;
  warn: THREE.Color;
  bg: THREE.Color;
  ground: THREE.Color;
  card: THREE.Color;
  text: THREE.Color;
  calib: THREE.Color;
}

const LABEL_FONT = '600 40px Inter, -apple-system, "Segoe UI", Roboto, sans-serif';
const LINE_WHITE = new THREE.Color(0xf4f7fa);
const WALL_RING = new THREE.Color(0xbfe6ff);
const TRAIL = 28;
const TRAIL_S = 0.45;
const RING_S = 0.8;
const FOLLOW_RATE = 0.06;
const OVERLAY_OPACITY = 0.55;
const BONES: [number, number][] = [[1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12], [1, 2], [7, 8]];
const BONE_RADIUS = [0.055, 0.045, 0.055, 0.045, 0.075, 0.06, 0.075, 0.06, 0.05, 0.06];
const Y_AXIS = new THREE.Vector3(0, 1, 0);
const LABEL_LIFT = new THREE.Vector3(0, 0.38, 0);

function readPalette(el: HTMLElement): Palette {
  const css = getComputedStyle(el);
  const c = (name: string, fallback: string) => new THREE.Color(css.getPropertyValue(name).trim() || fallback);
  return {
    t1: c('--t1', '#d4f542'),
    t2: c('--t2', '#4d8dff'),
    other: new THREE.Color(0xaaaaaa),
    ball: c('--key', '#d4f542'),
    warn: c('--clay', '#ff8a65'),
    bg: c('--bg', '#000000'),
    ground: c('--card', '#1c1c1e'),
    card: c('--card', '#1c1c1e'),
    text: c('--text', '#f5f5f7'),
    calib: c('--rp-calib', '#ff5ad1'),
  };
}

function canvasTexture(
  own: Own,
  w: number,
  h: number,
  draw: (g: CanvasRenderingContext2D, w: number, h: number) => void,
  repeat?: [number, number],
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const g = canvas.getContext('2d');
  if (g) draw(g, w, h);
  const tex = own(new THREE.CanvasTexture(canvas));
  tex.colorSpace = THREE.SRGBColorSpace;
  if (repeat) {
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(...repeat);
  }
  tex.anisotropy = 8;
  return tex;
}

function hex(c: THREE.Color): string {
  return `#${c.getHexString()}`;
}

/** Court, walls, net and lights (html:281-409). */
function buildCourt(scene: THREE.Scene, own: Own, pal: Palette) {
  scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x1a2430, 0.9));
  const sun = own(new THREE.DirectionalLight(0xffffff, 2.2));
  sun.position.set(6, 22, 9);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -9, right: 9, top: 13, bottom: -13, near: 1, far: 60 });
  sun.shadow.bias = -0.0004;
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x9fc4ff, 0.5);
  fill.position.set(-10, 12, -14);
  scene.add(fill);

  const turf = canvasTexture(own, 256, 256, (g, w, h) => {
    g.fillStyle = '#245a96';
    g.fillRect(0, 0, w, h);
    for (let i = 0; i < 9000; i++) {
      g.fillStyle = Math.random() < 0.5 ? 'rgba(10,30,70,0.18)' : 'rgba(120,170,230,0.10)';
      g.fillRect(Math.random() * w, Math.random() * h, 1.2, 1.2);
    }
  }, [8, 16]);
  const meshTex = canvasTexture(own, 64, 64, (g, w, h) => {
    g.strokeStyle = 'rgba(30,34,40,0.95)';
    g.lineWidth = 3;
    g.beginPath();
    g.moveTo(0, 0);
    g.lineTo(w, h);
    g.moveTo(w, 0);
    g.lineTo(0, h);
    g.stroke();
  }, [1, 1]);
  const netTex = canvasTexture(own, 64, 64, (g, w, h) => {
    g.strokeStyle = 'rgba(235,240,245,0.9)';
    g.lineWidth = 2;
    g.strokeRect(1, 1, w - 2, h - 2);
  }, [1, 1]);

  const court = new THREE.Group();
  scene.add(court);
  const flat = (geo: THREE.PlaneGeometry, mat: THREE.Material, y: number) => {
    const m = new THREE.Mesh(own(geo), own(mat));
    m.rotation.x = -Math.PI / 2;
    m.position.y = y;
    m.receiveShadow = true;
    court.add(m);
  };
  flat(new THREE.PlaneGeometry(220, 220), new THREE.MeshStandardMaterial({ color: pal.ground, roughness: 1 }), -0.01);
  flat(new THREE.PlaneGeometry(16, 26), new THREE.MeshStandardMaterial({ color: 0x1b3a5f, roughness: 0.95 }), -0.005);
  flat(new THREE.PlaneGeometry(10, 20), new THREE.MeshStandardMaterial({ map: turf, roughness: 0.92 }), 0);

  // court lines stay visible in the video overlay
  const lines = new THREE.Group();
  scene.add(lines);
  const lineMat = own(new THREE.MeshStandardMaterial({ color: LINE_WHITE, roughness: 0.7 }));
  const stripe = (x0: number, z0: number, x1: number, z1: number) => {
    const m = new THREE.Mesh(own(new THREE.PlaneGeometry(Math.hypot(x1 - x0, z1 - z0), 0.05)), lineMat);
    m.rotation.x = -Math.PI / 2;
    m.rotation.z = -Math.atan2(z1 - z0, x1 - x0);
    m.position.set((x0 + x1) / 2, 0.002, (z0 + z1) / 2);
    m.receiveShadow = true;
    lines.add(m);
  };
  stripe(-5, 6.95, 5, 6.95); // service lines, 6.95 m from the net
  stripe(-5, -6.95, 5, -6.95);
  stripe(0, -7.15, 0, 7.15); // centre service line, 20 cm past the service lines

  // overlay only: floor edges and corner edges, to check the calibration against the video
  const edges = new THREE.Group();
  edges.visible = false;
  scene.add(edges);
  const edgeMat = own(new THREE.LineBasicMaterial({ color: pal.calib, transparent: true, opacity: 0.9 }));
  const P = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
  const edgeGeo = (pts: THREE.Vector3[]) => own(new THREE.BufferGeometry().setFromPoints(pts));
  edges.add(new THREE.LineLoop(edgeGeo([P(-5, 0, 10), P(5, 0, 10), P(5, 0, -10), P(-5, 0, -10)]), edgeMat));
  for (const [x, z] of [[-5, 10], [5, 10], [-5, -10], [5, -10]]) edges.add(new THREE.Line(edgeGeo([P(x, 0, z), P(x, 3, z)]), edgeMat));
  for (const z of [10, -10]) edges.add(new THREE.Line(edgeGeo([P(-5, 3, z), P(5, 3, z)]), edgeMat));

  const glassMat = own(new THREE.MeshPhysicalMaterial({
    color: 0xbfe6ff, transparent: true, opacity: 0.1, roughness: 0.05, metalness: 0, side: THREE.DoubleSide, depthWrite: false,
  }));
  const meshMat = own(new THREE.MeshStandardMaterial({
    map: meshTex, transparent: true, alphaTest: 0.35, side: THREE.DoubleSide, roughness: 0.6, metalness: 0.4,
  }));
  const steel = own(new THREE.MeshStandardMaterial({ color: 0x23282f, roughness: 0.5, metalness: 0.6 }));
  const glass = (w: number, h: number, cx: number, cy: number, cz: number, rotY: number) => {
    const m = new THREE.Mesh(own(new THREE.PlaneGeometry(w, h)), glassMat);
    m.position.set(cx, cy, cz);
    m.rotation.y = rotY;
    court.add(m);
  };
  // wire mesh tiled at 7 cm: each panel gets its own material and texture repeat
  const fence = (w: number, h: number, cx: number, cy: number, cz: number, rotY: number) => {
    const map = own(meshTex.clone());
    map.needsUpdate = true;
    map.repeat.set(w / 0.07, h / 0.07);
    const mat = own(meshMat.clone());
    mat.map = map;
    const m = new THREE.Mesh(own(new THREE.PlaneGeometry(w, h)), mat);
    m.position.set(cx, cy, cz);
    m.rotation.y = rotY;
    court.add(m);
  };
  const postGeo = { 3: own(new THREE.BoxGeometry(0.06, 3, 0.06)), 4: own(new THREE.BoxGeometry(0.06, 4, 0.06)) };
  const post = (x: number, z: number, h: 3 | 4) => {
    const p = new THREE.Mesh(postGeo[h], steel);
    p.position.set(x, h / 2, z);
    p.castShadow = true;
    court.add(p);
  };
  // back walls: 3 m glass, mesh up to 4 m
  for (const z of [-10, 10]) {
    glass(10, 3, 0, 1.5, z, 0);
    fence(10, 1, 0, 3.5, z, 0);
    for (let x = -5; x <= 5.01; x += 2) post(x, z, 4);
  }
  // side walls: per end 2 m of 3 m glass, then 2 m of 2 m glass (stepped), then 12 m of mesh 3 m high
  for (const x of [-5, 5]) {
    const r = Math.PI / 2;
    for (const s of [-1, 1]) {
      glass(2, 3, x, 1.5, s * 9, r);
      fence(2, 1, x, 3.5, s * 9, r);
      glass(2, 2, x, 1, s * 7, r);
      fence(2, 1, x, 2.5, s * 7, r);
      post(x, s * 8, 4);
      post(x, s * 6, 3);
    }
    fence(12, 3, x, 1.5, 0, r);
    for (let z = -4; z <= 4.01; z += 2) post(x, z, 3);
  }

  // net: 0.88 m in the middle, 0.92 m at the posts
  const segs = 40;
  const top: THREE.Vector3[] = [];
  const pos: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  for (let i = 0; i <= segs; i++) {
    const x = -5 + (10 * i) / segs;
    const h = 0.88 + 0.04 * (x / 5) ** 2;
    top.push(new THREE.Vector3(x, h, 0));
    pos.push(x, 0.02, 0, x, h, 0);
    uv.push(((i / segs) * 10) / 0.05, 0, ((i / segs) * 10) / 0.05, h / 0.05);
    if (i < segs) {
      const a = 2 * i;
      idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  }
  const netGeo = own(new THREE.BufferGeometry());
  netGeo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  netGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  netGeo.setIndex(idx);
  netGeo.computeVertexNormals();
  const netMap = own(netTex.clone());
  netMap.needsUpdate = true;
  netMap.wrapS = netMap.wrapT = THREE.RepeatWrapping;
  const net = new THREE.Mesh(netGeo, own(new THREE.MeshStandardMaterial({
    map: netMap, transparent: true, alphaTest: 0.3, side: THREE.DoubleSide, color: 0xdfe6ee,
  })));
  net.castShadow = true;
  court.add(net);
  const tape = new THREE.Mesh(
    own(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(top), 40, 0.025, 6)),
    own(new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6 })),
  );
  tape.castShadow = true;
  lines.add(tape);
  const netPostGeo = own(new THREE.CylinderGeometry(0.04, 0.04, 1.0, 12));
  for (const x of [-5.05, 5.05]) {
    const p = new THREE.Mesh(netPostGeo, steel);
    p.position.set(x, 0.5, 0);
    court.add(p);
  }
  return { court, edges, lineMat };
}

interface Player3D {
  id: string;
  color: THREE.Color;
  group: THREE.Group;
  bones: THREE.Mesh[];
  joints: THREE.Mesh[];
  torso: THREE.Mesh;
  neck: THREE.Mesh;
  racket: THREE.Group;
  wrist: number;
  elbow: number;
  label: THREE.Sprite | null;
}

/** Bodies as cylinders and spheres, a team shirt and a racket in the playing hand (html:412-458). */
function buildPlayers(scene: THREE.Scene, own: Own, pal: Palette, data: SceneData) {
  const unitCyl = own(new THREE.CylinderGeometry(1, 1, 1, 12));
  const unitSph = own(new THREE.SphereGeometry(1, 16, 12));
  const unitBox = own(new THREE.BoxGeometry(1, 1, 1));
  const handleGeo = own(new THREE.CylinderGeometry(0.018, 0.018, 0.2, 8));
  const faceGeo = own(new THREE.CylinderGeometry(0.13, 0.13, 0.038, 28));
  const rimGeo = own(new THREE.TorusGeometry(0.13, 0.012, 8, 32));
  const skin = own(new THREE.MeshStandardMaterial({ color: 0xc89a7a, roughness: 0.7 }));
  const shoe = own(new THREE.MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.6 }));
  const leg = own(new THREE.MeshStandardMaterial({ color: 0x1d2530, roughness: 0.8 }));
  const handleMat = own(new THREE.MeshStandardMaterial({ color: 0x15191f }));
  const faceMat = own(new THREE.MeshStandardMaterial({ color: 0x20262e, roughness: 0.4, metalness: 0.2 }));
  /** Materials that turn translucent in the overlay. */
  const bodyMats: THREE.Material[] = [skin, shoe, leg];

  const players = data.players.map((p): Player3D => {
    const color = p.team === 'T_1' ? pal.t1 : p.team === 'T_2' ? pal.t2 : pal.other;
    const shirt = own(new THREE.MeshStandardMaterial({ color, roughness: 0.55 }));
    bodyMats.push(shirt);
    const group = new THREE.Group();
    scene.add(group);
    const mk = (geo: THREE.BufferGeometry, mat: THREE.Material) => {
      const m = new THREE.Mesh(geo, mat);
      m.castShadow = true;
      group.add(m);
      return m;
    };
    const bones = BONES.map((_, i) => mk(unitCyl, i < 4 || i === 8 ? shirt : leg));
    const joints = Array.from({ length: NJ }, (_, j) => {
      const mat = j === 0 ? skin : j >= 11 ? shoe : j === 5 || j === 6 ? skin : j >= 7 ? leg : shirt;
      const m = mk(unitSph, mat);
      m.scale.setScalar(j === 0 ? 0.11 : j >= 9 ? 0.06 : j >= 7 ? 0.075 : j >= 5 ? 0.045 : j >= 3 ? 0.05 : 0.065);
      return m;
    });
    const torso = mk(unitBox, shirt);
    const neck = mk(unitCyl, skin);
    const racket = new THREE.Group();
    group.add(racket);
    const handle = new THREE.Mesh(handleGeo, handleMat);
    handle.position.z = 0.1;
    handle.rotation.x = Math.PI / 2;
    handle.castShadow = true;
    racket.add(handle);
    const face = new THREE.Mesh(faceGeo, faceMat);
    face.scale.set(1, 1, 1.12);
    face.position.z = 0.35;
    face.castShadow = true;
    racket.add(face);
    const rim = new THREE.Mesh(rimGeo, own(new THREE.MeshStandardMaterial({ color, roughness: 0.4 })));
    rim.rotation.x = Math.PI / 2;
    rim.scale.set(1, 1.12, 1);
    rim.position.z = 0.35;
    racket.add(rim);
    const wrist = p.hand === 'left' ? 5 : 6;
    return { id: p.id, color, group, bones, joints, torso, neck, racket, wrist, elbow: wrist - 2, label: null };
  });
  return { players, bodyMats };
}

/** A name tag drawn on a canvas; call only once the label font has loaded. */
function labelSprite(own: Own, pal: Palette, text: string, color: THREE.Color): THREE.Sprite {
  const tex = canvasTexture(own, 256, 96, (g, w, h) => {
    g.fillStyle = hex(pal.card);
    g.globalAlpha = 0.86;
    g.beginPath();
    g.roundRect(8, 16, w - 16, h - 32, 14);
    g.fill();
    g.globalAlpha = 1;
    g.fillStyle = hex(color);
    g.fillRect(8, 16, 10, h - 32);
    g.fillStyle = hex(pal.text);
    g.font = LABEL_FONT;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(text, w / 2 + 6, h / 2 + 2);
  });
  const s = new THREE.Sprite(own(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true })));
  s.scale.set(0.62, 0.23, 1);
  s.renderOrder = 10;
  return s;
}

async function labelFontReady(): Promise<void> {
  const fonts: FontFaceSet | undefined = document.fonts;
  if (!fonts) return;
  try {
    await fonts.load(LABEL_FONT);
  } catch {
    // the fallback font is used
  }
  await fonts.ready;
}

export function createReplay(
  host: HTMLElement,
  video: HTMLVideoElement | null,
  data: SceneData,
  callbacks: ReplayCallbacks,
): ReplayHandle {
  const d = decodeScene(data);
  const owned: Disposable[] = [];
  const own: Own = (x) => {
    owned.push(x);
    return x;
  };
  const cleanups: (() => void)[] = [];
  const listen = (target: EventTarget, type: string, fn: EventListener) => {
    target.addEventListener(type, fn);
    cleanups.push(() => target.removeEventListener(type, fn));
  };
  const listeners = new Set<ReplayListener>();
  const pal = readPalette(host);

  // ---------- renderer, scene, camera ----------
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const background = pal.bg.clone();
  const fog = new THREE.Fog(pal.bg.clone(), 45, 110);
  scene.background = background;
  scene.fog = fog;
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 300);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = 2;
  controls.maxDistance = 80;

  const { court, edges, lineMat } = buildCourt(scene, own, pal);
  const { players, bodyMats } = buildPlayers(scene, own, pal, data);

  // ---------- ball, trail, height line, flight path, bounce rings ----------
  const glow = canvasTexture(own, 64, 64, (g, w, h) => {
    const r = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    r.addColorStop(0, 'rgba(255,255,255,1)');
    r.addColorStop(0.25, 'rgba(255,255,255,0.55)');
    r.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = r;
    g.fillRect(0, 0, w, h);
  });
  const ballMesh = new THREE.Mesh(
    own(new THREE.SphereGeometry(0.0325, 20, 14)),
    own(new THREE.MeshStandardMaterial({ color: pal.ball, emissive: pal.ball, emissiveIntensity: 0.6, roughness: 0.5 })),
  );
  ballMesh.castShadow = true;
  scene.add(ballMesh);
  const haloMat = own(new THREE.SpriteMaterial({ map: glow, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  const halo = new THREE.Sprite(haloMat);
  halo.scale.setScalar(0.42);
  scene.add(halo);
  const haloOk = pal.ball.clone().lerp(new THREE.Color(0xffffff), 0.5);
  const haloLow = pal.warn.clone().lerp(new THREE.Color(0xffffff), 0.3);

  const trailGeo = own(new THREE.BufferGeometry());
  const trailPos = new THREE.Float32BufferAttribute(new Float32Array(TRAIL * 3), 3);
  const trailCol = new THREE.Float32BufferAttribute(new Float32Array(TRAIL * 3), 3);
  trailGeo.setAttribute('position', trailPos);
  trailGeo.setAttribute('color', trailCol);
  const trail = new THREE.Line(trailGeo, own(new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  })));
  trail.frustumCulled = false;
  scene.add(trail);
  const guide = new THREE.Line(
    own(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()])),
    own(new THREE.LineDashedMaterial({ color: pal.ball, dashSize: 0.08, gapSize: 0.06, transparent: true, opacity: 0.7 })),
  );
  guide.frustumCulled = false;
  scene.add(guide);
  const spot = new THREE.Mesh(
    own(new THREE.RingGeometry(0.05, 0.075, 32)),
    own(new THREE.MeshBasicMaterial({ color: pal.ball, transparent: true, opacity: 0.8, depthWrite: false })),
  );
  spot.rotation.x = -Math.PI / 2;
  scene.add(spot);

  // only the current flight's path exists; it is rebuilt when the flight changes
  const pathMat = own(new THREE.LineBasicMaterial({ color: pal.ball, transparent: true, opacity: 0.5 }));
  const path = new THREE.Line(new THREE.BufferGeometry(), pathMat);
  path.frustumCulled = false;
  scene.add(path);

  const bounces = sortedBounces(data.flights);
  const ringGeo = own(new THREE.RingGeometry(0.8, 1, 40));
  const rings = Array.from({ length: 6 }, () => {
    const m = new THREE.Mesh(ringGeo, own(new THREE.MeshBasicMaterial({
      color: 0xffffff, transparent: true, depthWrite: false, side: THREE.DoubleSide,
    })));
    m.visible = false;
    scene.add(m);
    return m;
  });

  // ---------- state ----------
  let disposed = false;
  let t = 0;
  let playing = INITIAL_STATE.playing;
  let speed = INITIAL_STATE.speed;
  let view: ReplayState['view'] = INITIAL_STATE.view;
  let mode: VideoMode = INITIAL_STATE.mode;
  let layers: Layers = { ...INITIAL_STATE.layers };
  let flight = -1;
  let vidReady = false;
  let vidError = false;
  let frameTime: number | null = null;
  let tween: Tween | null = null;

  const snapshot = (): ReplayState => ({
    playing, speed, view, mode, layers: { ...layers }, flight, videoReady: vidReady, videoError: vidError,
  });
  const publish = () => {
    const state = snapshot();
    listeners.forEach((l) => l({ type: 'state', state }));
  };

  // ---------- cameras ----------
  const views = fixedViews(data.camera);
  const camRight = new THREE.Vector3(...data.camera.right);
  const camUp = new THREE.Vector3(...data.camera.up);
  const camBack = new THREE.Vector3(...data.camera.back);
  const basis = new THREE.Matrix4();
  const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;

  /** The exact calibrated camera, roll included (OrbitControls would drop the roll). */
  function lockBroadcast() {
    camera.position.copy(views.broadcast.pos);
    basis.makeBasis(camRight, camUp, camBack);
    camera.quaternion.setFromRotationMatrix(basis);
    camera.fov = fovFor(views.broadcast, camera.aspect, data.camera.aspect);
    camera.updateProjectionMatrix();
    controls.target.copy(views.broadcast.target);
  }

  function goTo(pos: THREE.Vector3, target: THREE.Vector3, fov: number) {
    if (reducedMotion()) {
      tween = null;
      camera.position.copy(pos);
      controls.target.copy(target);
      camera.fov = fov;
      camera.updateProjectionMatrix();
      return;
    }
    tween = {
      p0: camera.position.clone(), t0: controls.target.clone(), f0: camera.fov,
      p1: pos.clone(), t1: target.clone(), f1: fov, start: performance.now(),
    };
  }

  const ball = new THREE.Vector3();
  function applyView(name: ViewName) {
    view = name;
    if (name === 'follow') {
      const seen = ballAt(d, t * d.fps, ball) >= 0;
      const goal = followGoal(camera.position, controls.target, seen ? ball : null);
      goTo(goal.pos, goal.target, FOLLOW_FOV);
    } else {
      const v = views[name];
      goTo(v.pos, v.target, fovFor(v, camera.aspect, data.camera.aspect));
    }
    publish();
  }

  const onControlsStart = () => {
    tween = null;
    if (view !== 'follow' && view !== 'free') {
      view = 'free';
      publish();
    }
  };
  controls.addEventListener('start', onControlsStart);

  function resize() {
    const w = host.clientWidth;
    const h = host.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    if (view !== 'free' && view !== 'follow' && !tween) camera.fov = fovFor(views[view], camera.aspect, data.camera.aspect);
    camera.updateProjectionMatrix();
    if (mode === 'overlay') lockBroadcast();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(host);
  resize();
  camera.position.copy(views.broadcast.pos);
  controls.target.copy(views.broadcast.target);
  camera.fov = fovFor(views.broadcast, camera.aspect, data.camera.aspect);
  camera.updateProjectionMatrix();

  // ---------- video ----------
  const videoActive = () => vidReady && mode !== 'off';
  function seekVideo(s: number) {
    if (!video || !vidReady) return;
    video.currentTime = videoSeekTime(s, d.fps, video.duration);
    frameTime = null; // the last presented frame is from before the seek
  }
  function syncVideo() {
    if (!video) return;
    if (!videoActive()) {
      video.pause();
      return;
    }
    video.playbackRate = speed;
    if (playing) {
      if (video.paused) {
        seekVideo(t);
        video.play().catch(() => {});
      }
    } else {
      video.pause();
      seekVideo(t);
    }
  }

  function setMode(m: VideoMode) {
    if (m !== 'off' && !video) return;
    const was = mode;
    mode = m;
    const overlay = m === 'overlay';
    court.visible = !overlay;
    edges.visible = overlay;
    scene.background = overlay ? null : background;
    scene.fog = overlay ? null : fog;
    lineMat.color.copy(overlay ? pal.calib : LINE_WHITE);
    for (const mat of bodyMats) {
      mat.transparent = overlay;
      mat.opacity = overlay ? OVERLAY_OPACITY : 1;
      mat.needsUpdate = true;
    }
    controls.enabled = !overlay;
    resize();
    if (overlay) {
      tween = null;
      view = 'broadcast';
      lockBroadcast();
    } else if (was === 'overlay') {
      applyView('broadcast');
    }
    syncVideo();
    publish();
  }

  function onVideoLoaded() {
    vidReady = true;
    vidError = false;
    if (mode === 'off') setMode('side');
    else {
      syncVideo();
      publish();
    }
  }

  let frameRequest = 0;
  if (video) {
    listen(video, 'loadeddata', onVideoLoaded);
    listen(video, 'error', () => {
      vidReady = false;
      vidError = true;
      publish();
    });
    listen(video, 'emptied', () => {
      vidReady = false;
      publish();
    });
    if ('requestVideoFrameCallback' in video) {
      const onFrame: VideoFrameRequestCallback = (_now, meta) => {
        frameTime = meta.mediaTime;
        frameRequest = video.requestVideoFrameCallback(onFrame);
      };
      frameRequest = video.requestVideoFrameCallback(onFrame);
      cleanups.push(() => video.cancelVideoFrameCallback(frameRequest));
    }
    video.playbackRate = speed;
    if (video.error) vidError = true;
    else if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) onVideoLoaded();
  }

  // ---------- name tags, once the font is there ----------
  void labelFontReady().then(() => {
    if (disposed) return;
    for (const p of players) {
      p.label = labelSprite(own, pal, p.id.replace('_', ' '), p.color);
      p.label.visible = false;
      scene.add(p.label);
    }
  });

  // ---------- per-frame update ----------
  const jp = Array.from({ length: NJ }, () => new THREE.Vector3());
  const prev = new THREE.Vector3();
  const tmp = new THREE.Vector3();
  const cylDir = new THREE.Vector3();
  const sh = new THREE.Vector3();
  const hp = new THREE.Vector3();
  const up = new THREE.Vector3();
  const across = new THREE.Vector3();
  const depth = new THREE.Vector3();
  const dir = new THREE.Vector3();
  const nrm = new THREE.Vector3();
  const xax = new THREE.Vector3();
  const delta = new THREE.Vector3();

  function placeCyl(m: THREE.Mesh, a: THREE.Vector3, b: THREE.Vector3, r: number) {
    cylDir.subVectors(b, a);
    const len = cylDir.length();
    if (len < 1e-4) {
      m.visible = false;
      return;
    }
    m.visible = true;
    m.position.addVectors(a, b).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(Y_AXIS, cylDir.divideScalar(len));
    m.scale.set(r, len, r);
  }

  function updatePlayer(p: Player3D, index: number, fr: number) {
    const have = poseAt(d, index, fr, jp);
    p.group.visible = have;
    if (p.label) p.label.visible = have && layers.labels;
    if (!have) return;
    BONES.forEach(([i, k], b) => placeCyl(p.bones[b], jp[i], jp[k], BONE_RADIUS[b]));
    p.joints.forEach((m, j) => m.position.copy(jp[j]));
    // torso box: spine along y, shoulders along x
    sh.addVectors(jp[1], jp[2]).multiplyScalar(0.5);
    hp.addVectors(jp[7], jp[8]).multiplyScalar(0.5);
    up.subVectors(sh, hp);
    const len = up.length();
    up.divideScalar(Math.max(len, 1e-4));
    across.subVectors(jp[2], jp[1]);
    across.addScaledVector(up, -across.dot(up));
    const w = Math.max(across.length(), tmp.subVectors(jp[8], jp[7]).length(), 0.22) + 0.1;
    if (across.lengthSq() < 1e-6) across.set(1, 0, 0);
    across.normalize();
    depth.crossVectors(across, up).normalize();
    basis.makeBasis(across, up, depth);
    p.torso.quaternion.setFromRotationMatrix(basis);
    p.torso.position.addVectors(sh, hp).multiplyScalar(0.5);
    p.torso.scale.set(w, len + 0.08, 0.2);
    placeCyl(p.neck, sh, tmp.lerpVectors(sh, jp[0], 0.6), 0.045);
    // racket in the playing hand, continuing the forearm
    dir.subVectors(jp[p.wrist], jp[p.elbow]).normalize();
    nrm.crossVectors(dir, Y_AXIS);
    if (nrm.lengthSq() < 1e-4) nrm.set(1, 0, 0);
    nrm.normalize();
    xax.crossVectors(nrm, dir);
    basis.makeBasis(xax, nrm, dir);
    p.racket.quaternion.setFromRotationMatrix(basis);
    p.racket.position.copy(jp[p.wrist]);
    if (p.label) p.label.position.copy(jp[0]).add(LABEL_LIFT);
  }

  function showFlight(fi: number) {
    if (fi === flight) return;
    flight = fi;
    path.geometry.dispose();
    const geo = new THREE.BufferGeometry();
    const fl = data.flights[fi] as Flight | undefined;
    if (fl) {
      const flat = flightPoints(d, fi, fl);
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i + 2 < flat.length; i += 3) pts.push(new THREE.Vector3(flat[i], flat[i + 1], flat[i + 2]));
      if (pts.length >= 2) geo.setFromPoints(new THREE.CatmullRomCurve3(pts).getPoints(pts.length * 3));
      pathMat.color.copy(fl.ok ? pal.ball : pal.warn);
    }
    path.geometry = geo;
    publish();
  }

  /** Draws time t; returns the ball height on show, or null. */
  function update(): number | null {
    const fr = drawnFrame(t, d.fps, d.n, videoActive());
    players.forEach((p, i) => updatePlayer(p, i, fr));

    const owner = ballAt(d, fr, ball);
    const found = data.flights[owner] as Flight | undefined;
    const fl = found && (found.ok || layers.low) ? found : undefined;
    const fi = fl ? owner : -1;
    const show = fl !== undefined;
    showFlight(fi);
    ballMesh.visible = halo.visible = show;
    if (fl) {
      ballMesh.position.copy(ball);
      halo.position.copy(ball);
      haloMat.color.copy(fl.ok ? haloOk : haloLow);
    }

    // trail: the last ~0.45 s of the same flight
    trail.visible = show && layers.trail;
    if (fl && layers.trail) {
      const P = trailPos.array as Float32Array;
      const C = trailCol.array as Float32Array;
      const c = fl.ok ? pal.ball : pal.warn;
      let k = 0;
      for (let i = 0; i < TRAIL; i++) {
        if (ballAt(d, fr - (i * TRAIL_S * d.fps) / TRAIL, prev) !== fi) break;
        const fade = 1 - i / TRAIL;
        P[k * 3] = prev.x;
        P[k * 3 + 1] = prev.y;
        P[k * 3 + 2] = prev.z;
        C[k * 3] = c.r * fade;
        C[k * 3 + 1] = c.g * fade;
        C[k * 3 + 2] = c.b * fade;
        k++;
      }
      trailGeo.setDrawRange(0, k);
      trailPos.needsUpdate = true;
      trailCol.needsUpdate = true;
    }

    guide.visible = spot.visible = show && layers.guide;
    if (show && layers.guide) {
      const a = guide.geometry.attributes.position as THREE.BufferAttribute;
      a.setXYZ(0, ball.x, ball.y, ball.z);
      a.setXYZ(1, ball.x, 0.005, ball.z);
      a.needsUpdate = true;
      guide.computeLineDistances();
      spot.position.set(ball.x, 0.006, ball.z);
    }
    path.visible = show && layers.path;

    // bounce rings for 0.8 s after each bounce
    const [lo, hi] = bounceRange(bounces, t, RING_S);
    let r = 0;
    for (let i = lo; i < hi && r < rings.length; i++) {
      const b = bounces[i];
      if (!b.ok && !layers.low) continue;
      const m = rings[r++];
      const age = (t - b.t) / RING_S;
      const mat = m.material as THREE.MeshBasicMaterial;
      m.visible = true;
      mat.opacity = 0.9 * (1 - age);
      mat.color.copy(b.kind === 'ground' ? (b.ok ? pal.ball : pal.warn) : WALL_RING);
      m.position.set(b.p[0], b.p[1], b.p[2]);
      m.scale.setScalar(0.12 + 0.55 * age);
      m.rotation.set(0, 0, 0);
      if (b.kind === 'ground') {
        m.rotation.x = -Math.PI / 2;
        m.position.y = 0.008;
      } else if (Math.abs(Math.abs(b.p[0]) - 5) < Math.abs(Math.abs(b.p[2]) - 10)) {
        m.rotation.y = Math.PI / 2; // side wall
      }
    }
    for (; r < rings.length; r++) rings[r].visible = false;

    // follow cam: drag the orbit along with the ball
    if (view === 'follow' && show && !tween) {
      delta.subVectors(ball, controls.target).multiplyScalar(FOLLOW_RATE);
      controls.target.add(delta);
      camera.position.add(delta);
    }
    return show ? ball.y : null;
  }

  // ---------- loop ----------
  let raf = 0;
  let last = performance.now();
  let sentT = NaN;
  let sentY: number | null = NaN;
  function frame(now: number) {
    const dt = (now - last) / 1000;
    last = now;
    t = tick({
      t, dt, dur: d.dur, speed, playing,
      videoActive: videoActive(),
      videoPaused: video?.paused ?? true,
      frameTime,
      currentTime: video?.currentTime ?? 0,
    });
    if (tween) {
      const step = tweenAt(tween, now, camera.position, controls.target);
      camera.fov = step.fov;
      camera.updateProjectionMatrix();
      if (step.done) tween = null;
    }
    const ballY = update();
    if (mode !== 'overlay') controls.update();
    renderer.render(scene, camera);
    if (t !== sentT || ballY !== sentY) {
      sentT = t;
      sentY = ballY;
      listeners.forEach((l) => l({ type: 'time', t, ballY }));
    }
    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);

  listen(renderer.domElement, 'webglcontextlost', () => {
    cancelAnimationFrame(raf);
    callbacks.onError('The 3D view lost its graphics context. Reload the page to restart it.');
  });

  function dispose() {
    if (disposed) return;
    disposed = true;
    cancelAnimationFrame(raf);
    resizeObserver.disconnect();
    cleanups.forEach((f) => f());
    listeners.clear();
    video?.pause();
    controls.removeEventListener('start', onControlsStart);
    controls.dispose();
    path.geometry.dispose();
    owned.forEach((x) => x.dispose());
    renderer.dispose();
    renderer.forceContextLoss();
    renderer.domElement.remove();
  }

  return {
    setView: (name) => {
      if (mode !== 'overlay') applyView(name);
    },
    setLayers: (next) => {
      layers = { ...layers, ...next };
      publish();
    },
    setSpeed: (s) => {
      speed = s;
      if (video) video.playbackRate = s;
      publish();
    },
    setPlaying: (p) => {
      playing = p;
      syncVideo();
      publish();
    },
    setMode,
    seek: (s) => {
      t = Math.min(d.dur, Math.max(0, s));
      seekVideo(t);
    },
    dispose,
    subscribe: (listener) => {
      listeners.add(listener);
      listener({ type: 'state', state: snapshot() });
      return () => {
        listeners.delete(listener);
      };
    },
  };
}
