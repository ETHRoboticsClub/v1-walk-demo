// The page: loads MuJoCo, the policy bundle and the robot's meshes, runs the sim loop in real time
// and renders it with three.js. URL parameters:
//   ?bundle=<name>   a folder under models/ (default walkglide), or a URL of a bundle folder
//   ?ep=wasm|webgpu  force the onnxruntime execution provider (default: WebGPU when available)

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import * as ort from 'onnxruntime-web';
import loadMujoco from '@mujoco/mujoco';
import yaml from 'js-yaml';
import { WalkDemo } from './demo.js';
import * as quat from './quat.js';

// Demo speed limits: the walk policy does not step below ~0.4 m/s, so a held key commands the limit.
const WALK = { maxForward: 0.8, maxLateral: 0.6, maxYaw: 0.6 };
// A turn on the spot (Q/E alone) is a goal head that only rotates: the policy answers it by twisting
// the torso and hips (base_link, the goal's head body, is the upper torso) and barely steps (feet 1.4 of
// 2.9 rad over 5 s at 0.6 rad/s, torso 0.6 rad, hips -0.8 rad). With 0.35 m/s forward it steps round a
// ~0.6 m circle with the feet following and the torso straight (test/turn_trace.mjs), so Q/E alone walk that.
const TURN_FORWARD = 0.35;
// The goal's head point in base_link (ONNX metadata head_point_in_body), for the start offset.
const HEAD_IN_BODY = [0.05, 0, 0.9];

const $ = (id) => document.getElementById(id);
const status = (text, error = false) => {
  const el = $('status');
  el.textContent = text;
  el.classList.toggle('error', error);
  el.style.display = text ? '' : 'none';
};

const params = new URLSearchParams(location.search);
const bundleParam = params.get('bundle') || 'walkglide';
const bundleUrl = /^https?:/.test(bundleParam) ? bundleParam.replace(/\/?$/, '/') : new URL(`../models/${bundleParam}/`, import.meta.url).href;
const assetsUrl = new URL('../assets/', import.meta.url).href;

async function fetchBytes(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label}: HTTP ${res.status} for ${url}`);
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  // content-length is only a progress hint: with content-encoding (GitHub Pages gzips) it is the
  // compressed size, while the stream delivers the decoded bytes.
  const total = Number(res.headers.get('content-length')) || 0;
  const chunks = [];
  const reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) status(`Loading ${label}… ${Math.min(99, Math.round((100 * got) / total))}%`);
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

async function fetchText(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label}: HTTP ${res.status} for ${url}`);
  return res.text();
}

async function webgpuUsable() {
  if (!('gpu' in navigator)) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------------------------
// three.js scene

function buildScene() {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  $('view').appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a2129);
  scene.fog = new THREE.Fog(0x1a2129, 12, 30);
  const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.05, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(-2.6, -2.4, 1.6);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.8);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.minDistance = 0.8;
  controls.maxDistance = 12;
  controls.update();
  scene.add(new THREE.HemisphereLight(0xdfefff, 0x30363d, 1.4));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(3, -2, 6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 20 });
  scene.add(sun, sun.target);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x232c36, roughness: 0.95 }));
  floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.GridHelper(60, 120, 0x51606e, 0x34404c);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.001;
  scene.add(grid);
  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
  return { renderer, scene, camera, controls, sun };
}

async function buildRobot(scene, sim) {
  const [meta, bin] = await Promise.all([
    fetch(assetsUrl + 'v1_visual.json').then((r) => r.json()),
    fetchBytes(assetsUrl + 'v1_visual.bin', 'robot meshes'),
  ]);
  const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  const positions = new Float32Array(buf, 0, meta.positionsBytes / 4);
  const indices = new Uint32Array(buf, meta.positionsBytes, meta.indicesBytes / 4);
  const bodies = new Map();
  const { mj, model } = sim;
  for (const e of meta.meshes) {
    const id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, e.body);
    if (id < 0) continue;
    if (!bodies.has(id)) {
      const g = new THREE.Group();
      scene.add(g);
      bodies.set(id, g);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions.slice(3 * e.vertexOffset, 3 * (e.vertexOffset + e.vertexCount)), 3));
    geo.setIndex(new THREE.BufferAttribute(indices.slice(e.indexOffset, e.indexOffset + e.indexCount), 1));
    const shaded = toCreasedNormals(geo, Math.PI / 6);
    const [r, g, b, a] = e.rgba;
    const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(r, g, b), roughness: 0.55, metalness: 0.15, transparent: a < 1, opacity: a });
    const mesh = new THREE.Mesh(shaded, mat);
    mesh.castShadow = true;
    bodies.get(id).add(mesh);
  }
  return () => {
    const d = sim.data;
    for (const [id, g] of bodies) {
      g.position.set(d.xpos[3 * id], d.xpos[3 * id + 1], d.xpos[3 * id + 2]);
      g.quaternion.set(d.xquat[4 * id + 1], d.xquat[4 * id + 2], d.xquat[4 * id + 3], d.xquat[4 * id]);
    }
  };
}

// ---------------------------------------------------------------------------------------------

const KEYS = { KeyW: [1, 0, 0], KeyS: [-1, 0, 0], KeyA: [0, 1, 0], KeyD: [0, -1, 0], KeyQ: [0, 0, 1], KeyE: [0, 0, -1] };

async function main() {
  const view = buildScene();
  status('Loading MuJoCo…');
  const mujocoP = loadMujoco();
  const xmlP = fetchText(assetsUrl + 'v1_physics.xml', 'robot model');
  const manifest = yaml.load(await fetchText(bundleUrl + 'instanttrack.yaml', 'bundle manifest'));
  const onnx = await fetchBytes(bundleUrl + manifest.network.file, 'policy');
  const [mujoco, xml] = await Promise.all([mujocoP, xmlP]);

  ort.env.wasm.numThreads = 1; // GitHub Pages is not cross-origin isolated
  const want = params.get('ep');
  const tryWebgpu = want === 'webgpu' || (want !== 'wasm' && (await webgpuUsable()));
  let demo, backend;
  if (tryWebgpu) {
    status('Starting the policy on WebGPU…');
    try {
      demo = await WalkDemo.create({ ort, mujoco, xml, manifest, onnx, walk: WALK, sessionOptions: { executionProviders: ['webgpu'] } });
      await demo.tick(); // a first run compiles the kernels; fail over here rather than mid-walk
      demo.reset();
      backend = 'WebGPU';
    } catch (e) {
      console.warn('WebGPU failed, falling back to WASM', e);
      demo = null;
    }
  }
  if (!demo) {
    status('Starting the policy on WASM…');
    demo = await WalkDemo.create({ ort, mujoco, xml, manifest, onnx, walk: WALK, sessionOptions: { executionProviders: ['wasm'] } });
    backend = tryWebgpu ? 'WASM (WebGPU failed)' : 'WASM';
  }
  $('ep').textContent = backend;
  status('Loading robot meshes…');
  const syncRobot = await buildRobot(view.scene, demo.sim);

  // Faint markers: the goal's head point (sphere) and the virtual root on the floor (ring).
  const goalHead = new THREE.Mesh(new THREE.SphereGeometry(0.06, 20, 14), new THREE.MeshBasicMaterial({ color: 0x5cc8ff, transparent: true, opacity: 0.35, depthWrite: false }));
  const rootRing = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.2, 40), new THREE.MeshBasicMaterial({ color: 0x5cc8ff, transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false }));
  const rootArrow = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.14, 12), rootRing.material);
  rootArrow.rotation.z = -Math.PI / 2;
  rootArrow.position.x = 0.27;
  rootRing.add(rootArrow);
  rootRing.position.z = 0.003;
  view.scene.add(goalHead, rootRing);
  // The operation's frame in the world: yaw zero is the robot's heading at the reset, origin under
  // its head point then.
  let frame = { yaw: 0, origin: [0, 0] };
  const anchorFrame = () => {
    const s = demo.sim.state();
    const yaw = quat.heading(s.baseQuat);
    const p = demo.sim.basePosition();
    const h = quat.rotate(s.baseQuat, HEAD_IN_BODY);
    frame = { yaw, origin: [p[0] + h[0], p[1] + h[1]] };
  };
  const toWorld = (v) => {
    const c = Math.cos(frame.yaw), s = Math.sin(frame.yaw);
    return [frame.origin[0] + c * v[0] - s * v[1], frame.origin[1] + s * v[0] + c * v[1], v[2]];
  };
  anchorFrame();

  // ---- camera mode: free (follows position) or chase (also turns with the robot) ----
  const chase = {
    on: params.get('cam') === 'chase' || (params.get('cam') !== 'free' && (() => {
      try {
        return localStorage.getItem('v1walk.cam') === 'chase';
      } catch {
        return false;
      }
    })()),
    toggle() {
      this.on = !this.on;
      try {
        localStorage.setItem('v1walk.cam', this.on ? 'chase' : 'free');
      } catch {}
      this.show();
    },
    show() {
      const b = $('cam');
      if (b) {
        b.textContent = this.on ? 'chase' : 'free';
        b.classList.toggle('on', this.on);
      }
    },
  };
  chase.show();
  $('cam')?.addEventListener('click', () => chase.toggle());

  // ---- input: hold to drive ----
  const held = new Set();
  const updateCommand = () => {
    const v = [0, 0, 0];
    for (const k of held) KEYS[k].forEach((x, i) => (v[i] += x));
    const cmd = [v[0] * WALK.maxForward, v[1] * WALK.maxLateral, v[2] * WALK.maxYaw];
    if (cmd[2] !== 0 && cmd[0] === 0 && cmd[1] === 0) cmd[0] = TURN_FORWARD;
    demo.setCommand(cmd.map((x) => x + 0)); // +0: no -0
    document.querySelectorAll('kbd[data-k]').forEach((el) => el.classList.toggle('on', held.has(el.dataset.k)));
  };
  let resetRequested = false;
  addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (KEYS[e.code]) {
      held.add(e.code);
      updateCommand();
      e.preventDefault();
    } else if (e.code === 'Space') {
      held.clear();
      updateCommand();
      e.preventDefault();
    } else if (e.code === 'KeyR') resetRequested = true;
    else if (e.code === 'KeyC') chase.toggle();
  });
  addEventListener('keyup', (e) => {
    if (held.delete(e.code)) updateCommand();
  });
  addEventListener('blur', () => {
    held.clear();
    updateCommand();
  });

  // ---- the sim loop: real time, or slower when the browser cannot keep up (never skips steps) ----
  const dt = demo.bundle.controlPeriod;
  let anchorWall = performance.now(), anchorTicks = 0;
  const reanchor = () => {
    anchorWall = performance.now();
    anchorTicks = demo.tickCount;
  };
  let rtfWindow = { wall: performance.now(), ticks: 0 }, rtf = 1;
  let policyMs = 0;
  const actual = { f: 0, l: 0, y: 0 };
  let error = null;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  (async () => {
    try {
      for (;;) {
        if (resetRequested) {
          resetRequested = false;
          demo.reset();
          anchorFrame();
          reanchor();
          rtfWindow = { wall: performance.now(), ticks: 0 };
        }
        const now = performance.now();
        const ahead = (demo.tickCount - anchorTicks) * dt * 1000 - (now - anchorWall);
        if (ahead > 0) {
          await sleep(Math.max(1, ahead));
          continue;
        }
        if (ahead < -100) {
          // behind by more than 100 ms: let sim time fall behind wall time instead of catching up
          anchorWall = now - (demo.tickCount - anchorTicks) * dt * 1000;
        }
        const t0 = performance.now();
        await demo.tick();
        policyMs = 0.9 * policyMs + 0.1 * (performance.now() - t0);
        rtfWindow.ticks += 1;
        const w = performance.now() - rtfWindow.wall;
        if (w > 1000) {
          rtf = (rtfWindow.ticks * dt * 1000) / w;
          rtfWindow = { wall: performance.now(), ticks: 0 };
        }
        // actual speeds in the robot's heading frame, smoothed over ~0.5 s
        const v = demo.sim.baseVelocity();
        const yaw = demo.sim.baseYaw();
        const c = Math.cos(yaw), s = Math.sin(yaw);
        const st = demo.sim.state();
        const wz = quat.rotate(st.baseQuat, st.angVel)[2];
        const k = 0.04;
        actual.f += k * (c * v[0] + s * v[1] - actual.f);
        actual.l += k * (-s * v[0] + c * v[1] - actual.l);
        actual.y += k * (wz - actual.y);
        if (performance.now() - t0 > 8) await sleep(0); // let the page render
      }
    } catch (e) {
      error = e;
      console.error(e);
      status(`The sim stopped: ${e.message}`, true);
    }
  })();

  status('');
  const fmt = (x) => (Math.abs(x) < 0.005 ? '0.00' : x.toFixed(2));
  let last = { x: 0, y: 0 };
  {
    const p = demo.sim.basePosition();
    last = { x: p[0], y: p[1] };
  }
  // Chase camera (C, or ?cam=chase): the view also turns with the robot's heading, low-passed over
  // CHASE_TAU_S so the torso's sway and the turn-in-place twist don't shake it; orbiting still works
  // and is kept relative to the robot.
  const CHASE_TAU_S = 0.4;
  let camYaw = demo.sim.baseYaw();
  let lastFrame = performance.now();
  const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  view.renderer.setAnimationLoop(() => {
    syncRobot();
    const p = demo.sim.basePosition();
    // follow the robot: move the orbit target and camera with it, keeping the user's view
    const dx = p[0] - last.x, dy = p[1] - last.y;
    last = { x: p[0], y: p[1] };
    view.controls.target.x += dx;
    view.controls.target.y += dy;
    view.camera.position.x += dx;
    view.camera.position.y += dy;
    const now = performance.now();
    const frameS = Math.min(0.1, (now - lastFrame) / 1000);
    lastFrame = now;
    const dyaw = wrap(demo.sim.baseYaw() - camYaw) * (1 - Math.exp(-frameS / CHASE_TAU_S));
    camYaw = wrap(camYaw + dyaw);
    if (chase.on && dyaw) {
      // turn the camera about the vertical through the orbit target
      const t = view.controls.target, c = Math.cos(dyaw), s = Math.sin(dyaw);
      const ox = view.camera.position.x - t.x, oy = view.camera.position.y - t.y;
      view.camera.position.x = t.x + c * ox - s * oy;
      view.camera.position.y = t.y + s * ox + c * oy;
    }
    view.controls.target.z += 0.1 * (Math.max(p[2] + 0.15, 0.3) - view.controls.target.z);
    view.controls.update();
    view.sun.position.set(p[0] + 3, p[1] - 2, 6);
    view.sun.target.position.set(p[0], p[1], 0);
    const rt = demo.runtime;
    const g = rt.lastGoal;
    if (g) {
      const h = toWorld(g.frames[0].head);
      goalHead.position.set(h[0], h[1], h[2]);
    }
    const r = toWorld([rt.root.position[0], rt.root.position[1], 0]);
    rootRing.position.x = r[0];
    rootRing.position.y = r[1];
    rootRing.rotation.z = frame.yaw + rt.root.yaw;
    $('cf').textContent = fmt(rt.root.forward);
    $('cl').textContent = fmt(rt.root.left);
    $('cy').textContent = fmt(rt.root.yawRate);
    $('af').textContent = fmt(actual.f);
    $('al').textContent = fmt(actual.l);
    $('ay').textContent = fmt(actual.y);
    $('rtf').textContent = `${rtf.toFixed(2)}×`;
    $('rtf').style.color = rtf < 0.9 ? 'var(--warn)' : '';
    $('pms').textContent = `${policyMs.toFixed(1)} ms / 20 ms`;
    $('gs').textContent = g ? (g.state === 'live' && g.ramp < 1 ? `live (ramp ${Math.round(100 * g.ramp)}%)` : g.state) : '–';
    const banner = $('banner');
    banner.style.display = demo.fell && !error ? 'block' : 'none';
    banner.textContent = 'The robot fell. Press R to reset.';
    view.renderer.render(view.scene, view.camera);
  });
}

main().catch((e) => {
  console.error(e);
  status(`Failed to start: ${e.message}`, true);
});
