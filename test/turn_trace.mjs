// Turn diagnosis: base (pelvis) yaw, each sole's yaw, the root's yaw, and torso joint angles over a scripted command.
//   node test/turn_trace.mjs '2:0,0,0;4:0,0,0.6;3:0,0,0'
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web';
import yaml from 'js-yaml';
import { WalkDemo } from '../src/demo.js';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const bundleDir = path.resolve(ROOT, process.argv[3] || 'models/walkglide');
const mujoco = await loadMujoco();
ort.env.wasm.numThreads = 1;
const xml = fs.readFileSync(path.join(ROOT, 'assets/v1_physics.xml'), 'utf8');
const manifest = yaml.load(fs.readFileSync(path.join(bundleDir, 'instanttrack.yaml'), 'utf8'));
const onnx = new Uint8Array(fs.readFileSync(path.join(bundleDir, manifest.network.file)));
const demo = await WalkDemo.create({ ort, mujoco, xml, manifest, onnx, walk: { maxForward: 0.8, maxLateral: 0.6, maxYaw: 0.6 },
  sessionOptions: { executionProviders: ['wasm'] } });
const segs = process.argv[2].split(';').map((s) => { const [d, v] = s.split(':'); return [Number(d), v.split(',').map(Number)]; });
const total = segs.reduce((a, [d]) => a + d, 0);
const dt = demo.bundle.controlPeriod;
const yawOf = (R) => Math.atan2(R[3], R[0]);
const { model: m, data: d } = demo.sim;
const torso = demo.sim.jointNames.map((n, i) => [n, i]).filter(([n]) => /torso_axial|hip_internal/.test(n));
demo.reset();
console.log('t     cmd_yaw  root_yaw base_yaw soleL   soleR   ' + torso.map(([n]) => n.padEnd(10)).join(' '));
for (let f = 0; f < Math.round(total / dt); f++) {
  const t = f * dt;
  let a = 0, cmd = [0, 0, 0];
  for (const [du, v] of segs) { if (t < a + du - 1e-9) { cmd = v; break; } a += du; }
  demo.setCommand(cmd);
  await demo.tick();
  if (f % 25 === 0) {
    const q = demo.sim.state().q;
    const sl = yawOf(d.geom_xmat.subarray(9 * demo.sim.soles[0], 9 * demo.sim.soles[0] + 9));
    const sr = yawOf(d.geom_xmat.subarray(9 * demo.sim.soles[1], 9 * demo.sim.soles[1] + 9));
    const f2 = (x) => x.toFixed(2).padStart(7);
    console.log(t.toFixed(1).padEnd(5), f2(cmd[2]), f2(demo.runtime.root.yaw), f2(demo.sim.baseYaw()), f2(sl), f2(sr), torso.map(([, i]) => f2(q[i]).padEnd(10)).join(' '));
  }
  if (demo.fell) { console.log('fell', t); break; }
}
