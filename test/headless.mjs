// Headless check: the page's code (src/demo.js) with the same MuJoCo WASM and onnxruntime-web on its
// wasm (CPU) backend, under Node. Runs scripted velocity commands as tools/sim2sim_controls.py
// --velocity does (start standing; 'seconds:x,y,yaw;...', the pad's 0.1 s sends with 0.5 s expiry)
// and prints the base displacement at each segment's end in the start frame.
//
//   node test/headless.mjs                       # the reference suite
//   node test/headless.mjs '2:0,0,0;4:0.5,0,0;3:0,0,0' [--bundle models/walkglide2-2h]

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web';
import yaml from 'js-yaml';
import { WalkDemo } from '../src/demo.js';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const flag = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const bundleDir = path.resolve(ROOT, flag('--bundle', 'models/walkglide2-2h'));
const positional = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));

// The demo's walk limits (main.js); they cover every command below.
const WALK = { maxForward: 0.8, maxLateral: 0.6, maxYaw: 0.6 };
// tools/sim2sim_controls.py through the robot runtime (InstantTrackRuntime, controls plant with the
// parallel ankle), same bundle, EXP vmax 1.0 / vlat 0.8: robot_xy at the end of the command segment
// (t = 6 s) and at the end (8.98 s), robot_yaw at the end.
const SUITE = [
  ['forward 0.5', '2:0,0,0;4:0.5,0,0;3:0,0,0', { cmd: [1.849, 0], at6: [1.411, 0.085], end: [1.449, 0.031], yaw: 0.018 }],
  ['forward 0.8', '2:0,0,0;4:0.8,0,0;3:0,0,0', { cmd: [2.808, 0], at6: [1.997, 0.141], end: [2.197, 0.142], yaw: -0.047 }],
  ['left 0.6', '2:0,0,0;4:0,0.6,0;3:0,0,0', { cmd: [0, 2.181], at6: [-0.04, 0.879], end: [-0.01, 0.905], yaw: -0.056 }],
  ['forward 0.3', '2:0,0,0;4:0.3,0,0;3:0,0,0', { cmd: [1.147, 0], at6: [0.38, 0.011], end: [0.346, 0.01], yaw: -0.041 }],
  ['back 0.5', '2:0,0,0;4:-0.5,0,0;3:0,0,0', { cmd: [-1.849, 0], at6: [-0.101, 0.01], end: [-0.048, -0.002], yaw: -0.028 }],
  ['turn 0.6', '2:0,0,0;4:0,0,0.6;3:0,0,0', { cmd: [0, 0], at6: [-0.132, 0.003], end: [-0.276, -0.108], yaw: 2.063, rootYaw: 2.286 }],
  ['fwd 0.6 + turn 0.4', '2:0,0,0;4:0.6,0,0.4;3:0,0,0', { cmd: [1.35, 1.463], at6: [1.131, 0.964], end: [1.184, 1.028], yaw: 1.545, rootYaw: 1.551 }],
];

const mujoco = await loadMujoco();
ort.env.wasm.numThreads = 1;
const xml = fs.readFileSync(path.join(ROOT, 'assets/v1_physics.xml'), 'utf8');
const manifest = yaml.load(fs.readFileSync(path.join(bundleDir, 'instanttrack.yaml'), 'utf8'));
const onnx = new Uint8Array(fs.readFileSync(path.join(bundleDir, manifest.network.file)));
const demo = await WalkDemo.create({ ort, mujoco, xml, manifest, onnx, walk: WALK, sessionOptions: { executionProviders: ['wasm'] } });

async function run(script) {
  const segments = script.split(';').filter((s) => s.trim()).map((s) => {
    const [dur, v] = s.split(':');
    return [Number(dur), v.split(',').map(Number)];
  });
  const total = segments.reduce((a, [d]) => a + d, 0);
  const dt = demo.bundle.controlPeriod;
  const nSteps = Math.round(total / dt); // T - 1 policy steps, as the script
  demo.reset();
  const log = [];
  const ends = [];
  let acc = 0;
  for (const [d] of segments) ends.push((acc += d));
  const t0 = performance.now();
  for (let f = 0; f < nSteps; f++) {
    const t = f * dt;
    let a = 0, cmd = [0, 0, 0];
    for (const [d, v] of segments) {
      if (t < a + d - 1e-9) {
        cmd = v;
        break;
      }
      a += d;
    }
    demo.setCommand(cmd);
    const p = demo.sim.basePosition();
    log.push([t, p[0], p[1], demo.sim.baseYaw(), demo.runtime.root.position[0], demo.runtime.root.position[1], demo.runtime.root.yaw]);
    await demo.tick();
    if (demo.fell) break;
  }
  const wall = (performance.now() - t0) / 1000;
  // unwrap yaw, express in the start frame
  for (let i = 1; i < log.length; i++) {
    let dy = log[i][3] - log[i - 1][3];
    while (dy > Math.PI) dy -= 2 * Math.PI;
    while (dy < -Math.PI) dy += 2 * Math.PI;
    log[i][3] = log[i - 1][3] + dy;
  }
  const [, x0, y0, yaw0] = log[0];
  const c = Math.cos(yaw0), s = Math.sin(yaw0);
  const r3 = (v) => Math.round(v * 1000) / 1000;
  const walk = ends.map((end) => {
    let i = log.findIndex((row) => row[0] >= end - 1e-9);
    if (i < 0) i = log.length - 1;
    const dx = log[i][1] - x0, dy = log[i][2] - y0;
    return { t: r3(log[i][0]), robot_xy: [r3(c * dx + s * dy), r3(-s * dx + c * dy)], robot_yaw: r3(log[i][3] - yaw0),
      root_xy: [r3(log[i][4]), r3(log[i][5])], root_yaw: r3(log[i][6]) };
  });
  return { walk, fell: demo.fell ? r3(demo.sim.time) : null, rtf: r3(total / wall) };
}

if (positional.length) {
  console.log(JSON.stringify(await run(positional[0])));
} else {
  const rows = [];
  for (const [name, script, ref] of SUITE) {
    const r = await run(script);
    const [, at6, end] = r.walk;
    rows.push({ name, r, ref });
    console.log(`${name.padEnd(20)} commanded ${JSON.stringify(ref.cmd).padEnd(14)} | JS t=6 ${JSON.stringify(at6.robot_xy).padEnd(16)} end ${JSON.stringify(end.robot_xy).padEnd(16)} yaw ${String(end.robot_yaw).padEnd(6)} fell ${r.fell}` +
      ` | Python t=6 ${JSON.stringify(ref.at6).padEnd(16)} end ${JSON.stringify(ref.end).padEnd(16)} yaw ${ref.yaw} | ${r.rtf}x real time`);
  }
  fs.writeFileSync(path.join(ROOT, 'test/headless-results.json'), JSON.stringify(rows, null, 1));
}
