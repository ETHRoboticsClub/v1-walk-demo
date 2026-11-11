// One walk demo: the bundle's policy (onnxruntime) through the InstantTrack runtime, driving the V1
// in MuJoCo. Shared by the page (main.js) and the headless check (test/headless.mjs).

import { InstantTrackRuntime, loadBundle, NS, onnxMetadata } from './instanttrack.js';
import { V1Sim } from './sim.js';

// The dashboard's hold-to-drive pad: a nonzero velocity every 0.1 s with a 0.5 s expiry while
// held, one zero on release, nothing while released.
const SEND_PERIOD_S = 0.1;
const EXPIRY_S = 0.5;

export class WalkDemo {
  /**
   * @param o.ort       onnxruntime (web or node) module
   * @param o.mujoco    loaded @mujoco/mujoco module
   * @param o.xml       assets/v1_physics.xml text
   * @param o.manifest  parsed instanttrack.yaml
   * @param o.onnx      policy.onnx bytes (Uint8Array)
   * @param o.walk      walk limits {maxForward, maxLateral, maxYaw} (default: the bundle's)
   * @param o.sessionOptions  onnxruntime InferenceSession options (execution providers)
   */
  static async create(o) {
    const metadata = onnxMetadata(o.onnx);
    const bundle = loadBundle(o.manifest, metadata, o.walk ?? null);
    const session = await o.ort.InferenceSession.create(o.onnx, o.sessionOptions ?? {});
    const missing = ['obs_history', 'head_pos', 'head_quat', 'hand_pos', 'hand_valid'].filter((n) => !session.inputNames.includes(n));
    if (missing.length || !session.outputNames.includes('actions')) {
      throw new Error(`the network takes ${session.inputNames} -> ${session.outputNames}, not an instanttrack-live policy`);
    }
    return new WalkDemo(o, bundle, session, metadata);
  }

  constructor(o, bundle, session, metadata) {
    this.ort = o.ort;
    this.bundle = bundle;
    this.session = session;
    this.metadata = metadata;
    this.sim = new V1Sim(o.mujoco, o.xml, bundle.joints);
    this.decimation = Math.round(bundle.controlPeriod / this.sim.dt);
    if (Math.abs(this.decimation * this.sim.dt - bundle.controlPeriod) > 1e-9) throw new Error('the sim step must divide the control period');
    const run = async (feeds) => {
      const t = {};
      for (const [k, v] of Object.entries(feeds)) t[k] = new this.ort.Tensor('float32', v.data, v.dims);
      const out = await this.session.run(t, ['actions']);
      return out.actions.data;
    };
    this.runtime = new InstantTrackRuntime(bundle, run, this.sim.referenceBounds(bundle.joints));
    this.command = [0, 0, 0];
    this.reset();
  }

  reset() {
    this.sim.reset(this.runtime.defaults);
    this.tickCount = 0;
    this.sequence = 0;
    this.sent = [0, 0, 0];
    this.nextSendNs = 0;
    this.runtime.reset(this.sim.state());
    this.lastTargets = this.runtime.defaults.slice();
    this.fell = false;
  }

  /** The operator's velocity (forward m/s, left m/s, yaw rad/s), as long as it is held. */
  setCommand(v) {
    this.command = v.slice();
  }

  get nowNs() {
    return this.tickCount * Math.round(this.bundle.controlPeriod * NS);
  }

  deliver(nowNs) {
    const v = this.command;
    const zero = v.every((x) => x === 0);
    const wasZero = this.sent.every((x) => x === 0);
    const changed = v.some((x, i) => x !== this.sent[i]);
    let send = false;
    if (!zero) send = changed || nowNs >= this.nextSendNs;
    else send = !wasZero; // one zero on release
    if (!send) return;
    this.sequence += 1;
    this.runtime.setReference({ sequence: this.sequence, expiresNs: nowNs + Math.round(EXPIRY_S * NS), velocity: v.slice() });
    this.sent = v.slice();
    this.nextSendNs = nowNs + Math.round(SEND_PERIOD_S * NS);
  }

  /** One control period: the policy step, then the sim steps at its targets. */
  async tick() {
    const now = this.nowNs;
    this.deliver(now);
    const targets = await this.runtime.step(this.sim.state(), now);
    this.lastTargets = targets;
    this.sim.setTargets(targets);
    this.sim.step(this.decimation);
    this.tickCount += 1;
    if (this.sim.basePosition()[2] < 0.35) this.fell = true;
  }
}
