// The InstantTrack teleop runtime of the robot (ETHRC-Humanoid-Controls,
// ethrc_policy_core/instanttrack/{bundle,goal,walk,runtime}.py), ported to JS for the browser.
// Only the velocity path is used here: a planar velocity walks a virtual root that carries the
// bundle's standing goal; GoalStream keeps the last K goal frames, newest first.
// Times are integer nanoseconds (Numbers), as the Python runtime keeps them.

import * as quat from './quat.js';

export const NS = 1_000_000_000;
// goal.py
export const STALE_S = 0.25;
export const RAMP_S = 1.5;
export const MAX_HEAD_STEP_M = 0.15;
// walk.py
export const ACCEL_M_S2 = 0.8;
export const YAW_ACCEL_RAD_S2 = 1.5;
export const MAX_STEP_S = 0.1;
export const DEFAULT_WALK_LIMITS = { maxForward: 0.5, maxLateral: 0.3, maxYaw: 0.6 };
// bundle.py
export const NETWORK_INPUTS = ['obs_history', 'head_pos', 'head_quat', 'hand_pos', 'hand_valid'];
export const NETWORK_OUTPUT = 'actions';
export const observationDim = (joints) => 3 * joints + 6 + 6;

// ---------------------------------------------------------------------------------------------
// Bundle: instanttrack.yaml (parsed) + the ONNX's metadata (training gains)

/** The custom metadata_props of an ONNX ModelProto (field 14), read straight off the protobuf. */
export function onnxMetadata(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let p = 0;
  const varint = (end) => {
    let v = 0, mul = 1;
    for (;;) {
      if (p >= end) throw new Error('truncated varint');
      const b = u8[p++];
      v += (b & 0x7f) * mul;
      if (b < 0x80) return v;
      mul *= 128;
    }
  };
  const dec = new TextDecoder();
  const out = {};
  const end = u8.length;
  while (p < end) {
    const key = varint(end);
    const field = Math.floor(key / 8), wire = key & 7;
    if (wire === 0) varint(end);
    else if (wire === 1) p += 8;
    else if (wire === 5) p += 4;
    else if (wire === 2) {
      const len = varint(end);
      const start = p;
      p += len;
      if (field === 14) {
        let q = start, k = '', v = '';
        while (q < start + len) {
          const save = p;
          p = q;
          const kk = varint(start + len);
          const l = varint(start + len);
          const s = dec.decode(u8.subarray(p, p + l));
          if (kk >> 3 === 1) k = s;
          else if (kk >> 3 === 2) v = s;
          q = p + l;
          p = save;
        }
        out[k] = v;
      }
    } else throw new Error(`unsupported protobuf wire type ${wire}`);
  }
  return out;
}

export function loadBundle(manifest, metadata = {}, walk = null) {
  if (manifest.schema !== 'instanttrack/v1') throw new Error(`schema must be instanttrack/v1, got ${manifest.schema}`);
  const joints = manifest.joints.map((j) => ({
    name: j.name,
    default: j.default_position_rad,
    scale: j.action_scale_rad,
    kp: j.stiffness_nm_per_rad,
    kd: j.damping_nm_s_per_rad,
  }));
  // The PD the policy was trained with is in the ONNX metadata; instanttrack.yaml carries the
  // robot's deploy gains (and 0 for the linkage ankles, which ankle_motor_pd drives instead).
  const list = (k) => (metadata[k] ? metadata[k].split(',').map(Number) : null);
  const names = metadata.joint_names ? metadata.joint_names.split(',') : null;
  if (names && names.join() !== joints.map((j) => j.name).join()) throw new Error('ONNX joint order differs from instanttrack.yaml');
  const kp = list('joint_stiffness'), kd = list('joint_damping');
  const pd = manifest.ankle_motor_pd;
  joints.forEach((j, i) => {
    if (kp && kd) {
      j.kp = kp[i];
      j.kd = kd[i];
    } else if (j.kp === 0 && pd) {
      j.kp = pd.kp;
      j.kd = pd.kd;
    }
  });
  const g = manifest.goal;
  const st = g.standing;
  const w = { ...DEFAULT_WALK_LIMITS };
  if (manifest.walk) {
    if (manifest.walk.max_forward_m_s != null) w.maxForward = manifest.walk.max_forward_m_s;
    if (manifest.walk.max_lateral_m_s != null) w.maxLateral = manifest.walk.max_lateral_m_s;
    if (manifest.walk.max_yaw_rad_s != null) w.maxYaw = manifest.walk.max_yaw_rad_s;
  }
  return {
    source: manifest.source,
    controlPeriod: manifest.control_period_s,
    joints,
    observationHistory: manifest.observation_history,
    goalFrames: g.frames,
    headOrientation: g.head_orientation ?? 'full',
    standing: {
      head: st.head_position_m.slice(),
      left: st.left_hand_position_m.slice(),
      right: st.right_hand_position_m.slice(),
    },
    walk: walk ?? w,
    networkFile: manifest.network.file,
  };
}

// ---------------------------------------------------------------------------------------------
// goal.py

const points = (head, q, left, right, lt = true, rt = true) => ({ head, q, left, right, lt, rt });
export const standingPoints = (s) => points(s.head.slice(), quat.IDENTITY.slice(), s.left.slice(), s.right.slice());

const rz = (yaw, p) => {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]];
};
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

function applyAnchor(anchor, p) {
  const mv = (x) => add(rz(anchor.yaw, x), anchor.t);
  return { ...p, head: mv(p.head), q: quat.mul(quat.yaw(anchor.yaw), p.q), left: mv(p.left), right: mv(p.right) };
}

function anchorOnto(sample, target) {
  const yaw = quat.heading(target.q) - quat.heading(sample.q);
  const turned = rz(yaw, sample.head);
  return { yaw, t: [target.head[0] - turned[0], target.head[1] - turned[1], 0] };
}

function blend(a, b, t) {
  if (t >= 1) return b;
  const lerp = (x, y) => x.map((v, i) => (1 - t) * v + t * y[i]);
  return points(
    lerp(a.head, b.head),
    quat.slerp(a.q, b.q, t),
    lerp(a.left, b.left),
    lerp(a.right, b.right),
    t > 0 ? b.lt : a.lt,
    t > 0 ? b.rt : a.rt,
  );
}

const smoothstep = (x) => {
  x = Math.min(Math.max(x, 0), 1);
  return x * x * (3 - 2 * x);
};

export class GoalStream {
  constructor(standing, frames) {
    this.standing = standingPoints(standing);
    this.frames = frames;
    this.reset();
  }

  reset() {
    this.history = Array.from({ length: this.frames }, () => this.standing); // newest first
    this.state = 'standing';
    this.frozen = this.standing;
    this.sample = null;
    this.source = null;
    this.switched = false;
    this.liveUntil = 0;
    this.anchor = null;
    this.rampStart = 0;
    this.lastLiveHead = null;
    this.entries = 0;
    this.jumps = 0;
  }

  take(p, arrivedNs, expiresNs, source = 'operator') {
    if (this.source !== null && source !== this.source) this.switched = true;
    this.source = source;
    this.sample = p;
    this.liveUntil = Math.min(arrivedNs + Math.round(STALE_S * NS), expiresNs);
  }

  ramp(nowNs) {
    return smoothstep((nowNs - this.rampStart) / (RAMP_S * NS));
  }

  tick(nowNs) {
    const live = this.sample !== null && nowNs < this.liveUntil;
    const current = this.history[0];
    let frame;
    if (live) {
      if (this.state !== 'live' || this.switched) this.enter(this.sample, current, nowNs);
      let anchored = applyAnchor(this.anchor, this.sample);
      if (this.lastLiveHead !== null) {
        const d = anchored.head.map((v, i) => v - this.lastLiveHead[i]);
        if (Math.hypot(...d) > MAX_HEAD_STEP_M) {
          this.jumps += 1;
          this.enter(this.sample, current, nowNs);
          anchored = applyAnchor(this.anchor, this.sample);
        }
      }
      this.lastLiveHead = anchored.head;
      frame = blend(this.frozen, anchored, this.ramp(nowNs));
    } else {
      if (this.state === 'live') {
        this.state = 'frozen';
        this.frozen = { ...current, lt: true, rt: true };
      }
      frame = this.frozen;
    }
    this.history.unshift(frame);
    this.history.length = this.frames;
    return {
      frames: this.history,
      handValid: [frame.lt ? 1 : 0, frame.rt ? 1 : 0],
      state: this.state,
      ramp: this.state === 'live' ? this.ramp(nowNs) : 0,
    };
  }

  enter(sample, current, nowNs) {
    this.frozen = current;
    this.switched = false;
    this.anchor = anchorOnto(sample, current);
    this.rampStart = nowNs;
    this.lastLiveHead = null;
    this.state = 'live';
    this.entries += 1;
  }
}

// ---------------------------------------------------------------------------------------------
// walk.py

export function approach(value, target, rate, dt) {
  if (Math.abs(target) <= Math.abs(value) && target * value >= 0) return target;
  const step = rate * dt;
  return value + Math.max(-step, Math.min(step, target - value));
}

export class VirtualRoot {
  constructor(limits = DEFAULT_WALK_LIMITS) {
    this.limits = limits;
    this.reset();
  }

  reset() {
    this.position = [0, 0, 0];
    this.yaw = 0;
    this.forward = this.left = this.yawRate = 0;
  }

  stop() {
    this.forward = this.left = this.yawRate = 0;
  }

  step(forward, left, yawRate, dt) {
    dt = Math.min(Math.max(dt, 0), MAX_STEP_S);
    const L = this.limits;
    const clip = (v, m) => Math.max(-m, Math.min(m, v));
    this.forward = approach(this.forward, clip(forward, L.maxForward), ACCEL_M_S2, dt);
    this.left = approach(this.left, clip(left, L.maxLateral), ACCEL_M_S2, dt);
    this.yawRate = approach(this.yawRate, clip(yawRate, L.maxYaw), YAW_ACCEL_RAD_S2, dt);
    this.yaw += this.yawRate * dt;
    const d = rz(this.yaw, [this.forward, this.left, 0]);
    this.position = this.position.map((v, i) => v + d[i] * dt);
  }

  carry(p) {
    const mv = (x) => add(rz(this.yaw, x), this.position);
    return { ...p, head: mv(p.head), q: quat.mul(quat.yaw(this.yaw), p.q), left: mv(p.left), right: mv(p.right) };
  }
}

// ---------------------------------------------------------------------------------------------
// runtime.py (velocity input only)

export class InstantTrackRuntime {
  /**
   * @param bundle   loadBundle(...)
   * @param run      async (feeds: {name: {data: Float32Array, dims: number[]}}) => Float32Array actions
   * @param bounds   per-joint [lower, upper] target bounds (Infinity for none)
   */
  constructor(bundle, run, bounds) {
    this.bundle = bundle;
    this.run = run;
    this.J = bundle.joints.length;
    this.defaults = bundle.joints.map((j) => j.default);
    this.scales = bundle.joints.map((j) => j.scale);
    this.bounds = bounds ?? bundle.joints.map(() => [-Infinity, Infinity]);
    this.goal = new GoalStream(bundle.standing, bundle.goalFrames);
    this.standing = standingPoints(bundle.standing);
    this.root = new VirtualRoot(bundle.walk);
    this.lastGoal = null;
    this.resetState();
  }

  resetState() {
    this.history = [];
    this.lastAction = new Float64Array(this.J);
    this.velocity = null;
    this.lastStepNs = null;
    this.lastSequence = 0;
    this.headingRad = 0;
    this.goal.reset();
    this.root.reset();
  }

  /** A fresh run: the goal stands; the operation's yaw zero is the robot's heading now. */
  reset(state) {
    this.resetState();
    this.headingRad = quat.heading(state.baseQuat);
  }

  /** {sequence, expiresNs, velocity: [forward, left, yawRate]}; the newest wins. */
  setReference(ref) {
    if (ref.sequence <= this.lastSequence) return;
    this.lastSequence = ref.sequence;
    this.velocity = ref;
  }

  /** state: {q, qd (policy order), baseQuat (wxyz), angVel (base frame)} -> joint targets. */
  async step(state, nowNs) {
    this.advanceGoalInput(nowNs);
    const goal = (this.lastGoal = this.goal.tick(nowNs));
    const J = this.J, K = this.bundle.goalFrames, H = this.bundle.observationHistory;
    let head = goal.frames[0].q;
    if (this.bundle.headOrientation === 'yaw') head = quat.headingQuat(head);
    const headWorld = quat.mul(quat.yaw(this.headingRad), head);
    const R = quat.matrix(quat.mul(quat.conjugate(state.baseQuat), headWorld));
    const g = quat.gravity(state.baseQuat);
    const obs = new Float64Array(observationDim(J));
    for (let i = 0; i < J; i++) {
      obs[i] = state.q[i] - this.defaults[i];
      obs[J + i] = state.qd[i];
      obs[2 * J + 6 + i] = this.lastAction[i];
    }
    obs.set(state.angVel, 2 * J);
    obs.set(g, 2 * J + 3);
    obs.set([R[0], R[1], R[3], R[4], R[6], R[7]], 3 * J + 6); // R[:, :2] row-major
    if (this.history.length === 0) for (let h = 0; h < H; h++) this.history.push(obs);
    else {
      this.history.push(obs);
      this.history.shift();
    }
    const D = obs.length;
    const hist = new Float32Array(H * D);
    this.history.forEach((o, h) => hist.set(o, h * D)); // oldest first
    const hp = new Float32Array(K * 3), hq = new Float32Array(K * 4), hand = new Float32Array(K * 6);
    goal.frames.forEach((f, k) => {
      hp.set(f.head, 3 * k);
      hq.set(f.q, 4 * k);
      hand.set(f.left, 6 * k);
      hand.set(f.right, 6 * k + 3);
    });
    const action = await this.run({
      obs_history: { data: hist, dims: [1, H, D] },
      head_pos: { data: hp, dims: [1, K, 3] },
      head_quat: { data: hq, dims: [1, K, 4] },
      hand_pos: { data: hand, dims: [1, K, 2, 3] },
      hand_valid: { data: Float32Array.from(goal.handValid), dims: [1, 2] },
    });
    if (action.length !== J || !action.every(Number.isFinite)) throw new Error('the network returned a malformed action');
    const targets = new Float64Array(J);
    for (let i = 0; i < J; i++) {
      this.lastAction[i] = action[i];
      const t = this.defaults[i] + action[i] * this.scales[i];
      targets[i] = Math.min(Math.max(t, this.bounds[i][0]), this.bounds[i][1]);
    }
    return targets;
  }

  advanceGoalInput(nowNs) {
    const dt = this.lastStepNs === null ? 0 : (nowNs - this.lastStepNs) / NS;
    this.lastStepNs = nowNs;
    const v = this.velocity;
    if (v === null || nowNs >= v.expiresNs) {
      this.velocity = null;
      this.root.stop();
      return;
    }
    this.root.step(v.velocity[0], v.velocity[1], v.velocity[2], dt);
    this.goal.take(this.root.carry(this.standing), nowNs, v.expiresNs, 'walk');
  }
}
