// The V1 in MuJoCo (WASM): the training model (assets/v1_physics.xml) with one <position> actuator
// per joint, the PD the policy was trained with, stepped at the model's 5 ms.

export class V1Sim {
  /**
   * @param mujoco  the loaded @mujoco/mujoco module
   * @param xml     assets/v1_physics.xml
   * @param joints  bundle joints in policy order ({name, kp, kd})
   */
  constructor(mujoco, xml, joints) {
    this.mj = mujoco;
    this.model = mujoco.MjModel.from_xml_string(xml);
    this.data = new mujoco.MjData(this.model);
    const m = this.model;
    const id = (type, name) => {
      const i = mujoco.mj_name2id(m, mujoco.mjtObj[type].value, name);
      if (i < 0) throw new Error(`the model has no ${type} ${name}`);
      return i;
    };
    this.jointNames = joints.map((j) => j.name);
    const jid = joints.map((j) => id('mjOBJ_JOINT', j.name));
    this.qadr = jid.map((j) => m.jnt_qposadr[j]);
    this.vadr = jid.map((j) => m.jnt_dofadr[j]);
    this.act = joints.map((j) => id('mjOBJ_ACTUATOR', j.name));
    this.base = id('mjOBJ_BODY', 'base_link');
    this.range = jid.map((j) => [m.jnt_range[2 * j], m.jnt_range[2 * j + 1]]);
    this.effort = this.act.map((a) => m.actuator_forcerange[2 * a + 1]);
    // The policy's training PD on the position actuators (gainprm / biasprm rows of 10).
    joints.forEach((j, i) => {
      const a = this.act[i];
      m.actuator_gainprm[10 * a] = j.kp;
      m.actuator_biasprm[10 * a + 1] = -j.kp;
      m.actuator_biasprm[10 * a + 2] = -j.kd;
    });
    this.soles = ['sole_box_l', 'sole_box_r'].map((n) => id('mjOBJ_GEOM', n));
    this.dt = m.opt.timestep;
  }

  /** Standing in the default pose at the origin, facing +x, the soles on the floor. */
  reset(defaults) {
    const { mj, model: m, data: d } = this;
    mj.mj_resetData(m, d);
    d.qpos.set([0, 0, 1, 1, 0, 0, 0], 0);
    this.qadr.forEach((a, i) => (d.qpos[a] = defaults[i]));
    mj.mj_forward(m, d);
    let low = Infinity;
    for (const g of this.soles) {
      const p = d.geom_xpos.subarray(3 * g, 3 * g + 3), R = d.geom_xmat.subarray(9 * g, 9 * g + 9);
      const s = m.geom_size.subarray(3 * g, 3 * g + 3);
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
        low = Math.min(low, p[2] + R[6] * sx * s[0] + R[7] * sy * s[1] + R[8] * sz * s[2]);
      }
    }
    d.qpos[2] -= low;
    mj.mj_forward(m, d);
    this.setTargets(defaults);
  }

  setTargets(targets) {
    const ctrl = this.data.ctrl;
    this.act.forEach((a, i) => (ctrl[a] = targets[i]));
  }

  step(n = 1) {
    for (let i = 0; i < n; i++) this.mj.mj_step(this.model, this.data);
  }

  /** The policy's state: joint positions and velocities (policy order), base orientation (wxyz,
   * world) and angular velocity (base frame): an ideal IMU on base_link. */
  state() {
    const d = this.data;
    return {
      q: this.qadr.map((a) => d.qpos[a]),
      qd: this.vadr.map((a) => d.qvel[a]),
      baseQuat: Array.from(d.qpos.subarray(3, 7)),
      angVel: Array.from(d.qvel.subarray(3, 6)),
    };
  }

  get time() {
    return this.data.time;
  }

  basePosition() {
    return Array.from(this.data.qpos.subarray(0, 3));
  }

  /** Base linear velocity in the world frame (free joint qvel[0:3]). */
  baseVelocity() {
    return Array.from(this.data.qvel.subarray(0, 3));
  }

  baseYaw() {
    const q = this.data.qpos;
    const [w, x, y, z] = [q[3], q[4], q[5], q[6]];
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }

  /** Target bounds as the robot runtime's (drive_interface.reference_bound): the joint's range
   * widened by effort / kp on each side. The linkage ankles are left unbounded, as there. */
  referenceBounds(joints) {
    return joints.map((j, i) => {
      if (j.name.startsWith('ankle_')) return [-Infinity, Infinity];
      const slack = j.kp > 0 ? this.effort[i] / j.kp : Infinity;
      return [this.range[i][0] - slack, this.range[i][1] + slack];
    });
  }
}
