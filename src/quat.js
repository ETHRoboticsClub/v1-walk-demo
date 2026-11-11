// Quaternion arithmetic of ethrc_policy_core.sonic.quat (wxyz, double precision), operation for
// operation, so the observation matches the robot runtime's.

export const IDENTITY = [1, 0, 0, 0];
const EPS = 1e-12;
const SLERP_LINEAR_DOT = 0.9995;

export function rotate(q, v) {
  const [w, qx, qy, qz] = q;
  const sa = 2 * w * w - 1;
  const a = [v[0] * sa, v[1] * sa, v[2] * sa];
  const cross = [qy * v[2] - qz * v[1], qz * v[0] - qx * v[2], qx * v[1] - qy * v[0]];
  const b = [cross[0] * w * 2, cross[1] * w * 2, cross[2] * w * 2];
  const dot = qx * v[0] + qy * v[1] + qz * v[2];
  const c = [qx * dot * 2, qy * dot * 2, qz * dot * 2];
  return [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]];
}

export function mul(a, b) {
  const [w1, x1, y1, z1] = a;
  const [w2, x2, y2, z2] = b;
  const ww = (z1 + x1) * (x2 + y2);
  const yy = (w1 - y1) * (w2 + z2);
  const zz = (w1 + y1) * (w2 - z2);
  const xx = ww + yy + zz;
  const qq = 0.5 * (xx + (z1 - x1) * (x2 - y2));
  return [
    qq - ww + (z1 - y1) * (y2 - z2),
    qq - xx + (x1 + w1) * (x2 + w2),
    qq - yy + (w1 - x1) * (y2 + z2),
    qq - zz + (z1 + y1) * (w2 - x2),
  ];
}

export const conjugate = (q) => [q[0], -q[1], -q[2], -q[3]];

export function unit(q) {
  const n = Math.max(Math.hypot(q[0], q[1], q[2], q[3]), EPS);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function normalize(v) {
  const n = Math.max(Math.hypot(v[0], v[1], v[2]), EPS);
  return [v[0] / n, v[1] / n, v[2] / n];
}

export function fromAngleAxis(angle, axis) {
  const h = angle / 2;
  const s = Math.sin(h);
  const n = normalize(axis);
  return unit([Math.cos(h), n[0] * s, n[1] * s, n[2] * s]);
}

/** The yaw of q's rotated x axis. */
export function heading(q) {
  const [x, y] = rotate(q, [1, 0, 0]);
  return Math.atan2(y, x);
}

export const headingQuat = (q) => fromAngleAxis(heading(q), [0, 0, 1]);
export const yaw = (angle) => fromAngleAxis(angle, [0, 0, 1]);

export function slerp(q0, q1, t) {
  let dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
  if (dot < 0) {
    q1 = q1.map((v) => -v);
    dot = -dot;
  }
  if (dot > SLERP_LINEAR_DOT) return unit(q0.map((v, i) => v + t * (q1[i] - v)));
  const theta = Math.acos(Math.abs(dot));
  const st = Math.sin(theta);
  const f0 = Math.sin((1 - t) * theta) / st;
  const f1 = Math.sin(t * theta) / st;
  return q0.map((v, i) => f0 * v + f1 * q1[i]);
}

/** Rotation matrix of q (normalized first), rows, flat row-major [9]. */
export function matrix(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  const w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

/** Gravity's direction in the base frame. */
export const gravity = (base) => rotate(conjugate(base), [0, 0, -1]);
