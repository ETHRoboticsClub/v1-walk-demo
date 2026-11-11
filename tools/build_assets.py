"""Build the demo's robot assets from the instantTrack-v1 V1 model (data/v1_description/urdf/v1_rt.xml).

    uv run --with mujoco==3.12.0 --with numpy --with fast-simplification --with scipy \
        tools/build_assets.py <instantTrack-v1 checkout>

Writes
- assets/v1_physics.xml: the model as instantTrack-v1 trains on it (src/instanttrack/v1/v1_constants.py and
  env.py): the controls MJCF's bodies, joints, inertials, armature and passive damping/friction; only the
  ground-contact geoms (contype 0, conaffinity 1; self-collision off); one MuJoCo <position> actuator per joint
  with the policy's training gains (kp, kd from the ONNX metadata) and the drive's effort limit; a flat plane;
  5 ms timestep, implicitfast, pyramidal cones, Newton. Self-contained: the few collision meshes are inlined.
- assets/v1_visual.json + .bin: every visual mesh, decimated, merged per body and colour, in the body frame.
"""

import json
import math
import os
import sys

import fast_simplification
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

SRC = os.path.abspath(sys.argv[1])
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
MJCF = os.path.join(SRC, "data", "v1_description", "urdf", "v1_rt.xml")

# v1_constants.py: the drive families (effort N m, reflected armature, kp cap) and the gain rule.
HEJ90, HEJ70, HEJ50 = (140.0, 0.05, math.inf), (50.0, 0.34, 100.0), (30.0, 0.0056, math.inf)
DRIVES = {
    "torso_axial_rotation": HEJ90, "torso_lateral_bending": HEJ90, "hip_flexion": HEJ90, "hip_abduction": HEJ90,
    "hip_internal_rotation": HEJ90, "knee_flexion": HEJ90, "shoulder_flexion": HEJ70, "shoulder_abduction": HEJ70,
    "shoulder_internal_rotation": HEJ50, "elbow_flexion": HEJ50,
    "ankle_dorsiflexion": (30.0, 0.0104, math.inf), "ankle_inversion": (30.0, 0.0126, math.inf),
}  # fmt: skip
GROUND_GEOM_BODIES = (
    "base_link", "hej90_hip_abduction_1", "universal_joint_connector_l_Mirror_1", "universal_joint_connector_l_1",
    "connector_elbow_forearm_l_1", "connector_elbow_forearm_r_1", "L_Carpals_1", "R_Carpals_1",
)  # fmt: skip
W = 2.0 * math.pi * 10.0


def drive(name):
    return DRIVES[name[:-2] if name.endswith(("_l", "_r")) else name]


def gains(name):
    _, arm, cap = drive(name)
    kp = min(arm * W * W, cap)
    return kp, 4.0 * math.sqrt(kp * arm)


def body_frame_verts(m, g):
    """Vertices and faces of mesh geom g in its body's frame."""
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
    f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return v @ R.reshape(3, 3).T + m.geom_pos[g], f


spec = mujoco.MjSpec.from_file(MJCF)
for s in list(spec.sensors):
    spec.delete(s)
for a in list(spec.actuators):
    spec.delete(a)
m = spec.compile()
joints = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]

# The collision set of v1_constants.get_spec.
keep = {m.geom("sole_box_l").id, m.geom("sole_box_r").id}
for body in GROUND_GEOM_BODIES:
    bid = m.body(body).id
    ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    keep.add(max(ids, key=lambda g: m.geom_rbound[g]))

# ---- visuals: every mesh geom, decimated, merged per (body, colour) ----
groups = {}
for g in range(m.ngeom):
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        v, f = body_frame_verts(m, g)
    elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and g not in (m.geom("sole_box_l").id, m.geom("sole_box_r").id):
        sx, sy, sz = m.geom_size[g]
        c = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
                      [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])  # fmt: skip
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        v = c @ R.reshape(3, 3).T + m.geom_pos[g]
    else:
        continue
    if len(f) > 400:
        v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=0.75)
    key = (m.body(m.geom_bodyid[g]).name, tuple(np.round(m.geom_rgba[g], 3)))
    groups.setdefault(key, []).append((np.asarray(v, np.float32), np.asarray(f, np.uint32)))

entries, pos_chunks, idx_chunks = [], [], []
voff = ioff = 0
for (body, rgba), parts in sorted(groups.items()):
    base = 0
    vs, fs = [], []
    for v, f in parts:
        vs.append(v)
        fs.append(f + base)
        base += len(v)
    V, F = np.concatenate(vs), np.concatenate(fs).reshape(-1)
    entries.append({"body": body, "rgba": [float(x) for x in rgba], "vertexOffset": voff, "vertexCount": len(V),
                    "indexOffset": ioff, "indexCount": len(F)})  # fmt: skip
    pos_chunks.append(V.reshape(-1))
    idx_chunks.append(F)
    voff += len(V)
    ioff += len(F)
P = np.concatenate(pos_chunks).astype("<f4")
I = np.concatenate(idx_chunks).astype("<u4")
with open(os.path.join(OUT, "v1_visual.bin"), "wb") as fh:
    fh.write(P.tobytes())
    fh.write(I.tobytes())
json.dump({"positionsBytes": P.nbytes, "indicesBytes": I.nbytes, "meshes": entries},
          open(os.path.join(OUT, "v1_visual.json"), "w"), indent=0)  # fmt: skip
print(f"visual: {voff} vertices, {ioff // 3} triangles, {len(entries)} meshes", file=sys.stderr)

# ---- physics: drop every non-contact geom; inline the kept collision meshes in the body frame ----
inline = {}
for g in keep:
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        v, _ = body_frame_verts(m, g)
        hull = ConvexHull(v)  # MuJoCo collides a mesh as its convex hull: keep only that
        inline[g] = (v[hull.vertices], None)
geoms = list(spec.geoms)
for i, geom in enumerate(geoms):  # spec geoms are in compiled order
    if i in keep:
        geom.contype, geom.conaffinity = 0, 1
        if i in inline:
            v, _ = inline[i]
            name = f"collision_{i}"
            spec.add_mesh(name=name, uservert=v.reshape(-1).tolist())
            geom.meshname = name
            geom.pos, geom.quat = [0, 0, 0], [1, 0, 0, 0]
        geom.rgba = [0.9, 0.3, 0.2, 0.4]
    else:
        spec.delete(geom)
for mesh in list(spec.meshes):
    if not mesh.name.startswith("collision_"):
        spec.delete(mesh)
spec.meshdir = ""
for name in joints:
    kp, kd = gains(name)
    effort = drive(name)[0]
    a = spec.add_actuator(name=name, target=name)
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.dyntype = mujoco.mjtDyn.mjDYN_NONE
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.gainprm[0] = kp
    a.biasprm[1], a.biasprm[2] = -kp, -kd
    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    a.forcerange = [-effort, effort]
spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01])
o = spec.option
o.timestep = 0.005
o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
o.solver = mujoco.mjtSolver.mjSOL_NEWTON
o.iterations, o.ls_iterations, o.tolerance, o.ccd_iterations = 100, 50, 1e-6, 35
spec.compile()
xml = spec.to_xml()
open(os.path.join(OUT, "v1_physics.xml"), "w").write(xml)
check = mujoco.MjModel.from_xml_string(xml)
print(f"physics: {check.ngeom} geoms, {check.nu} actuators, {len(xml) // 1024} KiB; joints {joints}", file=sys.stderr)
