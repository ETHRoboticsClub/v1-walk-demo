# V1 walk demo

The ETHRC Humanoid V1 simulated in [MuJoCo](https://mujoco.org) (WebAssembly) in the browser and walked with the
keyboard. An InstantTrack teleop policy runs in [onnxruntime-web](https://onnxruntime.ai) on WebGPU, or on WASM
when WebGPU isn't available. The policy runs through a JavaScript port of the robot's own InstantTrack runtime.

**Live:** https://ethroboticsclub.github.io/v1-walk-demo/

## Controls

| Key | Action |
| --- | --- |
| hold <kbd>W</kbd> / <kbd>S</kbd> | walk forward / back (0.8 m/s) |
| hold <kbd>A</kbd> / <kbd>D</kbd> | walk left / right (0.6 m/s) |
| hold <kbd>Q</kbd> / <kbd>E</kbd> | turn left / right (0.6 rad/s) |
| <kbd>Space</kbd> | stop (release all keys) |
| <kbd>R</kbd> | reset to standing at the origin |
| mouse | drag to orbit, scroll to zoom (the camera follows the robot) |

Driving works like the robot dashboard's hold-to-drive pad. While a key is held, the page sends the velocity every
0.1 s with a 0.5 s expiry, and it sends one zero when the key is released. The virtual root ramps toward the command
at 0.8 m/s² (1.5 rad/s² for yaw). This policy doesn't step below about 0.4 m/s, so a held key commands the full
limit. Commands of 0.3 m/s or less barely move the robot, and walking backward doesn't work: the robot shuffles a few
centimetres.

The HUD shows:
- the commanded speeds (the virtual root's ramped speeds) next to the robot's actual speeds, in its heading frame
- the policy backend
- the sim's real-time factor
- the time per 20 ms control step (policy plus physics)
- the goal state: standing, live with its 1.5 s entry ramp, or frozen

A faint sphere marks the goal's head point and a ring marks the virtual root. If the browser can't keep up, sim time
slows down; policy steps are never skipped.

## What runs

- **Policy:** `models/walkglide/policy.onnx` with its `instanttrack.yaml`. It is the deploy bundle of
  [instantTrack-v1](https://github.com/ETHRoboticsClub/instantTrack-v1) commit `e5e1fd9`, run
  `v1-teleop-walkglide-s0` (iteration 33993). The policy was trained on BONES-SEED motion data. Only the exported
  network is published here.
- **Runtime** (`src/instanttrack.js`, `src/quat.js`): a port of `ethrc_policy_core/instanttrack/{runtime,goal,walk,bundle}.py`
  from ETHRC-Humanoid-Controls, limited to the velocity path:
  - A planar velocity walks a `VirtualRoot` (accel limits, turn then walk) that carries the bundle's standing goal.
  - `GoalStream` keeps the last K = 5 goal frames, newest first. It has the standing, live and frozen states, a 0.25 s
    staleness window, anchoring and the 1.5 s smoothstep entry ramp.
  - The observation is built as on the robot: joint positions minus defaults, joint velocities, base angular velocity,
    gravity in the base frame, the last action, and the first two columns of R_baseᵀ R_head. The head is yaw-only for
    this bundle. A 10-frame history (oldest first) feeds the network with `obs_history`, `head_pos`, `head_quat`
    (wxyz), `hand_pos` and `hand_valid`.
  - Targets are `default + action · scale`, clamped to the reference bounds.
  - The quaternion math copies the robot's `quat.py` operation for operation.
- **Sim** (`src/sim.js`, `assets/v1_physics.xml`): the V1 as instantTrack-v1 trains on it
  (`src/instanttrack/v1/v1_constants.py`, `env.py`):
  - the controls MJCF's bodies, inertials, joint armature, damping and friction
  - ground contact on the soles and the few bodies that touch the floor in falls, with no self-collision
  - one MuJoCo `<position>` actuator per joint with the policy's training PD (from the ONNX metadata) and the drive's
    effort limit
  - a flat plane, and a 5 ms step with 4 sim steps per 20 ms policy step (implicitfast, pyramidal cones, Newton)
  - an ideal IMU (ground-truth base orientation and rate) and no latency, as in training

  The robot starts standing in the default pose with its soles on the floor.

### Simplified compared with the robot (and with `tools/sim2sim_controls.py`)

- **Ankle:** a serial pitch/roll joint pair driven by joint-space PD, as in training. The real ankle is a parallel
  crank linkage. The robot runtime projects ankle targets onto what the linkage can reach. This demo doesn't, and leaves
  the ankle targets unbounded.
- **Drives:** ideal PD torques with the training gains, clipped at each drive's effort limit. There are no
  per-device signs, torque-speed curves, bus or drive delays, or crank stops.
- **Reference bounds:** joint range ± effort / kp (the robot's `reference_bound` at zero velocity and feed-forward,
  with the policy's kp). The robot uses the stiffer of the policy's gains and the drive's baseline gains.
- **Inputs:** no IMU noise, filter or latency, and no PICO/VR teleop. Only the keyboard velocity path is ported.
- **Model:** the visual meshes are decimated (about 200k triangles, merged per body). The physics model keeps only
  the contact geoms (the collision meshes as their convex hulls), which is what MuJoCo collides anyway.

## Headless check

`test/headless.mjs` runs the page's code (`src/demo.js`) under Node, with the same MuJoCo WASM and onnxruntime-web on
its wasm (CPU) backend. It plays the scripted velocity commands of `tools/sim2sim_controls.py --velocity`: 2 s still,
4 s of command, 3 s still. It prints the base displacement in the start frame.

```sh
npm install
npm run check                                   # the suite below
node test/headless.mjs '2:0,0,0;4:0.5,0,0;3:0,0,0'
```

The Python reference is `sim2sim_controls.py --bundle --velocity`, run through the robot's `InstantTrackRuntime` on
the controls plant (parallel ankle, drive models) with the walk limits lifted. Both columns are x, y in metres:

| command (4 s) | commanded | demo (JS) at t = 6 s | Python at t = 6 s | demo at 9 s | Python at 9 s | fell |
| --- | --- | --- | --- | --- | --- | --- |
| forward 0.5 m/s | 1.85, 0 | 1.51, 0.08 | 1.41, 0.09 | 1.73, 0.05 | 1.45, 0.03 | no / no |
| forward 0.8 m/s | 2.81, 0 | 2.41, −0.02 | 2.00, 0.14 | 2.65, −0.02 | 2.20, 0.14 | no / no |
| left 0.6 m/s | 0, 2.18 | −0.01, 0.64 | −0.04, 0.88 | 0.01, 0.64 | −0.01, 0.91 | no / no |
| forward 0.3 m/s | 1.15, 0 | 0.41, 0.03 | 0.38, 0.01 | 0.38, 0.01 | 0.35, 0.01 | no / no |
| back 0.5 m/s | −1.85, 0 | −0.11, 0 | −0.10, 0.01 | −0.02, 0 | −0.05, 0 | no / no |
| turn 0.6 rad/s (yaw at 9 s) | 2.29 rad | 1.93 rad | 2.06 rad | | | no / no |
| forward 0.6 + turn 0.4 | 1.35, 1.46 | 1.17, 1.20 | 1.13, 0.96 | 1.23, 1.37 | 1.18, 1.03 | no / no |

The demo's training-model sim walks a little further forward and a little less sideways than the controls plant.
The pattern is the same in both: it walks forward and sideways, turns, barely moves at 0.3 m/s or backward, and doesn't fall.

## Swapping in another bundle

Put an `instanttrack/v1` bundle (`instanttrack.yaml` plus its ONNX, from `tools/export_onnx.py --live --bundle`) in
`models/<name>/` and open `?bundle=<name>`. Any URL of a bundle folder that serves CORS works too:
`?bundle=https://…/my-bundle/`.

The joints must be the V1's 22, in the bundle's order. The PD gains come from the ONNX metadata
(`joint_stiffness`, `joint_damping`). Without them the yaml's gains apply, with `ankle_motor_pd` for the ankles.
`?ep=wasm` or `?ep=webgpu` forces the execution provider. With this small network, WASM is often faster than WebGPU
(about 3.5 ms against 12 ms per control step in a headless Chrome test).

## Layout

```
index.html               the page (import map: three, onnxruntime-web, @mujoco/mujoco, js-yaml from jsDelivr)
src/main.js              rendering, input, the real-time loop, HUD
src/demo.js              policy + runtime + sim, one control step at a time (shared with the headless check)
src/instanttrack.js      the runtime port: bundle, GoalStream, VirtualRoot, InstantTrackRuntime, ONNX metadata
src/quat.js              the robot runtime's quaternion math
src/sim.js               the MuJoCo model wrapper
assets/                  v1_physics.xml, v1_visual.{json,bin} (tools/build_assets.py)
models/walkglide/        the published policy bundle
test/headless.mjs        the headless check
tools/build_assets.py    rebuilds assets/ from an instantTrack-v1 checkout (data/v1_description)
```

It's a static site with no build step, served by GitHub Pages from `main`.
