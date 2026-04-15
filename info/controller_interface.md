# Controller Interface: Unified Motion Commander

## Overview

`arm_cmd.py` is a middleware process that bridges experiment/client scripts to the
C++ joint controller (`State_JointCmd`). It accepts either raw joint angles or
Cartesian end-effector targets, runs warm-started IK if needed, applies model-based
gravity compensation, and streams corrected joint commands to the controller at 50 Hz.

---

## Process Layout

```
  your_script.py  /  test_arm_cmd_joints.py  /  test_arm_cmd_cartesian.py
  any rate            50 Hz (--hz)               10 Hz (--hz)
  ─────────────────────────────────────────────────────────────────────────
         │  ZMQ PUB → connect  port 5557
         │  {"type":"joints",    "q":[29 floats]}
         │  {"type":"cartesian", "targets":{"right_eef":[x,y,z]}}
         │  optional  "hand": [12 floats, 0-1]   ← add to any message type
         ▼
╔═══════════════════════════════════════════════════════════════╗
║  scripts/arm_cmd.py                    50 Hz  (--hz flag)     ║
║                                                               ║
║  joints ──────────────────────────────────────► q_desired     ║
║                                                               ║
║  cartesian → [IK Solver]                                      ║
║              warm-started DLS  ~200 iters cold / ~10 warm     ║
║              --ik-config  ik/configs/g1_right_arm.yaml        ║
║                       │                                       ║
║                       ▼                                       ║
║              [joint name map]  MuJoCo name → index 0-28  ───► q_desired
║                                arm_cmd.py : MUJOCO_JOINT_TO_IDX ║
║                                                               ║
║  q_desired → [Gravity Comp]  q_sent = q_desired + τ_g/Kp     ║
║               qfrc_bias/Kp  ← g1_sitting.xml                 ║
║               kp            ← joint_cmd.yaml : kp            ║
║                                                               ║
║  hand cmd  → DDS rt/inspire/cmd  (if --hand-interface set)    ║
╚══════════╤═══════════════════════════════════╤════════════════╝
           │  ZMQ PUB → bind  port 5555        │  ZMQ SUB ← connect  port 5556
           │  {"q":[29 floats]}  @ 50 Hz       │  feedback @ 200 Hz
           ▼                                   │  {"tick","q_current","q_target"}
╔══════════════════════════════════════════════╧══════════════════════════╗
║  g1_ctrl  (State_JointCmd)                         1000 Hz  (dt_=0.001f)║
║  ZMQ SUB ← port 5555  receive joint commands                            ║
║  ZMQ PUB → port 5556  feedback  200 Hz  ← joint_cmd.yaml:               ║
║                                            feedback_decimation: 5       ║
║  ZMQ PUB → port 5558  plot pub  200 Hz  ← joint_cmd.yaml:               ║
║                                            zmq_plot_address             ║
╚══════════════════════════════╤═════════════════════════════════════════╝
                               │  Unitree SDK  (DDS / shared mem)
                               ▼
                    ┌─────────────────────┐
                    │  unitree_mujoco      │  ~500 Hz physics
                    │  (simulator)         │  ← XML <option timestep=...>
                    └─────────────────────┘

  Hand path (real robot only):
  ────────────────────────────────────────────────────────────────────────
  arm_cmd.py  ──DDS rt/inspire/cmd──►  inspire_g1  ──serial──►  Inspire hands
                                       (on robot,               /dev/ttyUSB1 (R)
                                        sudo required)          /dev/ttyUSB2 (L)

  scripts/test_hand.py  (standalone hand test, no body controller needed)
  ─────────────────────────────────────────────────────────────────────────
  DDS PUB → rt/inspire/cmd    (open/close/wave modes)
  DDS SUB ← rt/inspire/state  (live joint readback)


  port 5558 consumed by (optional):
  ──────────────────────────────────
  scripts/plot_joint_error.py   30 Hz plot  (--hz)
  ZMQ SUB ← connect  port 5558
  receives 200 Hz stream, renders at 30 Hz


  ALTERNATIVE — send_joint_cmd.py  (bypasses arm_cmd.py entirely):
  ─────────────────────────────────────────────────────────────────
  scripts/send_joint_cmd.py          50 Hz  (--hz)
  ZMQ PUB → bind   port 5555  →  controller
  ZMQ SUB ← connect port 5556  ←  feedback  (convergence detection)


  FREQUENCY REFERENCE
  ───────────────────────────────────────────────────────────────────────
  Process                    Default    Change in
  unitree_mujoco  physics    ~500 Hz    XML <option timestep="...">
  g1_ctrl  FSM tick          1000 Hz    State_JointCmd.h : dt_ = 0.001f
  feedback pub  (port 5556)  200 Hz     joint_cmd.yaml : feedback_decimation
  plot pub      (port 5558)  200 Hz     joint_cmd.yaml : feedback_decimation
  arm_cmd.py  loop            50 Hz     --hz  flag
  send_joint_cmd.py           50 Hz     --hz  flag
  plot_joint_error.py         30 Hz     --hz  flag
  inspire_g1 control loop    10 kHz     dfx_inspire_service/inspire_g1.cpp
```

`send_joint_cmd.py` is kept as a standalone test/debug tool and is unchanged.

---

## Message Format

All messages are JSON, exchanged over ZMQ PUB/SUB.

### Input to arm_cmd.py  (port 5557)

```json
// Joint mode — forward directly
{"type": "joints", "q": [29 floats]}

// Cartesian mode — IK solve first
{"type": "cartesian", "targets": {"right_eef": [x, y, z]}}
{"type": "cartesian", "targets": {"right_eef": [x,y,z], "left_eef": [x,y,z]}}

// Either mode + optional hand control
{"type": "joints", "q": [29 floats], "hand": [12 floats]}
```

- `targets` keys must match names defined in the IK config (e.g. `right_eef`, `left_eef`).
- If `type` is absent it defaults to `"joints"`.
- `"hand"` is optional and independent of `type`. If omitted, the last commanded hand pose is held.

**Hand field format** (`"hand"`: 12 floats, range `0.0`=closed → `1.0`=open):

```
Index  0- 5:  right hand  [pinky, ring, middle, index, thumb_bend, thumb_rot]
Index  6-11:  left  hand  [pinky, ring, middle, index, thumb_bend, thumb_rot]
```

Requires `--hand-interface` set and `inspire_g1` running on the robot (see Workflows).

### Output from arm_cmd.py → controller  (port 5555)

```json
{"q": [29 floats]}   // same format that send_joint_cmd.py already produces
```

### Feedback from controller  (port 5556, subscribed by arm_cmd.py)

```json
{"tick": N, "q_current": [29 floats], "q_target": [29 floats]}
```

Used for gravity compensation (needs live `q_current`).

### Plot stream from controller  (port 5558, subscribed by plot_joint_error.py)

```json
{"tick": N, "q_current": [29 floats], "q_target": [29 floats]}
```

Same JSON format as port 5556, published on a dedicated port so the plotter
is decoupled from the arm_cmd.py feedback channel.

---

## arm_cmd.py Design

### Startup args

| Flag | Default | Purpose |
|------|---------|---------|
| `--ik-config` | `ik/configs/g1_right_arm.yaml` | IK YAML (active joints, bodies, limits) |
| `--in-address` | `tcp://*:5557` | ZMQ bind address for incoming commands |
| `--out-address` | `tcp://localhost:5555` | ZMQ connect address for controller |
| `--feedback` | `tcp://localhost:5556` | ZMQ feedback address |
| `--hz` | `50` | Control loop rate |
| `--no-gravity-comp` | off | Disable model-based gravity compensation |
| `--no-ik` | off | Disable IK (cartesian commands rejected) |
| `--hand-interface` | *(disabled)* | Network interface for Inspire hand DDS (e.g. `enp4s0`). Requires `inspire_g1` running on the robot. |

### Control loop (50 Hz)

```
1. Drain fb_socket  → update q_current (latest only)
2. Drain in_socket  → take latest command (stale ones silently dropped)
3. Process command:
     "joints"    → q_desired = msg["q"]
     "cartesian" → run IKSolver.solve(targets, warm_start=last_qpos)
                   map result.joint_angles → q_desired (full 29-vector)
                   store result.qpos as warm_start for next call
     "hand" key  → hand_desired = msg["hand"]   (any type; persists if absent)
4. gravity_offset = GravityCompensator.compute(q_current)   (if enabled)
5. q_sent[i] = q_desired[i] + gravity_offset[i]
5b. hand_ctrl.set_angles(hand_desired)   (if --hand-interface set)
6. out_socket.send_string(json.dumps({"q": q_sent}))
7. sleep until next tick
```

---

## Inspire Hand Control (real robot only)

The G1's Inspire hands connect to the robot via serial (`/dev/ttyUSB1` right, `/dev/ttyUSB2` left).
The `inspire_g1` bridge process translates between serial and DDS.

### Architecture

```
arm_cmd.py / test_hand.py
    │  DDS  rt/inspire/cmd   (unitree_go::msg::dds_::MotorCmds_, 12 motors)
    ▼
inspire_g1   (must run as root on the robot)
    │  serial  115200 baud, binary protocol
    ├─► /dev/ttyUSB1   right hand
    └─► /dev/ttyUSB2   left hand
```

### Running the hand bridge

```bash
# On the robot (separate terminal, keep running)
ssh unitree@192.168.123.164
sudo ./dfx_inspire_service/build/inspire_g1
```

### DDS topics

| Topic | Direction | IDL type |
|-------|-----------|----------|
| `rt/inspire/cmd` | publish (command) | `unitree_go::msg::dds_::MotorCmds_` |
| `rt/inspire/state` | subscribe (feedback) | `unitree_go::msg::dds_::MotorStates_` |

12 motors total: indices 0-5 = right hand, 6-11 = left hand.  
Joint order per hand: `[pinky, ring, middle, index, thumb_bend, thumb_rot]`.  
Values: `0.0` = closed, `1.0` = open.

### Standalone test (no body controller needed)

```bash
# Open/close wave with live state readback
python3 scripts/test_hand.py --interface enp4s0

# Hold open
python3 scripts/test_hand.py --interface enp4s0 --open

# Hold closed
python3 scripts/test_hand.py --interface enp4s0 --close
```

`--interface` is your local Ethernet interface facing the robot (`enp4s0` at `192.168.123.222`).
DDS discovers across the subnet — the script does **not** need to run on the robot.

---

## IK Integration

### Why warm-starting is critical for real-time

| Call type | Iterations needed | Wall time |
|-----------|------------------|-----------|
| Cold start (first call) | ~200 | ~5–15 ms |
| Warm start (Δpos ≈ 1 cm) | ~5–20 | < 2 ms |

`solver.data.qpos` persists between calls inside `IKSolver`. The commander keeps
`warm_start = result.qpos` and passes it to the next `solver.solve()` call.

### IK config at startup (not per-message)

Active joints, body names, joint limits — all fixed at startup from the YAML config.
The message only needs to specify target position(s) by name. This avoids the overhead
of re-parsing configs on every command at streaming rates.

### Target name → body name

Defined in the IK YAML (`targets:` section). The commander reads this at startup:

```yaml
targets:
  - name: right_eef
    body: right_wrist_yaw_link
    ...
```

The message uses `right_eef` as the key; the commander looks up `right_wrist_yaw_link`.

---

## Joint Name Mapping

IK solver returns `dict[mujoco_joint_name → angle_rad]` for active joints only.
The commander overlays these onto a full 29-vector (starting from `q_default`).

```
MuJoCo joint name              → controller index
─────────────────────────────────────────────────
left_hip_pitch_joint           →  0
left_hip_roll_joint            →  1
left_hip_yaw_joint             →  2
left_knee_joint                →  3
left_ankle_pitch_joint         →  4
left_ankle_roll_joint          →  5
right_hip_pitch_joint          →  6
right_hip_roll_joint           →  7
right_hip_yaw_joint            →  8
right_knee_joint               →  9
right_ankle_pitch_joint        → 10
right_ankle_roll_joint         → 11
waist_yaw_joint                → 12
waist_roll_joint               → 13
waist_pitch_joint              → 14
left_shoulder_pitch_joint      → 15
left_shoulder_roll_joint       → 16
left_shoulder_yaw_joint        → 17
left_elbow_joint               → 18
left_wrist_roll_joint          → 19
left_wrist_pitch_joint         → 20
left_wrist_yaw_joint           → 21
right_shoulder_pitch_joint     → 22
right_shoulder_roll_joint      → 23
right_shoulder_yaw_joint       → 24
right_elbow_joint              → 25
right_wrist_roll_joint         → 26
right_wrist_pitch_joint        → 27
right_wrist_yaw_joint          → 28
```

Non-active joints stay at `q_default` (the sitting pose from `joint_cmd.yaml`).

---

## Files

| File | Role | Status |
|------|------|--------|
| `scripts/arm_cmd.py` | Unified motion commander (body + hands) | **updated** |
| `scripts/test_hand.py` | Standalone Inspire hand test | **new** |
| `scripts/send_joint_cmd.py` | Step/sine test sender | unchanged |
| `scripts/plot_joint_error.py` | Live error plotter | updated — default address → port 5558 |
| `ik/ik_solver.py` | IK solver library | unchanged |
| `ik/configs/g1_right_arm.yaml` | Right-arm IK config | unchanged |
| `ik/configs/g1_dual_arm.yaml` | Dual-arm IK config | unchanged |
| `deploy/robots/g1/config/joint_cmd.yaml` | PD gains, q_default | unchanged |
| `dump_scripts/test_arm_cmd_joints.py` | Joint-mode test client | **new** |
| `dump_scripts/test_arm_cmd_cartesian.py` | Cartesian streaming test client | **new** |
| `dfx_inspire_service/build/inspire_g1` | Hand serial↔DDS bridge (on robot) | external |

---

## Typical Workflows

Start the terminals in order. T0 is real-robot only; T1–T3 are always required; T4 is swapped depending on what you are testing; T5 is optional; T6 is only needed when running an experiment from your own script.

### T0 — Inspire hand bridge  *(real robot only)*

```bash
# On the robot
ssh unitree@192.168.123.164
sudo ./dfx_inspire_service/build/inspire_g1
```

### T1 — Simulator

```bash
./simulate/build/unitree_mujoco -s src/assets/robots/unitree_g1/xmls/g1_sitting.xml
```

### T2 — C++ controller  *(press [4] to enter JointCmd mode)*

```bash
./deploy/robots/g1/build/g1_ctrl -n lo
```

### T3 — Motion commander  *(always required)*

```bash
python3 scripts/arm_cmd.py
# With dual-arm IK config:
python3 scripts/arm_cmd.py --ik-config ik/configs/g1_dual_arm.yaml
# With Inspire hand control (real robot, requires T0):
python3 scripts/arm_cmd.py --hand-interface enp4s0
```

### T4a — Joint pass-through test client

```bash
# Oscillate right elbow ±0.4 rad, 2 s per phase
python3 dump_scripts/test_arm_cmd_joints.py --joint 25 --amp 0.4 --dwell 2.0
```

### T4b — Cartesian streaming test client

```bash
# Trace 8 cm circle in Y-Z plane over 8 s
python3 dump_scripts/test_arm_cmd_cartesian.py --radius 0.08 --period 8.0
```

### T5 — Live error plotter  *(optional)*

```bash
# Right elbow only
python3 scripts/plot_joint_error.py --joints 25
# Waist + full right arm
python3 scripts/plot_joint_error.py --joints 12 13 14 22 23 24 25 26 27 28
```

### T6 — Experiment script snippet  *(your own code)*

```python
import zmq, json, time

ctx  = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.connect("tcp://localhost:5557")
time.sleep(0.5)  # let SUB side register

# Joint mode — send a full 29-vector
sock.send_string(json.dumps({"type": "joints", "q": my_29_vector}))

# Cartesian mode — send end-effector target(s) by name
sock.send_string(json.dumps({
    "type": "cartesian",
    "targets": {"right_eef": [0.45, -0.25, 0.85]}
}))

# Dual-arm cartesian (requires --ik-config ik/configs/g1_dual_arm.yaml on T3)
sock.send_string(json.dumps({
    "type": "cartesian",
    "targets": {
        "right_eef": [0.45, -0.25, 0.85],
        "left_eef":  [0.45,  0.25, 0.85],
    }
}))

# With hand control (requires --hand-interface on T3 and inspire_g1 on T0)
# "hand": [R_pinky, R_ring, R_mid, R_idx, R_thumb_bend, R_thumb_rot,
#           L_pinky, L_ring, L_mid, L_idx, L_thumb_bend, L_thumb_rot]
# 0.0 = closed, 1.0 = open
sock.send_string(json.dumps({
    "type": "joints",
    "q": my_29_vector,
    "hand": [1.0]*6 + [1.0]*6,   # both hands fully open
}))
```

---

## Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| PUB/SUB (not REQ/REP) | Fire-and-forget; drain-and-take-latest pattern; no blocking |
| Drain + take latest | Real-time: stale targets are irrelevant |
| IK config at startup | Avoids re-parsing 300-iteration configs at streaming rate |
| Warm-starting mandatory | Makes IK fast enough for 50 Hz streaming (<2 ms per solve) |
| Gravity comp in commander | Pure PD controller has steady-state error = τ_gravity/Kp; fix in Python only |
| C++ controller unchanged | Single responsibility: just apply commanded joint angles |
| `q_default` as IK baseline | Non-active joints (legs) stay at their safe sitting pose |
