#!/usr/bin/env python3
"""
arm_cmd.py — Unified motion commander for the G1 arm + Inspire hands.

Binds a ZMQ SUB socket on port 5557 to receive high-level commands:
  {"type": "joints",    "q": [29 floats]}
  {"type": "joints",    "q": [29 floats], "hand": [12 floats]}
  {"type": "cartesian", "targets": {"right_eef": [x, y, z], ...}}
  {"type": "cartesian", "targets": {...},  "hand": [12 floats]}

The optional "hand" field controls the Inspire hands via the dfx_inspire_service
DDS bridge (rt/inspire/cmd).  Ordering: [R_pinky, R_ring, R_mid, R_idx,
R_thumb_bend, R_thumb_rot, L_pinky, L_ring, L_mid, L_idx, L_thumb_bend,
L_thumb_rot], each in range 0.0-1.0 (0=closed, 1=open).

Prerequisites: the inspire_g1 service must be running on the robot:
  sudo ./dfx_inspire_service/build/inspire_g1

Resolves cartesian commands via warm-started differential IK, computes
model-based gravity torque (qfrc_bias) as a per-joint feedforward, and
forwards both q_desired and tau_ff to the C++ controller via ZMQ PUB on
port 5555. The C++ side adds tau_ff straight into motor.tau(), which the
FSA actuator sums into its internal PD law (τ = kp·(q-q_des) + kd·(dq-dq_des)
+ tau_ff), cancelling gravity bias without any kp dependence.

Usage
-----
  python3 scripts/arm_cmd.py
  python3 scripts/arm_cmd.py --ik-config ik/configs/g1_right_arm.yaml --hz 50
  python3 scripts/arm_cmd.py --no-ik
  python3 scripts/arm_cmd.py --hand-interface eth0
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import yaml
import zmq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)

# Make ik/ importable
sys.path.insert(0, REPO_ROOT)

CONFIG_PATH        = os.path.join(REPO_ROOT, "deploy", "robots", "g1", "config", "joint_cmd.yaml")
PASSIVE_JOINTS_PATH = os.path.join(REPO_ROOT, "deploy", "robots", "g1", "config", "passive_joints.yaml")
XML_PATH    = os.path.join(REPO_ROOT, "src", "assets", "robots",
                           "unitree_g1", "xmls", "g1_sitting.xml")

FEEDBACK_TIMEOUT_S = 2.0   # seconds without feedback → warn about JointCmd mode
WARN_COOLDOWN_S    = 5.0   # min gap between repeated warnings

# ---------------------------------------------------------------------------
# Inspire Hand (DDS via dfx_inspire_service — rt/inspire/cmd)
# ---------------------------------------------------------------------------

class InspireHandDDS:
    """Publishes Inspire hand commands via DDS to the dfx_inspire_service bridge.

    The inspire_g1 service must be running on the robot; it bridges DDS ↔ serial.

    Motor ordering in rt/inspire/cmd (12 motors total):
      0-5:  right hand [pinky, ring, middle, index, thumb_bend, thumb_rot]
      6-11: left  hand [pinky, ring, middle, index, thumb_bend, thumb_rot]

    Values: 0.0 (closed) to 1.0 (open).
    """

    def __init__(self, network_interface: str = None):
        from unitree_sdk2py.core.channel import (
            ChannelPublisher, ChannelFactoryInitialize
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

        if network_interface:
            ChannelFactoryInitialize(0, network_interface)
        else:
            ChannelFactoryInitialize(0)

        self._cmd = MotorCmds_()
        for _ in range(12):
            self._cmd.cmds.append(unitree_go_msg_dds__MotorCmd_())

        self._pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self._pub.Init()

    def set_angles(self, values: list) -> None:
        """Send 12 finger values (clamped to [0,1]). Right hand first, then left."""
        if len(values) != 12:
            return
        for i in range(12):
            self._cmd.cmds[i].q = float(max(0.0, min(1.0, values[i])))
        self._pub.Write(self._cmd)

    def close(self) -> None:
        if hasattr(self, "_pub"):
            self._pub.Close()


# ---------------------------------------------------------------------------
# MuJoCo joint name → controller index (0-28)
# ---------------------------------------------------------------------------

MUJOCO_JOINT_TO_IDX = {
    "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2,
    "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8,
    "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16, "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19, "left_wrist_pitch_joint": 20, "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23, "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27, "right_wrist_yaw_joint": 28,
}

# ---------------------------------------------------------------------------
# Gravity compensator (verbatim from send_joint_cmd.py)
# ---------------------------------------------------------------------------

class GravityCompensator:
    """Computes per-joint gravity torque (MuJoCo qfrc_bias at qvel=0).

    With qvel=0, MuJoCo's qfrc_bias is the pure gravity torque at each joint.
    Forwarding it as motor.tau() makes the FSA actuator's internal PD law
    cancel gravity directly: τ = kp·(q_des-q) + kd·(dq_des-dq) + qfrc_bias.
    No kp dependence, so it works on compliant or zero-kp joints too.

    Joint ordering in g1_sitting.xml matches the controller order exactly:
      left_leg[0-5], right_leg[6-11], waist[12-14], left_arm[15-21], right_arm[22-28]
    """

    def __init__(self, xml_path: str):
        import mujoco
        self._mujoco = mujoco
        self.model   = mujoco.MjModel.from_xml_path(xml_path)
        self.data    = mujoco.MjData(self.model)
        self.data.qvel[:] = 0.0   # static — pure gravity, no Coriolis/centrifugal
        assert self.model.nq == 29, f"Expected 29 DOF, got {self.model.nq}"

    def compute(self, q_current: list) -> list:
        """Return per-joint gravity torque (Nm) at the given configuration."""
        self.data.qpos[:] = q_current
        self._mujoco.mj_forward(self.model, self.data)
        return [float(self.data.qfrc_bias[i]) for i in range(29)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_q_default() -> list:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    q = cfg["q_default"]
    assert len(q) == 29, f"q_default has {len(q)} joints, expected 29"
    return list(map(float, q))


def _qpos_from_q_current(ik_solver, ik_cfg, q_current: list) -> np.ndarray:
    """Build a full IK qpos vector seeded with the live 29-DOF joint state.

    The IK model has a free base joint (qpos[0:7]); we set it from the IK
    config's base_pos / base_quat. The 29 controller joints are written into
    their qpos addresses by name, so the IK starts from the actual robot pose
    and picks the kinematic branch nearest to it.
    """
    import mujoco  # type: ignore[import-untyped]
    qpos = np.zeros(ik_solver.model.nq)
    if ik_solver.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        qpos[0:3] = ik_cfg.base_pos
        qpos[3:7] = ik_cfg.base_quat
    for jname, idx in MUJOCO_JOINT_TO_IDX.items():
        jid = mujoco.mj_name2id(ik_solver.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            qpos[ik_solver.model.jnt_qposadr[jid]] = q_current[idx]
    return qpos


def drain_latest(sock):  # -> str | None
    """Drain all pending messages from a NOBLOCK SUB socket; return the latest raw string."""
    latest = None
    try:
        while True:
            latest = sock.recv_string(flags=zmq.NOBLOCK)
    except zmq.Again:
        pass
    return latest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified arm motion commander — receives high-level commands, runs IK, forwards to controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ik-config",   default=os.path.join(REPO_ROOT, "ik", "configs", "g1_right_arm.yaml"),
                        help="Path to IK YAML config (default: ik/configs/g1_right_arm.yaml)")
    parser.add_argument("--in-address",  default="tcp://*:5557",
                        help="ZMQ bind address for incoming commands (default: tcp://*:5557)")
    parser.add_argument("--out-address", default="tcp://*:5555",
                        help="ZMQ bind address for controller (default: tcp://*:5555)")
    parser.add_argument("--feedback",    default="tcp://localhost:5556",
                        help="ZMQ feedback address (default: tcp://localhost:5556)")
    parser.add_argument("--hz",          type=float, default=None,
                        help="Control loop rate in Hz. Defaults to joint_cmd.yaml's policy_hz "
                             "so each outgoing message carries one real policy step (required "
                             "for the C++ JointCmd's dq_ff feedforward to be correct).")
    parser.add_argument("--no-ik",       action="store_true",
                        help="Disable IK (cartesian commands will be rejected)")
    parser.add_argument("--hand-interface", default=None,
                        help="Network interface for Inspire hand DDS (e.g. eth0). "
                             "Requires inspire_g1 service running on the robot.")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg_yaml = yaml.safe_load(open(CONFIG_PATH))
    q_default = load_q_default()
    locked_joints: dict[int, float] = {
        int(k): float(v) for k, v in cfg_yaml.get("locked_joints", {}).items()
    }

    # arm_cmd must publish at the same rate as the C++ JointCmd's policy_hz so each
    # outgoing message advances by one real upstream step. If arm_cmd runs faster,
    # most messages carry only gravity-comp wobble and the C++ dq_ff feedforward
    # (which divides Δq by 1/policy_hz) degenerates into a 10 Hz pulse train.
    policy_hz = float(cfg_yaml["policy_hz"])
    if args.hz is None:
        args.hz = policy_hz
    else:
        assert abs(args.hz - policy_hz) < 1e-6, (
            f"--hz={args.hz} disagrees with policy_hz={policy_hz} in {CONFIG_PATH}; "
            f"arm_cmd must publish at policy_hz so the C++ dq_ff feedforward is correct."
        )

    # ── Inspire hands (DDS) ──────────────────────────────────────────────────
    hand_ctrl = InspireHandDDS(args.hand_interface) if args.hand_interface else None

    # ── Gravity compensator ──────────────────────────────────────────────────
    gc = None
    gravity_comp_enabled = cfg_yaml.get("gravity_comp", True)
    if gravity_comp_enabled:
        gc = GravityCompensator(XML_PATH)

    # ── Per-joint torque clamp (Nm) ───────────────────────────────────────────
    # Bound the gravity feedforward we publish so a bad MuJoCo state can't drive
    # the FSA past its ctrlrange. The C++ side already uses this for adaptive
    # interp duration; we reuse it here as a symmetric ±max clamp on tau_ff.
    max_torque = cfg_yaml.get("max_torque", None)
    if max_torque is not None:
        max_torque = list(map(float, max_torque))
        assert len(max_torque) == 29, f"max_torque has {len(max_torque)} entries, expected 29"

    # ── IK solver ────────────────────────────────────────────────────────────
    ik_solver = None
    ik_cfg    = None
    body_map  = {}   # target name → TargetFrame (from config)
    # frozen_q: index → value derived from IK config's frozen_leg_joints.
    # Used to initialise q_desired and to pin those indices in q_sent.
    frozen_q: dict[int, float] = {}
    if not args.no_ik:
        from ik.ik_solver import IKConfig, IKSolver  # type: ignore[reportMissingImports]
        ik_cfg    = IKConfig.from_yaml(args.ik_config)
        ik_solver = IKSolver(ik_cfg)
        body_map  = {t.name: t for t in ik_cfg.targets}
        for jname, val in ik_cfg.frozen_leg_joints.items():
            idx = MUJOCO_JOINT_TO_IDX.get(jname)
            if idx is not None:
                frozen_q[idx] = val

    # ── Passive joint indices ─────────────────────────────────────────────────
    # Loaded from passive_joints.yaml — single source of truth shared with the
    # C++ states (JointCmd, FixSit).  These joints are held at q_current each
    # tick so the controller never drives them back to q_default.
    passive_indices: list[int] = []
    if os.path.exists(PASSIVE_JOINTS_PATH):
        _pj = yaml.safe_load(open(PASSIVE_JOINTS_PATH))
        passive_indices = [idx for idx in _pj.get("passive_joints", []) if 0 <= idx < 29]

    # ── ZMQ sockets ──────────────────────────────────────────────────────────
    ctx = zmq.Context()

    in_sock = ctx.socket(zmq.SUB)
    in_sock.bind(args.in_address)
    in_sock.setsockopt_string(zmq.SUBSCRIBE, "")

    out_sock = ctx.socket(zmq.PUB)
    out_sock.bind(args.out_address)

    fb_sock = ctx.socket(zmq.SUB)
    fb_sock.connect(args.feedback)
    fb_sock.setsockopt_string(zmq.SUBSCRIBE, "")

    # ── Startup banner ───────────────────────────────────────────────────────
    print("=" * 60)
    print("[arm_cmd] G1 unified motion commander")
    print(f"  in  (commands)  : {args.in_address}")
    print(f"  out (controller): {args.out_address}")
    print(f"  feedback        : {args.feedback}")
    print(f"  loop rate       : {args.hz} Hz")
    print(f"  gravity comp    : {'enabled' if gc else 'disabled'}")
    print(f"  IK              : {'enabled' if ik_solver else 'disabled'}")
    if ik_solver is not None and ik_cfg is not None:
        print(f"  IK config       : {args.ik_config}")
        print(f"  IK active joints: {ik_cfg.active_joints}")
        print(f"  IK targets      : {list(body_map.keys())}")
    print(f"  Inspire hands   : {'DDS (' + args.hand_interface + ')' if hand_ctrl else 'disabled'}")
    if passive_indices:
        passive_names = [n for n, i in MUJOCO_JOINT_TO_IDX.items() if i in passive_indices]
        print(f"  passive joints  : {passive_names} → held at q_current each tick")
    if locked_joints:
        locked_names = {n: locked_joints[i] for n, i in MUJOCO_JOINT_TO_IDX.items() if i in locked_joints}
        print(f"  locked joints   : {locked_names} → pinned unconditionally")
    print("=" * 60)
    if hand_ctrl is None:
        print("\033[91m[arm_cmd] WARNING: hand commands will be DROPPED — pass --hand-interface <eth> to enable\033[0m")
        print("\033[91m          To enable: ssh unitree@192.168.123.164 then run:\033[0m")
        print("\033[91m            sudo /home/unitree/dfx_inspire_service/build/inspire_g1\033[0m")
        print("\033[91m          then restart arm_cmd.py with --hand-interface <eth>\033[0m")
    print("[arm_cmd] waiting 500 ms for subscribers to connect…")
    time.sleep(0.5)
    print("[arm_cmd] running — Ctrl-C to stop")

    # ── Control loop ─────────────────────────────────────────────────────────
    period    = 1.0 / args.hz
    next_t    = time.monotonic()
    q_desired = q_default[:]
    for idx, val in frozen_q.items():
        q_desired[idx] = val
    gravity_torque = [0.0] * 29
    q_current = None
    warm_start = None
    hand_desired = [1.0] * 12  # fully open; [R0..R5, L0..L5], range 0.0-1.0
    last_fb_time     = None   # monotonic time of last received feedback; None = never received
    last_mode_warn_t = 0.0

    try:
        while True:
            now = time.monotonic()

            # 1. Drain feedback → update q_current
            raw_fb = drain_latest(fb_sock)
            if raw_fb is not None:
                try:
                    q_current    = json.loads(raw_fb)["q_current"]
                    last_fb_time = now
                except (KeyError, json.JSONDecodeError):
                    pass
            else:
                if (last_fb_time is not None
                        and now - last_fb_time > FEEDBACK_TIMEOUT_S
                        and now - last_mode_warn_t > WARN_COOLDOWN_S):
                    print(f"\033[93m[arm_cmd] WARN: no feedback for {now - last_fb_time:.1f}s — "
                          "C++ controller is likely NOT in JointCmd mode "
                          "(activate with keyboard '4' or joystick LT+B)\033[0m")
                    last_mode_warn_t = now

            # 1b. Hold passive joints at their current position so the
            # controller doesn't drive them back to q_default.
            if q_current is not None:
                for i in passive_indices:
                    q_desired[i] = q_current[i]

            # 2. Drain in_socket → take latest command
            raw_cmd = drain_latest(in_sock)

            # 3. Process command
            if raw_cmd is not None:
                try:
                    cmd = json.loads(raw_cmd)
                    ctype = cmd.get("type", "joints")

                    if ctype == "joints":
                        q_new = cmd["q"]
                        if len(q_new) != 29:
                            print(f"[arm_cmd] WARN: joints command has {len(q_new)} values, expected 29 — ignored")
                        else:
                            q_desired = list(map(float, q_new))

                    elif ctype == "cartesian":
                        if ik_solver is None:
                            print("[arm_cmd] WARN: cartesian command received but IK is disabled — ignored")
                        else:
                            targets_raw = cmd.get("targets", {})
                            ik_targets = []
                            for tname, val in targets_raw.items():
                                if tname not in body_map:
                                    print(f"[arm_cmd] WARN: unknown target '{tname}' — ignored")
                                    continue
                                proto = body_map[tname]
                                if isinstance(val, dict):
                                    pos  = np.array(val["pos"],  dtype=float)
                                    quat = np.array(val["quat"], dtype=float) if "quat" in val else proto.orientation
                                    if quat is not None:
                                        norm = np.linalg.norm(quat)
                                        if norm > 0:
                                            quat = quat / norm
                                    ik_targets.append(dataclasses.replace(proto, position=pos, orientation=quat))
                                else:
                                    # backward compat: val is [x, y, z]
                                    ik_targets.append(dataclasses.replace(
                                        proto, position=np.array(val, dtype=float)
                                    ))
                            if ik_targets:
                                # Re-anchor warm-start to live q_current on every
                                # solve. Chaining from result.qpos drifts from
                                # reality if the controller rejects a cmd
                                # (e.g. C++ DISCARD): the IK keeps starting from
                                # its own divergent pose and produces ever-larger
                                # joint jumps. Anchoring to q_current keeps the
                                # IK in the local kinematic branch.
                                if q_current is not None:
                                    if warm_start is None:
                                        print("[arm_cmd] IK warm-started from live q_current")
                                    warm_start = _qpos_from_q_current(ik_solver, ik_cfg, q_current)
                                result = ik_solver.solve(targets=ik_targets, warm_start=warm_start)
                                # Merge IK solution into q_desired
                                for jname, angle in result.joint_angles.items():
                                    idx = MUJOCO_JOINT_TO_IDX.get(jname)
                                    if idx is not None:
                                        q_desired[idx] = angle

                    else:
                        print(f"[arm_cmd] WARN: unknown command type '{ctype}' — ignored")

                    # Optional hand command — valid for any ctype
                    if "hand" in cmd:
                        hand_raw = cmd["hand"]
                        if len(hand_raw) == 12:
                            hand_desired = [float(max(0.0, min(1.0, x))) for x in hand_raw]
                        else:
                            print(f"[arm_cmd] WARN: 'hand' has {len(hand_raw)} values, expected 12 — ignored")

                except (KeyError, json.JSONDecodeError, ValueError) as e:
                    print(f"[arm_cmd] WARN: malformed command — {e}")

            # 4. Gravity feedforward (qfrc_bias at q_current, qvel=0)
            if gc is not None and q_current is not None:
                gravity_torque = gc.compute(q_current)

            # 5. Build outgoing q_sent.  Pure user intent — no kp-dependent offset.
            # Frozen joints (legs + any frozen upper-body joints from IK config)
            # pin to their frozen_q values; locked joints override everything.
            q_sent = [
                frozen_q[i] if i in frozen_q else q_desired[i]
                for i in range(29)
            ]
            for idx, val in locked_joints.items():
                if 0 <= idx < 29:
                    q_sent[idx] = val

            # 5a. Build tau_ff. Mask out joints we don't want to gravity-comp:
            #   - frozen (legs sitting on platform — gravity partly supported externally)
            #   - locked (rigid pin via q_des; tau_ff would just fight the entry ramp)
            #   - passive (semantically "limp"; user wants no torque produced)
            # Then clamp symmetrically to max_torque so a stale q_current can't
            # drive the FSA past ctrlrange.
            tau_ff_sent = [0.0] * 29
            for i in range(29):
                if i in frozen_q or i in locked_joints or i in passive_indices:
                    continue
                t = gravity_torque[i]
                if max_torque is not None:
                    t = max(-max_torque[i], min(max_torque[i], t))
                tau_ff_sent[i] = t

            # 5b. Send hand commands
            if hand_ctrl is not None:
                hand_ctrl.set_angles(hand_desired)

            # 6. Forward to controller
            out_sock.send_string(json.dumps({"q": q_sent, "tau_ff": tau_ff_sent}))

            # 7. Rate-limit sleep
            next_t += period
            sleep_s = next_t - now
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[arm_cmd] interrupted — shutting down")
    finally:
        in_sock.close()
        out_sock.close()
        fb_sock.close()
        ctx.term()
        if hand_ctrl is not None:
            hand_ctrl.close()
        print("[arm_cmd] done")


if __name__ == "__main__":
    main()
