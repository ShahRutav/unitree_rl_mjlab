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

Resolves cartesian commands via warm-started differential IK, applies
model-based gravity compensation, and forwards the corrected 29-vector
to the C++ controller via ZMQ PUB on port 5555.

Usage
-----
  python3 scripts/arm_cmd.py
  python3 scripts/arm_cmd.py --ik-config ik/configs/g1_right_arm.yaml --hz 50
  python3 scripts/arm_cmd.py --no-gravity-comp
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

CONFIG_PATH = os.path.join(REPO_ROOT, "deploy", "robots", "g1", "config", "joint_cmd.yaml")
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
    """Computes per-joint position offset = qfrc_bias / Kp to cancel gravity error.

    With qvel=0, MuJoCo's qfrc_bias is the pure gravity torque at each joint.
    Dividing by Kp gives the position offset the PD controller needs to hold
    the joint against gravity — commanding q_desired + offset results in
    q_actual ≈ q_desired with near-zero steady-state error.

    Joint ordering in g1_sitting.xml matches the controller order exactly:
      left_leg[0-5], right_leg[6-11], waist[12-14], left_arm[15-21], right_arm[22-28]
    """

    def __init__(self, xml_path: str, kp: list):
        import mujoco
        self._mujoco = mujoco
        self.model   = mujoco.MjModel.from_xml_path(xml_path)
        self.data    = mujoco.MjData(self.model)
        self.data.qvel[:] = 0.0   # static — pure gravity, no Coriolis/centrifugal
        self.kp = kp
        assert self.model.nq == 29, f"Expected 29 DOF, got {self.model.nq}"

    def compute(self, q_current: list) -> list:
        """Return per-joint position offsets that cancel gravity-induced steady-state error."""
        self.data.qpos[:] = q_current
        self._mujoco.mj_forward(self.model, self.data)
        return [float(self.data.qfrc_bias[i]) / self.kp[i] for i in range(29)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_q_default() -> list:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    q = cfg["q_default"]
    assert len(q) == 29, f"q_default has {len(q)} joints, expected 29"
    return list(map(float, q))


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
    parser.add_argument("--hz",          type=float, default=50.0,
                        help="Control loop rate in Hz (default: 50.0)")
    parser.add_argument("--no-gravity-comp", action="store_true",
                        help="Disable model-based gravity compensation")
    parser.add_argument("--no-ik",       action="store_true",
                        help="Disable IK (cartesian commands will be rejected)")
    parser.add_argument("--hand-interface", default=None,
                        help="Network interface for Inspire hand DDS (e.g. eth0). "
                             "Requires inspire_g1 service running on the robot.")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg_yaml = yaml.safe_load(open(CONFIG_PATH))
    q_default = load_q_default()

    # ── Inspire hands (DDS) ──────────────────────────────────────────────────
    hand_ctrl = InspireHandDDS(args.hand_interface) if args.hand_interface else None

    # ── Gravity compensator ──────────────────────────────────────────────────
    gc = None
    if not args.no_gravity_comp:
        gc = GravityCompensator(XML_PATH, cfg_yaml["kp"])

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

    # ── Passive joint indices (non-frozen, non-IK-active) ────────────────────
    # When IK is enabled, these joints are held at their current position each
    # tick to prevent the controller from driving them back to q_default.
    ik_active_indices: set[int] = set()
    if ik_solver is not None and ik_cfg is not None:
        for jname in ik_cfg.active_joints:
            idx = MUJOCO_JOINT_TO_IDX.get(jname)
            if idx is not None:
                ik_active_indices.add(idx)

    passive_indices: list[int] = (
        [i for i in range(29) if i not in frozen_q and i not in ik_active_indices]
        if ik_solver is not None else []
    )

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
    gravity_offset = [0.0] * 29
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
                                result = ik_solver.solve(targets=ik_targets, warm_start=warm_start)
                                warm_start = result.qpos   # warm-start next call
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

            # 4. Gravity compensation (upper body only — waist + arms, indices 12+)
            if gc is not None and q_current is not None:
                gravity_offset = gc.compute(q_current)

            # 5. Build outgoing command.
            # Frozen joints (legs + any frozen upper-body joints from IK config) are
            # pinned to their frozen_q values — no gravity comp, no command override.
            # All other upper-body joints get gravity compensation applied.
            q_sent = [
                frozen_q[i] if i in frozen_q else q_desired[i] + gravity_offset[i]
                for i in range(29)
            ]

            # 5b. Send hand commands
            if hand_ctrl is not None:
                hand_ctrl.set_angles(hand_desired)

            # 6. Forward to controller
            out_sock.send_string(json.dumps({"q": q_sent}))

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
