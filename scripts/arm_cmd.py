#!/usr/bin/env python3
"""
arm_cmd.py — Unified motion commander for the G1 arm.

Binds a ZMQ SUB socket on port 5557 to receive high-level commands:
  {"type": "joints",    "q": [29 floats]}
  {"type": "cartesian", "targets": {"right_eef": [x, y, z], ...}}

Resolves cartesian commands via warm-started differential IK, applies
model-based gravity compensation, and forwards the corrected 29-vector
to the C++ controller via ZMQ PUB on port 5555.

Usage
-----
  python3 scripts/arm_cmd.py
  python3 scripts/arm_cmd.py --ik-config ik/configs/g1_right_arm.yaml --hz 50
  python3 scripts/arm_cmd.py --no-gravity-comp
  python3 scripts/arm_cmd.py --no-ik
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
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg_yaml = yaml.safe_load(open(CONFIG_PATH))
    q_default = load_q_default()

    # ── Gravity compensator ──────────────────────────────────────────────────
    gc = None
    if not args.no_gravity_comp:
        gc = GravityCompensator(XML_PATH, cfg_yaml["kp"])

    # ── IK solver ────────────────────────────────────────────────────────────
    ik_solver = None
    ik_cfg    = None
    body_map  = {}   # target name → TargetFrame (from config)
    if not args.no_ik:
        from ik.ik_solver import IKConfig, IKSolver  # type: ignore[reportMissingImports]
        ik_cfg    = IKConfig.from_yaml(args.ik_config)
        ik_solver = IKSolver(ik_cfg)
        body_map  = {t.name: t for t in ik_cfg.targets}

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
    print("=" * 60)
    print("[arm_cmd] waiting 500 ms for subscribers to connect…")
    time.sleep(0.5)
    print("[arm_cmd] running — Ctrl-C to stop")

    # ── Control loop ─────────────────────────────────────────────────────────
    period    = 1.0 / args.hz
    next_t    = time.monotonic()
    q_desired = q_default[:]
    gravity_offset = [0.0] * 29
    q_current = None
    warm_start = None

    try:
        while True:
            # 1. Drain feedback → update q_current
            raw_fb = drain_latest(fb_sock)
            if raw_fb is not None:
                try:
                    q_current = json.loads(raw_fb)["q_current"]
                except (KeyError, json.JSONDecodeError):
                    pass

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
                            for tname, pos in targets_raw.items():
                                if tname not in body_map:
                                    print(f"[arm_cmd] WARN: unknown target '{tname}' — ignored")
                                    continue
                                proto = body_map[tname]
                                # replace only the position; keep body/orientation/weights from config
                                ik_targets.append(dataclasses.replace(
                                    proto, position=np.array(pos, dtype=float)
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

                except (KeyError, json.JSONDecodeError, ValueError) as e:
                    print(f"[arm_cmd] WARN: malformed command — {e}")

            # 4. Gravity compensation
            if gc is not None and q_current is not None:
                gravity_offset = gc.compute(q_current)

            # 5. Build outgoing command
            q_sent = [q_desired[i] + gravity_offset[i] for i in range(29)]

            # 6. Forward to controller
            out_sock.send_string(json.dumps({"q": q_sent}))

            # 7. Rate-limit sleep
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[arm_cmd] interrupted — shutting down")
    finally:
        in_sock.close()
        out_sock.close()
        fb_sock.close()
        ctx.term()
        print("[arm_cmd] done")


if __name__ == "__main__":
    main()
