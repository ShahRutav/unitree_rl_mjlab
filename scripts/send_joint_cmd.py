#!/usr/bin/env python3
"""
JointCmd test sender.

Each mode oscillates (sine) or steps (step) specific joints around the robot's
default hold pose (read from joint_cmd.yaml — always in sync).
All other joints are held at their default values.

Waveforms
---------
  step (default)
      Sends pose-A until the robot converges (max joint error < --threshold),
      dwells for --dwell seconds, then flips to pose-B, and so on.
      Uses the ZMQ feedback publisher (port 5556) for convergence detection.
      Applies model-based gravity compensation (qfrc_bias/Kp) to cancel
      the PD steady-state error instantly, plus a small ki integrator for
      any residual model/hardware mismatch.

  sine
      Continuously oscillates the joints sinusoidally.  The robot will lag
      behind at high frequencies — useful for seeing bandwidth limits.

Modes (right arm + waist focus)
--------------------------------
  waist_yaw        joint 12   waist_roll       joint 13   waist_pitch      joint 14
  waist            joints 12-14

  r_shoulder_pitch joint 22   r_shoulder_roll  joint 23   r_shoulder_yaw   joint 24
  r_elbow          joint 25
  r_wrist_roll     joint 26   r_wrist_pitch    joint 27   r_wrist_yaw      joint 28
  r_arm            joints 22-28

Workflow
--------
  Terminal 1:  ./simulate/build/unitree_mujoco -s src/assets/robots/unitree_g1/xmls/g1_sitting.xml
  Terminal 2:  ./deploy/robots/g1/build/g1_ctrl -n lo   then press [4]
  Terminal 3:  python3 scripts/send_joint_cmd.py --mode r_elbow
  Terminal 4:  python3 scripts/plot_joint_error.py --joints 25
"""

import argparse
import json
import math
import os
import time

import yaml
import zmq

# ---------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(REPO_ROOT, "deploy", "robots", "g1", "config", "joint_cmd.yaml")
XML_PATH    = os.path.join(REPO_ROOT, "src", "assets", "robots",
                           "unitree_g1", "xmls", "g1_sitting.xml")

JOINT_NAMES = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",   #  0-5
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",   #  6-11
    "waist_yaw", "waist_roll", "waist_pitch",                                               # 12-14
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",                    # 15-18
    "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",                                        # 19-21
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",                    # 22-25
    "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw",                                        # 26-28
]

MODES = {
    # waist
    "waist_yaw":        [12],
    "waist_roll":       [13],
    "waist_pitch":      [14],
    "waist":            [12, 13, 14],
    # right arm — individual joints
    "r_shoulder_pitch": [22],
    "r_shoulder_roll":  [23],
    "r_shoulder_yaw":   [24],
    "r_elbow":          [25],
    "r_wrist_roll":     [26],
    "r_wrist_pitch":    [27],
    "r_wrist_yaw":      [28],
    # right arm — full
    "r_arm":            [22, 23, 24, 25, 26, 27, 28],
}


def load_q_default():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    q = cfg["q_default"]
    assert len(q) == 29, f"q_default has {len(q)} joints, expected 29"
    return list(map(float, q))


# ---------------------------------------------------------------------------
# Gravity compensator
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
# Step waveform
# ---------------------------------------------------------------------------

def run_step(socket, fb_socket, joints, q_default, gc, args):
    """Hold pose-A until converged, dwell, flip to pose-B, repeat.

    Gravity compensation (model-based, instant) + small ki integrator for
    residual model/hardware mismatch.  Both corrections reset on phase flip.
    """
    q_a = q_default[:]
    q_b = q_default[:]
    for j in joints:
        q_b[j] += args.amp

    phases      = [q_b, q_a]
    phase_tags  = ["→ B (target)", "→ A (default)"]
    phase_idx   = 0
    q_desired   = phases[0]
    converged_at = None

    correction     = [0.0] * 29   # ki integrator correction (resets on flip)
    gravity_offset = [0.0] * 29   # model-based gravity offset (updates each cycle)
    dt             = 1.0 / args.hz

    joint_str = ", ".join(f"{j}={JOINT_NAMES[j]}" for j in joints)
    print(f"[step] joints      : {joint_str}")
    print(f"[step] amplitude   : ±{args.amp} rad")
    print(f"[step] threshold   : {args.threshold} rad  dwell: {args.dwell} s  pub: {args.hz} Hz")
    print(f"[step] gravity comp: {'enabled' if gc else 'disabled'}")
    print(f"[step] ki={args.ki}  clamp=±{args.correction_clamp} rad")
    print(f"[step] starting phase B (target)…")

    period = 1.0 / args.hz
    next_t = time.monotonic()

    try:
        while True:
            # Drain all pending feedback messages, keep latest q_current
            q_current = None
            try:
                while True:
                    raw = fb_socket.recv_string(flags=zmq.NOBLOCK)
                    data = json.loads(raw)
                    q_current = data["q_current"]
            except zmq.Again:
                pass

            if q_current is not None:
                # Model-based gravity compensation (pose-aware, instant)
                if gc is not None:
                    gravity_offset = gc.compute(q_current)

                # Residual ki integrator for model/hardware mismatch
                if args.ki > 0.0:
                    for i in range(29):
                        correction[i] += args.ki * (q_desired[i] - q_current[i]) * dt
                        correction[i] = max(-args.correction_clamp,
                                            min(args.correction_clamp, correction[i]))

            q_sent = [q_desired[i] + gravity_offset[i] + correction[i] for i in range(29)]
            socket.send_string(json.dumps({"q": q_sent}))

            # Convergence check against the true desired pose (not the corrected command)
            if q_current is not None:
                max_err = max(abs(q_current[j] - q_desired[j]) for j in joints)

                if max_err < args.threshold:
                    if converged_at is None:
                        converged_at = time.monotonic()
                        corr_str = "  ".join(
                            f"{JOINT_NAMES[j]} err={q_current[j]-q_desired[j]:+.4f} "
                            f"grav={gravity_offset[j]:+.4f} ki={correction[j]:+.4f}"
                            for j in joints
                        )
                        print(f"[step] converged  max_err={max_err:.4f} rad  [{corr_str}]")

                    elif time.monotonic() - converged_at >= args.dwell:
                        phase_idx      = 1 - phase_idx
                        q_desired      = phases[phase_idx]
                        correction     = [0.0] * 29    # reset on phase flip
                        gravity_offset = [0.0] * 29    # will recompute next cycle
                        converged_at   = None
                        print(f"[step] switching {phase_tags[phase_idx]}")
                else:
                    converged_at = None  # reset if drifted away

            # Rate-limit publish
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[step] stopped.")


# ---------------------------------------------------------------------------
# Sine waveform (original)
# ---------------------------------------------------------------------------

def run_sine(socket, joints, q_default, args):
    """Continuously oscillate joints sinusoidally around the default pose."""
    joint_str = ", ".join(f"{j}={JOINT_NAMES[j]} (base={q_default[j]:.3f})" for j in joints)
    print(f"[sine] joints   : {joint_str}")
    print(f"[sine] amplitude: ±{args.amp} rad   freq: {args.freq} Hz   pub: {args.hz} Hz")

    period  = 1.0 / args.hz
    next_t  = time.monotonic()
    t_start = time.monotonic()

    try:
        while True:
            t_now = time.monotonic() - t_start
            swing = args.amp * math.sin(2 * math.pi * args.freq * t_now)

            q = q_default[:]
            for j in joints:
                q[j] += swing

            socket.send_string(json.dumps({"q": q}))

            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[sine] stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Send G1 joint commands (step or sine waveform)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Modes: " + "  ".join(MODES.keys()),
    )
    parser.add_argument("--mode",     default="r_elbow", choices=list(MODES.keys()),
                        help="Which joint(s) to move (default: r_elbow)")
    parser.add_argument("--waveform", default="step", choices=["step", "sine"],
                        help="step: hold until converged then flip; sine: continuous (default: step)")
    parser.add_argument("--amp",      type=float, default=0.3,
                        help="Amplitude in radians (default: 0.3)")
    parser.add_argument("--hz",       type=float, default=50.0,
                        help="Publish rate in Hz (default: 50)")
    # step-only args
    parser.add_argument("--threshold",       type=float, default=0.05,
                        help="[step] convergence threshold in rad (default: 0.05)")
    parser.add_argument("--dwell",           type=float, default=0.5,
                        help="[step] seconds to hold after converging before flipping (default: 0.5)")
    parser.add_argument("--feedback",        default="tcp://localhost:5556",
                        help="[step] ZMQ feedback address (default: tcp://localhost:5556)")
    parser.add_argument("--ki",              type=float, default=0.00,
                        help="[step] residual integrator gain (default: 0.05; 0=off)")
    parser.add_argument("--correction-clamp", type=float, default=0.3,
                        help="[step] max integrator correction in rad (default: 0.3)")
    parser.add_argument("--gravity-comp", action="store_true",
                        help="[step] enable model-based gravity compensation (default: disabled)")
    # sine-only args
    parser.add_argument("--freq",     type=float, default=0.5,
                        help="[sine] oscillation frequency in Hz (default: 0.5)")
    # addresses
    parser.add_argument("--address",  default="tcp://*:5555",
                        help="ZMQ bind address for commands (default: tcp://*:5555)")
    args = parser.parse_args()

    joints    = MODES[args.mode]
    cfg       = yaml.safe_load(open(CONFIG_PATH))
    q_default = list(map(float, cfg["q_default"]))
    assert len(q_default) == 29, f"q_default has {len(q_default)} joints, expected 29"
    use_gc    = cfg.get("gravity_comp", False) or args.gravity_comp

    ctx    = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.bind(args.address)

    print(f"[send_joint_cmd] mode={args.mode}  waveform={args.waveform}")
    print(f"[send_joint_cmd] bound to {args.address}, waiting 500 ms for subscriber…")
    time.sleep(0.5)

    try:
        if args.waveform == "step":
            fb_socket = ctx.socket(zmq.SUB)
            fb_socket.connect(args.feedback)
            fb_socket.setsockopt_string(zmq.SUBSCRIBE, "")

            gc = None
            if use_gc:
                gc = GravityCompensator(XML_PATH, cfg["kp"])
                print(f"[send_joint_cmd] gravity comp: enabled (xml={XML_PATH})")

            run_step(socket, fb_socket, joints, q_default, gc, args)
            fb_socket.close()
        else:
            run_sine(socket, joints, q_default, args)
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
