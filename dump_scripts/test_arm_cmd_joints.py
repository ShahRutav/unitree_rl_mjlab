#!/usr/bin/env python3
"""
Throwaway test: joint pass-through mode for arm_cmd.py.

Sends a step waveform via {"type": "joints", "q": [...]} to arm_cmd.py on port 5557.
Oscillates a single joint between q_default[j] and q_default[j] + amp.
Flips phase every --dwell seconds (time-based, no convergence detection).

Usage:
    python3 dump_scripts/test_arm_cmd_joints.py --joint 25 --amp 0.4 --dwell 2.0
"""

import argparse
import json
import os
import time

import yaml
import zmq

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(REPO_ROOT, "deploy", "robots", "g1", "config", "joint_cmd.yaml")


def load_q_default():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    q = cfg["q_default"]
    assert len(q) == 29, f"q_default has {len(q)} joints, expected 29"
    return list(map(float, q))


def main():
    parser = argparse.ArgumentParser(description="Send joint step waveform to arm_cmd.py")
    parser.add_argument("--joint",   type=int,   default=25,
                        help="Joint index 0-28 to oscillate (default: 25 = R_elbow)")
    parser.add_argument("--amp",     type=float, default=0.4,
                        help="Amplitude in radians (default: 0.4)")
    parser.add_argument("--dwell",   type=float, default=2.0,
                        help="Seconds to hold each phase before flipping (default: 2.0)")
    parser.add_argument("--hz",      type=float, default=50.0,
                        help="Publish rate in Hz (default: 50)")
    parser.add_argument("--address", default="tcp://localhost:5557",
                        help="ZMQ connect address for arm_cmd.py (default: tcp://localhost:5557)")
    args = parser.parse_args()

    q_default = load_q_default()

    q_a = q_default[:]                  # phase A: default pose
    q_b = q_default[:]                  # phase B: default + amp on chosen joint
    q_b[args.joint] += args.amp

    phases     = [q_b, q_a]
    phase_names = ["B (target = default + amp)", "A (default)"]
    phase_idx  = 0

    ctx    = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.connect(args.address)

    print(f"[test_joints] connected to {args.address}")
    print(f"[test_joints] joint={args.joint}  amp={args.amp} rad  dwell={args.dwell} s  hz={args.hz}")
    print(f"[test_joints] sleeping 500 ms for SUB side to register…")
    time.sleep(0.5)

    print(f"[test_joints] starting phase {phase_names[phase_idx]}")

    period     = 1.0 / args.hz
    next_t     = time.monotonic()
    phase_end  = time.monotonic() + args.dwell

    try:
        while True:
            now = time.monotonic()

            if now >= phase_end:
                phase_idx  = 1 - phase_idx
                phase_end  = now + args.dwell
                print(f"[test_joints] → phase {phase_names[phase_idx]}")

            q = phases[phase_idx]
            socket.send_string(json.dumps({"type": "joints", "q": q}))

            next_t  += period
            sleep_s  = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[test_joints] stopped.")
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
