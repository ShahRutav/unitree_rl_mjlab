#!/usr/bin/env python3
"""
Throwaway test: Cartesian streaming mode for arm_cmd.py.

Streams a slow circular trajectory via {"type": "cartesian", "targets": {"right_eef": [x,y,z]}}
to arm_cmd.py on port 5557.  Circle is traced in the Y-Z plane around a center point.

Usage:
    python3 dump_scripts/test_arm_cmd_cartesian.py
    python3 dump_scripts/test_arm_cmd_cartesian.py --radius 0.05 --period 12.0
"""

import argparse
import json
import math
import time

import zmq


def main():
    parser = argparse.ArgumentParser(description="Stream Cartesian circle trajectory to arm_cmd.py")
    parser.add_argument("--center", type=float, nargs=3, default=[0.45, -0.25, 0.85],
                        metavar=("X", "Y", "Z"),
                        help="Circle center x y z (default: 0.45 -0.25 0.85)")
    parser.add_argument("--radius", type=float, default=0.08,
                        help="Circle radius in metres (default: 0.08)")
    parser.add_argument("--period", type=float, default=8.0,
                        help="Seconds for one full circle (default: 8.0)")
    parser.add_argument("--hz",     type=float, default=10.0,
                        help="Publish rate in Hz (default: 10)")
    parser.add_argument("--address", default="tcp://localhost:5557",
                        help="ZMQ connect address for arm_cmd.py (default: tcp://localhost:5557)")
    args = parser.parse_args()

    cx, cy, cz = args.center

    ctx    = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.connect(args.address)

    print(f"[test_cartesian] connected to {args.address}")
    print(f"[test_cartesian] center=({cx:.3f}, {cy:.3f}, {cz:.3f})  "
          f"radius={args.radius} m  period={args.period} s  hz={args.hz}")
    print(f"[test_cartesian] sleeping 500 ms for SUB side to register…")
    time.sleep(0.5)

    print("[test_cartesian] streaming circle in Y-Z plane…")

    period    = 1.0 / args.hz
    next_t    = time.monotonic()
    t_start   = time.monotonic()
    last_print = t_start - 1.0   # ensure first-tick print

    try:
        while True:
            t_now  = time.monotonic() - t_start
            angle  = 2.0 * math.pi * t_now / args.period

            x = cx
            y = cy + args.radius * math.cos(angle)
            z = cz + args.radius * math.sin(angle)

            msg = {"type": "cartesian", "targets": {"right_eef": [x, y, z]}}
            socket.send_string(json.dumps(msg))

            if time.monotonic() - last_print >= 1.0:
                print(f"[test_cartesian] t={t_now:6.1f}s  target=({x:.4f}, {y:.4f}, {z:.4f})")
                last_print = time.monotonic()

            next_t  += period
            sleep_s  = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[test_cartesian] stopped.")
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
