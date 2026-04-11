#!/usr/bin/env python3
"""
Live rolling plot of joint tracking error: q_current - q_target

Subscribes to the ZMQ feedback PUB published by State_JointCmd (port 5556).
Maintains a rolling buffer of the last N samples and plots error per joint.

Usage:
  # all 29 joints (busy but complete)
  python3 scripts/plot_joint_error.py

  # right elbow only
  python3 scripts/plot_joint_error.py --joints 25

  # full right arm
  python3 scripts/plot_joint_error.py --joints 22 23 24 25 26 27 28

  # waist + right arm, last 5000 samples, fixed y-axis
  python3 scripts/plot_joint_error.py --joints 12 13 14 22 23 24 25 26 27 28 \\
                                      --buffer 5000 --yrange -0.5 0.5

  # save samples to CSV as well
  python3 scripts/plot_joint_error.py --joints 25 --csv /tmp/elbow_error.csv
"""

import argparse
import collections
import csv
import json
import sys
import time

import zmq
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------

JOINT_NAMES = [
    "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",   #  0-5
    "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",   #  6-11
    "waist_yaw", "waist_roll", "waist_pitch",                                               # 12-14
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",                    # 15-18
    "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",                                        # 19-21
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",                    # 22-25
    "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw",                                        # 26-28
]

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live joint tracking error plot (q_current - q_target)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--joints", type=int, nargs="+",
        default=[12, 13, 14, 22, 23, 24, 25, 26, 27, 28],
        metavar="J",
        help="Joint indices to plot 0-28 (default: waist + right arm, indices 12-14 and 22-28)",
    )
    parser.add_argument(
        "--buffer", type=int, default=10000,
        help="Rolling window size in samples (default: 10000)",
    )
    parser.add_argument(
        "--hz", type=float, default=30.0,
        help="Plot refresh rate in Hz (default: 30)",
    )
    parser.add_argument(
        "--address", default="tcp://localhost:5556",
        help="ZMQ address to connect to (default: tcp://localhost:5556)",
    )
    parser.add_argument(
        "--yrange", type=float, nargs=2, metavar=("YMIN", "YMAX"),
        help="Fixed y-axis range in radians (default: auto-scale)",
    )
    parser.add_argument(
        "--csv", metavar="FILE",
        help="Also write samples to CSV file (appends)",
    )
    args = parser.parse_args()

    joints = [j for j in args.joints if 0 <= j < 29]
    if not joints:
        print("No valid joint indices (must be 0-28).", file=sys.stderr)
        sys.exit(1)

    n = len(joints)
    print(f"[plot_joint_error] joints: {[JOINT_NAMES[j] for j in joints]}")
    print(f"[plot_joint_error] buffer={args.buffer}  hz={args.hz}  address={args.address}")

    # ZMQ subscriber -----------------------------------------------------------
    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.connect(args.address)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # Rolling buffers per joint ------------------------------------------------
    t_bufs   = [collections.deque(maxlen=args.buffer) for _ in joints]
    err_bufs = [collections.deque(maxlen=args.buffer) for _ in joints]
    t_start  = time.monotonic()

    # Optional CSV writer ------------------------------------------------------
    csv_file   = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["time_s", "tick"] + [f"err_{JOINT_NAMES[j]}" for j in joints])
        print(f"[plot_joint_error] writing CSV → {args.csv}")

    # Plot setup ---------------------------------------------------------------
    colors = [plt.cm.tab20((i % 20) / 20.0) for i in range(n)]

    fig, ax = plt.subplots(figsize=(13, 5))
    lines = []
    for i, j in enumerate(joints):
        (ln,) = ax.plot([], [], color=colors[i],
                        label=f"{j}: {JOINT_NAMES[j]}", lw=1.2)
        lines.append(ln)

    ax.axhline(0.0, color="black", lw=0.6, linestyle="--", zorder=0, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error  (rad)")
    ax.set_title("Joint Tracking Error  (q_current − q_target)")

    if args.yrange:
        ax.set_ylim(args.yrange)

    ncol = max(1, n // 12)
    ax.legend(loc="upper right", fontsize=6, ncol=ncol)
    fig.tight_layout()

    # Animation ----------------------------------------------------------------
    def update(_frame):
        # Drain all pending ZMQ messages (non-blocking)
        received = 0
        try:
            while True:
                raw  = socket.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(raw)
                t_now = time.monotonic() - t_start
                q_c   = data["q_current"]
                q_g   = data["q_target"]
                tick  = data.get("tick", -1)

                row_errs = []
                for bi, j in enumerate(joints):
                    err = q_c[j] - q_g[j]
                    t_bufs[bi].append(t_now)
                    err_bufs[bi].append(err)
                    row_errs.append(err)

                if csv_writer is not None:
                    csv_writer.writerow([f"{t_now:.4f}", tick] +
                                        [f"{e:.6f}" for e in row_errs])

                received += 1
                if received >= 1000:   # yield so the plot doesn't freeze
                    break
        except zmq.Again:
            pass

        # Update plot lines
        for i, ln in enumerate(lines):
            xs = list(t_bufs[i])
            ys = list(err_bufs[i])
            ln.set_data(xs, ys)

        # Slide x-axis to cover the current data window
        if any(t_bufs[i] for i in range(n)):
            x_min = min(t_bufs[i][0]  for i in range(n) if t_bufs[i])
            x_max = max(t_bufs[i][-1] for i in range(n) if t_bufs[i])
            ax.set_xlim(x_min, max(x_max, x_min + 1.0))

        if not args.yrange:
            ax.relim()
            ax.autoscale_view(scalex=False)

        return lines

    ani = animation.FuncAnimation(   # noqa: F841 (kept alive by reference)
        fig, update,
        interval=1000.0 / args.hz,
        blit=False,
        cache_frame_data=False,
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if csv_file is not None:
            csv_file.close()
            print(f"[plot_joint_error] CSV closed: {args.csv}")
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
