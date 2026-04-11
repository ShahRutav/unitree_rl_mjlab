#!/usr/bin/env python3
"""
Live rolling plot of joint tracking error: q_current - q_target

Three-panel layout with shared time axis:
  - Legs     (indices 0-11): Left leg solid blue tones, Right leg dashed red/orange tones
  - Waist    (indices 12-14)
  - Right Arm (indices 22-28)

Usage:
  python3 scripts/plot_joint_error.py                        # all three panels
  python3 scripts/plot_joint_error.py --groups arm           # right arm only
  python3 scripts/plot_joint_error.py --groups legs arm      # legs + arm, no waist
  python3 scripts/plot_joint_error.py --buffer 5000 --yrange -0.5 0.5
  python3 scripts/plot_joint_error.py --groups arm --csv /tmp/arm_error.csv
"""

import argparse
import collections
import csv
import json
import sys
import time

import zmq
import numpy as np
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

# Per-group color + linestyle definitions ------------------------------------
# Left leg:  solid, blue family
# Right leg: dashed, red/orange family
_LEFT_LEG_COLORS  = ["#1f77b4", "#17becf", "#0a6fc2", "#00aacc", "#005fa3", "#00c8c8"]
_RIGHT_LEG_COLORS = ["#d62728", "#ff7f0e", "#b22222", "#ff4500", "#e08000", "#8b0000"]
_WAIST_COLORS     = ["#9467bd", "#8c564b", "#e377c2"]
_ARM_COLORS       = ["#2ca02c", "#98df8a", "#17becf", "#aec7e8", "#ff9896", "#f7b6d2", "#c5b0d5"]

GROUP_DEFS = {
    "legs": {
        "title": "Leg Joints  (solid = Left · dashed = Right)",
        "joints": list(range(12)),
        "colors": _LEFT_LEG_COLORS + _RIGHT_LEG_COLORS,
        "linestyles": ["solid"] * 6 + ["dashed"] * 6,
        "legend_ncol": 2,
    },
    "waist": {
        "title": "Waist Joints",
        "joints": [12, 13, 14],
        "colors": _WAIST_COLORS,
        "linestyles": ["solid"] * 3,
        "legend_ncol": 1,
    },
    "arm": {
        "title": "Right Arm Joints",
        "joints": [22, 23, 24, 25, 26, 27, 28],
        "colors": _ARM_COLORS,
        "linestyles": ["solid"] * 7,
        "legend_ncol": 1,
    },
}

ALL_GROUPS = ["legs", "waist", "arm"]

# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live joint tracking error plot — grouped by body segment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--groups", nargs="+", choices=ALL_GROUPS, default=ALL_GROUPS,
        metavar="GROUP",
        help=f"Body groups to show: {ALL_GROUPS} (default: all three)",
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
        "--address", default="tcp://localhost:5558",
        help="ZMQ address to connect to (default: tcp://localhost:5558)",
    )
    parser.add_argument(
        "--yrange", type=float, nargs=2, metavar=("YMIN", "YMAX"),
        help="Fixed y-axis range in radians applied to all panels (default: auto-scale)",
    )
    parser.add_argument(
        "--csv", metavar="FILE",
        help="Also write all plotted joint errors to a CSV file",
    )
    args = parser.parse_args()

    # Deduplicate --groups while preserving order
    seen = set()
    active_groups = [g for g in args.groups if not (g in seen or seen.add(g))]
    if not active_groups:
        print("No valid groups selected.", file=sys.stderr)
        sys.exit(1)

    # Build per-group runtime state ------------------------------------------
    groups = []
    for gname in active_groups:
        gdef = GROUP_DEFS[gname]
        joints = gdef["joints"]
        groups.append({
            **gdef,
            "name": gname,
            "t_bufs":   [collections.deque(maxlen=args.buffer) for _ in joints],
            "err_bufs": [collections.deque(maxlen=args.buffer) for _ in joints],
        })

    all_joint_indices = sorted({j for g in groups for j in g["joints"]})

    print(f"[plot_joint_error] groups  : {active_groups}")
    print(f"[plot_joint_error] buffer  : {args.buffer}  hz={args.hz}  address={args.address}")

    # ZMQ subscriber ---------------------------------------------------------
    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.connect(args.address)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    t_start = time.monotonic()

    # Optional CSV writer ----------------------------------------------------
    csv_file   = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["time_s", "tick"] + [f"err_{JOINT_NAMES[j]}" for j in all_joint_indices]
        )
        print(f"[plot_joint_error] CSV     : {args.csv}")

    # Figure & subplots (shared x-axis) --------------------------------------
    n_panels = len(groups)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(13, 4.0 * n_panels),
        sharex=True,
        squeeze=False,
    )
    axes  = [row[0] for row in axes]          # flatten (n,1) → list of n
    axes2 = [ax.twinx() for ax in axes]       # right-hand rad axis per panel

    all_lines = []
    for g, ax, ax2 in zip(groups, axes, axes2):
        lines = []
        for i, j in enumerate(g["joints"]):
            (ln,) = ax.plot(
                [], [],
                color=g["colors"][i],
                linestyle=g["linestyles"][i],
                label=f"{j}: {JOINT_NAMES[j]}",
                lw=1.2,
            )
            lines.append(ln)
        g["lines"] = lines
        all_lines.extend(lines)

        ax.axhline(0.0, color="black", lw=0.6, linestyle="--", zorder=0, alpha=0.4)
        ax.set_ylabel("Error  (deg)")
        ax2.set_ylabel("Error  (rad)", labelpad=4)
        ax.set_title(g["title"], fontsize=9, loc="left", pad=3)

        if args.yrange:
            ax2.set_ylim(args.yrange)
            ax.set_ylim(np.degrees(args.yrange[0]), np.degrees(args.yrange[1]))

        ax.legend(loc="upper right", fontsize=6, ncol=g["legend_ncol"])

    axes[-1].set_xlabel("Time  (s)")
    fig.suptitle("Joint Tracking Error  (q_current − q_target)", fontsize=11)
    fig.tight_layout()

    # Animation --------------------------------------------------------------
    def update(_frame):
        # Drain pending ZMQ messages (non-blocking)
        received = 0
        try:
            while True:
                raw  = socket.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(raw)
                t_now = time.monotonic() - t_start
                q_c   = data["q_current"]
                q_g   = data["q_target"]
                tick  = data.get("tick", -1)

                csv_errs = {}
                for g in groups:
                    for bi, j in enumerate(g["joints"]):
                        err = q_c[j] - q_g[j]
                        g["t_bufs"][bi].append(t_now)
                        g["err_bufs"][bi].append(err)
                        csv_errs[j] = err

                if csv_writer is not None:
                    csv_writer.writerow(
                        [f"{t_now:.4f}", tick] +
                        [f"{csv_errs[j]:.6f}" for j in all_joint_indices]
                    )

                received += 1
                if received >= 1000:   # yield so the plot doesn't freeze
                    break
        except zmq.Again:
            pass

        # Update lines per panel
        for g, ax, ax2 in zip(groups, axes, axes2):
            for i, ln in enumerate(g["lines"]):
                xs = list(g["t_bufs"][i])
                ys = [np.degrees(v) for v in g["err_bufs"][i]]
                ln.set_data(xs, ys)

            if not args.yrange:
                ax.relim()
                ax.autoscale_view(scalex=False)

            # Keep rad axis in sync with deg axis
            ymin_deg, ymax_deg = ax.get_ylim()
            ax2.set_ylim(np.radians(ymin_deg), np.radians(ymax_deg))

        # Shared x-axis: compute from all active buffers
        all_bufs = [buf for g in groups for buf in g["t_bufs"] if buf]
        if all_bufs:
            x_min = min(b[0]  for b in all_bufs)
            x_max = max(b[-1] for b in all_bufs)
            axes[0].set_xlim(x_min, max(x_max, x_min + 1.0))

        return all_lines

    ani = animation.FuncAnimation(   # noqa: F841  (kept alive by reference)
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
