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
import json
import sys
import time

import zmq
import matplotlib.pyplot as plt

from scripts.utils.live_plot import ALL_GROUPS, JointErrorPlotter


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

    print(f"[plot_joint_error] groups  : {active_groups}")
    print(f"[plot_joint_error] buffer  : {args.buffer}  hz={args.hz}  address={args.address}")

    plotter = JointErrorPlotter(
        active_groups=active_groups,
        buffer=args.buffer,
        hz=args.hz,
        yrange=tuple(args.yrange) if args.yrange else None,
        csv_path=args.csv,
    )
    if args.csv:
        print(f"[plot_joint_error] CSV     : {args.csv}")

    # ZMQ subscriber ---------------------------------------------------------
    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.connect(args.address)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    t_start = time.monotonic()

    # Per-frame callback: drain ZMQ, push samples, then redraw ---------------
    def update(_frame):
        received = 0
        try:
            while True:
                raw  = socket.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(raw)
                t_now = time.monotonic() - t_start
                plotter.push(t_now, data["q_current"], data["q_target"],
                             data.get("tick", -1))
                received += 1
                if received >= 1000:   # yield so the plot doesn't freeze
                    break
        except zmq.Again:
            pass
        return plotter.redraw()

    ani = plotter.make_animation(update_fn=update)  # noqa: F841 — kept alive

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plotter.close()
        if args.csv:
            print(f"[plot_joint_error] CSV closed: {args.csv}")
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
