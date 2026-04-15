#!/usr/bin/env python3
"""
Throwaway test: Cartesian streaming mode for arm_cmd.py.

Streams a slow circular trajectory via
    {"type": "cartesian", "targets": {"right_eef": {"pos": [x,y,z], "quat": [1,0,0,0]}}}
to arm_cmd.py on port 5557.  Circle is traced in the Y-Z plane around a center point.

Flags:
    --center X Y Z     Circle center (default: 0.32 -0.25 0.15; x=0.32 reduces self-collision)
    --radius           Circle radius in metres (default: 0.08)
    --period           Seconds for one full circle (default: 8.0)
    --hz               Publish rate in Hz (default: 20)
    --plot             Enable BOTH the 6-panel Cartesian error plot AND the joint error plot
    --csv-dir          Directory for CSV output files (default: /tmp)

Usage:
    python3 dump_scripts/test_arm_cmd_cartesian.py
    python3 dump_scripts/test_arm_cmd_cartesian.py --radius 0.05 --period 12.0
    python3 dump_scripts/test_arm_cmd_cartesian.py --plot
"""

import argparse
import collections
import csv
import json
import math
import os
import sys
import time
from datetime import datetime

import zmq

# ---------------------------------------------------------------------------
# Repo-root sys.path setup — needed for scripts.utils.live_plot import
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from scripts.utils.live_plot import JointErrorPlotter, JOINT_NAMES  # noqa: E402
from scripts.utils.stats import print_error_summary    # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XML_PATH    = os.path.join(REPO_ROOT, "src", "assets", "robots", "unitree_g1", "xmls", "g1.xml")
CONFIG_PATH = os.path.join(REPO_ROOT, "deploy", "robots", "g1", "config", "joint_cmd.yaml")

# Right EEF body name — from ik/configs/g1_right_arm.yaml (targets[0].body)
RIGHT_EEF_BODY = "right_wrist_yaw_link"

# Identity quaternion — target orientation for IK (world frame, no rotation)
TARGET_QUAT = [1.0, 0.0, 0.0, 0.0]


def _load_kp_kd():
    """Parse kp/kd lists from joint_cmd.yaml without requiring a YAML library."""
    kp, kd = None, None
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        kp = cfg.get("kp", [])
        kd = cfg.get("kd", [])
    except Exception:
        # Fallback: read raw values we know from the file
        kp = [100,100,100,150,40,40, 100,100,100,150,40,40, 200,200,400,
              40,40,40,40,40,40,40, 40,40,40,80,40,40,40]
        kd = [2,2,2,4,2,2, 2,2,2,4,2,2, 5,5,5,
              10,10,10,10,10,10,10, 10,10,10,10,10,10,10]
    return kp, kd


def _ori_error_deg(xmat, target_quat_np):
    """Orientation error in degrees: 2*vec(q_target * q_body^-1), converted to degrees."""
    import mujoco
    import numpy as np
    q_body = np.zeros(4)
    mujoco.mju_mat2Quat(q_body, xmat)
    q_inv = np.array([q_body[0], -q_body[1], -q_body[2], -q_body[3]])
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, target_quat_np, q_inv)
    if q_err[0] < 0:
        q_err = -q_err
    return np.degrees(2.0 * q_err[1:])   # shape (3,)


def _write_csv_header_comments(f, kp, kd):
    """Write # kp / # kd / # Generated comment block to an open file handle."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"# kp: {kp}\n")
    f.write(f"# kd: {kd}\n")
    f.write(f"# Generated: {now_str}\n")


def main():
    parser = argparse.ArgumentParser(description="Stream Cartesian circle trajectory to arm_cmd.py")
    parser.add_argument("--center", type=float, nargs=3, default=[0.32, -0.25, 0.15],
                        metavar=("X", "Y", "Z"),
                        help="Circle center x y z in pelvis frame (default: 0.32 -0.25 0.15)")
    parser.add_argument("--radius", type=float, default=0.08,
                        help="Circle radius in metres (default: 0.08)")
    parser.add_argument("--period", type=float, default=8.0,
                        help="Seconds for one full circle (default: 8.0)")
    parser.add_argument("-t", "--time", type=float, default=None, metavar="SECONDS",
                        help="Total run time in seconds (excluding warm-up); omit to run indefinitely")
    parser.add_argument("--warmup", type=float, default=5.0, metavar="SECONDS",
                        help="Seconds to hold the circle start position before the trajectory begins (default: 5.0)")
    parser.add_argument("--hz",     type=float, default=20.0,
                        help="Publish rate in Hz (default: 20)")
    parser.add_argument("--address", default="tcp://localhost:5557",
                        help="ZMQ connect address for arm_cmd.py (default: tcp://localhost:5557)")
    parser.add_argument("--feedback-address", default="tcp://localhost:5558",
                        help="ZMQ subscribe address for joint feedback (default: tcp://localhost:5558)")
    parser.add_argument("--plot", action="store_true",
                        help="Enable both the 6-panel Cartesian error plot and the joint error plot")
    parser.add_argument("--csv-dir", default="/tmp",
                        help="Directory for CSV output files (default: /tmp)")
    args = parser.parse_args()

    cx, cy, cz = args.center
    plot_cartesian = args.plot
    plot_joints    = args.plot
    do_plot        = args.plot

    # -----------------------------------------------------------------------
    # MuJoCo model — always loaded for FK stats; also needed for plot
    # -----------------------------------------------------------------------
    import mujoco
    import numpy as np
    mj_model = None
    mj_data  = None
    eef_body_id = None

    if not os.path.isfile(XML_PATH):
        print(f"[test_cartesian] WARNING: XML not found at {XML_PATH}; Cartesian FK unavailable.")
        plot_cartesian = False
    else:
        mj_model    = mujoco.MjModel.from_xml_path(XML_PATH)
        mj_data     = mujoco.MjData(mj_model)
        eef_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EEF_BODY)
        if eef_body_id < 0:
            print(f"[test_cartesian] WARNING: body '{RIGHT_EEF_BODY}' not found in model; "
                  "Cartesian FK unavailable.")
            plot_cartesian = False

    # -----------------------------------------------------------------------
    # Matplotlib — only when --plot
    # -----------------------------------------------------------------------
    if do_plot:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

    # -----------------------------------------------------------------------
    # ZMQ setup
    # -----------------------------------------------------------------------
    ctx    = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.connect(args.address)

    # Feedback socket — always open for stats collection
    fb_socket = ctx.socket(zmq.SUB)
    fb_socket.connect(args.feedback_address)
    fb_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    fb_socket.setsockopt(zmq.RCVTIMEO, 0)   # non-blocking

    print(f"[test_cartesian] connected to {args.address}")
    print(f"[test_cartesian] feedback subscribed to {args.feedback_address}")
    print(f"[test_cartesian] center=({cx:.3f}, {cy:.3f}, {cz:.3f})  "
          f"radius={args.radius} m  period={args.period} s  hz={args.hz}")
    print("[test_cartesian] sleeping 500 ms for SUB side to register…")
    time.sleep(0.5)

    # -----------------------------------------------------------------------
    # Timestamp for CSV filenames
    # -----------------------------------------------------------------------
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    kp, kd = _load_kp_kd()

    # -----------------------------------------------------------------------
    # Cartesian error CSV
    # -----------------------------------------------------------------------
    cart_csv_file   = None
    cart_csv_writer = None
    if plot_cartesian:
        cart_csv_path = os.path.join(args.csv_dir, f"cartesian_errors_{ts_str}.csv")
        cart_csv_file = open(cart_csv_path, "w", newline="")
        _write_csv_header_comments(cart_csv_file, kp, kd)
        cart_csv_writer = csv.writer(cart_csv_file)
        cart_csv_writer.writerow([
            "time_s", "target_x", "target_y", "target_z",
            "actual_x", "actual_y", "actual_z",
            "err_x_cm", "err_y_cm", "err_z_cm", "err_norm_cm",
            "ori_rx_deg", "ori_ry_deg", "ori_rz_deg",
        ])
        print(f"[test_cartesian] Cartesian CSV: {cart_csv_path}")

    # -----------------------------------------------------------------------
    # Joint error plotter (JointErrorPlotter)
    # -----------------------------------------------------------------------
    joint_plotter    = None
    joint_csv_file   = None
    joint_csv_writer = None
    if plot_joints:
        joint_csv_path = os.path.join(args.csv_dir, f"joint_errors_{ts_str}.csv")
        # Write comment header first; then append CSV rows so the kp/kd block
        # precedes the column header.  Pass csv_path=None to JointErrorPlotter
        # so it does not open/overwrite the file itself.
        with open(joint_csv_path, "w") as jf:
            _write_csv_header_comments(jf, kp, kd)
        joint_plotter = JointErrorPlotter(
            active_groups=["waist", "arm"],
            buffer=6000,
            hz=30.0,
            csv_path=None,
        )
        joint_csv_file = open(joint_csv_path, "a", newline="")
        joint_csv_writer = csv.writer(joint_csv_file)
        joint_csv_writer.writerow(
            ["time_s", "tick",
             "err_waist_yaw", "err_waist_roll", "err_waist_pitch",
             "err_R_shoulder_pitch", "err_R_shoulder_roll", "err_R_shoulder_yaw",
             "err_R_elbow", "err_R_wrist_roll", "err_R_wrist_pitch", "err_R_wrist_yaw"]
        )
        print(f"[test_cartesian] Joint CSV: {joint_csv_path}")

    # -----------------------------------------------------------------------
    # Error buffers — always initialised for stats; figure only with --plot
    # 6 channels: indices 0-2 position error (cm), 3-5 orientation error (deg)
    # -----------------------------------------------------------------------
    # Joint stat buffers — always collected (waist 12-14, right arm 22-28)
    _STAT_JOINT_INDICES = [12, 13, 14, 22, 23, 24, 25, 26, 27, 28]
    joint_stat_bufs = {j: collections.deque() for j in _STAT_JOINT_INDICES}

    cart_t_bufs   = [collections.deque() for _ in range(6)]
    cart_err_bufs = [collections.deque() for _ in range(6)]

    fig_cart      = None
    axes_cart     = None
    cart_lines    = None

    if plot_cartesian:
        import matplotlib.pyplot as plt

        fig_cart, axes_cart = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
        cart_lines = []
        _cart_labels = [
            "X err (cm)", "Y err (cm)", "Z err (cm)",
            "Rx err (deg)", "Ry err (deg)", "Rz err (deg)",
        ]
        for i, label in enumerate(_cart_labels):
            (ln,) = axes_cart[i].plot([], [], lw=1.2)
            axes_cart[i].axhline(0.0, color='k', lw=0.6, linestyle='--', alpha=0.4)
            axes_cart[i].set_ylabel(label)
            cart_lines.append(ln)
        axes_cart[-1].set_xlabel("Time (s)")
        fig_cart.suptitle("Cartesian EEF Error  (actual - target)", fontsize=11)
        plt.tight_layout()

    def update_cart_plot():
        """Refresh Cartesian error plot from current buffers."""
        if cart_lines is None or cart_t_bufs is None or cart_err_bufs is None or axes_cart is None:
            return
        for i, ln in enumerate(cart_lines):
            xs = list(cart_t_bufs[i])
            ys = list(cart_err_bufs[i])
            ln.set_data(xs, ys)
            axes_cart[i].relim()
            axes_cart[i].autoscale_view()
        if cart_t_bufs[0]:
            axes_cart[0].set_xlim(
                cart_t_bufs[0][0],
                max(cart_t_bufs[0][-1], cart_t_bufs[0][0] + 1.0),
            )

    # -----------------------------------------------------------------------
    # Enable interactive mode if plotting
    # -----------------------------------------------------------------------
    if do_plot:
        import matplotlib.pyplot as plt
        plt.ion()

    # -----------------------------------------------------------------------
    # Main send loop
    # -----------------------------------------------------------------------
    print("[test_cartesian] streaming circle in Y-Z plane…")

    period     = 1.0 / args.hz

    # -----------------------------------------------------------------------
    # Warm-up phase — hold the circle's start position (angle=0) so the arm
    # has time to reach it before the timed trajectory begins.
    # angle=0 → y = cy + radius, z = cz  (cos(0)=1, sin(0)=0)
    # -----------------------------------------------------------------------
    x0, y0, z0 = cx, cy + args.radius, cz
    warmup_msg = json.dumps({"type": "cartesian", "targets": {
        "right_eef": {"pos": [x0, y0, z0], "quat": TARGET_QUAT}
    }})

    if args.warmup > 0:
        print(f"[test_cartesian] warm-up: holding start position "
              f"({x0:.3f}, {y0:.3f}, {z0:.3f}) for {args.warmup:.1f}s…")
        wu_start = time.monotonic()
        wu_next  = wu_start
        while (time.monotonic() - wu_start) < args.warmup:
            socket.send_string(warmup_msg)
            # Drain feedback during warm-up (discarded — not counted in stats)
            while True:
                try:
                    fb_socket.recv_string()
                except zmq.Again:
                    break
            wu_next += period
            sleep_s  = wu_next - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
        print("[test_cartesian] warm-up done — starting trajectory.")

    last_sent_target = [x0, y0, z0]   # warm-up left arm at start position

    next_t     = time.monotonic()
    t_start    = time.monotonic()
    last_print = t_start - 1.0   # ensure first-tick print

    # Track last FK result for printing
    last_actual = None

    try:
        while args.time is None or (time.monotonic() - t_start) < args.time:
            t_now  = time.monotonic() - t_start
            angle  = 2.0 * math.pi * t_now / args.period

            x = cx
            y = cy + args.radius * math.cos(angle)
            z = cz + args.radius * math.sin(angle)

            msg = {"type": "cartesian", "targets": {"right_eef": {"pos": [x, y, z], "quat": TARGET_QUAT}}}
            socket.send_string(json.dumps(msg))

            # ------------------------------------------------------------------
            # Drain feedback socket — always, for stats collection
            # q_current reflects the robot responding to last_sent_target, not
            # the target just sent above, so use last_sent_target for error.
            # ------------------------------------------------------------------
            while True:
                try:
                    raw = fb_socket.recv_string()
                    fb  = json.loads(raw)
                    q_current = fb.get("q_current", None)
                    q_target  = fb.get("q_target",  None)
                    tick      = fb.get("tick", -1)

                    if q_current is None:
                        continue

                    t_fb = time.monotonic() - t_start

                    # ---- Cartesian FK (always when model available) ----
                    # g1.xml has a free joint: qpos = [base_xyz(3), base_quat(4), joints(29)]
                    # Place pelvis at world origin so FK output is in pelvis frame,
                    # matching the IK convention (base_pos=[0,0,0]).
                    if mj_model is not None and mj_data is not None and last_sent_target is not None:
                        mj_data.qpos[0:3] = [0.0, 0.0, 0.0]   # pelvis at origin
                        mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]   # identity quat (wxyz)
                        mj_data.qpos[7:7 + len(q_current)] = q_current
                        mujoco.mj_forward(mj_model, mj_data)
                        actual_pos  = mj_data.xpos[eef_body_id].copy()
                        actual_xmat = mj_data.xmat[eef_body_id].copy()
                        target_pos  = np.array(last_sent_target)
                        pos_err_cm  = (actual_pos - target_pos) * 100.0
                        ori_err_deg = _ori_error_deg(actual_xmat, np.array(TARGET_QUAT))
                        err_norm_cm = float(np.linalg.norm(pos_err_cm))
                        last_actual = actual_pos

                        for i in range(3):
                            cart_t_bufs[i].append(t_fb)
                            cart_err_bufs[i].append(float(pos_err_cm[i]))
                        for i in range(3):
                            cart_t_bufs[3 + i].append(t_fb)
                            cart_err_bufs[3 + i].append(float(ori_err_deg[i]))

                        if cart_csv_writer is not None:
                            cart_csv_writer.writerow([
                                f"{t_fb:.4f}",
                                f"{target_pos[0]:.6f}", f"{target_pos[1]:.6f}", f"{target_pos[2]:.6f}",
                                f"{actual_pos[0]:.6f}", f"{actual_pos[1]:.6f}", f"{actual_pos[2]:.6f}",
                                f"{pos_err_cm[0]:.6f}", f"{pos_err_cm[1]:.6f}", f"{pos_err_cm[2]:.6f}",
                                f"{err_norm_cm:.6f}",
                                f"{ori_err_deg[0]:.6f}", f"{ori_err_deg[1]:.6f}", f"{ori_err_deg[2]:.6f}",
                            ])

                    # ---- Joint stats (always) ----
                    if q_target is not None:
                        for j in _STAT_JOINT_INDICES:
                            joint_stat_bufs[j].append(q_current[j] - q_target[j])

                    # ---- Joint plotter (only when --plot) ----
                    if (plot_joints and joint_plotter is not None and q_target is not None):
                        errs = joint_plotter.push(t_fb, q_current, q_target, tick=tick)
                        if joint_csv_writer is not None:
                            row_errs = [errs.get(j, 0.0) for j in [12, 13, 14, 22, 23, 24, 25, 26, 27, 28]]
                            joint_csv_writer.writerow(
                                [f"{t_fb:.4f}", tick] + [f"{e:.6f}" for e in row_errs]
                            )

                except zmq.Again:
                    break   # no more messages queued
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

            # ------------------------------------------------------------------
            # Console print
            # ------------------------------------------------------------------
            if time.monotonic() - last_print >= 1.0:
                if last_actual is not None:
                    print(f"[test_cartesian] t={t_now:6.1f}s  target=({x:.4f}, {y:.4f}, {z:.4f})  "
                          f"actual=({last_actual[0]:.4f}, {last_actual[1]:.4f}, {last_actual[2]:.4f})")
                else:
                    print(f"[test_cartesian] t={t_now:6.1f}s  target=({x:.4f}, {y:.4f}, {z:.4f})")
                last_print = time.monotonic()

            # Advance the reference target now that feedback has been drained
            last_sent_target = [x, y, z]

            # ------------------------------------------------------------------
            # Sleep / matplotlib event loop
            # ------------------------------------------------------------------
            next_t  += period
            sleep_s  = next_t - time.monotonic()

            if do_plot:
                import matplotlib.pyplot as plt
                if plot_cartesian:
                    update_cart_plot()
                if plot_joints and joint_plotter is not None:
                    joint_plotter.redraw()
                plt.pause(max(0.001, sleep_s))
            else:
                if sleep_s > 0:
                    time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[test_cartesian] stopped.")
    finally:
        # ── Summary statistics ──────────────────────────────────────────────
        n_cart = sum(len(b) for b in cart_err_bufs)
        n_joint = sum(len(b) for b in joint_stat_bufs.values())
        if n_cart > 0 or n_joint > 0:
            sep = "─" * 72
            print(f"\n{sep}")
            print(f"  Run summary  ({n_cart} Cartesian samples, {n_joint} joint samples)")
            print(f"  kp: {kp}")
            print(f"  kd: {kd}")
            print(sep)

        _CART_LABELS = ["X error", "Y error", "Z error", "Rx error", "Ry error", "Rz error"]
        if n_cart > 0:
            print_error_summary(
                {_CART_LABELS[i]: cart_err_bufs[i] for i in range(3)},
                title="Cartesian Position Error", unit="(cm)",
            )
            print_error_summary(
                {_CART_LABELS[i]: cart_err_bufs[i] for i in range(3, 6)},
                title="Cartesian Orientation Error", unit="(deg)",
            )
        if n_joint > 0:
            print_error_summary(
                {JOINT_NAMES[j]: [np.degrees(e) for e in joint_stat_bufs[j]]
                 for j in _STAT_JOINT_INDICES},
                title="Joint Tracking Error", unit="(deg)",
            )

        socket.close()
        fb_socket.close()
        ctx.term()

        if cart_csv_file is not None:
            cart_csv_file.close()
        if joint_csv_file is not None:
            joint_csv_file.close()
        if joint_plotter is not None:
            joint_plotter.close()


if __name__ == "__main__":
    main()
