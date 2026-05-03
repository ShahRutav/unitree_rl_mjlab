from __future__ import annotations

"""
Reusable live joint-error plotter.

Public API
----------
JOINT_NAMES   : list[str]   — 29-element list of joint names
GROUP_DEFS    : dict        — per-group color/style/index definitions
ALL_GROUPS    : list[str]   — ["legs", "waist", "arm"]
JointErrorPlotter           — class that owns buffers, figure, and CSV

Typical usage (caller owns the ZMQ loop)
-----------------------------------------
    plotter = JointErrorPlotter(active_groups=["legs", "arm"], buffer=10000, hz=30.0)

    # Supply a custom per-frame callback that feeds data before each redraw:
    def my_update(frame):
        # ... receive data, call plotter.push(...) one or more times ...
        return plotter.redraw()

    ani = plotter.make_animation(update_fn=my_update)
    plt.show()
    plotter.close()

    # Or use the built-in redraw-only animation (data pushed externally):
    ani = plotter.make_animation()   # just redraws from current buffers
    plt.show()
"""

import collections
import csv
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Constants
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

# Per-group color + linestyle definitions
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
# JointErrorPlotter
# ---------------------------------------------------------------------------

class JointErrorPlotter:
    """
    Manages rolling buffers, matplotlib figure, and optional CSV output for
    live joint tracking-error visualization.

    The caller is responsible for:
      - Feeding data via .push()
      - Creating the animation via .make_animation() and keeping the returned
        object alive (assign to a variable)
      - Calling plt.show() when ready
      - Calling .close() when done

    Parameters
    ----------
    active_groups : list[str]
        Subset of ALL_GROUPS to display, in display order.
    buffer : int
        Rolling window size (samples per joint).
    hz : float
        FuncAnimation refresh rate in Hz.
    yrange : (float, float) | None
        Fixed y-axis range in radians.  None → auto-scale.
    csv_path : str | None
        If given, every push() call appends a row to this CSV file.
    """

    def __init__(
        self,
        active_groups: list[str],
        buffer: int = 10000,
        hz: float = 30.0,
        yrange: Optional[tuple[float, float]] = None,
        csv_path: Optional[str] = None,
    ):
        self._hz     = hz
        self._yrange = yrange

        # Build per-group runtime state --------------------------------------
        self._groups = []
        for gname in active_groups:
            gdef = GROUP_DEFS[gname]
            joints = gdef["joints"]
            self._groups.append({
                **gdef,
                "name": gname,
                "t_bufs":   [collections.deque(maxlen=buffer) for _ in joints],
                "err_bufs": [collections.deque(maxlen=buffer) for _ in joints],
            })

        self._all_joint_indices = sorted({j for g in self._groups for j in g["joints"]})

        # Optional CSV -------------------------------------------------------
        self._csv_file   = None
        self._csv_writer = None
        if csv_path:
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["time_s", "tick"] + [f"err_{JOINT_NAMES[j]}" for j in self._all_joint_indices]
            )

        # Build figure -------------------------------------------------------
        n_panels = len(self._groups)
        self.fig, _axes = plt.subplots(
            n_panels, 1,
            figsize=(13, 4.0 * n_panels),
            sharex=True,
            squeeze=False,
        )
        self._axes  = [row[0] for row in _axes]
        self._axes2 = [ax.twinx() for ax in self._axes]

        self._all_lines = []
        for g, ax, ax2 in zip(self._groups, self._axes, self._axes2):
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
            self._all_lines.extend(lines)

            ax.axhline(0.0, color="black", lw=0.6, linestyle="--", zorder=0, alpha=0.4)
            ax.set_ylabel("Error  (deg)")
            ax2.set_ylabel("Error  (rad)", labelpad=4)
            ax.set_title(g["title"], fontsize=9, loc="left", pad=3)

            if yrange:
                ax2.set_ylim(yrange)
                ax.set_ylim(np.degrees(yrange[0]), np.degrees(yrange[1]))

            ax.legend(loc="upper right", fontsize=6, ncol=g["legend_ncol"])

        self._axes[-1].set_xlabel("Time  (s)")
        self.fig.suptitle("Joint Tracking Error  (q_current − q_target)", fontsize=11)
        plt.tight_layout()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push(
        self,
        t: float,
        q_current: list[float],
        q_target: list[float],
        tick: int = -1,
    ) -> dict[int, float]:
        """
        Append one sample to the rolling buffers.

        Parameters
        ----------
        t         : elapsed time in seconds (monotonic)
        q_current : 29-element list of current joint positions (rad)
        q_target  : 29-element list of target joint positions (rad)
        tick      : optional controller tick counter

        Returns
        -------
        dict {joint_index: error_radians} for every tracked joint.
        """
        csv_errs = {}
        for g in self._groups:
            for bi, j in enumerate(g["joints"]):
                err = q_current[j] - q_target[j]
                g["t_bufs"][bi].append(t)
                g["err_bufs"][bi].append(err)
                csv_errs[j] = err

        if self._csv_writer is not None:
            self._csv_writer.writerow(
                [f"{t:.4f}", tick] +
                [f"{csv_errs[j]:.6f}" for j in self._all_joint_indices]
            )

        return csv_errs

    def redraw(self) -> list:
        """
        Refresh line data and axis limits from current buffers.
        Returns the list of artist objects (for blit=False animations).
        This is called automatically by make_animation(); only needed
        directly if the caller manages its own animation loop.
        """
        for g, ax, ax2 in zip(self._groups, self._axes, self._axes2):
            for i, ln in enumerate(g["lines"]):
                xs = list(g["t_bufs"][i])
                ys = [np.degrees(v) for v in g["err_bufs"][i]]
                ln.set_data(xs, ys)

            if not self._yrange:
                ax.relim()
                ax.autoscale_view(scalex=False)

            # Keep rad axis in sync with deg axis
            ymin_deg, ymax_deg = ax.get_ylim()
            ax2.set_ylim(np.radians(ymin_deg), np.radians(ymax_deg))

        # Shared x-axis: compute from all active buffers
        all_bufs = [buf for g in self._groups for buf in g["t_bufs"] if buf]
        if all_bufs:
            x_min = min(b[0]  for b in all_bufs)
            x_max = max(b[-1] for b in all_bufs)
            self._axes[0].set_xlim(x_min, max(x_max, x_min + 1.0))

        return self._all_lines

    def make_animation(self, update_fn=None) -> animation.FuncAnimation:
        """
        Create and return a FuncAnimation tied to this plotter's figure.

        Parameters
        ----------
        update_fn : callable(frame) -> artists, optional
            Custom per-frame callback.  It should call self.push() as needed
            and then call self.redraw() (or return its result).
            If None, defaults to self.redraw (buffers must be filled externally).

        Returns
        -------
        FuncAnimation — caller MUST keep a reference to prevent garbage collection.
        """
        callback = update_fn if update_fn is not None else lambda _frame: self.redraw()
        return animation.FuncAnimation(
            self.fig,
            callback,
            interval=1000.0 / self._hz,
            blit=False,
            cache_frame_data=False,
        )

    def get_error_buffers(self) -> dict:
        """
        Return all accumulated error samples as plain lists.

        Returns
        -------
        dict  {joint_name: list[float]}  — one entry per tracked joint, in
        the order they were registered (group order, then joint order within
        each group).  Values are in radians (same unit as push() receives).
        """
        result = {}
        for g in self._groups:
            for bi, j in enumerate(g["joints"]):
                result[JOINT_NAMES[j]] = list(g["err_bufs"][bi])
        return result

    def reset_buffers(self):
        """Clear all rolling buffers and redraw blank axes (e.g. to start a fresh recording window)."""
        for g in self._groups:
            for buf in g["t_bufs"]:
                buf.clear()
            for buf in g["err_bufs"]:
                buf.clear()
        for ln in self._all_lines:
            ln.set_data([], [])
        self.fig.canvas.draw_idle()

    def close(self):
        """Close the CSV file (if open).  ZMQ cleanup is the caller's responsibility."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None
