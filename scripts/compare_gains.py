#!/usr/bin/env python3
"""Compare kp/kd PD gains across FSM states from config files."""

import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "deploy/robots/g1/config"

JOINT_NAMES = [
    "left  hip_pitch",  "left  hip_roll",  "left  hip_yaw",
    "left  knee",       "left  ankle_pitch","left  ankle_roll",
    "right hip_pitch",  "right hip_roll",   "right hip_yaw",
    "right knee",       "right ankle_pitch","right ankle_roll",
    "waist yaw",        "waist roll",       "waist pitch",
    "left  sh_pitch",   "left  sh_roll",    "left  sh_yaw",
    "left  elbow",      "left  wr_roll",    "left  wr_pitch",   "left  wr_yaw",
    "right sh_pitch",   "right sh_roll",    "right sh_yaw",
    "right elbow",      "right wr_roll",    "right wr_pitch",   "right wr_yaw",
]

def load_states():
    with open(CONFIG_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(CONFIG_DIR / "joint_cmd.yaml") as f:
        jcmd = yaml.safe_load(f)

    fsm = cfg.get("FSM", cfg)  # handle both top-level and nested
    states = {}
    for name, block in fsm.items():
        if isinstance(block, dict) and "kp" in block and "kd" in block:
            states[name] = {"kp": block["kp"], "kd": block["kd"]}
    states["JointCmd"] = {"kp": jcmd["kp"], "kd": jcmd["kd"]}
    return states

def print_table(name_a, name_b, states):
    a_kp = states[name_a]["kp"]
    b_kp = states[name_b]["kp"]
    a_kd = states[name_a]["kd"]
    b_kd = states[name_b]["kd"]

    n = max(len(a_kp), len(b_kp))
    col = max(len(name_a), len(name_b), 8)

    header = f"{'Joint':<22} {'[i]':>4}  {name_a:>{col}}  {name_b:>{col}}  {'kp diff':<16}  {name_a:>{col}}  {name_b:>{col}}  {'kd diff'}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(f"  kp / kd comparison:  {name_a}  vs  {name_b}")
    print(sep)
    print(f"{'Joint':<22} {'[i]':>4}  {'— kp —':>{col+col+4}}  {'— kd —':>{col+col+4}}")
    print(f"{'':22} {'':4}  {name_a:>{col}}  {name_b:>{col}}  {'diff':<16}  {name_a:>{col}}  {name_b:>{col}}  diff")
    print(sep)

    for i in range(n):
        jname = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"joint_{i}"
        kp_a = a_kp[i] if i < len(a_kp) else "—"
        kp_b = b_kp[i] if i < len(b_kp) else "—"
        kd_a = a_kd[i] if i < len(a_kd) else "—"
        kd_b = b_kd[i] if i < len(b_kd) else "—"

        def diff_str(va, vb):
            if va == "—" or vb == "—":
                return "—"
            if va == vb:
                return "same"
            ratio = vb / va if va != 0 else float("inf")
            direction = f"{name_b}" if vb > va else f"{name_a}"
            return f"{direction} {ratio:.1f}×" if ratio == int(ratio) else f"{direction} {ratio:.2f}×"

        kp_diff = diff_str(kp_a, kp_b)
        kd_diff = diff_str(kd_a, kd_b)

        kp_flag = " <" if kp_a != kp_b and kp_a != "—" else "  "
        kd_flag = " <" if kd_a != kd_b and kd_a != "—" else "  "

        print(f"{jname:<22} [{i:2d}]  {kp_a:>{col}}  {kp_b:>{col}}  {kp_diff:<16}{kp_flag}  "
              f"{kd_a:>{col}}  {kd_b:>{col}}  {kd_diff}{kd_flag}")

    print(sep)

def main():
    states = load_states()
    available = list(states.keys())

    if len(sys.argv) == 3:
        a, b = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        a, b = "FixSit", "JointCmd"
    else:
        print(f"Usage: {sys.argv[0]} [StateA StateB]")
        print(f"Available states: {', '.join(available)}")
        sys.exit(1)

    for name in (a, b):
        if name not in states:
            print(f"Unknown state '{name}'. Available: {', '.join(available)}")
            sys.exit(1)

    print_table(a, b, states)
    print(f"\nAvailable states: {', '.join(available)}")

if __name__ == "__main__":
    main()
