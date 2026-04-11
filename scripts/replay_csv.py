"""
Replay a motion CSV by setting joint angles directly (no physics, no policy).

Usage:
    python scripts/replay_csv.py
    python scripts/replay_csv.py --csv src/assets/motions/g1/arm_circle.csv
    python scripts/replay_csv.py --csv src/assets/motions/g1/arm_circle.csv --fps 30
"""

import argparse
import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH         = "src/assets/robots/unitree_g1/xmls/g1.xml"
XML_PATH_SITTING = "src/assets/robots/unitree_g1/xmls/g1_sitting.xml"

# Columns 7–35 map to these 29 joints in order.
CSV_JOINT_ORDER = [
    "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",         "waist_roll_joint",        "waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint",        "left_wrist_roll_joint",  "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",       "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def set_qpos_from_row(model, data, row, joint_ids):
    has_freejoint = model.nq > len(CSV_JOINT_ORDER)  # 36 DOF vs 29 DOF
    if has_freejoint:
        # Base position (cols 0–2) and quaternion xyzw→wxyz (cols 3–6).
        data.qpos[0:3] = row[0:3]
        data.qpos[3]   = row[6]   # w
        data.qpos[4]   = row[3]   # x
        data.qpos[5]   = row[4]   # y
        data.qpos[6]   = row[5]   # z
    # Cols 7–35: joint angles (same regardless of freejoint).
    for jid, val in zip(joint_ids, row[7:]):
        data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="src/assets/motions/g1/arm_circle.csv")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true", help="Loop indefinitely")
    parser.add_argument(
        "--xml",
        default=None,
        help=(
            "Path to the MuJoCo XML to load (default: g1.xml for normal CSVs, "
            "g1_sitting.xml auto-selected when --csv contains 'sit')."
        ),
    )
    args = parser.parse_args()

    traj = np.loadtxt(args.csv, delimiter=",")
    print(f"Loaded {traj.shape[0]} frames x {traj.shape[1]} cols from {args.csv}")

    if args.xml is not None:
        xml_path = Path(args.xml)
    elif "sit" in Path(args.csv).stem:
        xml_path = Path(XML_PATH_SITTING)
        print(f"Auto-selected sitting XML: {xml_path}")
    else:
        xml_path = Path(XML_PATH)
    xml = xml_path.read_text()
    xml = xml.replace(
        "<worldbody>",
        '<worldbody>\n    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1" contype="1" conaffinity="1"/>',
    )
    import os
    prev_dir = os.getcwd()
    os.chdir(xml_path.parent)
    model = mujoco.MjModel.from_xml_string(xml)
    os.chdir(prev_dir)
    data  = mujoco.MjData(model)

    # Pre-resolve joint IDs once.
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in CSV_JOINT_ORDER
    ]

    dt = 1.0 / args.fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while True:
            for row in traj:
                if not viewer.is_running():
                    return
                set_qpos_from_row(model, data, row, joint_ids)
                viewer.sync()
                time.sleep(dt)
            if not args.loop:
                break
        # Hold last frame until window is closed.
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.016)


if __name__ == "__main__":
    main()
