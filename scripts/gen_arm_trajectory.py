"""
gen_arm_trajectory.py

Generates a circle (or straight-line) trajectory for the G1 robot's right
end-effector, solves IK for each waypoint with the robot standing in its
default (HOME) pose, and writes a CSV motion file.

CSV format: 36 columns per row
  cols  0-2 : base position xyz  (qpos[0:3] of the free joint)
  cols  3-6 : base quaternion xyzw  (MuJoCo stores wxyz, so we reorder)
  cols 7-35 : 29 joint angles in CSV_JOINT_ORDER

Usage:
    python scripts/gen_arm_trajectory.py
    python scripts/gen_arm_trajectory.py --straight-line
    python scripts/gen_arm_trajectory.py --output src/assets/motions/g1/my_traj.csv
    python scripts/gen_arm_trajectory.py --n-points 60
"""

import os
import argparse
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
XML_PATH            = "src/assets/robots/unitree_g1/xmls/g1.xml"
XML_PATH_SITTING    = "src/assets/robots/unitree_g1/xmls/g1_sitting.xml"
FROZEN_LEGS_XML     = "/tmp/g1_frozen_legs.xml"

# ---------------------------------------------------------------------------
# Joint name lists
# ---------------------------------------------------------------------------
WAIST_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
IK_JOINTS = WAIST_JOINTS + RIGHT_ARM_JOINTS

EEF_BODY = "right_wrist_yaw_link"

# ---------------------------------------------------------------------------
# Frozen-legs config (matches ik_demo.py)
# ---------------------------------------------------------------------------
_H = math.pi / 2  # 90 deg

LEG_JOINTS: dict[str, float] = {
    "left_hip_pitch_joint":  -_H, "left_hip_roll_joint":  0.0, "left_hip_yaw_joint":  0.0,
    "left_knee_joint":        _H,
    "right_hip_pitch_joint": -_H, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint":       _H,
}


def generate_frozen_legs_xml(src: str = XML_PATH, dst: str = FROZEN_LEGS_XML,
                              joints: dict[str, float] = LEG_JOINTS) -> str:
    """
    Write a modified XML with:
      - pelvis welded to world (robot doesn't fall during IK)
      - equality/joint constraints pinning each leg joint to its frozen value
    Matches the implementation in dump_scripts/ik_demo.py.
    """
    ET.register_namespace("", "")
    tree = ET.parse(src)
    root = tree.getroot()

    # Rewrite meshdir to absolute so the XML loads from any working directory.
    src_dir = os.path.abspath(os.path.dirname(src))
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")
        if meshdir and not os.path.isabs(meshdir):
            compiler.set("meshdir", os.path.join(src_dir, meshdir))

    # Add a ground plane if not already present.
    worldbody = root.find("worldbody")
    if worldbody is not None:
        if not any(g.get("name") == "floor" for g in worldbody.findall("geom")):
            ET.SubElement(worldbody, "geom", name="floor", type="plane",
                          size="3 3 0.1", pos="0 0 0", rgba="0.8 0.8 0.8 1",
                          condim="3")

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    # Weld pelvis to world so the robot stays upright during IK.
    existing_welds = {(el.get("body1"), el.get("body2")) for el in equality.findall("weld")}
    if ("world", "pelvis") not in existing_welds:
        ET.SubElement(equality, "weld", body1="world", body2="pelvis")

    # Pin each leg joint to its frozen value.
    existing_joints = {el.get("joint1") for el in equality.findall("joint")}
    added = []
    for name, val in joints.items():
        if name not in existing_joints:
            ET.SubElement(equality, "joint", joint1=name,
                          polycoef=f"{val:.6f} 0 0 0 0")
            added.append(name)

    ET.indent(tree, space="  ")
    tree.write(dst, encoding="unicode", xml_declaration=False)
    print(f"[freeze-legs] pelvis welded to world; pinned {len(added)} joints → {dst}")
    return dst

# Exact joint order for the 29 DOF in the CSV (cols 7-35)
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

# ---------------------------------------------------------------------------
# Home keyframe (regex patterns resolved manually)
# ---------------------------------------------------------------------------
HOME_KEYFRAME_JOINTS = {
    # .*_hip_pitch_joint
    "left_hip_pitch_joint":        -0.1,
    "right_hip_pitch_joint":       -0.1,
    # .*_knee_joint
    "left_knee_joint":              0.3,
    "right_knee_joint":             0.3,
    # .*_ankle_pitch_joint
    "left_ankle_pitch_joint":      -0.2,
    "right_ankle_pitch_joint":     -0.2,
    # .*_shoulder_pitch_joint
    "left_shoulder_pitch_joint":    0.35,
    "right_shoulder_pitch_joint":   0.35,
    # .*_elbow_joint
    "left_elbow_joint":             0.87,
    "right_elbow_joint":            0.87,
    # shoulder_roll (explicit)
    "left_shoulder_roll_joint":     0.18,
    "right_shoulder_roll_joint":   -0.18,
}


def set_home_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """
    Reset data.qpos to the HOME keyframe.
    Base pos = (0, 0, 0.8), base quat = (1, 0, 0, 0) wxyz.
    All joints default to 0 except those in HOME_KEYFRAME_JOINTS.
    """
    data.qpos[:] = 0.0

    # Free joint: position (indices 0-2) + quaternion wxyz (indices 3-6)
    data.qpos[0] = 0.0   # x
    data.qpos[1] = 0.0   # y
    data.qpos[2] = 0.8   # z
    data.qpos[3] = 1.0   # w  (identity quaternion)
    data.qpos[4] = 0.0   # x
    data.qpos[5] = 0.0   # y
    data.qpos[6] = 0.0   # z

    # Set named joint angles
    for jname, val in HOME_KEYFRAME_JOINTS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = val

    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# IK utilities (DLS — Damped Least Squares)
# ---------------------------------------------------------------------------

def get_dof_ids(model: mujoco.MjModel, joint_names: list) -> np.ndarray:
    return np.array([
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in joint_names
    ])


def orientation_error(xmat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    """3-vector orientation error: 2 * vec(q_target * inv(q_body))."""
    q_body = np.zeros(4)
    mujoco.mju_mat2Quat(q_body, xmat)
    q_inv = np.array([q_body[0], -q_body[1], -q_body[2], -q_body[3]])
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, target_quat, q_inv)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:]


def ik_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    eef_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    dof_ids: np.ndarray,
    damping: float = 0.05,
    max_dq: float = 0.5,
) -> np.ndarray:
    """One DLS IK step; returns delta-q clipped to [-max_dq, max_dq]."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, eef_id)

    Jp = jacp[:, dof_ids]
    Jr = jacr[:, dof_ids]

    ep = target_pos - data.xpos[eef_id]
    er = orientation_error(data.xmat[eef_id], target_quat)

    n = len(dof_ids)
    JTJ  = Jp.T @ Jp + Jr.T @ Jr + damping ** 2 * np.eye(n)
    JTdx = Jp.T @ ep  + Jr.T @ er

    return np.clip(np.linalg.solve(JTJ, JTdx), -max_dq, max_dq)


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    eef_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    dof_ids: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-3,
):
    """
    Iterative DLS IK.  data.qpos is used as the warm start (caller sets it).
    Returns (qpos_copy, final_pos_error).
    """
    mujoco.mj_forward(model, data)
    for _ in range(max_iter):
        pos_err = np.linalg.norm(target_pos - data.xpos[eef_id])
        if pos_err < tol:
            break
        dq = ik_step(model, data, eef_id, target_pos, target_quat, dof_ids)
        dv = np.zeros(model.nv)
        dv[dof_ids] = dq
        mujoco.mj_integratePos(model, data.qpos, dv, 1.0)
        mujoco.mj_forward(model, data)

    final_err = np.linalg.norm(target_pos - data.xpos[eef_id])
    return data.qpos.copy(), final_err


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def make_circle_waypoints(n_points: int = 120) -> np.ndarray:
    """
    120 points (default) in the YZ plane at arm reach.
    center = (0.45, -0.2, 0.9), radius = 0.15.
    """
    center = np.array([0.45, -0.2, 0.9])
    radius = 0.15
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    waypoints = np.stack([
        np.full(n_points, center[0]),
        center[1] + radius * np.cos(angles),
        center[2] + radius * np.sin(angles),
    ], axis=1)
    return waypoints


def make_straight_waypoints(n_points: int = 60) -> np.ndarray:
    """
    60 points: 30 from (0.4, -0.1, 0.8) to (0.4, -0.4, 0.8),
               then 30 back.
    """
    start = np.array([0.4, -0.1, 0.8])
    end   = np.array([0.4, -0.4, 0.8])
    half  = n_points // 2
    fwd   = np.linspace(start, end, half, endpoint=True)
    bwd   = np.linspace(end, start, half, endpoint=True)
    return np.concatenate([fwd, bwd], axis=0)


# ---------------------------------------------------------------------------
# CSV row builder
# ---------------------------------------------------------------------------

def build_row(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Construct one 36-element CSV row from current data.qpos.

    Columns:
      0-2  : base position xyz         (qpos[0:3] when freejoint present)
      3-6  : base quaternion xyzw      (MuJoCo wxyz → reordered to xyzw)
      7-35 : 29 joint angles in CSV_JOINT_ORDER

    When the model has no freejoint (e.g. g1_sitting.xml, pelvis welded),
    qpos starts directly at joint 0 — so we read pelvis world position from
    data.xpos and use an identity quaternion instead.
    """
    has_freejoint = model.nq > len(CSV_JOINT_ORDER)  # 36 DOF vs 29 DOF
    if has_freejoint:
        base_pos  = data.qpos[0:3].copy()
        q_wxyz    = data.qpos[3:7]
        base_quat_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    else:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        base_pos  = data.xpos[pelvis_id].copy()
        base_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])  # identity xyzw

    joint_angles = np.zeros(len(CSV_JOINT_ORDER))
    for idx, jname in enumerate(CSV_JOINT_ORDER):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        adr = model.jnt_qposadr[jid]
        joint_angles[idx] = data.qpos[adr]

    return np.concatenate([base_pos, base_quat_xyzw, joint_angles])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Resolve absolute XML path (script may be invoked from any cwd)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)
    xml_path   = os.path.join(repo_root, XML_PATH)

    if args.freeze_legs:
        xml_path = os.path.join(repo_root, XML_PATH_SITTING)
        print(f"Freeze legs : ON  (using sitting XML: {xml_path})")
    else:
        print(f"Freeze legs : OFF")

    print(f"Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # EEF body id
    eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EEF_BODY)
    if eef_id < 0:
        raise ValueError(f"EEF body '{EEF_BODY}' not found in model.")

    # DOF ids for IK-controlled joints
    dof_ids = get_dof_ids(model, IK_JOINTS)

    # Target orientation: identity quaternion (wxyz)
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # Build waypoint list
    if args.straight_line:
        waypoints = make_straight_waypoints(args.n_points)
        if args.output == "src/assets/motions/g1/arm_circle.csv":
            args.output = "src/assets/motions/g1/arm_straight.csv"
        traj_name = "straight-line"
    else:
        waypoints = make_circle_waypoints(args.n_points)
        if args.freeze_legs:
            waypoints[:, 0] -= 0.10  # bring circle 10 cm closer (sitting pose)
            if args.output == "src/assets/motions/g1/arm_circle.csv":
                args.output = "src/assets/motions/g1/arm_circle_sit.csv"
        traj_name = "circle"

    n_waypoints = len(waypoints)
    print(f"Trajectory  : {traj_name}  ({n_waypoints} waypoints)")

    # Initialise robot pose.
    # For the sitting model, load the built-in "sitting" keyframe so leg joints
    # start at the values the XML already defines (hip_pitch=-π/2, knee=+π/2, etc.)
    # rather than any hardcoded standing defaults.
    # For the normal model, use the HOME keyframe.
    if args.freeze_legs:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "sitting")
        if key_id < 0:
            raise ValueError("'sitting' keyframe not found in g1_sitting.xml")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        print(f"Init pose   : 'sitting' keyframe from XML")
    else:
        set_home_keyframe(model, data)

    # Capture the interpolation start row (only needed when not using freeze-legs,
    # since the interpolation block is skipped in that case).
    if not args.freeze_legs:
        if args.init_joints is not None:
            init_vals = [float(v) for v in args.init_joints.split(",")]
            if len(init_vals) != 29:
                raise ValueError(f"--init-joints expects 29 values, got {len(init_vals)}")
            # Base pos / quat stay at HOME defaults; only joints are overridden.
            base_pos       = np.array([0.0, 0.0, 0.8])
            base_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw identity
            fix_stand_row  = np.concatenate([base_pos, base_quat_xyzw, np.array(init_vals)])
            print(f"Init pose   : from --init-joints (actual velocity-mode joints)")
        else:
            fix_stand_row = build_row(model, data)
            print(f"Init pose   : HOME keyframe")

    rows   = []
    errors = []

    for i, target_pos in enumerate(waypoints):
        if i % 20 == 0:
            print(f"  IK waypoint {i:4d}/{n_waypoints}  target={target_pos}")

        # Warm-start: data.qpos already holds the previous solution (or HOME
        # for i=0).  solve_ik updates data.qpos in place.
        qpos_sol, err = solve_ik(
            model, data,
            eef_id, target_pos, target_quat, dof_ids,
        )

        # Build CSV row from current (solved) state
        row = build_row(model, data)
        rows.append(row)
        errors.append(err)

    errors = np.array(errors)
    print(f"\nIK position errors — max: {errors.max():.5f}  mean: {errors.mean():.5f}")

    # Prepend a 2-second linear interpolation from fix-stand → first IK waypoint
    # (skipped when --freeze-legs is used).
    if not args.freeze_legs:
        n_interp = int(args.fps * 2.0)
        print(f"Prepending {n_interp} interpolation frames (fix-stand → first waypoint, 2 s @ {args.fps} fps)")
        first_row = rows[0]
        interp_rows = [
            (1.0 - alpha) * fix_stand_row + alpha * first_row
            for alpha in np.linspace(0.0, 1.0, n_interp, endpoint=False)
        ]
        rows = interp_rows + rows
    else:
        print("Interpolation skipped (freeze-legs mode)")

    # Write CSV
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(repo_root, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    csv_array = np.array(rows)  # shape (N, 36)
    np.savetxt(output_path, csv_array, delimiter=",", fmt="%.6f")

    print(f"Wrote {csv_array.shape[0]} rows x {csv_array.shape[1]} cols to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an arm trajectory CSV for the Unitree G1 robot."
    )
    parser.add_argument(
        "--straight-line",
        action="store_true",
        help="Generate a straight-line trajectory instead of a circle.",
    )
    parser.add_argument(
        "--freeze-legs",
        action="store_true",
        help=(
            "Freeze leg joints via MuJoCo equality constraints during IK "
            "(writes a modified XML to /tmp/g1_frozen_legs.xml). "
            "Leg joints are set to a tucked pose (hip_pitch=-90°, knee=+90°)."
        ),
    )
    parser.add_argument(
        "--output",
        default="src/assets/motions/g1/arm_circle.csv",
        help="Output CSV path (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=120,
        help="Number of trajectory waypoints (default: 120 for circle, 60 for straight).",
    )
    parser.add_argument(
        "--init-joints",
        default=None,
        help=(
            "29 comma-separated joint angles in CSV_JOINT_ORDER to use as the "
            "start of the fix-stand → first-waypoint interpolation. "
            "Copy the values from the '[Mimic enter] Current joint angles' log "
            "to match the actual robot pose in velocity mode."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help=(
            "Playback FPS used to compute the 2-second fix-stand interpolation "
            "prepended to the trajectory (default: 30). Must match the FPS used "
            "by replay_csv.py or the motion player."
        ),
    )
    args = parser.parse_args()
    main(args)
