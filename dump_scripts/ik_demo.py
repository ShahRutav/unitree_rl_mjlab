"""
Script 2: Differential IK (DLS) for G1 right end-effector.

Implements the same damped-least-squares algorithm as mjlab's
DifferentialIKAction, using plain mujoco Python bindings.

By default, only waist + right arm joints move (freeze=True).
Viewer always shows: pelvis frame, right EEF frame, camera frame,
and a red target sphere during IK.

Usage:
    python dump_scripts/ik_demo.py                            # freeze=True (default)
    python dump_scripts/ik_demo.py --no-freeze                # all joints active
    python dump_scripts/ik_demo.py --frozen-legs              # remove leg joints from XML (writes /tmp/g1_edited.xml)
    python dump_scripts/ik_demo.py --frozen-legs --no-freeze  # frozen legs, all other joints active
"""
import argparse
import math
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH        = "src/assets/robots/unitree_g1/xmls/g1.xml"
FROZEN_LEGS_XML = "/tmp/g1_edited.xml"
EEF_BODY    = "right_wrist_yaw_link"
BASE_BODY   = "pelvis"
CAMERA_NAME = "head_camera"

WAIST_JOINTS = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_roll_joint",     "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
_H = math.pi / 2   # 90 deg
LEG_JOINTS: dict[str, float] = {
    "left_hip_pitch_joint":  -_H,  "left_hip_roll_joint":  0.0,  "left_hip_yaw_joint":  0.0,
    "left_knee_joint":        _H,
    "right_hip_pitch_joint": -_H,  "right_hip_roll_joint": 0.0,  "right_hip_yaw_joint": 0.0,
    "right_knee_joint":       _H,
}

# ── XML manipulation ──────────────────────────────────────────────────────────

def generate_frozen_legs_xml(src: str = XML_PATH, dst: str = FROZEN_LEGS_XML,
                             joints: dict[str, float] = LEG_JOINTS) -> str:
    """
    Parse *src* XML, add an equality/joint constraint for each joint in
    *joints* pinning it to the given value (radians), and write to *dst*.
    Called every run so *dst* is always in sync with *src*.

    MuJoCo equality constraint (single-joint form):
        <joint joint="NAME" polycoef="v 0 0 0 0"/>
    enforces  q = v.  The DOF stays in the model so qpos indices are stable.

    Returns the path of the written file.
    """
    import os
    ET.register_namespace("", "")   # suppress ns0: prefixes
    tree = ET.parse(src)
    root = tree.getroot()

    # Rewrite meshdir to an absolute path so the XML loads correctly from
    # any location (e.g. /tmp/).
    src_dir = os.path.abspath(os.path.dirname(src))
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")
        if meshdir and not os.path.isabs(meshdir):
            compiler.set("meshdir", os.path.join(src_dir, meshdir))

    # Add a ground plane to the worldbody if not already present.
    worldbody = root.find("worldbody")
    if worldbody is not None:
        if not any(g.get("name") == "floor" for g in worldbody.findall("geom")):
            ET.SubElement(worldbody, "geom", name="floor", type="plane",
                          size="3 3 0.1", pos="0 0 0", rgba="0.8 0.8 0.8 1",
                          condim="3")

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    # Weld pelvis to world so the robot doesn't fall.
    # This is equivalent to fixing the feet — with frozen legs the whole
    # lower body is rigid, so fixing either the pelvis or the feet is the same.
    # The freejoint stays in the model (qpos indices unchanged for IK).
    existing_welds = {(el.get("body1"), el.get("body2")) for el in equality.findall("weld")}
    if ("world", "pelvis") not in existing_welds:
        ET.SubElement(equality, "weld", body1="world", body2="pelvis")

    existing_joints = {el.get("joint1") for el in equality.findall("joint")}
    added = []
    for name, val in joints.items():
        if name not in existing_joints:
            ET.SubElement(equality, "joint", joint1=name,
                          polycoef=f"{val:.6f} 0 0 0 0")
            added.append(f"{name}={val:.4f}")

    ET.indent(tree, space="  ")
    tree.write(dst, encoding="unicode", xml_declaration=False)
    print(f"[frozen-legs] pelvis welded to world; pinned {len(added)} joints → {dst}")
    return dst


# ── Visualization helpers ─────────────────────────────────────────────────────

def draw_sphere(scene, pos: np.ndarray, size: float, rgba) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, size, size]),
                        pos, np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


def draw_frame(scene, pos: np.ndarray, mat: np.ndarray,
               scale: float = 0.10, label: str = "") -> None:
    """Draw XYZ axes as colored arrows (X=red, Y=green, Z=blue)."""
    mat33 = mat.reshape(3, 3)
    colors = [
        np.array([1, 0, 0, 1], dtype=np.float32),
        np.array([0, 1, 0, 1], dtype=np.float32),
        np.array([0, 0, 1, 1], dtype=np.float32),
    ]
    for i, rgba in enumerate(colors):
        if scene.ngeom >= len(scene.geoms):
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                            np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006,
                             pos, pos + mat33[:, i] * scale)
        if i == 0 and label:
            g.label = label
        scene.ngeom += 1


def draw_frame_with_primary(scene, pos: np.ndarray, mat: np.ndarray,
                            primary_dir: np.ndarray, primary_rgba,
                            scale: float = 0.10) -> None:
    """Draw XYZ axes + a thick primary-direction arrow (white/yellow per frame type)."""
    draw_frame(scene, pos, mat, scale=scale)
    if scene.ngeom >= len(scene.geoms):
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        np.array(primary_rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.010,
                         pos, pos + primary_dir * scale * 1.5)
    scene.ngeom += 1


def update_scene(viewer, data, base_id, eef_id, cam_id, target_pos=None) -> None:
    """Refresh user_scn: pelvis + EEF + camera frames, optional target sphere."""
    scn = viewer.user_scn
    scn.ngeom = 0
    R_base = data.xmat[base_id].reshape(3, 3)
    R_eef  = data.xmat[eef_id].reshape(3, 3)
    R_cam  = data.cam_xmat[cam_id].reshape(3, 3)

    # Pelvis — white sphere; white arrow = robot forward (+X)
    draw_sphere(scn, data.xpos[base_id], 0.025, [1, 1, 1, 1])
    draw_frame_with_primary(scn, data.xpos[base_id], R_base,
                            primary_dir=R_base[:, 0],
                            primary_rgba=[1, 1, 1, 1], scale=0.15)

    # Right EEF — cyan sphere; white arrow = hand approach (+X)
    draw_sphere(scn, data.xpos[eef_id], 0.020, [0, 0.8, 1, 1])
    draw_frame_with_primary(scn, data.xpos[eef_id], R_eef,
                            primary_dir=R_eef[:, 0],
                            primary_rgba=[1, 1, 1, 1], scale=0.12)

    # Camera — orange sphere; yellow arrow = viewing direction (-Z)
    draw_sphere(scn, data.cam_xpos[cam_id], 0.020, [1, 0.5, 0, 1])
    draw_frame_with_primary(scn, data.cam_xpos[cam_id], R_cam,
                            primary_dir=-R_cam[:, 2],
                            primary_rgba=[1, 1, 0, 1], scale=0.12)

    if target_pos is not None:
        draw_sphere(scn, target_pos, 0.03, [1, 0.2, 0.2, 0.8])


# ── IK core ───────────────────────────────────────────────────────────────────

def get_dof_ids(model, joint_names: list[str]) -> np.ndarray:
    return np.array([
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in joint_names
    ])


def orientation_error(xmat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    """3-vector orientation error: 2 * vec(q_target * inv(q_body))."""
    q_body = np.zeros(4)
    mujoco.mju_mat2Quat(q_body, xmat)
    q_inv  = np.array([q_body[0], -q_body[1], -q_body[2], -q_body[3]])
    q_err  = np.zeros(4)
    mujoco.mju_mulQuat(q_err, target_quat, q_inv)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:]


def ik_step(model, data, eef_id, target_pos, target_quat, dof_ids,
            damping=0.05, max_dq=0.5, pos_weight=1.0, ori_weight=1.0) -> np.ndarray:
    """One DLS IK step using the EEF body Jacobian."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, eef_id)

    Jp = jacp[:, dof_ids]
    Jr = jacr[:, dof_ids]

    ep = target_pos - data.xpos[eef_id]
    er = orientation_error(data.xmat[eef_id], target_quat)

    wp2, wo2, lam2 = pos_weight**2, ori_weight**2, damping**2
    n    = len(dof_ids)
    JTJ  = wp2 * (Jp.T @ Jp) + wo2 * (Jr.T @ Jr) + lam2 * np.eye(n)
    JTdx = wp2 * (Jp.T @ ep) + wo2 * (Jr.T @ er)

    return np.clip(np.linalg.solve(JTJ, JTdx), -max_dq, max_dq)


def solve_ik(model, data, target_pos, target_quat,
             freeze=True, max_iter=200, tol=1e-3,
             damping=0.05, max_dq=0.5,
             viewer=None, base_id=None, eef_id=None, cam_id=None) -> np.ndarray:
    """
    Iterative differential IK for the right end-effector (right_wrist_yaw_link).

    Args:
        freeze: if True, only waist + right arm joints move.
        viewer: optional passive viewer for live animation.
    Returns:
        Final qpos array.
    """
    joint_names = (WAIST_JOINTS + RIGHT_ARM_JOINTS) if freeze else None
    dof_ids = (get_dof_ids(model, joint_names) if joint_names is not None
               else np.arange(6, model.nv))

    mujoco.mj_forward(model, data)

    for i in range(max_iter):
        pos_err = np.linalg.norm(target_pos - data.xpos[eef_id])

        if viewer is not None and viewer.is_running():
            update_scene(viewer, data, base_id, eef_id, cam_id, target_pos)
            viewer.sync()
            time.sleep(0.04)

        if pos_err < tol:
            print(f"Converged at iter {i}, pos_error={pos_err:.5f}")
            break

        dq = ik_step(model, data, eef_id, target_pos, target_quat, dof_ids, damping, max_dq)
        dv = np.zeros(model.nv)
        dv[dof_ids] = dq
        mujoco.mj_integratePos(model, data.qpos, dv, 1.0)
        mujoco.mj_forward(model, data)
    else:
        print(f"Max iters reached, pos_error={np.linalg.norm(target_pos - data.xpos[eef_id]):.5f}")

    return data.qpos.copy()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-freeze", dest="freeze", action="store_false",
                        help="Allow all joints to move (default: waist + right arm only)")
    parser.add_argument("--frozen-legs", dest="frozen_legs", action="store_true",
                        help="Remove leg joints from XML at runtime (writes to /tmp/g1_edited.xml)")
    parser.set_defaults(freeze=True, frozen_legs=False)
    args = parser.parse_args()

    # Generate the XML with leg joints stripped out, or use the original.
    xml_path = generate_frozen_legs_xml() if args.frozen_legs else XML_PATH
    print(f"XML         : {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # Equality constraints are only enforced during mj_step (dynamics).
    # The IK loop uses mj_forward only, so we must set qpos directly to
    # put the robot in the desired initial pose.
    if args.frozen_legs:
        for name, val in LEG_JOINTS.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = val

    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   BASE_BODY)
    eef_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   EEF_BODY)
    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)

    target_pos  = np.array([0.4, -0.3, 0.8])
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])

    print(f"Freeze mode : {args.freeze}")
    print(f"Frozen legs : {args.frozen_legs}")
    print(f"Target pos  : {target_pos}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        qpos = solve_ik(model, data, target_pos, target_quat,
                        freeze=args.freeze, viewer=viewer,
                        base_id=base_id, eef_id=eef_id, cam_id=cam_id)

        print("IK done — close window to exit.")
        while viewer.is_running():
            update_scene(viewer, data, base_id, eef_id, cam_id, target_pos)
            viewer.sync()
            time.sleep(0.016)

    np.set_printoptions(precision=4, suppress=True)
    print("\nFinal joint positions (qpos):")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        adr  = model.jnt_qposadr[i]
        nq   = 7 if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE else 1
        print(f"  {name:40s}: {qpos[adr:adr+nq]}")
