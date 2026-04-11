"""
Script 1: Transforms from robot camera / right EEF to robot base.
Viewer always shows: pelvis frame, right EEF frame, camera frame.

Bodies used:
  - pelvis             : robot base
  - right_wrist_yaw_link : right end-effector (last wrist body)
  - head_camera        : camera on torso_link (head position)
"""
import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH    = "src/assets/robots/unitree_g1/xmls/g1.xml"
CAMERA_NAME = "head_camera"
EEF_BODY    = "right_wrist_yaw_link"
BASE_BODY   = "pelvis"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_T(pos: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position and 3x3 rotation."""
    T = np.eye(4)
    T[:3, :3] = mat.reshape(3, 3)
    T[:3,  3] = pos
    return T


def draw_frame(scene, pos: np.ndarray, mat: np.ndarray,
               scale: float = 0.12, label: str = "") -> None:
    """Draw XYZ coordinate axes as colored arrows (X=red, Y=green, Z=blue)."""
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


def draw_sphere(scene, pos: np.ndarray, size: float, rgba) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, size, size]),
                        pos, np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


def draw_frame_with_primary(scene, pos: np.ndarray, mat: np.ndarray,
                            primary_dir: np.ndarray, primary_rgba,
                            scale: float = 0.12) -> None:
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


# ── Transform functions ───────────────────────────────────────────────────────

def get_camera_to_base(model, data) -> np.ndarray:
    """Return T_base_camera (4x4): head_camera frame expressed in pelvis frame."""
    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   BASE_BODY)
    T_world_cam  = make_T(data.cam_xpos[cam_id], data.cam_xmat[cam_id])
    T_world_base = make_T(data.xpos[base_id],    data.xmat[base_id])
    return np.linalg.inv(T_world_base) @ T_world_cam


def get_right_eef_to_base(model, data) -> np.ndarray:
    """Return T_base_eef (4x4): right_wrist_yaw_link frame expressed in pelvis frame."""
    eef_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EEF_BODY)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    T_world_eef  = make_T(data.xpos[eef_id],  data.xmat[eef_id])
    T_world_base = make_T(data.xpos[base_id], data.xmat[base_id])
    return np.linalg.inv(T_world_base) @ T_world_eef


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    np.set_printoptions(precision=4, suppress=True)

    T_cam = get_camera_to_base(model, data)
    print("=== Camera → Base (T_base_camera) ===")
    print(T_cam)

    T_eef = get_right_eef_to_base(model, data)
    print("\n=== Right EEF → Base (T_base_eef) ===")
    print(T_eef)

    # Cache IDs for viewer loop
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   BASE_BODY)
    eef_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   EEF_BODY)
    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)

    print("\nViewer open — close window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            scn = viewer.user_scn
            scn.ngeom = 0

            R_base = data.xmat[base_id].reshape(3, 3)
            R_eef  = data.xmat[eef_id].reshape(3, 3)
            R_cam  = data.cam_xmat[cam_id].reshape(3, 3)

            # 1. Pelvis — white sphere; white arrow = robot forward (+X)
            draw_sphere(scn, data.xpos[base_id], 0.025, [1, 1, 1, 1])
            draw_frame_with_primary(scn, data.xpos[base_id], R_base,
                                    primary_dir=R_base[:, 0],
                                    primary_rgba=[1, 1, 1, 1], scale=0.15)

            # 2. Right EEF — cyan sphere; white arrow = hand approach (+X)
            draw_sphere(scn, data.xpos[eef_id], 0.020, [0, 0.8, 1, 1])
            draw_frame_with_primary(scn, data.xpos[eef_id], R_eef,
                                    primary_dir=R_eef[:, 0],
                                    primary_rgba=[1, 1, 1, 1], scale=0.12)

            # 3. Camera — orange sphere; yellow arrow = viewing direction (-Z)
            draw_sphere(scn, data.cam_xpos[cam_id], 0.020, [1, 0.5, 0, 1])
            draw_frame_with_primary(scn, data.cam_xpos[cam_id], R_cam,
                                    primary_dir=-R_cam[:, 2],
                                    primary_rgba=[1, 1, 0, 1], scale=0.12)

            viewer.sync()
            time.sleep(0.016)
