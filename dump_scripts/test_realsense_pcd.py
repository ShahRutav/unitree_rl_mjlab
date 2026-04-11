"""
Capture RGB + Depth from a RealSense camera connected via USB-C, transform
the point cloud into the robot base (pelvis) frame, and visualize with Plotly.

Usage (from repo root):
    python dump_scripts/test_realsense_pcd.py

Dependencies:
    pip install pyrealsense2 plotly mujoco numpy
"""

import numpy as np
import pyrealsense2 as rs  # type: ignore
import mujoco
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

XML_PATH    = "src/assets/robots/unitree_g1/xmls/g1.xml"
CAMERA_NAME = "head_camera"
BASE_BODY   = "pelvis"


# ── RealSense ─────────────────────────────────────────────────────────────────

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, warmup=10):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16,  fps)

        profile          = self.pipeline.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align       = rs.align(rs.stream.color)

        intr       = (profile.get_stream(rs.stream.color)
                              .as_video_stream_profile().get_intrinsics())
        self.fx, self.fy = intr.fx, intr.fy
        self.cx, self.cy = intr.ppx, intr.ppy
        self.W, self.H   = width, height

        print(f"[RealSense] {width}x{height}@{fps}fps | depth_scale={self.depth_scale:.5f} m/unit")
        print(f"[RealSense] fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}")

        for _ in range(warmup):
            self.pipeline.wait_for_frames()

    def capture(self):
        """Returns (rgb (H,W,3) uint8, depth (H,W) float32 in metres)."""
        frames  = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        rgb   = np.asanyarray(aligned.get_color_frame().get_data())
        depth = (np.asanyarray(aligned.get_depth_frame().get_data())
                   .astype(np.float32) * self.depth_scale)
        return rgb, depth

    def to_pointcloud(self, rgb, depth, max_depth=3.0):
        """Back-project to camera frame (X right, Y down, Z forward).
        Returns points (N,3) float32 and colors (N,3) uint8."""
        uu, vv = np.meshgrid(np.arange(self.W, dtype=np.float32),
                             np.arange(self.H, dtype=np.float32))
        z = depth
        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy
        pts    = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = rgb.reshape(-1, 3)
        valid  = (z.ravel() > 0) & (z.ravel() < max_depth)
        return pts[valid].astype(np.float32), colors[valid]

    def stop(self):
        self.pipeline.stop()


# ── Camera-to-base transform ──────────────────────────────────────────────────

def get_camera_to_base(xml_path=XML_PATH, joint_positions=None):
    """Return T_base_camera (4x4) from the MuJoCo model."""
    model = mujoco.MjModel.from_xml_path(xml_path)  # type: ignore
    data  = mujoco.MjData(model)                     # type: ignore
    if joint_positions is not None:
        data.qpos[:len(joint_positions)] = joint_positions
    mujoco.mj_forward(model, data)

    def T(pos, mat):
        M = np.eye(4)
        M[:3, :3] = mat.reshape(3, 3)
        M[:3,  3] = pos
        return M

    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)  # type: ignore
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   BASE_BODY)    # type: ignore
    return np.linalg.inv(T(data.xpos[base_id], data.xmat[base_id])) \
           @ T(data.cam_xpos[cam_id], data.cam_xmat[cam_id])


# ── Plotly visualization ──────────────────────────────────────────────────────

def plotly_draw_3d_pcd(pcd_points, pcd_colors=None, addition_points=None,
                        marker_size=3, equal_axis=True, title="",
                        offline=False, no_background=False,
                        default_rgb_str="(255,0,0)",
                        additional_point_draw_lines=False, uniform_color=False):
    if pcd_colors is None:
        color_str = [f'rgb{default_rgb_str}'] * len(pcd_points)
    else:
        color_str = [f'rgb({r},{g},{b})' for r, g, b in pcd_colors]

    data = [go.Scatter3d(
        x=pcd_points[:, 0], y=pcd_points[:, 1], z=pcd_points[:, 2],
        mode='markers', marker=dict(size=3, color=color_str, opacity=0.8),
    )]

    if addition_points is not None:
        assert addition_points.shape[-1] == 3
        if addition_points.ndim == 2:
            addition_points = [addition_points]
        for pts in addition_points:
            md = dict(size=marker_size, opacity=0.8)
            if uniform_color:
                md["color"] = f'rgb{default_rgb_str}'
            data.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="lines+markers" if additional_point_draw_lines else "markers",
                marker=md,
            ))

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(aspectmode='data') if equal_axis else {},
            title=dict(text=title, automargin=True),
        ),
    )
    if no_background:
        ax = dict(showbackground=False, zeroline=False, showgrid=False,
                  showticklabels=False, showaxeslabels=False, visible=False)
        fig.update_layout(
            scene=dict(xaxis=ax, yaxis=ax, zaxis=ax),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, b=0, t=0), showlegend=False,
        )
    return fig if offline else fig.show()


# ── Capture helper ────────────────────────────────────────────────────────────

def capture_pcd(cam: RealSenseCamera, T_base_cam: np.ndarray,
                max_depth=3.0, max_pts=50_000):
    rgb, depth      = cam.capture()
    pts_cam, colors = cam.to_pointcloud(rgb, depth, max_depth=max_depth)

    # OpenCV convention (X right, Y down, Z fwd) → MuJoCo camera convention (X right, Y up, Z bwd)
    pts_cam_muj = pts_cam * np.array([1, -1, -1], dtype=np.float32)
    ones        = np.ones((len(pts_cam_muj), 1), dtype=np.float32)
    pts_base    = (T_base_cam @ np.hstack([pts_cam_muj, ones]).T).T[:, :3]

    if len(pts_base) > max_pts:
        idx      = np.random.choice(len(pts_base), max_pts, replace=False)
        pts_base = pts_base[idx]
        colors   = colors[idx]

    return pts_base, colors


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    REFRESH_MS = 15_000   # milliseconds
    MAX_DEPTH  = 3.0
    MAX_PTS    = 50_000

    print("Computing camera→base transform ...")
    T_base_cam = get_camera_to_base()
    np.set_printoptions(precision=4, suppress=True)
    print("T_base_camera =\n", T_base_cam)

    cam = RealSenseCamera()

    # ── Dash app ──────────────────────────────────────────────────────────────
    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id="pcd", style={"height": "95vh"}),
        dcc.Interval(id="timer", interval=REFRESH_MS, n_intervals=0),
    ])

    @app.callback(Output("pcd", "figure"), Input("timer", "n_intervals"))
    def update(_):
        pts, colors = capture_pcd(cam, T_base_cam, MAX_DEPTH, MAX_PTS)
        print(f"  refreshed: {len(pts):,} points")
        return plotly_draw_3d_pcd(pts, pcd_colors=colors,
                                   title="Point Cloud — Robot Base (Pelvis) Frame",
                                   offline=True)

    print(f"\nDash server running → http://127.0.0.1:8050  (refresh every {REFRESH_MS//1000}s)")
    app.run(debug=False)
