"""
ik_solver.py — Config-driven differential IK for MuJoCo robots.

Reads a YAML config (or an IKConfig dataclass) specifying:
  - robot XML path and optional leg-freeze settings
  - one or more target frames  (body name + desired position / orientation)
  - which joints are active (to optimize)
  - initial robot pose
  - DLS solver hyperparameters

Multiple simultaneous targets are handled by stacking Jacobians so all
targets are minimized together in a single DLS solve per iteration.

Public API
----------
    cfg = IKConfig.from_yaml("ik/configs/g1_right_arm.yaml")
    solver = IKSolver(cfg)
    result = solver.solve()          # use targets from config
    result = solver.solve(targets=[  # override at call time
        TargetFrame("right_eef", "right_wrist_yaw_link",
                    np.array([0.4, -0.3, 0.8]))
    ])
    print(result.joint_angles)       # dict {name: angle_rad}
    print(result.qpos)               # full qpos array
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import mujoco
import yaml


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TargetFrame:
    """One IK target: move *body* to *position* (and optionally *orientation*)."""
    name: str
    body: str
    position: np.ndarray      # shape (3,) — world-frame xyz
    orientation: Optional[np.ndarray] = None  # shape (4,) — wxyz; None = unconstrained
    pos_weight: float = 1.0
    ori_weight: float = 0.5

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.orientation is not None:
            self.orientation = np.asarray(self.orientation, dtype=float)
            norm = np.linalg.norm(self.orientation)
            if norm > 0:
                self.orientation /= norm

    @classmethod
    def from_dict(cls, d: dict) -> "TargetFrame":
        ori_raw = d.get("orientation")
        ori = np.array(ori_raw, dtype=float) if ori_raw is not None else None
        return cls(
            name=str(d["name"]),
            body=str(d["body"]),
            position=np.array(d["position"], dtype=float),
            orientation=ori,
            pos_weight=float(d.get("pos_weight", 1.0)),
            ori_weight=float(d.get("ori_weight", 0.5)),
        )


@dataclass
class IKResult:
    """Output of a single IK solve."""
    joint_angles: dict[str, float]   # active joint name → angle (rad)
    qpos: np.ndarray                  # full qpos vector (all joints)
    pos_errors: dict[str, float]      # target name → final position error (m)
    converged: bool
    iterations: int


@dataclass
class IKConfig:
    """Complete specification for one IK problem."""
    xml_path: str
    targets: list[TargetFrame]
    active_joints: list[str]
    initial_joints: dict[str, float]  # joint_name → initial angle (rad)
    base_pos: np.ndarray              # xyz for free joint
    base_quat: np.ndarray             # wxyz for free joint
    max_iter: int = 200
    tolerance: float = 1e-3
    damping: float = 0.05
    max_dq: float = 0.5
    freeze_legs: bool = False
    frozen_leg_joints: dict[str, float] = field(default_factory=dict)

    # ── default frozen-leg angles (standing pose with locked knees) ──────────
    _DEFAULT_FROZEN_LEGS: dict[str, float] = field(default_factory=lambda: {
        "left_hip_pitch_joint":  -math.pi / 2,
        "left_hip_roll_joint":    0.0,
        "left_hip_yaw_joint":     0.0,
        "left_knee_joint":        math.pi / 2,
        "right_hip_pitch_joint": -math.pi / 2,
        "right_hip_roll_joint":   0.0,
        "right_hip_yaw_joint":    0.0,
        "right_knee_joint":       math.pi / 2,
    })

    @classmethod
    def from_yaml(cls, path: str, repo_root: Optional[str] = None) -> "IKConfig":
        """Load from a YAML file.
        Relative paths inside the YAML are resolved against *repo_root*
        (defaults to two levels above the YAML, i.e. ik/ → repo root)."""
        path = os.path.abspath(path)
        with open(path) as f:
            cfg = yaml.safe_load(f)

        if repo_root is None:
            # ik/configs/foo.yaml → ik/ → repo root
            repo_root = str(Path(path).parent.parent.parent)

        # Robot section
        robot_cfg = cfg.get("robot", {})
        xml_path = robot_cfg.get("xml", "src/assets/robots/unitree_g1/xmls/g1.xml")
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(repo_root, xml_path)
        freeze_legs = bool(robot_cfg.get("freeze_legs", False))

        # Default frozen-leg angles, overridable from config
        _H = math.pi / 2
        frozen_leg_joints: dict[str, float] = {
            "left_hip_pitch_joint":  -_H, "left_hip_roll_joint":  0.0, "left_hip_yaw_joint":  0.0,
            "left_knee_joint":        _H,
            "right_hip_pitch_joint": -_H, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
            "right_knee_joint":       _H,
        }
        frozen_leg_joints.update(
            {k: float(v) for k, v in robot_cfg.get("frozen_leg_joints", {}).items()}
        )

        # Targets
        targets = [TargetFrame.from_dict(t) for t in cfg.get("targets", [])]

        # Active joints
        active_joints = list(cfg.get("joints", {}).get("active", []))

        # Initial pose
        init_cfg = cfg.get("initial_pose", {})
        base_pos  = np.array(init_cfg.get("base_pos",  [0.0, 0.0, 0.8]), dtype=float)
        base_quat = np.array(init_cfg.get("base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)
        initial_joints = {k: float(v) for k, v in init_cfg.get("joints", {}).items()}

        # Solver params
        solver_cfg = cfg.get("solver", {})

        return cls(
            xml_path=xml_path,
            targets=targets,
            active_joints=active_joints,
            initial_joints=initial_joints,
            base_pos=base_pos,
            base_quat=base_quat,
            max_iter=int(solver_cfg.get("max_iter", 200)),
            tolerance=float(solver_cfg.get("tolerance", 1e-3)),
            damping=float(solver_cfg.get("damping", 0.05)),
            max_dq=float(solver_cfg.get("max_dq", 0.5)),
            freeze_legs=freeze_legs,
            frozen_leg_joints=frozen_leg_joints,
        )


# ── XML helpers ───────────────────────────────────────────────────────────────

def _build_freeze_legs_xml(src: str, frozen_joints: dict[str, float],
                            dst: str = "/tmp/ik_freeze_legs.xml") -> str:
    """
    Write *src* XML to *dst* with:
      - meshdir rewritten to an absolute path
      - a floor geom added (if absent)
      - a weld constraint fixing pelvis to world
      - equality/joint constraints pinning each leg joint to its given value

    The free joint remains in the model so qpos indices are stable.
    Returns *dst*.
    """
    ET.register_namespace("", "")
    tree = ET.parse(src)
    root = tree.getroot()

    src_dir = os.path.abspath(os.path.dirname(src))
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")
        if meshdir and not os.path.isabs(meshdir):
            compiler.set("meshdir", os.path.join(src_dir, meshdir))

    worldbody = root.find("worldbody")
    if worldbody is not None and not any(
        g.get("name") == "floor" for g in worldbody.findall("geom")
    ):
        ET.SubElement(worldbody, "geom", name="floor", type="plane",
                      size="3 3 0.1", pos="0 0 0", rgba="0.8 0.8 0.8 1", condim="3")

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    existing_welds = {(el.get("body1"), el.get("body2")) for el in equality.findall("weld")}
    if ("world", "pelvis") not in existing_welds:
        ET.SubElement(equality, "weld", body1="world", body2="pelvis")

    existing_joints = {el.get("joint1") for el in equality.findall("joint")}
    for jname, val in frozen_joints.items():
        if jname not in existing_joints:
            ET.SubElement(equality, "joint", joint1=jname,
                          polycoef=f"{val:.6f} 0 0 0 0")

    tree.write(dst, encoding="unicode", xml_declaration=False)
    return dst


# ── Math helpers ──────────────────────────────────────────────────────────────

def _orientation_error(xmat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    """3-vector orientation error: 2 * vec(q_target * inv(q_body))."""
    q_body = np.zeros(4)
    mujoco.mju_mat2Quat(q_body, xmat)
    q_inv = np.array([q_body[0], -q_body[1], -q_body[2], -q_body[3]])
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, target_quat, q_inv)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:]


def _ik_step_multi(model: mujoco.MjModel, data: mujoco.MjData,
                   targets: list[TargetFrame], body_ids: list[int],
                   dof_ids: np.ndarray, damping: float, max_dq: float) -> np.ndarray:
    """
    One DLS IK step for multiple simultaneous targets.

    Jacobians from all targets are stacked (weighted) and solved together:
        JTJ = sum_i (wp_i² Jp_i^T Jp_i + wo_i² Jr_i^T Jr_i) + λ² I
        JTe = sum_i (wp_i² Jp_i^T ep_i + wo_i² Jr_i^T er_i)
        dq  = JTJ \ JTe   (clipped to ±max_dq)
    """
    n = len(dof_ids)
    JTJ  = damping ** 2 * np.eye(n)
    JTdx = np.zeros(n)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for tgt, bid in zip(targets, body_ids):
        mujoco.mj_jacBody(model, data, jacp, jacr, bid)
        Jp = jacp[:, dof_ids]
        Jr = jacr[:, dof_ids]

        ep = tgt.position - data.xpos[bid]
        wp2 = tgt.pos_weight ** 2
        JTJ  += wp2 * (Jp.T @ Jp)
        JTdx += wp2 * (Jp.T @ ep)

        if tgt.orientation is not None:
            er = _orientation_error(data.xmat[bid], tgt.orientation)
            wo2 = tgt.ori_weight ** 2
            JTJ  += wo2 * (Jr.T @ Jr)
            JTdx += wo2 * (Jr.T @ er)

    dq = np.linalg.solve(JTJ, JTdx)
    return np.clip(dq, -max_dq, max_dq)


# ── IK Solver ─────────────────────────────────────────────────────────────────

class IKSolver:
    """
    Config-driven differential IK solver.

    Parameters
    ----------
    config : IKConfig
        Fully-populated config (load with IKConfig.from_yaml or build manually).
    verbose : bool
        Print per-iteration progress.
    """

    def __init__(self, config: IKConfig, verbose: bool = False):
        self.cfg = config
        self.verbose = verbose

        # Optionally patch the XML to freeze legs
        xml_path = config.xml_path
        if config.freeze_legs:
            xml_path = _build_freeze_legs_xml(xml_path, config.frozen_leg_joints)
            if verbose:
                print(f"[IKSolver] freeze_legs=True → patched XML: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # Pre-resolve body IDs for the targets defined in config
        self._config_body_ids = self._resolve_body_ids(config.targets)

        # Pre-resolve DOF ids for active joints
        self.dof_ids = self._resolve_dof_ids(config.active_joints)

        if verbose:
            print(f"[IKSolver] active joints ({len(config.active_joints)}): "
                  f"{config.active_joints}")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _resolve_body_ids(self, targets: list[TargetFrame]) -> list[int]:
        ids = []
        for t in targets:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, t.body)
            if bid < 0:
                raise ValueError(f"Body '{t.body}' not found in model.")
            ids.append(bid)
        return ids

    def _resolve_dof_ids(self, joint_names: list[str]) -> np.ndarray:
        ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in model.")
            ids.append(self.model.jnt_dofadr[jid])
        return np.array(ids, dtype=int)

    def _set_initial_pose(self) -> None:
        """Reset data.qpos to the configured initial pose."""
        self.data.qpos[:] = 0.0

        # Free joint (if present): set base position and orientation
        if self.model.nq > self.model.njnt - 1:
            # Check if first joint is a free joint
            if self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
                self.data.qpos[0:3] = self.cfg.base_pos
                self.data.qpos[3:7] = self.cfg.base_quat  # wxyz

        # Set named joint angles
        for jname, val in self.cfg.initial_joints.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = val

        # For freeze_legs mode, also set the frozen leg joints in qpos
        # (equality constraints are only enforced during mj_step, not mj_forward)
        if self.cfg.freeze_legs:
            for jname, val in self.cfg.frozen_leg_joints.items():
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = val

        mujoco.mj_forward(self.model, self.data)

    def _collect_joint_angles(self, joint_names: list[str]) -> dict[str, float]:
        return {
            name: float(self.data.qpos[
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            ])
            for name in joint_names
        }

    # ── public API ────────────────────────────────────────────────────────────

    def solve(
        self,
        targets: Optional[list[TargetFrame]] = None,
        warm_start: Optional[np.ndarray] = None,
    ) -> IKResult:
        """
        Run the IK solver.

        Parameters
        ----------
        targets : list[TargetFrame] | None
            Override the targets from the config.  If None, uses config targets.
        warm_start : np.ndarray | None
            Initial qpos.  If None, uses the configured initial_pose.

        Returns
        -------
        IKResult
            .joint_angles  — dict {joint_name: angle_rad} for active joints
            .qpos          — full qpos array
            .pos_errors    — dict {target_name: final_position_error_m}
            .converged     — True if all targets reached tolerance
            .iterations    — number of iterations used
        """
        if targets is None:
            targets     = self.cfg.targets
            body_ids    = self._config_body_ids
        else:
            body_ids = self._resolve_body_ids(targets)

        if not targets:
            raise ValueError("No targets specified.")

        if warm_start is not None:
            self.data.qpos[:] = warm_start
            mujoco.mj_forward(self.model, self.data)
        else:
            self._set_initial_pose()

        converged  = False
        iterations = 0

        for i in range(self.cfg.max_iter):
            iterations = i + 1
            max_pos_err = max(
                np.linalg.norm(t.position - self.data.xpos[bid])
                for t, bid in zip(targets, body_ids)
            )

            if self.verbose and i % 20 == 0:
                print(f"  iter {i:4d}  max_pos_err={max_pos_err:.5f} m")

            if max_pos_err < self.cfg.tolerance:
                converged = True
                if self.verbose:
                    print(f"  Converged at iter {i}  max_pos_err={max_pos_err:.5f} m")
                break

            dq = _ik_step_multi(
                self.model, self.data, targets, body_ids,
                self.dof_ids, self.cfg.damping, self.cfg.max_dq,
            )
            dv = np.zeros(self.model.nv)
            dv[self.dof_ids] = dq
            mujoco.mj_integratePos(self.model, self.data.qpos, dv, 1.0)
            mujoco.mj_forward(self.model, self.data)

        if not converged and self.verbose:
            max_pos_err = max(
                np.linalg.norm(t.position - self.data.xpos[bid])
                for t, bid in zip(targets, body_ids)
            )
            print(f"  Max iters reached  max_pos_err={max_pos_err:.5f} m")

        pos_errors = {
            t.name: float(np.linalg.norm(t.position - self.data.xpos[bid]))
            for t, bid in zip(targets, body_ids)
        }

        return IKResult(
            joint_angles=self._collect_joint_angles(self.cfg.active_joints),
            qpos=self.data.qpos.copy(),
            pos_errors=pos_errors,
            converged=converged,
            iterations=iterations,
        )

    def solve_batch(
        self,
        targets_list: list[list[TargetFrame]],
        warm_chain: bool = True,
    ) -> list[IKResult]:
        """
        Solve IK for a sequence of target sets.

        Parameters
        ----------
        targets_list : list of list[TargetFrame]
            Each element is a set of targets for one waypoint.
        warm_chain : bool
            If True, use the previous solution as the warm start for the next.

        Returns
        -------
        list[IKResult]
        """
        results: list[IKResult] = []
        warm_start = None

        for i, targets in enumerate(targets_list):
            result = self.solve(targets=targets, warm_start=warm_start)
            results.append(result)
            if warm_chain:
                warm_start = result.qpos
            if self.verbose:
                status = "OK" if result.converged else "!"
                errs = "  ".join(
                    f"{k}={v:.4f}" for k, v in result.pos_errors.items()
                )
                print(f"  [{status}] waypoint {i:4d}  {errs}")

        return results

    # ── convenience: all joint names in the model ─────────────────────────────

    def joint_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]

    def body_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.model.nbody)
        ]
