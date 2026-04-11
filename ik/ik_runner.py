"""
ik_runner.py — CLI interface for the config-driven IK solver.

Loads an IK YAML config, optionally overrides target positions from the
command line, runs the solver, and prints the resulting joint angles.
Can also launch a MuJoCo passive viewer to visualise the solve.

Usage examples
--------------
# Solve with config defaults:
    python ik/ik_runner.py ik/configs/g1_right_arm.yaml

# Override a single target's position at the command line:
    python ik/ik_runner.py ik/configs/g1_right_arm.yaml \
        --target right_eef 0.45 -0.25 0.85

# Override multiple targets:
    python ik/ik_runner.py ik/configs/g1_dual_arm.yaml \
        --target right_eef 0.45 -0.25 0.85 \
        --target left_eef  0.45  0.25 0.85

# Live viewer:
    python ik/ik_runner.py ik/configs/g1_right_arm.yaml --viewer

# Show available bodies / joints in the model, then exit:
    python ik/ik_runner.py ik/configs/g1_right_arm.yaml --list-bodies
    python ik/ik_runner.py ik/configs/g1_right_arm.yaml --list-joints
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer as mj_viewer

from ik.ik_solver import IKConfig, IKResult, IKSolver, TargetFrame


# ── Viewer helpers ────────────────────────────────────────────────────────────

def _draw_sphere(scene, pos: np.ndarray, size: float, rgba) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, size, size]),
                        pos, np.eye(3).flatten(),
                        np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


def _draw_frame(scene, pos: np.ndarray, mat: np.ndarray,
                scale: float = 0.10, label: str = "") -> None:
    mat33 = mat.reshape(3, 3)
    colors = [
        [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
    ]
    for i, rgba in enumerate(colors):
        if scene.ngeom >= len(scene.geoms):
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                            np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006,
                             pos, pos + mat33[:, i] * scale)
        if i == 0 and label:
            g.label = label[:30]
        scene.ngeom += 1


def _update_scene(viewer, data, targets: list[TargetFrame], body_ids: list[int]) -> None:
    """Refresh user_scn: EEF frames + red target spheres."""
    scn = viewer.user_scn
    scn.ngeom = 0
    for tgt, bid in zip(targets, body_ids):
        # Current EEF frame (cyan sphere + axes)
        _draw_sphere(scn, data.xpos[bid], 0.018, [0, 0.8, 1, 1])
        _draw_frame(scn, data.xpos[bid], data.xmat[bid], scale=0.10, label=tgt.name)
        # Target sphere (red)
        _draw_sphere(scn, tgt.position, 0.025, [1, 0.2, 0.2, 0.8])


# ── Solve with optional viewer ────────────────────────────────────────────────

def solve_with_viewer(solver: IKSolver, targets: list[TargetFrame]) -> IKResult:
    """Run IK with a live MuJoCo viewer open, syncing every iteration."""
    body_ids = solver._resolve_body_ids(targets)
    solver._set_initial_pose()

    # Kick off a passive viewer
    with mj_viewer.launch_passive(solver.model, solver.data) as viewer:
        viewer.sync()

        converged  = False
        iterations = 0
        cfg = solver.cfg

        for i in range(cfg.max_iter):
            iterations = i + 1
            max_pos_err = max(
                np.linalg.norm(t.position - solver.data.xpos[bid])
                for t, bid in zip(targets, body_ids)
            )

            if viewer.is_running():
                _update_scene(viewer, solver.data, targets, body_ids)
                viewer.sync()
                time.sleep(0.03)

            if max_pos_err < cfg.tolerance:
                converged = True
                print(f"  Converged at iter {i}  max_pos_err={max_pos_err:.5f} m")
                break

            from ik.ik_solver import _ik_step_multi
            dq = _ik_step_multi(
                solver.model, solver.data, targets, body_ids,
                solver.dof_ids, cfg.damping, cfg.max_dq,
            )
            dv = np.zeros(solver.model.nv)
            dv[solver.dof_ids] = dq
            mujoco.mj_integratePos(solver.model, solver.data.qpos, dv, 1.0)
            mujoco.mj_forward(solver.model, solver.data)

        if not converged:
            max_pos_err = max(
                np.linalg.norm(t.position - solver.data.xpos[bid])
                for t, bid in zip(targets, body_ids)
            )
            print(f"  Max iters reached  max_pos_err={max_pos_err:.5f} m")

        pos_errors = {
            t.name: float(np.linalg.norm(t.position - solver.data.xpos[bid]))
            for t, bid in zip(targets, body_ids)
        }

        result = IKResult(
            joint_angles=solver._collect_joint_angles(cfg.active_joints),
            qpos=solver.data.qpos.copy(),
            pos_errors=pos_errors,
            converged=converged,
            iterations=iterations,
        )

        print("\nIK done — close viewer window to exit.")
        while viewer.is_running():
            _update_scene(viewer, solver.data, targets, body_ids)
            viewer.sync()
            time.sleep(0.016)

    return result


# ── Result printing ───────────────────────────────────────────────────────────

def print_result(result: IKResult) -> None:
    print("\n" + "=" * 60)
    status = "CONVERGED" if result.converged else "DID NOT CONVERGE"
    print(f"  IK result: {status}  (iters={result.iterations})")
    print()
    print("  Position errors:")
    for name, err in result.pos_errors.items():
        flag = "✓" if err < 1e-3 else "✗"
        print(f"    {flag}  {name:30s}  {err:.5f} m")
    print()
    print("  Joint angles (active joints):")
    for name, angle in result.joint_angles.items():
        print(f"    {name:40s}  {angle:+.6f} rad  ({np.degrees(angle):+.2f} deg)")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Config-driven IK solver for MuJoCo robots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", help="Path to IK YAML config file.")
    p.add_argument(
        "--target", metavar=("NAME", "X", "Y", "Z"),
        nargs=4, action="append", default=[],
        help=(
            "Override target position: NAME X Y Z (in metres). "
            "NAME must match a target defined in the config. "
            "Repeat for multiple targets."
        ),
    )
    p.add_argument(
        "--viewer", action="store_true",
        help="Open a MuJoCo passive viewer during the solve.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-iteration progress.",
    )
    p.add_argument(
        "--list-joints", action="store_true",
        help="Print all joint names in the model and exit.",
    )
    p.add_argument(
        "--list-bodies", action="store_true",
        help="Print all body names in the model and exit.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = IKConfig.from_yaml(args.config)
    solver = IKSolver(cfg, verbose=args.verbose)

    # --list-* queries
    if args.list_joints:
        print("Joints in model:")
        for name in solver.joint_names():
            print(f"  {name}")
        return
    if args.list_bodies:
        print("Bodies in model:")
        for name in solver.body_names():
            print(f"  {name}")
        return

    # Build target list (start from config defaults)
    targets: list[TargetFrame] = list(cfg.targets)

    # Apply --target overrides
    overrides: dict[str, np.ndarray] = {}
    for name, x, y, z in args.target:
        overrides[name] = np.array([float(x), float(y), float(z)])

    if overrides:
        updated = []
        for t in targets:
            if t.name in overrides:
                import dataclasses
                t = dataclasses.replace(t, position=overrides.pop(t.name))
            updated.append(t)
        targets = updated
        if overrides:
            print(f"WARNING: unknown target name(s) in --target: {list(overrides)}")

    if not targets:
        print("ERROR: no targets specified (add them to the config or via --target).")
        sys.exit(1)

    # Print what we're solving
    print(f"Config  : {args.config}")
    print(f"XML     : {cfg.xml_path}")
    print(f"Targets ({len(targets)}):")
    for t in targets:
        print(f"  {t.name:30s}  body={t.body}  pos={t.position}")
    print(f"Active joints ({len(cfg.active_joints)}): {cfg.active_joints}")
    print()

    # Solve
    if args.viewer:
        result = solve_with_viewer(solver, targets)
    else:
        result = solver.solve(targets=targets)

    print_result(result)


if __name__ == "__main__":
    main()
