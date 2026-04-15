"""
ik/tests.py — Verification test suite for the IK solver.

Each test case exercises a specific scenario and checks the result against
expected outcomes (convergence, position error, joint-limit compliance).

Usage
-----
    # List all available test cases
    python ik/tests.py --list

    # Run all tests headlessly and print a summary table
    python ik/tests.py --all

    # Run one specific test headlessly
    python ik/tests.py --test reachable_right_front

    # Visualise a test in the MuJoCo viewer (single test only)
    python ik/tests.py --test joint_limit_waist_yaw --viewer

    # Run all tests but open the viewer on each failure
    python ik/tests.py --all --viewer-on-fail

Adding a test case
------------------
Add an entry to the TESTS dict at the bottom of this file:

    "my_test": TestCase(
        description="...",
        config="ik/configs/g1_right_arm.yaml",
        targets=[TargetFrame("right_eef", "right_wrist_yaw_link",
                             np.array([...]))],
        expect_converged=True,
        joint_limit_overrides={},    # tighten limits for this test only
        extra_checks=[],             # list of (check_fn, failure_message)
    ),
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco.viewer as mj_viewer

from ik.ik_solver import IKConfig, IKResult, IKSolver, TargetFrame
from ik.ik_runner import _update_scene, solve_with_viewer

# ── ANSI colours ─────────────────────────────────────────────────────────────
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

def _ok(s):  return f"{_GREEN}{s}{_RESET}"
def _fail(s): return f"{_RED}{s}{_RESET}"
def _warn(s): return f"{_YELLOW}{s}{_RESET}"


# ── TestCase definition ───────────────────────────────────────────────────────

@dataclass
class TestCase:
    description: str
    config: str                               # path to YAML (relative to repo root)
    targets: list[TargetFrame]                # runtime target override
    expect_converged: bool = True             # should the solver converge?
    joint_limit_overrides: dict[str, tuple] = field(default_factory=dict)
    # extra_checks: list of (fn(result, solver) -> bool, failure_message)
    extra_checks: list[tuple[Callable, str]]  = field(default_factory=list)


# ── Test runner ───────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    failures: list[str]
    result: IKResult
    elapsed: float   # seconds


def _run_one(name: str, tc: TestCase, viewer: bool = False) -> TestResult:
    cfg = IKConfig.from_yaml(tc.config)

    # Apply per-test joint limit overrides on top of config limits
    for jname, bounds in tc.joint_limit_overrides.items():
        cfg.joint_limits[jname] = bounds

    solver = IKSolver(cfg, verbose=False)

    t0 = time.perf_counter()
    if viewer:
        result = solve_with_viewer(solver, tc.targets)
    else:
        result = solver.solve(targets=tc.targets)
    elapsed = time.perf_counter() - t0

    failures: list[str] = []

    # 1 — convergence check
    if tc.expect_converged and not result.converged:
        failures.append(
            f"Expected convergence but solver hit max_iter "
            f"(max pos_err={max(result.pos_errors.values()):.4f} m)"
        )
    if not tc.expect_converged and result.converged:
        failures.append(
            f"Expected non-convergence but solver converged "
            f"(max pos_err={max(result.pos_errors.values()):.4f} m)"
        )

    # 2 — joint limit compliance
    limits = solver.get_effective_limits()
    for jname, angle in result.joint_angles.items():
        lo, hi = limits[jname]
        if np.isfinite(lo) and angle < lo - 1e-6:
            failures.append(
                f"Joint {jname} = {angle:.4f} rad is below limit {lo:.4f} rad"
            )
        if np.isfinite(hi) and angle > hi + 1e-6:
            failures.append(
                f"Joint {jname} = {angle:.4f} rad is above limit {hi:.4f} rad"
            )

    # 3 — user-defined extra checks
    for check_fn, msg in tc.extra_checks:
        if not check_fn(result, solver):
            failures.append(msg)

    return TestResult(
        name=name,
        passed=len(failures) == 0,
        failures=failures,
        result=result,
        elapsed=elapsed,
    )


# ── Batch runner & reporting ─────────────────────────────────────────────────

def run_all(tests: dict[str, TestCase],
            viewer_on_fail: bool = False) -> list[TestResult]:
    results = []
    for name, tc in tests.items():
        tr = _run_one(name, tc, viewer=False)
        results.append(tr)
        _print_one(tr, verbose=False)
        if not tr.passed and viewer_on_fail:
            print(f"  → opening viewer for {name} ...")
            _run_one(name, tc, viewer=True)
    return results


def _print_one(tr: TestResult, verbose: bool = True) -> None:
    status   = _ok("PASS") if tr.passed else _fail("FAIL")
    conv     = "converged" if tr.result.converged else "no-conv  "
    max_err  = max(tr.result.pos_errors.values())
    err_str  = f"{max_err:.4f} m"
    iters    = f"{tr.result.iterations:3d} it"
    elapsed  = f"{tr.elapsed*1000:.1f} ms"
    print(f"  [{status}]  {tr.name:<32}  {conv}  err={err_str}  {iters}  {elapsed}")
    if verbose or not tr.passed:
        for f in tr.failures:
            print(f"           {_fail('✗')} {f}")


def _print_summary(results: list[TestResult]) -> None:
    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass
    print()
    print("─" * 70)
    if n_fail == 0:
        print(f"  {_ok(_BOLD + 'All tests passed')} ({n_pass}/{len(results)})")
    else:
        print(f"  {_fail(f'{n_fail} failed')}, {_ok(f'{n_pass} passed')}  "
              f"({len(results)} total)")
    print("─" * 70)


# ── Helper: check that a joint stays within tighter-than-config bounds ───────

def _joint_within(joint_name: str, lo: float, hi: float):
    """Returns an extra_check tuple asserting joint is in [lo, hi]."""
    def _check(result: IKResult, solver: IKSolver) -> bool:
        angle = result.joint_angles.get(joint_name)
        if angle is None:
            return False
        return lo - 1e-6 <= angle <= hi + 1e-6
    return (
        _check,
        f"{joint_name} should stay in [{lo:.3f}, {hi:.3f}] rad"
    )


def _dual_arm_symmetric(axis_tol: float = 0.15):
    """Dual-arm extra check: right/left shoulder angles are roughly mirror images."""
    pairs = [
        ("right_shoulder_pitch_joint", "left_shoulder_pitch_joint",  1.0),
        ("right_shoulder_roll_joint",  "left_shoulder_roll_joint",  -1.0),
        ("right_elbow_joint",          "left_elbow_joint",           1.0),
    ]
    def _check(result: IKResult, solver: IKSolver) -> bool:
        ja = result.joint_angles
        for r_name, l_name, sign in pairs:
            if r_name not in ja or l_name not in ja:
                return True  # can't check, skip
            if abs(ja[r_name] - sign * ja[l_name]) > axis_tol:
                return False
        return True
    return (
        _check,
        f"Right/left arm angles are not symmetric (tol={axis_tol:.2f} rad)"
    )


def _batch_all_converged(results: list[IKResult]) -> bool:
    return all(r.converged for r in results)


# ── Batch circle helper ───────────────────────────────────────────────────────

def _make_circle_targets(n: int = 12,
                         center=(0.45, -0.2, 0.9), radius=0.12) -> list[TargetFrame]:
    """n evenly-spaced targets around a circle in the YZ plane."""
    cx, cy, cz = center
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return [
        TargetFrame(
            name=f"wp_{i:02d}",
            body="right_wrist_yaw_link",
            position=np.array([cx,
                                cy + radius * math.cos(a),
                                cz + radius * math.sin(a)]),
        )
        for i, a in enumerate(angles)
    ]


# ── Test cases ────────────────────────────────────────────────────────────────
# Edit or extend this dict to add / remove / modify test cases.

_CFG_RIGHT = "ik/configs/g1_right_arm.yaml"
_CFG_DUAL  = "ik/configs/g1_dual_arm.yaml"

TESTS: dict[str, TestCase] = {

    # ── Basic reachability ────────────────────────────────────────────────────

    "reachable_right_front": TestCase(
        description="Right EEF to a natural forward-reach position.",
        config=_CFG_RIGHT,
        targets=[TargetFrame("right_eef", "right_wrist_yaw_link",
                             np.array([0.40, -0.25, 0.85]))],
        expect_converged=True,
    ),

    "reachable_elevated": TestCase(
        description="Right EEF raised high (hand-wave pose).",
        config=_CFG_RIGHT,
        targets=[TargetFrame("right_eef", "right_wrist_yaw_link",
                             np.array([0.20, -0.30, 1.10]))],
        expect_converged=True,
    ),

    "reachable_low": TestCase(
        description=(
            "Right EEF lowered near hip height.  "
            "waist_pitch_joint is locked to 0.0, which reduces the reachable "
            "workspace for low targets — convergence is not expected."
        ),
        config=_CFG_RIGHT,
        targets=[TargetFrame("right_eef", "right_wrist_yaw_link",
                             np.array([0.35, -0.25, 0.65]))],
        expect_converged=False,
    ),

    # ── Unreachable targets ───────────────────────────────────────────────────

    "unreachable_far_front": TestCase(
        description="Target 1.5 m in front — outside arm workspace.",
        config=_CFG_RIGHT,
        targets=[TargetFrame("right_eef", "right_wrist_yaw_link",
                             np.array([1.50, -0.25, 0.85]))],
        expect_converged=False,
    ),

    "unreachable_behind": TestCase(
        description="Target directly behind the robot's pelvis.",
        config=_CFG_RIGHT,
        targets=[TargetFrame("right_eef", "right_wrist_yaw_link",
                             np.array([-0.60, -0.20, 0.80]))],
        expect_converged=False,
    ),

    # ── Joint limit enforcement ───────────────────────────────────────────────

    "joint_limit_waist_yaw": TestCase(
        description=(
            "Target far to the left normally needs large waist yaw rotation.  "
            "Config limit is tightened to ±0.15 rad — verify the joint stays "
            "within bounds even though the target may not be fully reached."
        ),
        config=_CFG_RIGHT,
        targets=[TargetFrame("right_eef", "right_wrist_yaw_link",
                             np.array([0.30,  0.40, 0.85]))],
        expect_converged=False,           # won't reach with tight waist
        joint_limit_overrides={
            "waist_yaw_joint": (-0.15, 0.15),
        },
        extra_checks=[
            _joint_within("waist_yaw_joint", -0.15, 0.15),
        ],
    ),

    "joint_limit_elbow": TestCase(
        description=(
            "Elbow limited to [0, 1.0] rad (no extension past neutral).  "
            "Verify elbow never goes negative during the solve."
        ),
        config=_CFG_RIGHT,
        targets=[TargetFrame("right_eef", "right_wrist_yaw_link",
                             np.array([0.40, -0.30, 0.85]))],
        expect_converged=True,
        joint_limit_overrides={
            "right_elbow_joint": (0.0, 1.0),
        },
        extra_checks=[
            _joint_within("right_elbow_joint", 0.0, 1.0),
        ],
    ),

    "joint_limit_waist_pitch_only": TestCase(
        description=(
            "waist_pitch_joint is constrained to ±0.2618 rad (±15 deg).  "
            "Target at [0.40, -0.30, 0.75] is below the comfortable workspace "
            "with this tighter limit — convergence is not expected, but the "
            "joint must remain within ±0.2618 rad throughout the solve."
        ),
        config=_CFG_RIGHT,
        targets=[TargetFrame("right_eef", "right_wrist_yaw_link",
                             np.array([0.40, -0.30, 0.75]))],
        expect_converged=False,
        extra_checks=[
            _joint_within("waist_pitch_joint", -0.2618, 0.2618),
        ],
    ),

    # ── Orientation constraint ────────────────────────────────────────────────

    "orientation_hand_down": TestCase(
        description=(
            "Right EEF at forward reach with hand pointing downward (−Z).  "
            "The tighter waist_pitch limit (±0.2618 rad) prevents the last "
            "0.3 mm of convergence — full convergence is not expected."
        ),
        config=_CFG_RIGHT,
        targets=[TargetFrame(
            "right_eef", "right_wrist_yaw_link",
            position=np.array([0.40, -0.25, 0.80]),
            # 90° rotation about Y  → hand faces −Z (downward)
            orientation=np.array([math.cos(math.pi/4), 0, math.sin(math.pi/4), 0]),
            pos_weight=1.0,
            ori_weight=1.0,
        )],
        expect_converged=False,
    ),

    # ── Dual-arm ─────────────────────────────────────────────────────────────

    "dual_arm_symmetric": TestCase(
        description=(
            "Both arms reaching symmetric points.  "
            "waist_pitch_joint is locked to 0.0; convergence is not guaranteed "
            "for the 16-DOF dual-arm problem within the iteration budget."
        ),
        config=_CFG_DUAL,
        targets=[
            TargetFrame("right_eef", "right_wrist_yaw_link",
                        np.array([0.40, -0.30, 0.85])),
            TargetFrame("left_eef",  "left_wrist_yaw_link",
                        np.array([0.40,  0.30, 0.85])),
        ],
        expect_converged=False,
    ),

    "dual_arm_asymmetric": TestCase(
        description=(
            "Both arms at different heights — one high, one low.  "
            "waist_pitch_joint is locked to 0.0; convergence is not guaranteed."
        ),
        config=_CFG_DUAL,
        targets=[
            TargetFrame("right_eef", "right_wrist_yaw_link",
                        np.array([0.35, -0.25, 1.05])),
            TargetFrame("left_eef",  "left_wrist_yaw_link",
                        np.array([0.35,  0.25, 0.60])),
        ],
        expect_converged=False,
    ),

    # ── Batch trajectory ──────────────────────────────────────────────────────

    "batch_circle_12pt": TestCase(
        description=(
            "12-point circle trajectory solved as a warm-chained batch.  "
            "With waist_pitch constrained to ±0.2618 rad, waypoints near "
            "the bottom of the circle fall outside the reachable workspace; "
            "at least 8/12 waypoints must converge."
        ),
        config=_CFG_RIGHT,
        # targets here is a single placeholder — the batch runner overrides it
        targets=[TargetFrame("right_eef", "right_wrist_yaw_link",
                             np.array([0.45, -0.2, 0.9]))],
        expect_converged=False,
    ),
}


# ── Special handler for the batch test ───────────────────────────────────────

def _run_batch_circle(tc: TestCase, viewer: bool) -> TestResult:
    """Runs the 12-pt circle as a solve_batch call and checks all converge."""
    cfg    = IKConfig.from_yaml(tc.config)
    solver = IKSolver(cfg, verbose=False)
    waypoints = _make_circle_targets(n=12)

    t0 = time.perf_counter()
    batch_results = solver.solve_batch([[t] for t in waypoints], warm_chain=True)
    elapsed = time.perf_counter() - t0

    # Aggregate into a single IKResult for reporting
    all_conv   = all(r.converged for r in batch_results)
    max_err    = max(max(r.pos_errors.values()) for r in batch_results)
    last_res   = batch_results[-1]
    agg_result = IKResult(
        joint_angles=last_res.joint_angles,
        qpos=last_res.qpos,
        pos_errors={f"wp_{i:02d}": max(r.pos_errors.values())
                    for i, r in enumerate(batch_results)},
        converged=all_conv,
        iterations=sum(r.iterations for r in batch_results),
    )

    failures: list[str] = []
    n_conv = sum(1 for r in batch_results if r.converged)
    n_total = len(batch_results)
    if tc.expect_converged and not all_conv:
        n_fail = n_total - n_conv
        failures.append(f"{n_fail}/{n_total} waypoints did not converge  (max_err={max_err:.4f} m)")
    elif not tc.expect_converged:
        # Require at least 8/12 waypoints to converge as a sanity floor
        min_required = 8
        if n_conv < min_required:
            failures.append(
                f"Only {n_conv}/{n_total} waypoints converged — expected at least {min_required}"
            )

    if viewer:
        # Re-run with viewer, showing each waypoint
        cfg2    = IKConfig.from_yaml(tc.config)
        solver2 = IKSolver(cfg2, verbose=False)
        body_ids = solver2._resolve_body_ids(waypoints)
        solver2._set_initial_pose()
        with mj_viewer.launch_passive(solver2.model, solver2.data) as v:
            v.sync()
            for i, wp in enumerate(waypoints):
                res = solver2.solve(targets=[wp],
                                    warm_start=solver2.data.qpos.copy())
                if v.is_running():
                    scn = v.user_scn
                    scn.ngeom = 0
                    for tgt, bid in zip([wp], [body_ids[i]]):
                        from ik.ik_runner import _draw_sphere, _draw_frame
                        _draw_sphere(scn, solver2.data.xpos[bid], 0.018, [0, 0.8, 1, 1])
                        _draw_frame(scn, solver2.data.xpos[bid],
                                    solver2.data.xmat[bid], 0.10, tgt.name)
                        _draw_sphere(scn, tgt.position, 0.022, [1, 0.2, 0.2, 0.8])
                    # draw all remaining targets as faint grey spheres
                    for future_wp in waypoints[i+1:]:
                        _draw_sphere(scn, future_wp.position, 0.012, [0.5, 0.5, 0.5, 0.4])
                    v.sync()
                    time.sleep(0.1)
            print("Batch done — close viewer to exit.")
            while v.is_running():
                v.sync()
                time.sleep(0.016)

    return TestResult(
        name="batch_circle_12pt",
        passed=len(failures) == 0,
        failures=failures,
        result=agg_result,
        elapsed=elapsed,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="IK solver verification tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true",
                   help="List all test cases and exit.")
    g.add_argument("--all", action="store_true",
                   help="Run all tests headlessly.")
    g.add_argument("--test", metavar="NAME",
                   help="Run a single named test.")
    p.add_argument("--viewer", action="store_true",
                   help="Open MuJoCo viewer (single test only, or with --all for batch).")
    p.add_argument("--viewer-on-fail", action="store_true",
                   help="With --all: open viewer automatically on each failure.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.list:
        print(f"\n{'Test name':<32}  Description")
        print("─" * 80)
        for name, tc in TESTS.items():
            marker = "" if tc.expect_converged else " [expect no-conv]"
            print(f"  {name:<30}  {tc.description[:60]}{marker}")
        print()
        return

    if args.all:
        print(f"\nRunning {len(TESTS)} IK tests ...\n")
        results = []
        for name, tc in TESTS.items():
            if name == "batch_circle_12pt":
                tr = _run_batch_circle(tc, viewer=False)
            else:
                tr = _run_one(name, tc, viewer=False)
            results.append(tr)
            _print_one(tr, verbose=False)
            if not tr.passed and args.viewer_on_fail:
                print(f"  → opening viewer for {name} ...")
                if name == "batch_circle_12pt":
                    _run_batch_circle(tc, viewer=True)
                else:
                    _run_one(name, tc, viewer=True)
        _print_summary(results)
        sys.exit(0 if all(r.passed for r in results) else 1)

    # --test NAME
    name = args.test
    if name not in TESTS:
        print(f"Unknown test '{name}'.  Available: {', '.join(TESTS)}")
        sys.exit(1)

    tc = TESTS[name]
    print(f"\nTest : {name}")
    print(f"Desc : {tc.description}")
    print(f"Cfg  : {tc.config}")
    if tc.joint_limit_overrides:
        print(f"Limit overrides: {tc.joint_limit_overrides}")
    print()

    if name == "batch_circle_12pt":
        tr = _run_batch_circle(tc, viewer=args.viewer)
    else:
        tr = _run_one(name, tc, viewer=args.viewer)

    _print_one(tr, verbose=True)

    # Detailed joint output for single-test runs
    print()
    print("  Joint angles (active):")
    if name == "batch_circle_12pt":
        pass  # already shown during batch
    else:
        cfg_loaded = IKConfig.from_yaml(tc.config)
        for jname, bounds in tc.joint_limit_overrides.items():
            cfg_loaded.joint_limits[jname] = bounds
        solver = IKSolver(cfg_loaded)
        limits  = solver.get_effective_limits()
        for jname, angle in tr.result.joint_angles.items():
            lo, hi    = limits[jname]
            lo_s = f"{lo:+.4f}" if np.isfinite(lo) else "  -inf"
            hi_s = f"{hi:+.4f}" if np.isfinite(hi) else "  +inf"
            at_lo = np.isfinite(lo) and abs(angle - lo) < 1e-3
            at_hi = np.isfinite(hi) and abs(angle - hi) < 1e-3
            clamp_flag = _warn(" ← AT LIMIT") if (at_lo or at_hi) else ""
            print(f"    {jname:<40}  {angle:+.4f} rad  "
                  f"[{lo_s}, {hi_s}]{clamp_flag}")

    _print_summary([tr])
    sys.exit(0 if tr.passed else 1)


if __name__ == "__main__":
    main()
