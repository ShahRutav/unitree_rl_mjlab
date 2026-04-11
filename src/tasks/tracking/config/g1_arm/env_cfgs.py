"""Unitree G1 arm tracking environment configurations.

All 29 joints are available to the policy. The motion reference only moves
the right arm / waist; leg joints stay at their HOME_KEYFRAME defaults in
the reference trajectory, so the policy learns to keep them still naturally.
"""

from src.assets.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_g1_arm_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 arm tracking configuration.

  All 29 joints are controlled. The motion reference only moves the right
  arm and waist; legs stay at HOME_KEYFRAME positions in the reference so
  the policy is implicitly encouraged to keep them still.
  """
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  # Full 29-DOF action space — no joint restrictions.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = "src/assets/motions/g1/arm_circle.npz"
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "torso_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Drop velocity-based rewards — noisy with only 4 tracked bodies.
  cfg.rewards.pop("motion_body_lin_vel", None)
  cfg.rewards.pop("motion_body_ang_vel", None)

  # Remove leg end-effector termination — legs are not actively tracked.
  cfg.terminations.pop("ee_body_pos", None)

  cfg.viewer.body_name = "torso_link"

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
