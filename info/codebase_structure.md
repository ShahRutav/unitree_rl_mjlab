# Codebase Structure

## Top-Level Layout

```
unitree_rl_mjlab/
├── src/
│   ├── tasks/
│   │   ├── velocity/       # Locomotion with velocity commands
│   │   └── tracking/       # Motion capture tracking
│   └── assets/
│       ├── robots/         # 8 Unitree robot variants (XML models + configs)
│       └── motions/        # Reference motion NPZ/CSV files (g1, g1_23dof)
├── simulate/               # C++ simulation utilities
├── deploy/                 # Deployment utilities
└── dump_scripts/           # Utility scripts
```

## Task Directory Structure

Each task (`velocity/` or `tracking/`) follows the same layout:

```
velocity/ (or tracking/)
├── mdp/
│   ├── rewards.py          ← reward functions defined here
│   ├── observations.py
│   ├── terminations.py
│   └── curriculums.py
├── config/
│   └── <robot>/
│       ├── env_cfgs.py     ← reward terms registered/overridden here
│       └── rl_cfg.py       ← PPO hyperparameters
├── velocity_env_cfg.py     ← base reward dict (factory function)
└── rl/runner.py            ← training loop
```

---

## Reward Terms

### Function Definitions

| Task | File | Key Rewards |
|------|------|-------------|
| Velocity | `src/tasks/velocity/mdp/rewards.py` | `track_linear_velocity`, `track_angular_velocity`, `body_orientation_l2`, `self_collision_cost`, `body_angular_velocity_penalty`, `angular_momentum_penalty`, `feet_air_time`, `feet_clearance`, `feet_gait`, `feet_swing_height`, `feet_slip`, `soft_landing`, `variable_posture`, `stand_still` (14 total) |
| Tracking | `src/tasks/tracking/mdp/rewards.py` | `motion_global_anchor_position_error_exp`, `motion_global_anchor_orientation_error_exp`, `motion_relative_body_position_error_exp`, `motion_relative_body_orientation_error_exp`, `motion_global_body_linear_velocity_error_exp`, `motion_global_body_angular_velocity_error_exp`, `self_collision_cost` (7 total) |

### Registration (Wiring into Training)

Rewards are registered as `RewardTermCfg` objects in two layers:

**Layer 1 — Base defaults** in the factory function:
- `src/tasks/velocity/velocity_env_cfg.py:258-354` — `make_velocity_env_cfg()`
- `src/tasks/tracking/tracking_env_cfg.py:210-253` — `make_tracking_env_cfg()`

**Layer 2 — Robot-specific overrides** in env configs:
- `src/tasks/velocity/config/g1/env_cfgs.py` — adds/modifies rewards for G1
- Similarly for each robot under `config/<robot>/env_cfgs.py`

Each entry looks like:
```python
cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": ..., "force_threshold": 10.0},
)
```

The `RewardManager` (from `mjlab`) iterates all registered terms each step, calls the function, multiplies by `weight`, and sums them into a scalar reward.

---

## Velocity vs. Tracking Tasks

### Velocity Task
- **Command**: 3D twist `(lin_vel_x, lin_vel_y, ang_vel_z)`, resampled every 3-8s via `UniformVelocityCommandCfg`
- **Rewards**: Penalize gait irregularity, posture deviation, and collisions; reward velocity tracking and appropriate foot contact
- **Training**: 10,001 max iterations, lr=1e-3
- **Runner**: `src/tasks/velocity/rl/runner.py` — exports policy to ONNX

### Tracking Task
- **Command**: Full-body reference motion loaded from NPZ files via `MotionCommand` (`src/tasks/tracking/mdp/commands.py`)
- **Rewards**: Measure position/orientation/velocity errors per body part relative to the reference motion
- **Metrics**: MPKPE, R-MPKPE, EE errors (`src/tasks/tracking/mdp/metrics.py`)
- **Training**: 30,001 max iterations (3x longer), lr=1e-3, lower entropy coeff (0.005 vs 0.01)
- **Runner**: `src/tasks/tracking/rl/runner.py` — embeds motion reference data into ONNX

---

## Supported Robots

| Robot | Velocity Config | Tracking Config |
|-------|----------------|-----------------|
| Unitree Go2 | `config/go2/` | — |
| Unitree G1 | `config/g1/` | `config/g1/` |
| Unitree G1 23-DOF | `config/g1_23dof/` | `config/g1_23dof/` |
| Unitree H1-2 | `config/h1_2/` | — |
| Unitree H2 | `config/h2/` | — |
| Unitree R1 | `config/r1/` | — |
| Unitree A2 | `config/a2/` | — |
| Unitree As2 | `config/as2/` | — |

---

## Key File Reference

| Component | Path |
|-----------|------|
| Velocity reward functions | `src/tasks/velocity/mdp/rewards.py` |
| Velocity base config | `src/tasks/velocity/velocity_env_cfg.py` |
| Velocity G1 robot config | `src/tasks/velocity/config/g1/env_cfgs.py` |
| Tracking reward functions | `src/tasks/tracking/mdp/rewards.py` |
| Tracking base config | `src/tasks/tracking/tracking_env_cfg.py` |
| Tracking motion command | `src/tasks/tracking/mdp/commands.py` |
| Tracking metrics | `src/tasks/tracking/mdp/metrics.py` |
| Robot XML models | `src/assets/robots/<robot>/xmls/` |
| Reference motions | `src/assets/motions/` |
