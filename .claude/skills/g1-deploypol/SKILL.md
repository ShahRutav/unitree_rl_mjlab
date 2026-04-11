---
name: g1-deploypol
description: Deploy a G1 mimic policy given a path to a .pt or .onnx file. Refreshes the motion NPZ, clears the old exported policy, copies the new one, and recompiles.
user-invocable: true
allowed-tools: Bash(ls *) Bash(rm *) Bash(cp *) Bash(readlink *) Bash(make *)
argument-hint: <path/to/model.pt or path/to/policy.onnx>
---

Deploy a G1 mimic policy.

The argument (`$ARGUMENTS`) is a path to either a `.pt` checkpoint or a `.onnx` file from a training run.

Follow these steps exactly:

## Step 1 — Resolve the ONNX path

- If `$ARGUMENTS` ends in `.onnx`, use it directly as the source ONNX.
- If `$ARGUMENTS` ends in `.pt`, look for a `.onnx` file in the **same directory**. There should be exactly one — use it. If it is a symlink, resolve it with `readlink -f` to get the real path.

## Step 2 — Identify the policy directory

The policy directory (`<policy_dir>`) is the deploy config folder for the task being deployed, e.g.:

```
deploy/robots/g1/config/policy/mimic/<task_name>/
```

Determine `<policy_dir>` from the ONNX path or from context (the task name matches the subdirectory under `mimic/`).

## Step 3 — Refresh the motion NPZ

The params directory `<policy_dir>/params/` contains a `.npz` motion file. The canonical source for that file lives in `src/assets/motions/g1/` under the same filename. Copy it over to keep them in sync:

```bash
cp src/assets/motions/g1/<motion_name>.npz <policy_dir>/params/<motion_name>.npz
```

## Step 4 — Clear the old exported policy

Remove all files currently in `<policy_dir>/exported/`.

## Step 5 — Copy the new policy

Copy the resolved ONNX file into `<policy_dir>/exported/` as two files:
- `policy.onnx` — the name the deployment binary loads
- A timestamped copy preserving the original filename (e.g. `2026-04-10_15-53-18.onnx`) for reference

Use `cp` with the resolved (non-symlink) path so the destination files are real copies, not symlinks.

## Step 6 — Recompile

```bash
cd deploy/robots/g1/build && make -j$(nproc)
```

## Step 7 — Confirm

List the contents of `<policy_dir>/exported/` and `<policy_dir>/params/`, and report what was deployed and the build result.
