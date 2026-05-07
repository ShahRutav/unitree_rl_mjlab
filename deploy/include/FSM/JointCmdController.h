// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "LinearInterpolator.h"

enum class CmdMode { DIRECT, INTERPOLATE, DISCARD };

struct JointCmdController {
    float threshold_direct          = 0.1f;                    // max L-inf error for DIRECT (rad)
    float threshold_discard_indiv   = static_cast<float>(M_PI); // DISCARD if any joint > this (rad, default 180°)
    float threshold_discard_sum     = 2.0f * static_cast<float>(M_PI); // DISCARD if sum(|err|) > this (rad, default 360°)
    float interp_duration           = 2.0f;                    // interpolation window (seconds)
    bool  disable_discard           = false;                   // if true, never discard — interpolate everything
    bool  disable_interp            = false;                   // if true, never interpolate — send directly always

    // Joint indices excluded from classify() and active_max_err() error computation.
    // Passive joints (kp=kd=0) and locked joints are added here so their drift
    // never triggers INTERPOLATE or DISCARD on the active joints.
    std::unordered_set<int> skip_joints;

    // Classify a desired joint command relative to the current joint state.
    // Only non-skip joints contribute to the error metrics.
    // Returns:
    //   DIRECT      if max(|err|) <= threshold_direct
    //   DISCARD     if max(|err|) > threshold_discard_indiv
    //               OR sum(|err|) > threshold_discard_sum
    //               OR sizes differ
    //   INTERPOLATE otherwise
    CmdMode classify(const std::vector<float>& q_current,
                     const std::vector<float>& q_target) const
    {
        if (q_current.size() != q_target.size()) {
            return CmdMode::DISCARD;
        }

        float max_err = 0.0f;
        float sum_err = 0.0f;
        for (std::size_t i = 0; i < q_current.size(); ++i) {
            if (skip_joints.count(static_cast<int>(i))) continue;
            float d = std::fabs(q_target[i] - q_current[i]);
            max_err  = std::max(max_err, d);
            sum_err += d;
        }

        if (!disable_discard && (max_err > threshold_discard_indiv || sum_err > threshold_discard_sum)) {
            return CmdMode::DISCARD;
        }
        if (max_err <= threshold_direct || disable_interp) {
            return CmdMode::DIRECT;
        }
        return CmdMode::INTERPOLATE;
    }

    // L-inf error over non-skip joints only — used for ramp-restart decisions.
    float active_max_err(const std::vector<float>& a, const std::vector<float>& b) const
    {
        float e = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (skip_joints.count(static_cast<int>(i))) continue;
            e = std::max(e, std::fabs(a[i] - b[i]));
        }
        return e;
    }

    // Return interpolated joint positions at time t along the straight-line
    // path from q_start (t=0) to q_target (t=interp_duration).
    // Values are clamped to the endpoints by linear_interpolate for t outside
    // [0, interp_duration].
    std::vector<float> interp_step(const std::vector<float>& q_start,
                                   const std::vector<float>& q_target,
                                   float t) const
    {
        return linear_interpolate(t,
                                  {0.0f, interp_duration},
                                  {q_start, q_target});
    }
};
