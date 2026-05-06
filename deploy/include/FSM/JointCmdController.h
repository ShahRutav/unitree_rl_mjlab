// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
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

    // Classify a desired joint command relative to the current joint state.
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

    // Return interpolated joint positions at time t.
    // Uses a cubic Hermite spline so that successive segments are
    // velocity-continuous: pass v_start = velocity of the outgoing ramp at
    // the moment of restart; zero on the very first segment.
    // The segment arrives at q_target with zero velocity (v_end = 0).
    std::vector<float> interp_step(const std::vector<float>& q_start,
                                   const std::vector<float>& v_start,
                                   const std::vector<float>& q_target,
                                   float t) const
    {
        const std::vector<float> v_zero(q_start.size(), 0.0f);
        return cubic_hermite(t, interp_duration, q_start, v_start, q_target, v_zero);
    }

    // Velocity of the current ramp at time t (used to seed the next segment).
    std::vector<float> interp_velocity(const std::vector<float>& q_start,
                                       const std::vector<float>& v_start,
                                       const std::vector<float>& q_target,
                                       float t) const
    {
        const std::vector<float> v_zero(q_start.size(), 0.0f);
        return cubic_hermite_velocity(t, interp_duration, q_start, v_start, q_target, v_zero);
    }
};
