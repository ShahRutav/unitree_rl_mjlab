// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "LinearInterpolator.h"
#include <iostream>

// ---------------------------------------------------------------------------
// State_FixSit
//
// Interpolates all joints from their current positions to a sitting pose
// (hip_pitch = -π/2, knee = +π/2, everything else held).
//
// SAFETY CHECK: the state refuses to execute if the robot appears to be
// standing (knees nearly straight AND hips nearly upright).  When rejected
// the robot is held in a passive damping-hold and a red error is printed.
// The user must transition to Passive (LT+B or key [0]) before retrying.
// ---------------------------------------------------------------------------

class State_FixSit : public FSMState
{
public:
    State_FixSit(int state, std::string state_string = "FixSit")
    : FSMState(state, state_string), rejected_(false), warn_ticks_(0)
    {
        ts_ = param::config["FSM"]["FixSit"]["ts"].as<std::vector<float>>();
        qs_ = param::config["FSM"]["FixSit"]["qs"].as<std::vector<std::vector<float>>>();
        assert(ts_.size() == qs_.size());

        // Optional thresholds — fall back to built-in defaults if not in config
        auto sc = param::config["FSM"]["FixSit"]["standing_check"];
        if (sc && sc["knee_threshold"])
            knee_threshold_ = sc["knee_threshold"].as<float>();
        if (sc && sc["hip_pitch_threshold"])
            hip_threshold_ = sc["hip_pitch_threshold"].as<float>();
    }

    void enter()
    {
        int n = static_cast<int>(lowstate->msg_.motor_state().size());

        // Read actual sensor joint positions
        std::vector<float> q0(n);
        for (int i = 0; i < n; ++i)
            q0[i] = lowstate->msg_.motor_state()[i].q();

        // ------------------------------------------------------------------
        // Standing check
        // Joint order: left_leg(6) right_leg(6) waist(3) left_arm(7) right_arm(7)
        //   index 0 = left_hip_pitch    index 3 = left_knee
        //   index 6 = right_hip_pitch   index 9 = right_knee
        // Reject when BOTH knees are nearly straight AND hips are nearly upright.
        // ------------------------------------------------------------------
        if (n > 9)
        {
            float avg_knee      = (q0[3] + q0[9]) * 0.5f;
            float avg_hip_pitch = (q0[0] + q0[6]) * 0.5f;
            bool  knees_straight = avg_knee      < knee_threshold_;
            bool  hips_upright   = avg_hip_pitch > hip_threshold_;

            if (knees_straight && hips_upright)
            {
                std::cerr
                    << "\033[31m[FixSit] REJECTED: robot appears to be standing "
                       "(avg_knee=" << avg_knee << " rad < " << knee_threshold_
                    << ", avg_hip_pitch=" << avg_hip_pitch << " rad > " << hip_threshold_
                    << ").\n"
                    << "         FixSit is only safe from a crouched / pre-lowered "
                       "starting position.\n"
                    << "         Press LT+B or key [0] to return to Passive first."
                    << "\033[0m\n" << std::flush;

                // Passive-like damping hold — no position stiffness, no motion
                for (int i = 0; i < n; ++i)
                {
                    auto& m  = lowcmd->msg_.motor_cmd()[i];
                    m.kp()  = 0.0f;
                    m.kd()  = 3.0f;
                    m.dq()  = 0.0f;
                    m.tau() = 0.0f;
                }
                rejected_   = true;
                warn_ticks_ = 0;
                return;
            }
        }

        // ------------------------------------------------------------------
        // Normal path: apply PD gains and set first waypoint to current pose
        // ------------------------------------------------------------------
        rejected_   = false;
        warn_ticks_ = 0;

        static auto kp = param::config["FSM"]["FixSit"]["kp"].as<std::vector<float>>();
        static auto kd = param::config["FSM"]["FixSit"]["kd"].as<std::vector<float>>();
        for (int i = 0; i < static_cast<int>(kp.size()); ++i)
        {
            auto& m  = lowcmd->msg_.motor_cmd()[i];
            m.kp()  = kp[i];
            m.kd()  = kd[i];
            m.dq()  = 0.0f;
            m.tau() = 0.0f;
        }

        qs_[0] = q0;   // start from current sensor pose
        t0_    = static_cast<double>(unitree::common::GetCurrentTimeMillisecond()) * 1e-3;
    }

    void run()
    {
        if (rejected_)
        {
            // Mirror sensor values — hold position, no commanded motion
            int n = static_cast<int>(lowcmd->msg_.motor_cmd().size());
            for (int i = 0; i < n; ++i)
                lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();

            // Remind user every ~2 s (1 kHz FSM loop → 2000 ticks)
            if ((++warn_ticks_) % 2000 == 0)
            {
                std::cerr
                    << "\033[31m[FixSit] Still rejected — press LT+B or key [0] "
                       "to return to Passive.\033[0m\n"
                    << std::flush;
            }
            return;
        }

        float t = static_cast<float>(
            static_cast<double>(unitree::common::GetCurrentTimeMillisecond()) * 1e-3 - t0_
        );
        auto q = linear_interpolate(t, ts_, qs_);
        for (int i = 0; i < static_cast<int>(q.size()); ++i)
            lowcmd->msg_.motor_cmd()[i].q() = q[i];
    }

private:
    bool    rejected_;
    int     warn_ticks_;
    double  t0_;
    std::vector<float>              ts_;
    std::vector<std::vector<float>> qs_;

    // Standing-check thresholds (rad).
    // Reject if avg_knee < knee_threshold_ AND avg_hip_pitch > hip_threshold_.
    float knee_threshold_ = 0.5f;   // below this → legs are nearly straight
    float hip_threshold_  = -0.5f;  // above this → hips are nearly upright
};

REGISTER_FSM(State_FixSit)
