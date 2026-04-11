// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

// State_SingleJointCmd: reads current joint positions on entry, then moves one
// joint by a relative delta while holding all others fixed.  The target is
// computed as:  target = q_current + delta_q_deg * (π/180)
//
// Config (in FSM.SingleJointCmd of config.yaml):
//   joint_index:     25          # SDK joint index to move (25 = R_elbow)
//   delta_q_deg:     20.0        # offset from current position in degrees
//   interp_duration: 2.0         # seconds to reach the target
//   kp: [...]                    # 29-element kp array for all joints
//   kd: [...]                    # 29-element kd array for all joints
class State_SingleJointCmd : public FSMState
{
public:
    State_SingleJointCmd(int state, std::string state_string = "SingleJointCmd")
    : FSMState(state, state_string)
    {
        auto cfg = param::config["FSM"]["SingleJointCmd"];
        joint_index_     = cfg["joint_index"].as<int>();
        delta_q_deg_     = cfg["delta_q_deg"].as<float>();
        interp_duration_ = cfg["interp_duration"] ? cfg["interp_duration"].as<float>() : 2.0f;
        kp_              = cfg["kp"].as<std::vector<float>>();
        kd_              = cfg["kd"].as<std::vector<float>>();
    }

    void enter() override
    {
        t_interp_ = 0.0f;
        tick_     = 0;

        // Snapshot current joint positions
        int n = static_cast<int>(kp_.size());
        q_start_.resize(n);
        for (int i = 0; i < n; ++i)
            q_start_[i] = lowstate->msg_.motor_state()[i].q();

        // Target is current position + delta — computed fresh each entry
        target_q_ = q_start_[joint_index_] + delta_q_deg_ * (M_PI / 180.0f);

        // Set PD gains for all joints once
        for (int i = 0; i < n; ++i) {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp()  = kp_[i];
            motor.kd()  = kd_[i];
            motor.dq()  = motor.tau() = 0.0f;
            motor.q()   = q_start_[i];
        }

        spdlog::info("[SingleJointCmd] joint[{}]({})  current={:+.4f} rad ({:+.1f}°)  delta={:+.1f}°  target={:+.4f} rad ({:+.1f}°)  duration={:.1f}s",
                     joint_index_, joint_name(joint_index_),
                     q_start_[joint_index_], q_start_[joint_index_] * (180.0f / M_PI),
                     delta_q_deg_,
                     target_q_, target_q_ * (180.0f / M_PI),
                     interp_duration_);
    }

    void run() override
    {
        tick_++;
        t_interp_ += dt_;
        float alpha = std::min(t_interp_ / interp_duration_, 1.0f);

        int n = static_cast<int>(q_start_.size());
        for (int i = 0; i < n; ++i) {
            float q_cmd = (i == joint_index_)
                          ? q_start_[i] + alpha * (target_q_ - q_start_[i])
                          : q_start_[i];
            lowcmd->msg_.motor_cmd()[i].q() = q_cmd;
        }

        if (tick_ % 500 == 0) {
            float q_actual = lowstate->msg_.motor_state()[joint_index_].q();
            float q_cmd    = q_start_[joint_index_] + alpha * (target_q_ - q_start_[joint_index_]);
            spdlog::info("[SingleJointCmd] t={:.2f}/{:.1f}s  actual={:+.4f} ({:+.1f}°)  cmd={:+.4f}  target={:+.4f} ({:+.1f}°)",
                         t_interp_, interp_duration_,
                         q_actual, q_actual * (180.0f / M_PI),
                         q_cmd,
                         target_q_, target_q_ * (180.0f / M_PI));
        }
    }

private:
    static const char* joint_name(int i)
    {
        static const char* names[29] = {
            "L_hip_pitch", "L_hip_roll",  "L_hip_yaw",
            "L_knee",      "L_ankle_pitch","L_ankle_roll",
            "R_hip_pitch", "R_hip_roll",  "R_hip_yaw",
            "R_knee",      "R_ankle_pitch","R_ankle_roll",
            "waist_yaw",   "waist_roll",  "waist_pitch",
            "L_shoulder_pitch","L_shoulder_roll","L_shoulder_yaw",
            "L_elbow",     "L_wrist_roll","L_wrist_pitch","L_wrist_yaw",
            "R_shoulder_pitch","R_shoulder_roll","R_shoulder_yaw",
            "R_elbow",     "R_wrist_roll","R_wrist_pitch","R_wrist_yaw",
        };
        return (i >= 0 && i < 29) ? names[i] : "unknown";
    }

    int   joint_index_     = 25;
    float delta_q_deg_     = 20.0f;
    float target_q_        = 0.0f;   // computed in enter() from q_start + delta
    float interp_duration_ = 2.0f;

    std::vector<float> kp_;
    std::vector<float> kd_;

    std::vector<float> q_start_;
    float    t_interp_ = 0.0f;
    uint32_t tick_     = 0;

    static constexpr float dt_ = 0.001f;  // matches FSM tick rate (1 kHz)
};

REGISTER_FSM(State_SingleJointCmd)
