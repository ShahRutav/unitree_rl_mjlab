// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

// State_PrintJoints: live in-place table of all joint positions.
//
// On enter(), reserves vertical space once. Each update cycle the cursor is
// moved back up with ANSI escape \033[NA and the table is overwritten in
// place — no terminal spam.
//
// Config (in FSM.PrintJoints of config.yaml):
//   print_every_n_ticks: 200   # refresh rate (1 kHz FSM => 200 = 5 Hz)
class State_PrintJoints : public FSMState
{
    static constexpr int N_JOINTS = 29;
    // Lines printed by print_table():
    //   top border + col header + mid border + N joints + bot border + tick line
    static constexpr int TABLE_LINES = N_JOINTS + 5;

public:
    State_PrintJoints(int state, std::string state_string = "PrintJoints")
    : FSMState(state, state_string)
    {
        auto cfg = param::config["FSM"]["PrintJoints"];
        print_every_ = (cfg && cfg["print_every_n_ticks"])
                       ? cfg["print_every_n_ticks"].as<int>()
                       : 200;
    }

    void enter() override
    {
        tick_ = 0;

        // Passive hold: kd-only, tracks current q
        static auto kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        int n = static_cast<int>(kd.size());
        for (int i = 0; i < n; ++i) {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp()  = 0.0f;
            motor.kd()  = kd[i];
            motor.dq()  = motor.tau() = 0.0f;
            motor.q()   = lowstate->msg_.motor_state()[i].q();
        }

        // Log before reserving space so the message sits above the live table
        spdlog::info("[PrintJoints] live table — {:.0f} Hz | [0] -> Passive",
                     1000.0f / print_every_);

        // Reserve vertical space; each update overwrites these lines in place
        for (int i = 0; i < TABLE_LINES; ++i) std::putchar('\n');
        std::fflush(stdout);
    }

    void run() override
    {
        tick_++;

        // Passive hold — only the 29 actuated joints (indices 0-28).
        // motor_cmd() has 35 slots; [29] is the ArmSdk weight, [30-34] reserved.
        for (int i = 0; i < N_JOINTS; ++i)
            lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();

        if (tick_ % static_cast<uint32_t>(print_every_) == 0) {
            // Move cursor back up to the start of the reserved block, then overwrite
            std::printf("\033[%dA", TABLE_LINES);
            print_table();
        }
    }

private:
    void print_table() const
    {
        int n = std::min(static_cast<int>(lowcmd->msg_.motor_cmd().size()), N_JOINTS);

        // Each content line is exactly 53 chars wide (+ \n):
        //   "| [%2d] %-22s  %+10.4f  %+8.2f |"
        //    2 + 5 + 22 + 2 + 10 + 2 + 8 + 2 = 53
        std::printf("+---------------------------------------------------+\n");
        std::printf("| %-5s%-22s  %10s  %8s |\n", "idx  ", "joint", "q (rad)", "q (deg)");
        std::printf("+---------------------------------------------------+\n");
        for (int i = 0; i < N_JOINTS; ++i) {
            float q = (i < n) ? lowstate->msg_.motor_state()[i].q() : 0.0f;
            std::printf("| [%2d] %-22s  %+10.4f  %+8.2f |\n",
                        i, joint_name(i), q, q * (180.0f / 3.14159265f));
        }
        std::printf("+---------------------------------------------------+\n");
        std::printf("  tick: %6u\n", tick_);
        std::fflush(stdout);
    }

    static const char* joint_name(int i)
    {
        static const char* names[29] = {
            "L_hip_pitch",     "L_hip_roll",      "L_hip_yaw",
            "L_knee",          "L_ankle_pitch",   "L_ankle_roll",
            "R_hip_pitch",     "R_hip_roll",       "R_hip_yaw",
            "R_knee",          "R_ankle_pitch",   "R_ankle_roll",
            "waist_yaw",       "waist_roll",       "waist_pitch",
            "L_shoulder_pitch","L_shoulder_roll",  "L_shoulder_yaw",
            "L_elbow",         "L_wrist_roll",     "L_wrist_pitch",   "L_wrist_yaw",
            "R_shoulder_pitch","R_shoulder_roll",  "R_shoulder_yaw",
            "R_elbow",         "R_wrist_roll",     "R_wrist_pitch",   "R_wrist_yaw",
        };
        return (i >= 0 && i < 29) ? names[i] : "unknown";
    }

    int      print_every_ = 200;
    uint32_t tick_        = 0;
};

REGISTER_FSM(State_PrintJoints)
