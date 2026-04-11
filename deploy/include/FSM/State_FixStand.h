// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "LinearInterpolator.h"

class State_FixStand : public FSMState
{
public:
    State_FixStand(int state, std::string state_string = "FixStand") 
    : FSMState(state, state_string) 
    {
        ts_ = param::config["FSM"]["FixStand"]["ts"].as<std::vector<float>>();
        qs_ = param::config["FSM"]["FixStand"]["qs"].as<std::vector<std::vector<float>>>();
        assert(ts_.size() == qs_.size());
    }

    void enter()
    {
        int n = static_cast<int>(lowstate->msg_.motor_state().size());

        // Read current sensor positions
        std::vector<float> q0(n);
        for (int i = 0; i < n; ++i)
            q0[i] = lowstate->msg_.motor_state()[i].q();

        // Crouched check: refuse if knees are significantly bent.
        // Joint order: left_leg(6) right_leg(6) ...  index 3 = left_knee, 9 = right_knee
        if (n > 9)
        {
            float avg_knee = (q0[3] + q0[9]) * 0.5f;
            if (avg_knee > knee_threshold_)
            {
                std::cerr
                    << "\033[31m[FixStand] REJECTED: robot appears to be crouched/sitting "
                       "(avg_knee=" << avg_knee << " rad > " << knee_threshold_ << " rad).\n"
                    << "         FixStand requires a standing or nearly-standing starting position.\n"
                    << "         Press [0] to return to Passive first.\033[0m\n" << std::flush;

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

        rejected_   = false;
        warn_ticks_ = 0;

        // set gain
        static auto kp = param::config["FSM"]["FixStand"]["kp"].as<std::vector<float>>();
        static auto kd = param::config["FSM"]["FixStand"]["kd"].as<std::vector<float>>();
        for(int i(0); i < (int)kp.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp[i];
            motor.kd() = kd[i];
            motor.dq() = motor.tau() = 0;
        }

        qs_[0] = q0;
        t0_ = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
    }

    void run()
    {
        if (rejected_)
        {
            int n = static_cast<int>(lowcmd->msg_.motor_cmd().size());
            for (int i = 0; i < n; ++i)
                lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();

            if ((++warn_ticks_) % 2000 == 0)
            {
                std::cerr << "\033[31m[FixStand] Still rejected — press [0] to return to Passive.\033[0m\n"
                          << std::flush;
            }
            return;
        }

        float t = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3 - t0_;
        auto q = linear_interpolate(t, ts_, qs_);
        for(int i(0); i < (int)q.size(); ++i)
            lowcmd->msg_.motor_cmd()[i].q() = q[i];
    }

private:
    bool   rejected_   = false;
    int    warn_ticks_ = 0;
    double t0_;
    std::vector<float> ts_;
    std::vector<std::vector<float>> qs_;
    float knee_threshold_ = 0.5f;  // above this → knees are bent (crouched/sitting)
};

REGISTER_FSM(State_FixStand)