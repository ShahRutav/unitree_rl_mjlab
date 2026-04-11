// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        int n = static_cast<int>(lowstate->msg_.motor_state().size());

        // Crouched check: refuse if knees are significantly bent.
        // index 3 = left_knee, 9 = right_knee
        if (n > 9)
        {
            float avg_knee = (lowstate->msg_.motor_state()[3].q() +
                              lowstate->msg_.motor_state()[9].q()) * 0.5f;
            if (avg_knee > knee_threshold_)
            {
                std::cerr
                    << "\033[31m[Velocity] REJECTED: robot appears to be crouched/sitting "
                       "(avg_knee=" << avg_knee << " rad > " << knee_threshold_ << " rad).\n"
                    << "         Velocity policy requires a standing starting position.\n"
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
        for (int i = 0; i < (int)env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();

    void exit()
    {
        if (rejected_) return;  // policy thread was never started
        policy_thread_running = false;
        if (policy_thread.joinable())
            policy_thread.join();
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
    bool rejected_   = false;
    int  warn_ticks_ = 0;
    float knee_threshold_ = 0.5f;  // above this → knees are bent (crouched/sitting)
};

REGISTER_FSM(State_RLBase)
