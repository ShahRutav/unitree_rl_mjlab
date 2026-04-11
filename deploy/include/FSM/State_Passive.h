// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"

class State_Passive : public FSMState
{
public:
    static constexpr const char* keyboard_help =
        "\033[36mKeyboard controls:\033[0m\n"
        "  [0] -> Passive          (from any state)\n"
        "  [1] -> FixStand         (from Passive)\n"
        "  [2] -> FixSit           (from Passive or FixStand; rejected if standing)\n"
        "  [3] -> Velocity         (from FixStand)\n"
        "  [4] -> JointCmd         (from Passive, FixStand, or FixSit)\n"
        "  [8] -> PrintJoints      (from Passive)\n"
        "  [9] -> SingleJointCmd   (from Passive)\n";

    State_Passive(int state, std::string state_string = "Passive")
    : FSMState(state, state_string) 
    {
        auto motor_mode = param::config["FSM"]["Passive"]["mode"];
        if(motor_mode.IsDefined())
        {
            auto values = motor_mode.as<std::vector<int>>();
            for(int i(0); i<values.size(); ++i)
            {
                lowcmd->msg_.motor_cmd()[i].mode() = values[i];
            }
        }
    } 

    void enter()
    {
        // set gain
        static auto kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        for(int i(0); i < kd.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = 0;
            motor.kd() = kd[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }

        std::cout << "\n[Passive] " << keyboard_help << std::flush;
    }

    void run()
    {
        for(int i(0); i < lowcmd->msg_.motor_cmd().size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
        }
    }
};

REGISTER_FSM(State_Passive)
