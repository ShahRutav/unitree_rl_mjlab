// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "JointCmdController.h"
#include "JointCmdReceiver.h"
#include <spdlog/spdlog.h>
#include <memory>
#include <vector>
#include <cmath>
#include <cstdio>
#include <zmq.hpp>

class State_JointCmd : public FSMState
{
public:
    State_JointCmd(int state, std::string state_string = "JointCmd")
    : FSMState(state, state_string)
    {
        auto jc_path = (param::config_dir / "joint_cmd.yaml").string();
        auto cfg = YAML::LoadFile(jc_path);

        kp_ = cfg["kp"].as<std::vector<float>>();
        kd_ = cfg["kd"].as<std::vector<float>>();

        threshold_direct_         = cfg["threshold_direct"]         ? cfg["threshold_direct"].as<float>()         : 0.1f;
        threshold_discard_indiv_  = cfg["threshold_discard_indiv"]  ? cfg["threshold_discard_indiv"].as<float>()  : static_cast<float>(M_PI);
        threshold_discard_sum_    = cfg["threshold_discard_sum"]    ? cfg["threshold_discard_sum"].as<float>()    : 2.0f * static_cast<float>(M_PI);
        disable_discard_          = cfg["disable_discard"]          ? cfg["disable_discard"].as<bool>()           : false;
        disable_interp_           = cfg["disable_interp"]           ? cfg["disable_interp"].as<bool>()            : false;
        interp_duration_          = cfg["interp_duration"]          ? cfg["interp_duration"].as<float>()          : 2.0f;
        zmq_address_       = cfg["zmq_address"]        ? cfg["zmq_address"].as<std::string>()  : "tcp://localhost:5555";
        zmq_rcvtimeo_ms_   = cfg["zmq_rcvtimeo_ms"]   ? cfg["zmq_rcvtimeo_ms"].as<int>()      : 100;
        zmq_poll_sleep_ms_ = cfg["zmq_poll_sleep_ms"]  ? cfg["zmq_poll_sleep_ms"].as<int>()    : 1;
        q_default_         = cfg["q_default"].as<std::vector<float>>();

        enable_feedback_pub_  = cfg["enable_feedback_pub"]  ? cfg["enable_feedback_pub"].as<bool>()         : true;
        zmq_feedback_address_ = cfg["zmq_feedback_address"] ? cfg["zmq_feedback_address"].as<std::string>() : "tcp://*:5556";
        feedback_decimation_  = cfg["feedback_decimation"]  ? cfg["feedback_decimation"].as<int>()          : 5;
        zmq_plot_address_     = cfg["zmq_plot_address"]     ? cfg["zmq_plot_address"].as<std::string>()     : "tcp://*:5558";
    }

    // -----------------------------------------------------------------------
    void enter() override
    {
        // 1. Set motor gains
        for (int i = 0; i < static_cast<int>(kp_.size()); ++i)
        {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp()  = kp_[i];
            motor.kd()  = kd_[i];
            motor.dq()  = motor.tau() = 0;
        }

        // 2. Seed the hold pose from q_default (matches the FixSit target exactly).
        //    Do NOT seed legs from the actual sensor readings: on real hardware
        //    gravity causes a steady-state error in FixSit (actual != target),
        //    so seeding from actual would zero the restoring PD torque and cause
        //    the legs to drift under gravity on entry.
        q_hold_ = q_default_;

        // 3. Reset state
        in_interp_ = false;
        t_interp_  = 0.0f;
        tick_                = 0;
        last_discard_log_tick_ = 0;
        last_direct_log_tick_  = 0;

        // 4. Create receiver
        receiver_ = std::make_unique<JointCmdReceiver>(zmq_address_, zmq_rcvtimeo_ms_, zmq_poll_sleep_ms_);

        // 5. Create feedback publisher (if enabled) and plot publisher
        if (enable_feedback_pub_) {
            feedback_ctx_    = std::make_unique<zmq::context_t>(1);
            feedback_socket_ = std::make_unique<zmq::socket_t>(*feedback_ctx_, zmq::socket_type::pub);
            feedback_socket_->bind(zmq_feedback_address_);
            spdlog::info("[JointCmd] feedback pub bound to {}", zmq_feedback_address_);
        }
        plot_ctx_    = std::make_unique<zmq::context_t>(1);
        plot_socket_ = std::make_unique<zmq::socket_t>(*plot_ctx_, zmq::socket_type::pub);
        plot_socket_->bind(zmq_plot_address_);
        spdlog::info("[JointCmd] plot pub bound to {}", zmq_plot_address_);

        // 6. Initialise controller fields
        controller_.threshold_direct         = threshold_direct_;
        controller_.threshold_discard_indiv  = threshold_discard_indiv_;
        controller_.threshold_discard_sum    = threshold_discard_sum_;
        controller_.disable_discard          = disable_discard_;
        controller_.disable_interp           = disable_interp_;
        controller_.interp_duration          = interp_duration_;

        if (disable_discard_)
            spdlog::warn("[JointCmd] DISCARD disabled — all commands will be interpolated regardless of error");
        if (disable_interp_)
            spdlog::warn("[JointCmd] INTERP disabled — all commands sent directly to target");
        spdlog::info("[JointCmd] entered, zmq={}", zmq_address_);
    }

    // -----------------------------------------------------------------------
    void run() override
    {
        tick_++;

        // Read current joint positions
        std::vector<float> q_current(kp_.size());
        for (int i = 0; i < static_cast<int>(kp_.size()); ++i)
            q_current[i] = lowstate->msg_.motor_state()[i].q();

        // Step 1: check for a new command from the receiver
        std::vector<float> q_new;
        if (receiver_->get_latest(q_new))
        {
            if (q_new.size() == kp_.size())
            {
                CmdMode mode = controller_.classify(q_current, q_new);
                float err    = max_err(q_current, q_new);

                switch (mode)
                {
                    case CmdMode::DIRECT:
                        q_hold_    = q_new;
                        in_interp_ = false;
                        t_interp_  = 0.0f;
                        if (tick_ - last_direct_log_tick_ >= 1000) {
                            spdlog::info("[JointCmd][direct] max_err={:.4f} rad", err);
                            last_direct_log_tick_ = tick_;
                        }
                        break;

                    case CmdMode::INTERPOLATE:
                        // Only restart the ramp if the target has changed meaningfully.
                        // This lets a sender stream the same pose at high rate without
                        // perpetually resetting a 2-second interpolation.
                        if (!in_interp_ || max_err(q_target_, q_new) > controller_.threshold_direct) {
                            q_start_   = q_current;
                            q_target_  = q_new;
                            in_interp_ = true;
                            t_interp_  = 0.0f;
                            spdlog::info("[JointCmd][interp] started: max_err={:.4f} rad, duration={:.1f}s",
                                         err, controller_.interp_duration);
                        }
                        break;

                    case CmdMode::DISCARD:
                        if (tick_ - last_discard_log_tick_ >= 200) {
                            float sum_err = sum_abs_err(q_current, q_new);
                            int   worst   = worst_joint(q_current, q_new);
                            spdlog::error("[JointCmd][discard] max={:.3f}rad({}={}) sum={:.3f}rad  limits: indiv>{:.3f} sum>{:.3f}",
                                          err, joint_name(worst), worst,
                                          sum_err,
                                          controller_.threshold_discard_indiv,
                                          controller_.threshold_discard_sum);
                            last_discard_log_tick_ = tick_;
                        }
                        break;
                }
            }
            // silently ignore wrong-size messages
        }

        // Step 2 / Step 3: send command
        if (in_interp_)
        {
            t_interp_ += dt_;
            auto q_cmd = controller_.interp_step(q_start_, q_target_, t_interp_);

            for (int i = 0; i < static_cast<int>(q_cmd.size()); ++i)
                lowcmd->msg_.motor_cmd()[i].q() = q_cmd[i];

            if (t_interp_ >= controller_.interp_duration)
            {
                // interpolation finished
                q_hold_    = q_target_;
                in_interp_ = false;
                spdlog::info("[JointCmd][interp] done");
            }
            else if (tick_ % 500 == 0)
            {
                spdlog::info("[JointCmd][interp] t={:.3f}/{:.1f}s",
                             t_interp_, controller_.interp_duration);
            }
        }
        else
        {
            for (int i = 0; i < static_cast<int>(q_hold_.size()); ++i)
                lowcmd->msg_.motor_cmd()[i].q() = q_hold_[i];
        }

        // Publish feedback + plot at the configured decimation rate
        if (tick_ % static_cast<uint32_t>(feedback_decimation_) == 0)
        {
            const auto& q_goal = in_interp_ ? q_target_ : q_hold_;
            auto s = make_feedback_json(tick_, q_current, q_goal);
            if (enable_feedback_pub_ && feedback_socket_)
                feedback_socket_->send(zmq::buffer(s), zmq::send_flags::dontwait);
            if (plot_socket_)
                plot_socket_->send(zmq::buffer(s), zmq::send_flags::dontwait);
        }
    }

    // -----------------------------------------------------------------------
    void exit() override
    {
        if (feedback_socket_) {
            feedback_socket_->close();
            feedback_socket_.reset();
            feedback_ctx_.reset();
        }
        if (plot_socket_) {
            plot_socket_->close();
            plot_socket_.reset();
            plot_ctx_.reset();
        }
        receiver_.reset();  // destructor joins thread
        spdlog::info("[JointCmd] exited");
    }

private:
    // Serialize feedback as JSON string (no external dependency)
    static std::string make_feedback_json(uint32_t tick,
                                          const std::vector<float>& q_current,
                                          const std::vector<float>& q_target)
    {
        auto arr_str = [](const std::vector<float>& v) {
            std::string s = "[";
            for (std::size_t i = 0; i < v.size(); ++i) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.6g", v[i]);
                s += buf;
                if (i + 1 < v.size()) s += ',';
            }
            return s + ']';
        };
        return "{\"tick\":" + std::to_string(tick) +
               ",\"q_current\":" + arr_str(q_current) +
               ",\"q_target\":" + arr_str(q_target) + "}";
    }

    // Sum of absolute errors
    float sum_abs_err(const std::vector<float>& a, const std::vector<float>& b) const
    {
        float s = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i)
            s += std::fabs(a[i] - b[i]);
        return s;
    }

    // Compute L-inf error between two equal-length vectors
    float max_err(const std::vector<float>& a, const std::vector<float>& b) const
    {
        float e = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i)
            e = std::max(e, std::fabs(a[i] - b[i]));
        return e;
    }

    // Index of the joint with the largest error
    int worst_joint(const std::vector<float>& a, const std::vector<float>& b) const
    {
        int idx = 0;
        float e = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            float d = std::fabs(a[i] - b[i]);
            if (d > e) { e = d; idx = static_cast<int>(i); }
        }
        return idx;
    }

    // Human-readable joint name for index 0-28
    // Order: left-leg(6), right-leg(6), waist(3), left-arm(7), right-arm(7)
    static const char* joint_name(int i)
    {
        static const char* names[29] = {
            "L_hip_pitch","L_hip_roll","L_hip_yaw","L_knee","L_ankle_pitch","L_ankle_roll",
            "R_hip_pitch","R_hip_roll","R_hip_yaw","R_knee","R_ankle_pitch","R_ankle_roll",
            "waist_yaw","waist_roll","waist_pitch",
            "L_shoulder_pitch","L_shoulder_roll","L_shoulder_yaw","L_elbow",
            "L_wrist_roll","L_wrist_pitch","L_wrist_yaw",
            "R_shoulder_pitch","R_shoulder_roll","R_shoulder_yaw","R_elbow",
            "R_wrist_roll","R_wrist_pitch","R_wrist_yaw",
        };
        return (i >= 0 && i < 29) ? names[i] : "unknown";
    }

    // Config (read from joint_cmd.yaml in constructor, used in enter())
    std::vector<float> kp_;
    std::vector<float> kd_;
    std::vector<float> q_default_;      // hold pose until first ZMQ command
    float       threshold_direct_;
    float       threshold_discard_indiv_;
    float       threshold_discard_sum_;
    bool        disable_discard_;
    bool        disable_interp_;
    float       interp_duration_;
    std::string zmq_address_;
    int         zmq_rcvtimeo_ms_;
    int         zmq_poll_sleep_ms_;

    // Runtime members
    std::unique_ptr<JointCmdReceiver> receiver_;
    JointCmdController controller_;

    std::vector<float> q_hold_;    // last successfully applied command
    std::vector<float> q_start_;   // start of current interpolation
    std::vector<float> q_target_;  // target of current interpolation
    float    t_interp_  = 0.0f;   // current interpolation time (seconds)
    bool     in_interp_ = false;   // are we currently interpolating?
    uint32_t tick_               = 0;
    uint32_t last_discard_log_tick_ = 0;
    uint32_t last_direct_log_tick_  = 0;

    // Feedback publisher (arm_cmd.py reads q_current from here)
    bool        enable_feedback_pub_  = false;
    std::string zmq_feedback_address_ = "tcp://*:5556";
    int         feedback_decimation_  = 5;
    std::unique_ptr<zmq::context_t>  feedback_ctx_;
    std::unique_ptr<zmq::socket_t>   feedback_socket_;

    // Plot publisher (plot_joint_error.py reads q_current+q_target from here)
    std::string zmq_plot_address_ = "tcp://*:5558";
    std::unique_ptr<zmq::context_t>  plot_ctx_;
    std::unique_ptr<zmq::socket_t>   plot_socket_;

    static constexpr float dt_ = 0.001f;
};

REGISTER_FSM(State_JointCmd)
