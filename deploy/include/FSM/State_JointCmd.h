// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "JointCmdController.h"
#include "JointCmdReceiver.h"
#include <spdlog/spdlog.h>
#include <map>
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

        // Apply passive joint overrides (kp=kd=0) from passive_joints.yaml if present.
        // Zeroing here means enter() naturally writes 0 gains for those joints.
        auto passive_path = param::config_dir / "passive_joints.yaml";
        if (std::filesystem::exists(passive_path)) {
            auto pcfg = YAML::LoadFile(passive_path.string());
            if (pcfg["passive_joints"]) {
                auto indices = pcfg["passive_joints"].as<std::vector<int>>();
                for (int idx : indices) {
                    if (idx >= 0 && idx < static_cast<int>(kp_.size())) {
                        kp_[idx] = 0.0f;
                        kd_[idx] = 0.0f;
                        passive_indices_.push_back(idx);
                    }
                }
                spdlog::info("[JointCmd] passive_joints.yaml: {} joints set to kp=kd=0", indices.size());
            }
        }

        // Load locked joints: pinned at a fixed value every tick regardless of incoming commands.
        if (cfg["locked_joints"]) {
            auto lj = cfg["locked_joints"];
            for (auto it = lj.begin(); it != lj.end(); ++it) {
                int   idx = it->first.as<int>();
                float val = it->second.as<float>();
                if (idx >= 0 && idx < static_cast<int>(kp_.size()))
                    locked_joints_[idx] = val;
            }
            spdlog::info("[JointCmd] locked_joints: {} joints pinned", locked_joints_.size());
        }

        float policy_hz           = cfg["policy_hz"] ? cfg["policy_hz"].as<float>() : 10.0f;
        policy_dt_                = (policy_hz > 0.0f) ? (1.0f / policy_hz) : 0.1f;

        threshold_direct_         = cfg["threshold_direct"]         ? cfg["threshold_direct"].as<float>()         : 0.1f;
        threshold_discard_indiv_  = cfg["threshold_discard_indiv"]  ? cfg["threshold_discard_indiv"].as<float>()  : static_cast<float>(M_PI);
        threshold_discard_sum_    = cfg["threshold_discard_sum"]    ? cfg["threshold_discard_sum"].as<float>()    : 2.0f * static_cast<float>(M_PI);
        disable_discard_          = cfg["disable_discard"]          ? cfg["disable_discard"].as<bool>()           : false;
        disable_interp_           = cfg["disable_interp"]           ? cfg["disable_interp"].as<bool>()            : false;
        interp_duration_          = cfg["interp_duration"]          ? cfg["interp_duration"].as<float>()          : 2.0f;
        if (cfg["max_torque"])
            max_torque_ = cfg["max_torque"].as<std::vector<float>>();
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

        // Lagged-ramp state — q_ramp_start_ doesn't matter until begin_ramp() runs;
        // initialize to q_default_ to match q_hold_ so any pre-message current_q_des()
        // call returns a sane vector.
        q_ramp_start_     = q_default_;
        dq_ff_.assign(kp_.size(), 0.0f);
        have_prev_policy_ = false;
        last_policy_tick_ = 0;

        // Timing watchdog
        have_msg_           = false;
        last_msg_tick_      = 0;
        last_late_log_tick_ = 0;

        // Seed locked-joint entry ramp from current hardware position so the
        // transition from FixSit doesn't produce a torque step.
        locked_entry_q_start_.clear();
        for (auto& [idx, val] : locked_joints_)
            locked_entry_q_start_[idx] = lowstate->msg_.motor_state()[idx].q();
        locked_entry_t_    = 0.0f;
        locked_entry_done_ = false;
        spdlog::info("[JointCmd] locked joints: entry ramp started ({:.2f}s)", interp_duration_);

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

        // Passive and locked joints are excluded from classify()/active_max_err()
        // so their drift never triggers INTERPOLATE on the active joints.
        controller_.skip_joints.clear();
        for (int idx : passive_indices_)
            controller_.skip_joints.insert(idx);
        for (auto& [idx, val] : locked_joints_)
            controller_.skip_joints.insert(idx);

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
                // Timing watchdog: incoming command rate should match policy_hz so the
                // ramp slope (Δq / policy_dt) reflects one real period of motion. If
                // the gap is outside ±50% of policy_dt, something upstream is jittering
                // (sender stall, scheduler stutter, or arm_cmd --hz mismatch).
                if (have_msg_) {
                    const float gap = (tick_ - last_msg_tick_) * dt_;
                    if (std::fabs(gap - policy_dt_) > 0.5f * policy_dt_ &&
                        tick_ - last_late_log_tick_ >= 1000) {
                        spdlog::warn("[JointCmd] message gap {:.3f}s vs policy_dt {:.3f}s "
                                     "(Δ={:+.3f}s) — dq_ff scaling will be off",
                                     gap, policy_dt_, gap - policy_dt_);
                        last_late_log_tick_ = tick_;
                    }
                }
                last_msg_tick_ = tick_;
                have_msg_      = true;

                CmdMode mode = controller_.classify(q_current, q_new);
                float err    = max_err(q_current, q_new);

                switch (mode)
                {
                    case CmdMode::DIRECT:
                        // Don't abort an active ramp when q_current has gotten within
                        // threshold_direct of the ramp's own target — that's a sensor
                        // reading racing ahead of the ramp, not a new command. Switching
                        // immediately would jump the commanded position from the current
                        // ramp value to q_new (up to threshold_direct gap) and cause a jerk.
                        if (in_interp_ && controller_.active_max_err(q_target_, q_new) <= controller_.threshold_direct) {
                            break;
                        }
                        begin_ramp(q_new);
                        in_interp_ = false;
                        t_interp_  = 0.0f;
                        if (tick_ - last_direct_log_tick_ >= 1000) {
                            spdlog::info("[JointCmd][direct] max_err={:.4f} rad", err);
                            last_direct_log_tick_ = tick_;
                        }
                        break;

                    case CmdMode::INTERPOLATE:
                        // Only restart the ramp if the active-joint target has changed
                        // meaningfully. Uses active_max_err so passive/locked joint drift
                        // does not perpetually reset a running interpolation.
                        if (!in_interp_ || controller_.active_max_err(q_target_, q_new) > controller_.threshold_direct) {
                            q_start_   = q_current;
                            q_target_  = q_new;
                            in_interp_ = true;
                            t_interp_  = 0.0f;
                            // Adaptive duration: T = max(interp_duration_, max over joints of
                            //   (kp*dt_ + kd) * delta / max_torque)
                            // This ensures the first-step PD torque (spring + damping) stays
                            // within max_torque for every joint.
                            float adaptive_duration = interp_duration_;
                            if (!max_torque_.empty()) {
                                for (std::size_t i = 0; i < q_current.size(); ++i) {
                                    if (max_torque_[i] <= 0.0f) continue;
                                    float delta    = std::fabs(q_new[i] - q_current[i]);
                                    float T_needed = (kp_[i] * dt_ + kd_[i]) * delta / max_torque_[i];
                                    if (T_needed > adaptive_duration)
                                        adaptive_duration = T_needed;
                                }
                            }
                            controller_.interp_duration = adaptive_duration;
                            spdlog::info("[JointCmd][interp] started: max_err={:.4f} rad, {}duration={:.2f}s{}",
                                         err,
                                         adaptive_duration > interp_duration_ ? "\033[33m" : "",
                                         adaptive_duration,
                                         adaptive_duration > interp_duration_ ? "\033[0m" : "");
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

        // Step 2 / Step 3: build final command, then write to motors
        std::vector<float> q_cmd;
        if (in_interp_)
        {
            t_interp_ += dt_;
            q_cmd = controller_.interp_step(q_start_, q_target_, t_interp_);

            if (t_interp_ >= controller_.interp_duration)
            {
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
        else if (have_prev_policy_)
        {
            // Lagged linear ramp: q_des moves from q_ramp_start_ → q_hold_ over policy_dt,
            // then saturates at q_hold_. Slope of this ramp matches dq_ff_ by construction,
            // so q_des and dq_des describe the same motion (no Kd·v/Kp settling offset, no
            // wrong-segment-slope overshoot during decel).
            const float age   = (tick_ - last_policy_tick_) * dt_;
            const float alpha = std::clamp(age / policy_dt_, 0.0f, 1.0f);
            q_cmd.resize(q_hold_.size());
            for (std::size_t i = 0; i < q_cmd.size(); ++i)
                q_cmd[i] = q_ramp_start_[i] + alpha * (q_hold_[i] - q_ramp_start_[i]);
        }
        else
        {
            q_cmd = q_hold_;
        }


        // Locked joints: ramp from entry position to locked value over interp_duration,
        // then hard-pin forever. Runs after the torque clamp.
        if (!locked_entry_done_) {
            locked_entry_t_ += dt_;
            float alpha = std::min(locked_entry_t_ / interp_duration_, 1.0f);
            for (auto& [idx, val] : locked_joints_)
                if (idx < static_cast<int>(q_cmd.size()))
                    q_cmd[idx] = locked_entry_q_start_[idx] + alpha * (val - locked_entry_q_start_[idx]);
            if (locked_entry_t_ >= interp_duration_) {
                locked_entry_done_ = true;
                spdlog::info("[JointCmd] locked joints: entry ramp done");
            }
        } else {
            for (auto& [idx, val] : locked_joints_)
                if (idx < static_cast<int>(q_cmd.size()))
                    q_cmd[idx] = val;
        }

        // Velocity feedforward: hold the last policy-derived dq_ff while it's fresh,
        // then decay to zero so a stalled sender doesn't leave a steady-state offset.
        // Suppressed during the safety ramp (in_interp_) and on locked joints.
        // FF is on only while the lagged ramp is still progressing (α < 1, equivalently
        // age < policy_dt). Once we've reached q_hold_, dq_des = 0 — same condition as
        // q_cmd saturating at q_hold_, so q_des and dq_des go flat together.
        const float ff_age = (tick_ - last_policy_tick_) * dt_;
        const bool  ff_active = have_prev_policy_ && !in_interp_ && ff_age < policy_dt_;

        for (int i = 0; i < static_cast<int>(q_cmd.size()); ++i) {
            lowcmd->msg_.motor_cmd()[i].q() = q_cmd[i];
            float v_ff = ff_active ? dq_ff_[i] : 0.0f;
            if (locked_joints_.count(i)) v_ff = 0.0f;
            lowcmd->msg_.motor_cmd()[i].dq() = v_ff;
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
    // Compute the current commanded q_des at this tick.
    //   - During the safety ramp (in_interp_): the 2 s linear interp from q_start_ → q_target_.
    //   - Before the first policy message: q_hold_ (= q_default_).
    //   - Otherwise: linear ramp from q_ramp_start_ → q_hold_ over policy_dt, saturating at q_hold_.
    // Called from begin_ramp() so a new message arriving mid-ramp anchors the next ramp's start
    // to the current q_des, keeping the commanded trajectory C0-continuous under sender jitter.
    std::vector<float> current_q_des() const
    {
        if (in_interp_)
            return controller_.interp_step(q_start_, q_target_, t_interp_);
        if (!have_prev_policy_)
            return q_hold_;
        const float age   = (tick_ - last_policy_tick_) * dt_;
        const float alpha = std::clamp(age / policy_dt_, 0.0f, 1.0f);
        std::vector<float> q(q_hold_.size());
        for (std::size_t i = 0; i < q.size(); ++i)
            q[i] = q_ramp_start_[i] + alpha * (q_hold_[i] - q_ramp_start_[i]);
        return q;
    }

    // Begin a new linear ramp of q_des from current q_des → q_new over policy_dt.
    // dq_ff = (q_new - q_ramp_start) / policy_dt matches the ramp's slope exactly,
    // so q_des and dq_des describe the same segment — no Kd·v/Kp settling offset
    // and no wrong-segment-slope overshoot when the policy decelerates.
    void begin_ramp(const std::vector<float>& q_new)
    {
        if (q_new.size() != q_hold_.size()) return;
        q_ramp_start_ = current_q_des();
        q_hold_       = q_new;
        const float inv_dt = 1.0f / policy_dt_;
        for (std::size_t i = 0; i < q_hold_.size(); ++i)
            dq_ff_[i] = (q_hold_[i] - q_ramp_start_[i]) * inv_dt;
        have_prev_policy_ = true;
        last_policy_tick_ = tick_;
    }

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
    std::vector<float>   kp_;
    std::vector<float>   kd_;
    std::vector<float>   q_default_;      // hold pose until first ZMQ command
    std::vector<float>   max_torque_;     // per-joint torque clamp (Nm); empty = disabled
    std::vector<int>     passive_indices_;            // joints with kp=kd=0
    std::map<int, float> locked_joints_;             // joints pinned at a fixed value every tick
    std::map<int, float> locked_entry_q_start_;      // hardware position at enter() for entry ramp
    float                locked_entry_t_    = 0.0f;  // elapsed time into entry ramp
    bool                 locked_entry_done_ = true;  // true once ramp completes (or no locked joints)
    float       policy_dt_ = 0.1f;     // 1 / policy_hz; used for dq_ff = Δq / policy_dt
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

    // Lagged-ramp state — see begin_ramp() / current_q_des().
    // q_des ramps linearly from q_ramp_start_ → q_hold_ over policy_dt; dq_ff_ is the
    // matching constant slope and is written to motor.dq() while ff_age < policy_dt.
    std::vector<float> q_ramp_start_;      // q_des at the start of the current 100 ms segment
    std::vector<float> dq_ff_;             // (q_hold_ - q_ramp_start_) / policy_dt, written to motor.dq()
    bool               have_prev_policy_ = false;
    uint32_t           last_policy_tick_ = 0;  // tick_ at which the current ramp began

    // Timing watchdog — independent of the ramp anchor; ticks on every received command.
    bool     have_msg_           = false;
    uint32_t last_msg_tick_      = 0;
    uint32_t last_late_log_tick_ = 0;

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
