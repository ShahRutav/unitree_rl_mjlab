#pragma once

#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <optional>

class JointCmdReceiver {
public:
    struct Cmd {
        std::vector<float> q;
        std::vector<float> tau_ff;  // empty if absent in the JSON message
    };

    explicit JointCmdReceiver(const std::string& address,
                              int rcvtimeo_ms  = 100,
                              int poll_sleep_ms = 1)
        : ctx_(1)
        , socket_(ctx_, zmq::socket_type::sub)
        , running_(true)
        , poll_sleep_ms_(poll_sleep_ms)
    {
        socket_.setsockopt(ZMQ_RCVTIMEO, rcvtimeo_ms); // timeout for clean shutdown
        socket_.setsockopt(ZMQ_SUBSCRIBE, "", 0);       // subscribe to all topics
        socket_.connect(address);
        receiver_thread_ = std::thread(&JointCmdReceiver::receive_loop, this);
    }

    ~JointCmdReceiver() {
        running_.store(false);
        if (receiver_thread_.joinable()) {
            receiver_thread_.join();
        }
    }

    // Non-copyable, non-movable (owns thread and socket)
    JointCmdReceiver(const JointCmdReceiver&) = delete;
    JointCmdReceiver& operator=(const JointCmdReceiver&) = delete;
    JointCmdReceiver(JointCmdReceiver&&) = delete;
    JointCmdReceiver& operator=(JointCmdReceiver&&) = delete;

    /// Thread-safe. Returns true and copies the latest message into out if a
    /// new one has arrived since the last call. Returns false otherwise.
    /// out.tau_ff is empty if the sender did not include the field.
    bool get_latest(Cmd& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_new_) {
            return false;
        }
        out = latest_;
        has_new_ = false;
        return true;
    }

    /// Parses JSON {"q": [f0, ..., fN], "tau_ff": [f0, ..., fN]}. tau_ff is
    /// optional; if present it must be the same length as q. Returns nullopt
    /// for malformed messages (missing/non-array q, length mismatch on tau_ff,
    /// non-numeric elements).
    static std::optional<Cmd> parse_message(const std::string& msg) {
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(msg);
        } catch (const nlohmann::json::parse_error&) {
            return std::nullopt;
        }

        if (!j.contains("q")) return std::nullopt;
        const auto& q_val = j["q"];
        if (!q_val.is_array() || q_val.empty()) return std::nullopt;

        Cmd cmd;
        cmd.q.reserve(q_val.size());
        for (const auto& elem : q_val) {
            if (!elem.is_number()) return std::nullopt;
            cmd.q.push_back(elem.get<float>());
        }

        if (j.contains("tau_ff")) {
            const auto& t_val = j["tau_ff"];
            if (!t_val.is_array() || t_val.size() != cmd.q.size()) {
                return std::nullopt;
            }
            cmd.tau_ff.reserve(t_val.size());
            for (const auto& elem : t_val) {
                if (!elem.is_number()) return std::nullopt;
                cmd.tau_ff.push_back(elem.get<float>());
            }
        }

        return cmd;
    }

private:
    void receive_loop() {
        // Use sleep_until (deadline-based) instead of sleep_for (duration-based).
        // sleep_for accumulates drift: each tick sleeps N ms *after* the body
        // finishes, so the effective period is N + body_time + scheduler_jitter.
        // sleep_until keeps a fixed deadline: body overrun is absorbed and the
        // next wakeup stays on the grid, preventing long-term drift.
        const auto period = std::chrono::microseconds(poll_sleep_ms_ * 1000);
        auto next_wake    = std::chrono::steady_clock::now() + period;

        while (running_.load()) {
            zmq::message_t msg;
            auto rc = socket_.recv(msg, zmq::recv_flags::dontwait);
            if (rc.has_value()) {
                std::string data(static_cast<char*>(msg.data()), msg.size());
                auto parsed = parse_message(data);
                if (parsed.has_value()) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    latest_ = std::move(*parsed);
                    has_new_ = true;
                }
            }
            std::this_thread::sleep_until(next_wake);
            next_wake += period;
        }
    }

    zmq::context_t ctx_;
    zmq::socket_t  socket_;
    std::atomic<bool> running_;
    std::thread receiver_thread_;
    int poll_sleep_ms_;

    std::mutex mutex_;
    Cmd        latest_;
    bool       has_new_{false};
};
