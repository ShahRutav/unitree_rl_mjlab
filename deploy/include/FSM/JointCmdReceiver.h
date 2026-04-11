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

    /// Thread-safe. Returns true and copies latest_q_ to out if a new message
    /// has arrived since the last call. Returns false otherwise.
    bool get_latest(std::vector<float>& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_new_) {
            return false;
        }
        out = latest_q_;
        has_new_ = false;
        return true;
    }

    /// Parses JSON {"q": [f0, ..., fN]}. Returns the float vector if valid,
    /// std::nullopt if malformed (wrong key, not array, not floats, size < 1).
    static std::optional<std::vector<float>> parse_message(const std::string& msg) {
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(msg);
        } catch (const nlohmann::json::parse_error&) {
            return std::nullopt;
        }

        // Must have key "q"
        if (!j.contains("q")) {
            return std::nullopt;
        }

        const auto& q_val = j["q"];

        // Must be an array
        if (!q_val.is_array()) {
            return std::nullopt;
        }

        // Must have at least one element
        if (q_val.empty()) {
            return std::nullopt;
        }

        // All elements must be numeric (float-compatible)
        std::vector<float> result;
        result.reserve(q_val.size());
        for (const auto& elem : q_val) {
            if (!elem.is_number()) {
                return std::nullopt;
            }
            result.push_back(elem.get<float>());
        }

        return result;
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
                    latest_q_ = std::move(*parsed);
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
    std::vector<float> latest_q_;
    bool has_new_{false};
};
