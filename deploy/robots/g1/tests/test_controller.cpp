// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// Compile example (from repo root):
//   g++ -std=c++17 -I/tmp/unitree_joint_cmd/deploy/include \
//       deploy/robots/g1/tests/test_controller.cpp \
//       -lgtest -lgtest_main -lpthread -o test_controller && ./test_controller

#include "FSM/JointCmdController.h"

#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// Shared controller configuration used across all tests.
// ---------------------------------------------------------------------------
static JointCmdController make_ctrl()
{
    JointCmdController c;
    c.threshold_direct         = 0.1f;
    c.threshold_discard_indiv  = static_cast<float>(M_PI);       // 180°
    c.threshold_discard_sum    = 2.0f * static_cast<float>(M_PI); // 360°
    c.interp_duration          = 2.0f;
    return c;
}

// ---------------------------------------------------------------------------
// classify() tests
// ---------------------------------------------------------------------------

// 1. max_err <= threshold_direct  =>  DIRECT
TEST(JointCmdControllerTest, ClassifyDirect)
{
    auto ctrl = make_ctrl();
    std::vector<float> q_cur = {0.0f, 0.0f, 0.0f};
    // max error = 0.05, well within 0.1
    std::vector<float> q_tgt = {0.05f, -0.03f, 0.01f};
    EXPECT_EQ(ctrl.classify(q_cur, q_tgt), CmdMode::DIRECT);
}

// 2. threshold_direct < max_err, both thresholds satisfied  =>  INTERPOLATE
TEST(JointCmdControllerTest, ClassifyInterpolate)
{
    auto ctrl = make_ctrl();
    std::vector<float> q_cur = {0.0f, 0.0f, 0.0f};
    // max error = 0.3 (< π), sum = 0.4 (< 2π)  =>  INTERPOLATE
    std::vector<float> q_tgt = {0.3f, 0.1f, 0.0f};
    EXPECT_EQ(ctrl.classify(q_cur, q_tgt), CmdMode::INTERPOLATE);
}

// 3a. Single joint error > π  =>  DISCARD
TEST(JointCmdControllerTest, ClassifyDiscardIndividual)
{
    auto ctrl = make_ctrl();
    std::vector<float> q_cur = {0.0f, 0.0f, 0.0f};
    // max error = 3.2 > π ≈ 3.14159
    std::vector<float> q_tgt = {3.2f, 0.0f, 0.0f};
    EXPECT_EQ(ctrl.classify(q_cur, q_tgt), CmdMode::DISCARD);
}

// 3b. sum(|err|) > 2π even though each individual is < π  =>  DISCARD
TEST(JointCmdControllerTest, ClassifyDiscardSum)
{
    auto ctrl = make_ctrl();
    std::vector<float> q_cur = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // each error = 1.4 rad (<π), but sum = 7.0 > 2π ≈ 6.283
    std::vector<float> q_tgt = {1.4f, 1.4f, 1.4f, 1.4f, 1.4f};
    EXPECT_EQ(ctrl.classify(q_cur, q_tgt), CmdMode::DISCARD);
}

// 4a. max_err == threshold_direct exactly  =>  DIRECT
// 4b. max_err just above threshold_discard_indiv  =>  DISCARD
TEST(JointCmdControllerTest, ClassifyAtExactBoundary)
{
    auto ctrl = make_ctrl();
    std::vector<float> q_cur = {0.0f};

    // Exactly at threshold_direct
    std::vector<float> q_tgt_direct = {0.1f};
    EXPECT_EQ(ctrl.classify(q_cur, q_tgt_direct), CmdMode::DIRECT);

    // Just above threshold_discard_indiv (π + small delta)
    float just_above = static_cast<float>(M_PI) + 1e-3f;
    std::vector<float> q_tgt_discard = {just_above};
    EXPECT_EQ(ctrl.classify(q_cur, q_tgt_discard), CmdMode::DISCARD);
}

// 5. Size mismatch  =>  DISCARD
TEST(JointCmdControllerTest, ClassifySizeMismatch)
{
    auto ctrl = make_ctrl();
    std::vector<float> q_cur = {0.0f, 0.0f, 0.0f};
    std::vector<float> q_tgt = {0.0f, 0.0f};  // one element short
    EXPECT_EQ(ctrl.classify(q_cur, q_tgt), CmdMode::DISCARD);
}

// 6. q_current == q_target  =>  max_err = 0  =>  DIRECT
TEST(JointCmdControllerTest, ClassifyIdentical)
{
    auto ctrl = make_ctrl();
    std::vector<float> q = {1.0f, -2.0f, 0.5f, 0.0f};
    EXPECT_EQ(ctrl.classify(q, q), CmdMode::DIRECT);
}

// ---------------------------------------------------------------------------
// interp_step() tests — cubic Hermite, v_start / v_end both zero unless noted
// ---------------------------------------------------------------------------

// Helper: zero-velocity vector of given length.
static std::vector<float> zeros(std::size_t n) { return std::vector<float>(n, 0.0f); }

// 7. t = 0  =>  returns q_start (regardless of v_start)
TEST(JointCmdControllerTest, InterpStepAtZero)
{
    auto ctrl = make_ctrl();
    std::vector<float> q_start  = {1.0f, 2.0f, 3.0f};
    std::vector<float> q_target = {4.0f, 5.0f, 6.0f};

    auto result = ctrl.interp_step(q_start, zeros(3), q_target, 0.0f);

    ASSERT_EQ(result.size(), q_start.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], q_start[i], 1e-5f);
}

// 8. t = interp_duration  =>  returns q_target
TEST(JointCmdControllerTest, InterpStepAtEnd)
{
    auto ctrl = make_ctrl();
    std::vector<float> q_start  = {1.0f, 2.0f, 3.0f};
    std::vector<float> q_target = {4.0f, 5.0f, 6.0f};

    auto result = ctrl.interp_step(q_start, zeros(3), q_target, ctrl.interp_duration);

    ASSERT_EQ(result.size(), q_target.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], q_target[i], 1e-5f);
}

// 9. t = interp_duration / 2, v_start = v_end = 0  =>  returns midpoint.
// With both velocities zero the h10/h11 terms vanish and the cubic Hermite
// reduces to a linear blend at tau = 0.5.
TEST(JointCmdControllerTest, InterpStepMidpoint)
{
    auto ctrl = make_ctrl();
    std::vector<float> q_start  = {0.0f, 0.0f, 0.0f};
    std::vector<float> q_target = {2.0f, 4.0f, -2.0f};
    float t_mid = ctrl.interp_duration / 2.0f;  // 1.0 s  =>  tau = 0.5

    auto result = ctrl.interp_step(q_start, zeros(3), q_target, t_mid);

    ASSERT_EQ(result.size(), q_target.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        float expected = (q_start[i] + q_target[i]) / 2.0f;
        EXPECT_NEAR(result[i], expected, 1e-5f);
    }
}

// 10. t > interp_duration  =>  clamps to q_target
TEST(JointCmdControllerTest, InterpStepBeyondEnd)
{
    auto ctrl = make_ctrl();
    std::vector<float> q_start  = {1.0f, 2.0f, 3.0f};
    std::vector<float> q_target = {4.0f, 5.0f, 6.0f};
    float t_beyond = ctrl.interp_duration + 10.0f;

    auto result = ctrl.interp_step(q_start, zeros(3), q_target, t_beyond);

    ASSERT_EQ(result.size(), q_target.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_NEAR(result[i], q_target[i], 1e-5f);
}

// 11. Velocity continuity: interp_velocity at the restart point equals v_start
// of the new segment, so the commanded position slope is continuous.
TEST(JointCmdControllerTest, VelocityContinuityAtRestart)
{
    auto ctrl = make_ctrl();
    std::vector<float> q_start  = {0.0f, 0.0f};
    std::vector<float> q_target = {2.0f, 2.0f};
    float t_restart = 0.5f;  // restart mid-ramp

    // Velocity of old ramp at the restart moment
    auto v_at_restart = ctrl.interp_velocity(q_start, zeros(2), q_target, t_restart);

    // New ramp starting from that velocity — velocity at t=0 must equal v_at_restart
    std::vector<float> q_start2  = {1.0f, 1.0f};  // some mid-ramp position
    std::vector<float> q_target2 = {3.0f, 3.0f};
    auto v_new_start = ctrl.interp_velocity(q_start2, v_at_restart, q_target2, 0.0f);

    // At t=0, cubic_hermite_velocity returns v_start exactly
    for (std::size_t i = 0; i < v_at_restart.size(); ++i)
        EXPECT_NEAR(v_new_start[i], v_at_restart[i], 1e-5f);
}
