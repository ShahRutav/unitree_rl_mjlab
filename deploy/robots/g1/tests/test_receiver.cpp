#include <gtest/gtest.h>
#include "FSM/JointCmdReceiver.h"

#include <string>
#include <vector>
#include <sstream>

// ---------------------------------------------------------------------------
// Helper: build a JSON string with N float values
// ---------------------------------------------------------------------------
static std::string make_q_json(const std::vector<float>& vals) {
    std::ostringstream oss;
    oss << "{\"q\": [";
    for (std::size_t i = 0; i < vals.size(); ++i) {
        oss << vals[i];
        if (i + 1 < vals.size()) oss << ", ";
    }
    oss << "]}";
    return oss.str();
}

// ---------------------------------------------------------------------------
// 1. ParseValid29 — 29-element vector, spot-check first and last
// ---------------------------------------------------------------------------
TEST(JointCmdReceiverTest, ParseValid29) {
    std::vector<float> expected;
    expected.reserve(29);
    for (int i = 0; i < 28; ++i) {
        expected.push_back(static_cast<float>(i + 1) * 0.1f);
    }
    expected.push_back(0.0f); // last element

    const std::string json = make_q_json(expected);
    auto result = JointCmdReceiver::parse_message(json);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->q.size(), 29u);
    EXPECT_NEAR(result->q[0],  0.1f, 1e-5f);
    EXPECT_NEAR(result->q[28], 0.0f, 1e-5f);
    EXPECT_TRUE(result->tau_ff.empty());
}

// ---------------------------------------------------------------------------
// 2. ParseValidSmall — receiver does not enforce size == 29
// ---------------------------------------------------------------------------
TEST(JointCmdReceiverTest, ParseValidSmall) {
    const std::string json = R"({"q": [1.0, 2.0, 3.0, 4.0, 5.0]})";
    auto result = JointCmdReceiver::parse_message(json);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->q.size(), 5u);
    EXPECT_NEAR(result->q[0], 1.0f, 1e-5f);
    EXPECT_NEAR(result->q[4], 5.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// 2b. ParseWithTauFf — both q and tau_ff present, equal lengths
// ---------------------------------------------------------------------------
TEST(JointCmdReceiverTest, ParseWithTauFf) {
    const std::string json =
        R"({"q": [1.0, 2.0, 3.0], "tau_ff": [0.1, -0.2, 0.3]})";
    auto result = JointCmdReceiver::parse_message(json);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->q.size(), 3u);
    ASSERT_EQ(result->tau_ff.size(), 3u);
    EXPECT_NEAR(result->tau_ff[0],  0.1f, 1e-5f);
    EXPECT_NEAR(result->tau_ff[1], -0.2f, 1e-5f);
    EXPECT_NEAR(result->tau_ff[2],  0.3f, 1e-5f);
}

// ---------------------------------------------------------------------------
// 2c. ParseTauFfLengthMismatch — tau_ff present but wrong length → nullopt
// ---------------------------------------------------------------------------
TEST(JointCmdReceiverTest, ParseTauFfLengthMismatch) {
    const std::string json = R"({"q": [1.0, 2.0, 3.0], "tau_ff": [0.1, 0.2]})";
    auto result = JointCmdReceiver::parse_message(json);

    ASSERT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// 3. ParseMissingKey — wrong key name → nullopt
// ---------------------------------------------------------------------------
TEST(JointCmdReceiverTest, ParseMissingKey) {
    const std::string json = R"({"joints": [1.0, 2.0]})";
    auto result = JointCmdReceiver::parse_message(json);

    ASSERT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// 4. ParseNotArray — value of "q" is a scalar, not an array → nullopt
// ---------------------------------------------------------------------------
TEST(JointCmdReceiverTest, ParseNotArray) {
    const std::string json = R"({"q": 1.0})";
    auto result = JointCmdReceiver::parse_message(json);

    ASSERT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// 5. ParseNonFloat — array contains strings instead of numbers → nullopt
// ---------------------------------------------------------------------------
TEST(JointCmdReceiverTest, ParseNonFloat) {
    const std::string json = R"({"q": ["a", "b"]})";
    auto result = JointCmdReceiver::parse_message(json);

    ASSERT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// 6. ParseEmptyArray — size < 1 rule → nullopt
// ---------------------------------------------------------------------------
TEST(JointCmdReceiverTest, ParseEmptyArray) {
    const std::string json = R"({"q": []})";
    auto result = JointCmdReceiver::parse_message(json);

    ASSERT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// 7. ParseMalformedJSON — not valid JSON at all → nullopt
// ---------------------------------------------------------------------------
TEST(JointCmdReceiverTest, ParseMalformedJSON) {
    const std::string json = "not json at all";
    auto result = JointCmdReceiver::parse_message(json);

    ASSERT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// 8. ParseEmptyString — empty input → nullopt
// ---------------------------------------------------------------------------
TEST(JointCmdReceiverTest, ParseEmptyString) {
    const std::string json = "";
    auto result = JointCmdReceiver::parse_message(json);

    ASSERT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
