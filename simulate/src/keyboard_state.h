#pragma once

#include <GLFW/glfw3.h>
#include <array>

namespace keyboard {
    // Shared key state updated by the GLFW key callback and read by KeyboardJoystick.
    inline std::array<bool, GLFW_KEY_LAST + 1> key_state{};
}
