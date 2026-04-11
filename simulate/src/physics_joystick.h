#pragma once

#include <iostream>
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include "joystick/joystick.h"
#include "keyboard_state.h"
#include <memory>


class XBoxJoystick : public unitree::common::UnitreeJoystick
{
public:
    XBoxJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Warning: XBox joystick not found at " << device << ". Continuing without joystick." << std::endl;
		}
        max_value_ = 1 << (bits - 1);
	}

    void update() override
    {
        if(!js_->isFound()) return;
        js_->getState();
        back(js_->button_[6]);
        start(js_->button_[7]);
        LB(js_->button_[4]);
        RB(js_->button_[5]);
        A(js_->button_[0]);
        B(js_->button_[1]); 
        X(js_->button_[2]);
        Y(js_->button_[3]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[2] > 0);
        RT(js_->axis_[5] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[3]) / max_value_);
        ry(-double(js_->axis_[4]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};


class SwitchJoystick : public unitree::common::UnitreeJoystick
{
public:
    SwitchJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Warning: Switch joystick not found at " << device << ". Continuing without joystick." << std::endl;
		}
        max_value_ = 1 << (bits - 1);
	}

    void update() override
    {
        if(!js_->isFound()) return;
        js_->getState();
        back(js_->button_[10]);
        start(js_->button_[11]);
        LB(js_->button_[6]);
        RB(js_->button_[7]);
        A(js_->button_[0]);
        B(js_->button_[1]); 
        X(js_->button_[3]);
        Y(js_->button_[4]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[5] > 0);
        RT(js_->axis_[4] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[2]) / max_value_);
        ry(-double(js_->axis_[3]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};


// Keyboard mapping:
//   W/S        → ly +/-   (walk forward/back)
//   A/D        → lx -/+   (strafe left/right)
//   Q/E        → rx -/+   (turn left/right)
//   Arrow keys → D-pad up/down/left/right
//   Left Ctrl  → LT
//   Right Ctrl → RT
//   Left Shift → LB
//   Right Shift→ RB
//   J          → A
//   K          → B
//   U          → X
//   I          → Y
class KeyboardJoystick : public unitree::common::UnitreeJoystick
{
public:
    KeyboardJoystick() : unitree::common::UnitreeJoystick() {}

    void update() override
    {
        using namespace keyboard;
        A(key_state[GLFW_KEY_J]);
        B(key_state[GLFW_KEY_K]);
        X(key_state[GLFW_KEY_U]);
        Y(key_state[GLFW_KEY_I]);
        LB(key_state[GLFW_KEY_LEFT_SHIFT]);
        RB(key_state[GLFW_KEY_RIGHT_SHIFT]);
        LT(key_state[GLFW_KEY_LEFT_CONTROL]);
        RT(key_state[GLFW_KEY_RIGHT_CONTROL]);
        back(key_state[GLFW_KEY_TAB]);
        start(key_state[GLFW_KEY_ENTER]);
        up(key_state[GLFW_KEY_UP]);
        down(key_state[GLFW_KEY_DOWN]);
        left(key_state[GLFW_KEY_LEFT]);
        right(key_state[GLFW_KEY_RIGHT]);
        lx((key_state[GLFW_KEY_D] ? 1.0 : 0.0) - (key_state[GLFW_KEY_A] ? 1.0 : 0.0));
        ly((key_state[GLFW_KEY_W] ? 1.0 : 0.0) - (key_state[GLFW_KEY_S] ? 1.0 : 0.0));
        rx((key_state[GLFW_KEY_E] ? 1.0 : 0.0) - (key_state[GLFW_KEY_Q] ? 1.0 : 0.0));
        ry(0.0);
    }
};