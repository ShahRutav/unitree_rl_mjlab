// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <vector>
#include <cassert>
#include <algorithm>

inline std::vector<float> linear_interpolate(float t, const std::vector<float>& ts, const std::vector<std::vector<float>>& ys)
{
    assert(ts.size() == ys.size() && !ys.empty() && ts.size() > 1 && ys[0].size() > 0);

    if (t <= ts[0]) return ys[0];
    if (t >= ts[ts.size() - 1]) return ys[ts.size() - 1];

    for (int i = 0; i < ts.size() - 1; ++i)
    {
        if (t >= ts[i] && t <= ts[i + 1])
        {
            float alpha = (t - ts[i]) / (ts[i + 1] - ts[i]);
            std::vector<float> result(ys[i].size());
            for (int j = 0; j < ys[i].size(); ++j)
            {
                result[j] = ys[i][j] * (1 - alpha) + ys[i + 1][j] * alpha;
            }
            return result;
        }
    }

    return std::vector<float>(ys[0].size(), 0.0f); // Fallback, should not reach here
}

// Cubic Hermite interpolation: position-and-velocity continuous.
//
// Matches p0,v0 at t=0 and p1,v1 at t=T. Set v1=zeros to arrive at rest.
// Using v0 = velocity of the outgoing ramp at the transition point makes
// successive segments velocity-continuous (no jerk at ramp restarts).
//
// Basis polynomials (tau = t/T):
//   h00 =  2tau^3 - 3tau^2 + 1   (blends p0)
//   h10 =   tau^3 - 2tau^2 + tau  (blends v0, scaled by T)
//   h01 = -2tau^3 + 3tau^2        (blends p1)
//   h11 =   tau^3 - tau^2         (blends v1, scaled by T)
inline std::vector<float> cubic_hermite(
    float t, float T,
    const std::vector<float>& p0, const std::vector<float>& v0,
    const std::vector<float>& p1, const std::vector<float>& v1)
{
    float tau  = std::clamp(t / T, 0.0f, 1.0f);
    float tau2 = tau * tau;
    float tau3 = tau2 * tau;
    float h00  =  2.0f*tau3 - 3.0f*tau2 + 1.0f;
    float h10  =       tau3 - 2.0f*tau2 + tau;
    float h01  = -2.0f*tau3 + 3.0f*tau2;
    float h11  =       tau3 - tau2;
    std::vector<float> r(p0.size());
    for (size_t i = 0; i < r.size(); ++i)
        r[i] = h00*p0[i] + h10*T*v0[i] + h01*p1[i] + h11*T*v1[i];
    return r;
}

// Velocity of the cubic Hermite segment at time t (dp/dt).
// Call this just before restarting a ramp to capture the outgoing velocity.
inline std::vector<float> cubic_hermite_velocity(
    float t, float T,
    const std::vector<float>& p0, const std::vector<float>& v0,
    const std::vector<float>& p1, const std::vector<float>& v1)
{
    float tau  = std::clamp(t / T, 0.0f, 1.0f);
    float tau2 = tau * tau;
    // d/dt of each basis: multiply d/dtau by 1/T, except h10/h11 which already absorbed T
    float dh00 = ( 6.0f*tau2 - 6.0f*tau) / T;
    float dh10 =   3.0f*tau2 - 4.0f*tau + 1.0f;
    float dh01 = (-6.0f*tau2 + 6.0f*tau) / T;
    float dh11 =   3.0f*tau2 - 2.0f*tau;
    std::vector<float> r(p0.size());
    for (size_t i = 0; i < r.size(); ++i)
        r[i] = dh00*p0[i] + dh10*v0[i] + dh01*p1[i] + dh11*v1[i];
    return r;
}
