#pragma once

#include "extended_kalman_filter.hpp"
#include <ceres/jet.h>
#include <cmath>

namespace armor_ekf
{

// 状态维数 / 量测维数
constexpr int N_x = 10;
constexpr int N_z = 4;

// 运动模型：
// - CONSTANT_VELOCITY：只考虑平移匀速
// - CONSTANT_ROTATION：只考虑自转匀速
// - CONSTANT_VEL_ROT：平移 + 自转都匀速
enum class MotionModel
{
    CONSTANT_VELOCITY,
    CONSTANT_ROTATION,
    CONSTANT_VEL_ROT
};

inline constexpr double OFFSET_SIGN = 1.0;

// 过程模型 f(x
// 单位：位置 mm，时间 s，角度 rad
struct Predict
{
    double dt;
    MotionModel model;

    Predict(double t = 0.005,
            MotionModel m = MotionModel::CONSTANT_VEL_ROT)
        : dt(t), model(m) {}

    template <typename T>
    void operator()(const T x_in[N_x], T x_out[N_x]) const
    {
        for (int i = 0; i < N_x; ++i)
            x_out[i] = x_in[i];

        // 平移匀速
        if (model == MotionModel::CONSTANT_VELOCITY ||
            model == MotionModel::CONSTANT_VEL_ROT)
        {
            x_out[0] = x_in[0] + x_in[1] * T(dt); // xc
            x_out[2] = x_in[2] + x_in[3] * T(dt); // yc
            x_out[4] = x_in[4] + x_in[5] * T(dt); // zc
        }

        // 自转匀速
        if (model == MotionModel::CONSTANT_ROTATION ||
            model == MotionModel::CONSTANT_VEL_ROT)
        {
            x_out[6] = x_in[6] + x_in[7] * T(dt); // yaw
        }

        // 速度和形状参数
        x_out[1] = x_in[1]; // vxc
        x_out[3] = x_in[3]; // vyc
        x_out[5] = x_in[5]; // vzc
        x_out[7] = x_in[7]; // vyaw
        x_out[8] = x_in[8]; // r
        x_out[9] = x_in[9]; // dz
    }
};

// 中心状态 → 装甲板的坐标（RestFrame, mm）
struct Measure
{
    template <typename T>
    void operator()(const T x_in[N_x], T z_out[N_z]) const
    {
        const T &xc  = x_in[0];
        const T &yc  = x_in[2];
        const T &zc  = x_in[4];
        const T &yaw = x_in[6];
        const T &r   = x_in[8];
        const T &dz  = x_in[9];

        z_out[0] = xc + T(OFFSET_SIGN) * r * ceres::sin(yaw); // xa
        z_out[1] = yc - T(OFFSET_SIGN) * r * ceres::cos(yaw); // ya
        z_out[2] = zc + dz;                                   // za
        z_out[3] = yaw;                                       // yaw_a
    }
};

using RobotEKF = ExtendedKalmanFilter<N_x, N_z, Predict, Measure>;

} // namespace armor_ekf
