// #pragma once

// #include "extended_kalman_filter.hpp"
// #include <ceres/jet.h>
// #include <cmath>

// namespace armor_ekf
// {

// // 状态维数 / 量测维数
// constexpr int N_x = 10;
// constexpr int N_z = 4;

// // 运动模型：
// // - CONSTANT_VELOCITY：只考虑平移匀速
// // - CONSTANT_ROTATION：只考虑自转匀速
// // - CONSTANT_VEL_ROT：平移 + 自转都匀速
// enum class MotionModel
// {
//     CONSTANT_VELOCITY,
//     CONSTANT_ROTATION,
//     CONSTANT_VEL_ROT
// };

// inline constexpr double OFFSET_SIGN = 1.0;

// // 过程模型 f(x
// // 单位：位置 mm，时间 s，角度 rad
// struct Predict
// {
//     double dt;
//     MotionModel model;

//     Predict(double t = 0.005,
//             MotionModel m = MotionModel::CONSTANT_VEL_ROT)
//         : dt(t), model(m) {}

//     template <typename T>
//     void operator()(const T x_in[N_x], T x_out[N_x]) const
//     {
//         for (int i = 0; i < N_x; ++i)
//             x_out[i] = x_in[i];

//         // 平移匀速
//         if (model == MotionModel::CONSTANT_VELOCITY ||
//             model == MotionModel::CONSTANT_VEL_ROT)
//         {
//             x_out[0] = x_in[0] + x_in[1] * T(dt); // xc
//             x_out[2] = x_in[2] + x_in[3] * T(dt); // yc
//             x_out[4] = x_in[4] + x_in[5] * T(dt); // zc
//         }

//         // 自转匀速
//         if (model == MotionModel::CONSTANT_ROTATION ||
//             model == MotionModel::CONSTANT_VEL_ROT)
//         {
//             x_out[6] = x_in[6] + x_in[7] * T(dt); // yaw
//         }

//         // 速度和形状参数
//         x_out[1] = x_in[1]; // vxc
//         x_out[3] = x_in[3]; // vyc
//         x_out[5] = x_in[5]; // vzc
//         x_out[7] = x_in[7]; // vyaw
//         x_out[8] = x_in[8]; // r
//         x_out[9] = x_in[9]; // dz
//     }
// };

// // 中心状态 → 装甲板的坐标（RestFrame, mm）
// struct Measure
// {
//     template <typename T>
//     void operator()(const T x_in[N_x], T z_out[N_z]) const
//     {
//         const T &xc  = x_in[0];
//         const T &yc  = x_in[2];
//         const T &zc  = x_in[4];
//         const T &yaw = x_in[6];
//         const T &r   = x_in[8];
//         const T &dz  = x_in[9];

//         z_out[0] = xc + T(OFFSET_SIGN) * r * ceres::sin(yaw); // xa
//         z_out[1] = yc - T(OFFSET_SIGN) * r * ceres::cos(yaw); // ya
//         z_out[2] = zc + dz;                                   // za
//         z_out[3] = yaw;                                       // yaw_a
//     }
// };

// using RobotEKF = ExtendedKalmanFilter<N_x, N_z, Predict, Measure>;

// } // namespace armor_ekf



#pragma once

#include "extended_kalman_filter.hpp"
#include <ceres/jet.h>
#include <cmath>

namespace armor_ekf
{

// ===================== 旧 10 维整体模型（仍保留以兼容其它代码） =====================

// 状态维数 / 量测维数
constexpr int N_x = 10;
constexpr int N_z = 4;

// 运动模型：
// - CONSTANT_VELOCITY：只考虑平移匀速 + 自转匀速
// - CONSTANT_ROTATION：只考虑自转匀速
// - CONSTANT_VEL_ROT：平移 + 自转都匀速（默认）
enum class MotionModel
{
    CONSTANT_VELOCITY,
    CONSTANT_ROTATION,
    CONSTANT_VEL_ROT
};

inline constexpr double OFFSET_SIGN = 1.0;

// 过程模型 f(x)：
// 状态： [0] xc, [1] vxc,
//       [2] yc, [3] vyc,
//       [4] zc, [5] vzc,
//       [6] yaw, [7] vyaw,
//       [8] r,   [9] dz
// 单位：位置 mm，时间 s，角度 rad
struct Predict
{
    double      dt;
    MotionModel model;

    Predict(double t = 0.005,
            MotionModel m = MotionModel::CONSTANT_VEL_ROT)
        : dt(t), model(m) {}

    template <typename T>
    void operator()(const T x_in[N_x], T x_out[N_x]) const
    {
        // 默认先把所有状态拷贝过去
        for (int i = 0; i < N_x; ++i)
            x_out[i] = x_in[i];

        const T dt_T = T(dt);

        const T &xc   = x_in[0];
        const T &vxc  = x_in[1];
        const T &yc   = x_in[2];
        const T &vyc  = x_in[3];
        const T &zc   = x_in[4];
        const T &vzc  = x_in[5];
        const T &yaw  = x_in[6];
        const T &vyaw = x_in[7];

        switch (model)
        {
        case MotionModel::CONSTANT_VELOCITY:
        case MotionModel::CONSTANT_VEL_ROT:
            // 平移匀速
            x_out[0] = xc + vxc * dt_T;
            x_out[2] = yc + vyc * dt_T;
            x_out[4] = zc + vzc * dt_T;
            // 自转匀速
            x_out[6] = yaw + vyaw * dt_T;
            break;
        case MotionModel::CONSTANT_ROTATION:
            x_out[6] = yaw + vyaw * dt_T;
            break;
        default:
            x_out[0] = xc + vxc * dt_T;
            x_out[2] = yc + vyc * dt_T;
            x_out[4] = zc + vzc * dt_T;
            x_out[6] = yaw + vyaw * dt_T;
            break;
        }

        // 速度和形状参数维持不变
        x_out[1] = vxc;
        x_out[3] = vyc;
        x_out[5] = vzc;
        x_out[7] = vyaw;
        x_out[8] = x_in[8];
        x_out[9] = x_in[9];
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

// ===================== 新拆分的两个 EKF =====================

// 位置 EKF：只管中心平移 [xc, vxc, yc, vyc, zc, vzc]
constexpr int N_x_pos = 6;
constexpr int N_z_pos = 3;

// 几何 EKF：只管装甲几何 [yaw, vyaw, r, dz]
constexpr int N_x_geo = 4;
constexpr int N_z_geo = 3;

// 位置 EKF 过程模型
struct PosPredict
{
    double      dt;
    MotionModel model;

    PosPredict(double t = 0.005,
               MotionModel m = MotionModel::CONSTANT_VEL_ROT)
        : dt(t), model(m) {}

    template <typename T>
    void operator()(const T x_in[N_x_pos], T x_out[N_x_pos]) const
    {
        for (int i = 0; i < N_x_pos; ++i)
            x_out[i] = x_in[i];

        const T dt_T = T(dt);

        const T &xc  = x_in[0];
        const T &vxc = x_in[1];
        const T &yc  = x_in[2];
        const T &vyc = x_in[3];
        const T &zc  = x_in[4];
        const T &vzc = x_in[5];

        (void)model; // 目前只用常速度模型

        x_out[0] = xc + vxc * dt_T;
        x_out[2] = yc + vyc * dt_T;
        x_out[4] = zc + vzc * dt_T;

        x_out[1] = vxc;
        x_out[3] = vyc;
        x_out[5] = vzc;
    }
};

// 位置 EKF 量测模型：z_pos = [xc, yc, zc]
struct PosMeasure
{
    template <typename T>
    void operator()(const T x_in[N_x_pos], T z_out[N_z_pos]) const
    {
        z_out[0] = x_in[0]; // xc
        z_out[1] = x_in[2]; // yc
        z_out[2] = x_in[4]; // zc
    }
};

// 几何 EKF 过程模型：x_geo = [yaw, vyaw, r, dz]
struct GeoPredict
{
    double dt;

    explicit GeoPredict(double t = 0.005)
        : dt(t) {}

    template <typename T>
    void operator()(const T x_in[N_x_geo], T x_out[N_x_geo]) const
    {
        for (int i = 0; i < N_x_geo; ++i)
            x_out[i] = x_in[i];

        const T dt_T = T(dt);

        const T &yaw  = x_in[0];
        const T &vyaw = x_in[1];

        x_out[0] = yaw + vyaw * dt_T; // yaw
        x_out[1] = vyaw;              // vyaw
        x_out[2] = x_in[2];           // r  近似常值
        x_out[3] = x_in[3];           // dz 近似常值
    }
};

// 几何 EKF 量测模型：z_geo = [yaw, r, dz]
struct GeoMeasure
{
    template <typename T>
    void operator()(const T x_in[N_x_geo], T z_out[N_z_geo]) const
    {
        z_out[0] = x_in[0]; // yaw
        z_out[1] = x_in[2]; // r
        z_out[2] = x_in[3]; // dz
    }
};

using PosEKF = ExtendedKalmanFilter<N_x_pos, N_z_pos, PosPredict, PosMeasure>;
using GeoEKF = ExtendedKalmanFilter<N_x_geo, N_z_geo, GeoPredict, GeoMeasure>;

} // namespace armor_ekf
