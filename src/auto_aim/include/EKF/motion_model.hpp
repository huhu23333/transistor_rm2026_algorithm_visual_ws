/* #ifndef ARMOR_SOLVER_MOTION_MODEL_HPP_
#define ARMOR_SOLVER_MOTION_MODEL_HPP_

#include "extended_kalman_filter.hpp" // 继续使用您的EKF头文件
#include <ceres/jet.h>

// 新模型：状态维数 N_x = 6, 测量维数 N_z = 3
constexpr int N_x = 6;
constexpr int N_z = 3;

// 状态向量定义: 直接描述装甲板
// 0: 装甲板位置 x (xa)
// 1: 装甲板速度 x (v_xa)
// 2: 装甲板位置 y (ya)
// 3: 装甲板速度 y (v_ya)
// 4: 装甲板位置 z (za)
// 5: 装甲板速度 z (v_za)
struct Predict {
    double dt;
    explicit Predict(double t) : dt(t) {}

    template <typename T>
    void operator()(const T x_in[N_x], T x_out[N_x]) const {
        // 匀速运动模型
        x_out[0] = x_in[0] + x_in[1] * dt; // x' = x + vx * dt
        x_out[2] = x_in[2] + x_in[3] * dt; // y' = y + vy * dt
        x_out[4] = x_in[4] + x_in[5] * dt; // z' = z + vz * dt
        
        // 速度假设不变
        x_out[1] = x_in[1]; // vx
        x_out[3] = x_in[3]; // vy
        x_out[5] = x_in[5]; // vz
    }
};

struct Measure {
    template <typename T>
    void operator()(const T x_in[N_x], T z_out[N_z]) const {
        // 测量即为状态中的位置
        z_out[0] = x_in[0]; // z_x = x
        z_out[1] = x_in[2]; // z_y = y
        z_out[2] = x_in[4]; // z_z = z
    }
};

using RobotEKF = ExtendedKalmanFilter<N_x, N_z, Predict, Measure>;

#endif // ARMOR_SOLVER_MOTION_MODEL_HPP_ */



#ifndef ARMOR_SOLVER_MOTION_MODEL_HPP_
#define ARMOR_SOLVER_MOTION_MODEL_HPP_

#include "extended_kalman_filter.hpp"
#include <ceres/jet.h>

// 新模型：追踪旋转中心 (9维状态, 4维测量)
constexpr int N_x = 9;
constexpr int N_z = 4;

// 状态向量 x:
// 0: 中心位置 xc
// 1: 中心速度 v_xc
// 2: 中心位置 yc
// 3: 中心速度 v_yc
// 4: 中心位置 zc
// 5: 中心速度 v_zc
// 6: 机器人Yaw角 (绕Y轴)
// 7: 机器人Yaw角速度 (v_yaw)
// 8: 机器人半径 (中心到装甲板)
struct Predict {
    double dt;
    explicit Predict(double t) : dt(t) {}

    template <typename T>
    void operator()(const T x_in[N_x], T x_out[N_x]) const {
        // 中心位置和角度进行匀速预测
        x_out[0] = x_in[0] + x_in[1] * dt;
        x_out[2] = x_in[2] + x_in[3] * dt;
        x_out[4] = x_in[4] + x_in[5] * dt;
        x_out[6] = x_in[6] + x_in[7] * dt;
        
        // 速度和半径假设不变
        x_out[1] = x_in[1];
        x_out[3] = x_in[3];
        x_out[5] = x_in[5];
        x_out[7] = x_in[7];
        x_out[8] = x_in[8];
    }
};

// 测量向量 z: [xa, ya, za, yaw_a] (装甲板的观测)
struct Measure {
    template <typename T>
    void operator()(const T x_in[N_x], T z_out[N_z]) const {
        T xc = x_in[0], yc = x_in[2], zc = x_in[4];
        T yaw = x_in[6], r = x_in[8];

        // 从机器人中心状态，反解出装甲板的位置
        z_out[0] = xc + r * ceres::sin(yaw); // xa
        z_out[1] = yc - r * ceres::cos(yaw); // ya
        z_out[2] = zc;                       // za
        z_out[3] = yaw;                      // yaw_a
    }
};

using RobotEKF = ExtendedKalmanFilter<N_x, N_z, Predict, Measure>;

#endif // ARMOR_SOLVER_MOTION_MODEL_HPP_