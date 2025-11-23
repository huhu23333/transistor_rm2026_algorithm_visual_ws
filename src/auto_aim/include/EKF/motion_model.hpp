// #ifndef ARMOR_SOLVER_MOTION_MODEL_HPP_
// #define ARMOR_SOLVER_MOTION_MODEL_HPP_

// #include "extended_kalman_filter.hpp" // 继续使用您的EKF头文件

// // 状态维数 N_x = 10, 测量维数 N_z = 4
// constexpr int N_x = 10;
// constexpr int N_z = 4;

// // 状态向量定义 (Y轴垂直)
// // 0: 机器人中心位置 x (xc)
// // 1: 机器人中心速度 x (v_xc)
// // 2: 机器人中心垂直位置/高度 y (yc)
// // 3: 机器人中心垂直速度 y (v_yc)
// // 4: 机器人中心位置 z (zc)
// // 5: 机器人中心速度 z (v_zc)
// // 6: 机器人Yaw角 (绕Y轴)
// // 7: 机器人Yaw角速度
// // 8: 机器人半径 (中心到装甲板)
// // 9: 装甲板相对中心的垂直偏移 (d_yc)
// struct Predict {
//     double dt;
//     explicit Predict(double t) : dt(t) {}

//     template <typename T>
//     void operator()(const T x_in[N_x], T x_out[N_x]) const {
//         // 根据新的 (位置, 速度) 配对更新预测
//         x_out[0] = x_in[0] + x_in[1] * dt; // xc' = xc + v_xc * dt
//         x_out[2] = x_in[2] + x_in[3] * dt; // yc' = yc + v_yc * dt
//         x_out[4] = x_in[4] + x_in[5] * dt; // zc' = zc + v_zc * dt
//         x_out[6] = x_in[6] + x_in[7] * dt; // yaw' = yaw + v_yaw * dt
        
//         // 速度和几何参数假设为匀速或不变
//         x_out[1] = x_in[1]; // v_xc
//         x_out[3] = x_in[3]; // v_yc
//         x_out[5] = x_in[5]; // v_zc
//         x_out[7] = x_in[7]; // v_yaw
//         x_out[8] = x_in[8]; // r
//         x_out[9] = x_in[9]; // d_yc
//     }
// };

// struct Measure {
//     template <typename T>
//     void operator()(const T x_in[N_x], T z_out[N_z]) const {
//         T xc = x_in[0], yc = x_in[2], zc = x_in[4];
//         T yaw = x_in[6], r = x_in[8];

//         z_out[0] = xc - r * ceres::cos(yaw); // xa = xc - r*cos(yaw)
//         z_out[1] = yc + x_in[9];                 // ya = yc + d_yc
//         z_out[2] = zc - r * ceres::sin(yaw); // za = zc - r*sin(yaw)
//         z_out[3] = yaw;                       // yaw_a = yaw
//     }
// };

// using RobotEKF = ExtendedKalmanFilter<N_x, N_z, Predict, Measure>;

// #endif // ARMOR_SOLVER_MOTION_MODEL_HPP_




#ifndef ARMOR_SOLVER_MOTION_MODEL_HPP_
#define ARMOR_SOLVER_MOTION_MODEL_HPP_

#include "extended_kalman_filter.hpp" 
#include <ceres/jet.h>

// 9状态 CA模型 (匀加速模型)
// 状态顺序: [x, vx, ax, y, vy, ay, z, vz, az]
constexpr int N_x = 9;
constexpr int N_z = 3;

// 索引定义方便后续使用
struct Idx {
    static constexpr int X = 0, Vx = 1, Ax = 2;
    static constexpr int Y = 3, Vy = 4, Ay = 5;
    static constexpr int Z = 6, Vz = 7, Az = 8;
};

struct Predict {
    double dt;
    explicit Predict(double t) : dt(t) {}

    template <typename T>
    void operator()(const T x_in[N_x], T x_out[N_x]) const {
        // 预计算时间幂
        T dt_t = T(dt);
        T dt2_half = T(0.5 * dt * dt);

        // X轴预测: x = x + v*t + 0.5*a*t^2
        x_out[Idx::X]  = x_in[Idx::X]  + x_in[Idx::Vx] * dt_t + x_in[Idx::Ax] * dt2_half;
        x_out[Idx::Vx] = x_in[Idx::Vx] + x_in[Idx::Ax] * dt_t;
        x_out[Idx::Ax] = x_in[Idx::Ax];

        // Y轴预测
        x_out[Idx::Y]  = x_in[Idx::Y]  + x_in[Idx::Vy] * dt_t + x_in[Idx::Ay] * dt2_half;
        x_out[Idx::Vy] = x_in[Idx::Vy] + x_in[Idx::Ay] * dt_t;
        x_out[Idx::Ay] = x_in[Idx::Ay];

        // Z轴预测
        x_out[Idx::Z]  = x_in[Idx::Z]  + x_in[Idx::Vz] * dt_t + x_in[Idx::Az] * dt2_half;
        x_out[Idx::Vz] = x_in[Idx::Vz] + x_in[Idx::Az] * dt_t;
        x_out[Idx::Az] = x_in[Idx::Az];
    }
};

struct Measure {
    template <typename T>
    void operator()(const T x_in[N_x], T z_out[N_z]) const {
        // 测量仅包含位置
        z_out[0] = x_in[Idx::X];
        z_out[1] = x_in[Idx::Y];
        z_out[2] = x_in[Idx::Z];
    }
};

using RobotEKF = ExtendedKalmanFilter<N_x, N_z, Predict, Measure>;

#endif // ARMOR_SOLVER_MOTION_MODEL_HPP_



// #ifndef ARMOR_SOLVER_MOTION_MODEL_HPP_
// #define ARMOR_SOLVER_MOTION_MODEL_HPP_

// #include "extended_kalman_filter.hpp"
// #include <ceres/jet.h>

// // 新模型：追踪旋转中心 (9维状态, 4维测量)
// constexpr int N_x = 9;
// constexpr int N_z = 4;

// // 状态向量 x:
// // 0: 中心位置 xc
// // 1: 中心速度 v_xc
// // 2: 中心位置 yc
// // 3: 中心速度 v_yc
// // 4: 中心位置 zc
// // 5: 中心速度 v_zc
// // 6: 机器人Yaw角 (绕Y轴)
// // 7: 机器人Yaw角速度 (v_yaw)
// // 8: 机器人半径 (中心到装甲板)
// struct Predict {
//     double dt;
//     explicit Predict(double t) : dt(t) {}

//     template <typename T>
//     void operator()(const T x_in[N_x], T x_out[N_x]) const {
//         // 中心位置和角度进行匀速预测
//         x_out[0] = x_in[0] + x_in[1] * dt;
//         x_out[2] = x_in[2] + x_in[3] * dt;
//         x_out[4] = x_in[4] + x_in[5] * dt;
//         x_out[6] = x_in[6] + x_in[7] * dt;
        
//         // 速度和半径假设不变
//         x_out[1] = x_in[1];
//         x_out[3] = x_in[3];
//         x_out[5] = x_in[5];
//         x_out[7] = x_in[7];
//         x_out[8] = x_in[8];
//     }
// };

// // 测量向量 z: [xa, ya, za, yaw_a] (装甲板的观测)
// struct Measure {
//     template <typename T>
//     void operator()(const T x_in[N_x], T z_out[N_z]) const {
//         T xc = x_in[0], yc = x_in[2], zc = x_in[4];
//         T yaw = x_in[6], r = x_in[8];

//         // 从机器人中心状态，反解出装甲板的位置
//         z_out[0] = xc - r * ceres::cos(yaw); // xa = xc - r*cos(yaw)
//         z_out[1] = yc + r * ceres::sin(yaw); // ya
//         z_out[2] = zc;                       // za
//         z_out[3] = yaw;                      // yaw_a
//     }
// };

// using RobotEKF = ExtendedKalmanFilter<N_x, N_z, Predict, Measure>;

// #endif // ARMOR_SOLVER_MOTION_MODEL_HPP_