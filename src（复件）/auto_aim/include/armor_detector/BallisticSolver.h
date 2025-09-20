// BallisticSolver.h
#ifndef BALLISTIC_SOLVER_H
#define BALLISTIC_SOLVER_H

#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>

// 结构体声明
struct BallisticInfo {
    float delta_pitch_rad;  // pitch需要转动的角度
    float target_yaw_rad;    // yaw最终的角度（逆时针为正）
    bool valid;
};

class BallisticSolver {
public:
    BallisticSolver(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node) : 
    config_file_ptr(config_file_ptr), node(node) {}
    // 函数声明
    BallisticInfo calcBallisticAngle(float x, float y, float z, float deltax, float deltay, float deltaz, 
                                    float v_bullet, float cur_pitch, float cur_yaw);
                                    
private:
    
    std::shared_ptr<YAML::Node> config_file_ptr;
    rclcpp::Node* node;

    struct CalcPitchInfo {
        float target_pitch_result_smaller;
        float target_pitch_result_larger;
        bool valid;
    };
    float normalizeRad(float rad);
    float shortestRadDiff(float target, float current);
    CalcPitchInfo calcTargetPitch(float horizontal_distance, float vertical_distance, float v_bullet);

    struct BallisticParams {
        float drag_coeff;      // 阻力系数
        float air_density;     // 空气密度
        float bullet_diameter; // 弹丸直径
        float bullet_mass;     // 弹丸质量
        // 添加默认构造函数
        BallisticParams() : drag_coeff(0.47f), air_density(1.225f), bullet_diameter(17.0*1e-3), bullet_mass(3.2*1e-3) {}
    };
    struct SimulateTrajectoryInfo {
        float hit_height;
        bool valid;
    };
    struct TrajectoryInfo {
        float pitch;
        float hit_height;
    };
    struct RefineInfo {
        TrajectoryInfo lower_pitch_trajectory;
        TrajectoryInfo upper_pitch_trajectory;
        bool rising;
        bool valid = false;
    };
    BallisticParams ballisticParams;
    CalcPitchInfo calcTargetPitchWithAirResistance(float horizontal_distance, float vertical_height, float v_bullet);
    SimulateTrajectoryInfo simulateTrajectory(double v_bullet, double pitch_rad, double horizontal_distance,
                                              double MAX_FLIGHT_TIME, double DT, double MIN_HEIGHT);
};

#endif // BALLISTIC_SOLVER_H


// BallisticSolver.h


// #pragma once

// // 结构体定义
// struct BallisticInfo {
//     float pitch_angle;  // pitch需要转动的角度
//     float yaw_angle;    // yaw最终的角度（逆时针为正）
//     bool valid;
// };

// // 弹道参数配置结构体
// struct BallisticParams {
//     float drag_coeff;      // 阻力系数
//     float air_density;     // 空气密度
//     float projectile_dia;  // 弹丸直径
    
//     // 添加默认构造函数
//     BallisticParams() : drag_coeff(0.47f), air_density(1.225f), projectile_dia(0.0425f) {}
// };

// // 函数声明 - 确保与实现完全匹配
// BallisticInfo calcBallisticAngle(float x, float y, float z, 
//                                 float deltax, float deltay, float deltaz,
//                                 float v, float cur_pitch, float cur_yaw,
//                                 const BallisticParams& params = BallisticParams());