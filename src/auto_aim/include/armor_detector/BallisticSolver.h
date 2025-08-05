#pragma once

// 结构体声明
struct BallisticInfo {
    float pitch_angle;  // pitch需要转动的角度
    float yaw_angle;    // yaw最终的角度（逆时针为正）
    bool valid;
};

// 函数声明
BallisticInfo calcBallisticAngle(float x, float y, float z, float deltax, float deltay, float deltaz, 
                                float v, float cur_pitch, float cur_yaw);

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