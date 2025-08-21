// BallisticSolver.cpp
#include "armor_detector/BallisticSolver.h"  // 包含头文件

using namespace std;

// 辅助函数：将角度限制在[-180, 180]范围内
float normalizeRad(float rad) {
    while (rad > M_PI) rad -= 2 * M_PI;
    while (rad < -M_PI) rad += 2 * M_PI;
    return rad;
}

// 辅助函数：计算最短角度差
float shortestRadDiff(float target, float current) {
    float diff = normalizeRad(target - current);
    return diff;
}

BallisticInfo calcBallisticAngle(float x_camera, float y_camera, float z_camera, float deltax_camera, float deltay_camera, float deltaz_camera, 
                                  float v, float cur_pitch, float cur_yaw) {
    BallisticInfo result;
    result.valid = false;
    
    // 转换单位：mm到m
    x_camera = (x_camera + deltax_camera) / 1000.0f; // 向左
    y_camera = (y_camera + deltay_camera) / 1000.0f; // 向下
    z_camera = (z_camera + deltaz_camera) / 1000.0f; // 向前

    // 3. 转弧度
    float pitch_rad = cur_pitch * M_PI / 180.0f;

    float x_standard = -x_camera;                                               // 向右
    float y_standard = z_camera*sin(pitch_rad) - y_camera*cos(pitch_rad);       // 向上
    float z_standard = z_camera*cos(pitch_rad) + y_camera*sin(pitch_rad);       // 向前
    float r_standard = sqrt(x_standard*x_standard + z_standard*z_standard);

    // 1. 计算目标yaw弧度
    float target_delta_yaw = atan2(x_standard, z_standard);
    float target_yaw = normalizeRad(target_delta_yaw + cur_yaw);  // 标准化到[-M_PI, M_PI]
    
    // 4. 求解弹道方程
    float g = 9.8f;
    float denominator = g * r_standard;
    float v_square = v * v;
    float numerator_part1 = v_square;
    float numerator_part2_square = v_square * v_square - g * (g * r_standard * r_standard + 2 * y_standard * v_square);
    if (numerator_part2_square < 0) {
        return result;  // 返回无效结果
    }
    float tan_angle1 = (numerator_part1 + sqrt(numerator_part2_square)) / denominator;
    float tan_angle2 = (numerator_part1 - sqrt(numerator_part2_square)) / denominator;
    
    // 计算两个可能的pitch角
    float angle1 = atan(tan_angle1);
    float angle2 = atan(tan_angle2);

    // 选择较小的仰角
    angle1 = angle1 * 180.0f / M_PI;
    angle2 = angle2 * 180.0f / M_PI;
    float final_pitch = abs(angle1 - cur_pitch) < abs(angle2 - cur_pitch) ? angle1 : angle2;

    // 5. 计算需要转动的角度
    result.pitch_angle = (final_pitch - cur_pitch) * M_PI / 180;
    result.yaw_angle = target_yaw;
    
    result.valid = true;
    return result;
}
