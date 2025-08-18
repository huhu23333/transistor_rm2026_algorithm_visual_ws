#include <math.h>
#include <iostream>
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
    x_camera = (x_camera + deltax_camera) / 1000.0f;
    y_camera = (y_camera + deltay_camera) / 1000.0f;
    z_camera = (z_camera + deltaz_camera) / 1000.0f;

    // 3. 转弧度
    float pitch_rad = cur_pitch * M_PI / 180.0f;

    float x_standard = x_camera;
    float y_standard = z_camera*sin(pitch_rad) + y_camera*cos(pitch_rad);
    float z_standard = z_camera*cos(pitch_rad) - y_camera*sin(pitch_rad);
    float r_standard = sqrt(x_standard*x_standard + z_standard*z_standard);

    // 1. 计算目标yaw弧度
    float target_delta_yaw = atan2(-x_standard, z_standard);
    float target_yaw = normalizeRad(target_delta_yaw + cur_yaw);  // 标准化到[-M_PI, M_PI]
    
    // pitch变换
    float y_g = y_standard;
    float r_g = r_standard;
    
    // 4. 求解弹道方程
    float g = 9.8f;
    float v2 = v * v;
    float temp1 = v2 / r_g / g ;
    float temp2 = (2 * v2 * y_g ) / (r_g * r_g * g);
    float delta = temp1 * temp1 + temp2 - 1;

    if (delta < 0) {
        return result;  // 返回无效结果
    }
    
    // 计算两个可能的pitch角
    float angle1 = atan(-temp1 + sqrt(delta));
    float angle2 = atan(-temp1 - sqrt(delta));

    // 选择较小的仰角
    angle1 = angle1 * 180.0f / M_PI;
    angle2 = angle2 * 180.0f / M_PI;
    float final_pitch = abs(angle1 - cur_pitch) < abs(angle2 - cur_pitch) ? angle1 : angle2;

    // 5. 计算需要转动的角度
    result.pitch_angle = final_pitch - cur_pitch;
    result.yaw_angle = target_yaw;
    
    result.valid = true;
    return result;
}
