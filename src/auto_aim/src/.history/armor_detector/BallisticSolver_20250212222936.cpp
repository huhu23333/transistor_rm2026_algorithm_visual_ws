#include <math.h>
#include <stdio.h>
#include <iostream>
using namespace std;
struct BallisticInfo {
    float pitch_angle;  // pitch需要转动的角度
    float yaw_angle;    // yaw最终的角度（逆时针为正）
    bool valid;
};

// 辅助函数：将角度限制在[-180, 180]范围内
float normalizeAngle(float angle) {
    while(angle > 180.0f) angle -= 360.0f;
    while(angle < -180.0f) angle += 360.0f;
    return angle;
}

// 辅助函数：计算最短角度差
float shortestAngleDiff(float target, float current) {
    float diff = normalizeAngle(target - current);
    return diff;
}

BallisticInfo calcBallisticAngle(float x, float y, float z, float deltax, float deltay, float deltaz, 
                                float v, float cur_pitch, float cur_yaw) {
    BallisticInfo result;
    result.valid = false;
    
    // 转换单位：mm到m
    x = (x + deltax) / 1000.0f;
    y = (y + deltay) / 1000.0f;
    z = (z + deltaz) / 1000.0f;
    
    // 1. 计算目标yaw角度
    float target_yaw = atan2(-x, z) * 180.0f / M_PI;
    target_yaw = normalizeAngle(target_yaw + cur_yaw);  // 标准化到[-180, 180]
    
    // 2. 计算水平距离(使用原始坐标)
    float r = sqrt(x*x + z*z);
    
    // 3. 转弧度
    float pitch_rad = cur_pitch * M_PI / 180.0f;
    
    // pitch变换
    float y_g = r*sin(pitch_rad) + y*cos(pitch_rad);
    float r_g = r*cos(pitch_rad) - y*sin(pitch_rad);
    
    // 4. 求解弹道方程
    float g = 9.8f;
    float v2 = v * v;
    float temp1 = v2 / r_g / g ;
    float temp2 = (2 * v2 * y_g ) / (r_g * r_g * g);
    float delta = temp1 * temp1 + temp2 - 1;
    // cout << "temp1:" << temp1 << "delta:" << delta << endl;
    if (delta < 0) {
        return result;
    }
    // 计算两个可能的pitch角
    float angle1 = atan(-temp1 + sqrt(delta));
    float angle2 = atan(-temp1 - sqrt(delta));
    // cout << angle1 << " " << angle2 << endl;
    // cout << angle1 << " " << angle2 << endl;
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





// int main() {
//     float x = 74.91, y = 126.93, z = 739.06, v = 25, cur_pitch = -1.47, cur_yaw = -153,delta_x = 0,delta_y = -44.0f, delta_z= 49.0f;
//     BallisticInfo result;
//     result = calcBallisticAngle(x, y, z,delta_x, delta_y, delta_z, v, cur_pitch, cur_yaw);
//     cout << result.pitch_angle << endl;
//     cout << result.yaw_angle << endl;
//     cout << result.valid << endl;
//     // 处理结果
//     return 0;
// }
