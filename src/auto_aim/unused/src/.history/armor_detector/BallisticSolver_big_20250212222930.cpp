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


// 弹道参数配置
struct BallisticParams {
    float drag_coeff;      // 阻力系数Cd（球形弹丸约0.47，锥形约0.15-0.3）
    float air_density;     // 空气密度kg/m³（1.225 at sea level）
    float projectile_dia;  // 弹丸直径(m)
};


// 辅助函数：计算最短角度差
float shortestAngleDiff(float target, float current) {
    float diff = normalizeAngle(target - current);
    return diff;
}

// 带空气阻力的弹道模拟
int simulateTrajectory(float v0, float theta, float target_r, float target_y,
                        const BallisticParams& params,  float MAX_FLIGHT_TIME = 5.0f, float DT = 0.01f) {
    const float area = M_PI * pow(params.projectile_dia/2, 2);
    const float k = 0.5f * params.air_density * params.drag_coeff * area;
    
    float vx = v0 * cos(theta);
    float vy = v0 * sin(theta);
    float x = 0, y = 0, t = 0;
    
    while(t < MAX_FLIGHT_TIME) {
        float speed = sqrt(vx*vx + vy*vy);
        if(speed < 0.1f) break; // 速度过小停止计算
        
        // 空气阻力加速度
        float ax = -k * vx * speed;
        float ay = -9.8f - k * vy * speed;
        
        // 欧拉法积分
        vx += ax * DT;
        vy += ay * DT;
        x += vx * DT;
        y += vy * DT;
        t += DT;
        
        // 检查是否过目标点
        if(x >= target_r) {
            // 线性插值修正
            float overshoot = x - target_r;
            float back_step = overshoot / fabs(vx);
            y -= vy * back_step;
            if (fabs(y - target_y) < 0.01f) return 0; // 0.01米精度容差
            else return 1;
        }
    }
    return -1;
}

BallisticInfo calcBallisticAngle(float x, float y, float z, float deltax, float deltay, float deltaz, 
                                float v, float cur_pitch, float cur_yaw, const BallisticParams& params = {0.47f, 1.225f, 0.0425f}) {
    BallisticInfo result;
    result.valid = false;
  

    // 数值积分步长控制（秒）
    const float DT = 0.01f;
    const float MAX_FLIGHT_TIME = 5.0f;


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
    if (delta < 0) {
        return result;
    }
    // 计算两个可能的pitch角
    float angle1 = atan(-temp1 + sqrt(delta));
    float angle2 = atan(-temp1 - sqrt(delta));
    angle1 = angle1 * 180.0f / M_PI;
    angle2 = angle2 * 180.0f / M_PI;
    float final_pitch = abs(angle1 - cur_pitch) < abs(angle2 - cur_pitch) ? angle1 : angle2;

    float pitch_min = final_pitch - 5;
    float pitch_max = final_pitch + 5;   
    int hit = 1;
        // 二分法搜索最佳角度
    for(int i = 0; i < 10; ++i) {
        float theta = (pitch_min + pitch_max) / 2.0f;
        float test_pitch;
        hit = simulateTrajectory(v, theta, r_g, y_g, params);
        if(hit == 0) {
            result.pitch_angle = theta - cur_pitch;
            break;
        } else if(hit == 1) {
            pitch_max = theta;
        } else pitch_min = theta;
    }

    // 5. 计算需要转动的角度
    if (hit != 0) result.pitch_angle = pitch_max - cur_pitch;
    result.yaw_angle = target_yaw;
    
    result.valid = true;
    return result;
}





// int main() {
//     float x = 74.91, y = 126.93, z = 739.06, v = 25, cur_pitch = -1.47, cur_yaw = -153, delta_x = 0, delta_y = -44.0f, delta_z= 49.0f;
//     BallisticInfo result;
//     result = calcBallisticAngle(x, y, z, delta_x, delta_y, delta_z, v, cur_pitch, cur_yaw);
//     cout << result.pitch_angle << endl;
//     cout << result.yaw_angle << endl;
//     cout << result.valid << endl;
//     // 处理结果
//     return 0;
// }
