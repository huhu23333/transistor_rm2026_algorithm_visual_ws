// BallisticSolver.cpp
#include "armor_detector/BallisticSolver.h"  // 包含头文件



// 辅助函数：将角度限制在[-180, 180]范围内
float BallisticSolver::normalizeRad(float rad) {
    while (rad > M_PI) rad -= 2 * M_PI;
    while (rad < -M_PI) rad += 2 * M_PI;
    return rad;
}

// 辅助函数：计算最短角度差
float BallisticSolver::shortestRadDiff(float target, float current) {
    float diff = normalizeRad(target - current);
    return diff;
}

BallisticSolver::CalcPitchInfo BallisticSolver::calcTargetPitch(float horizontal_distance, float vertical_height, float v_bullet) {
    CalcPitchInfo result;
    result.valid = false;
    
    // 4. 求解弹道方程
    float g = 9.8f;
    float denominator = g * horizontal_distance;
    float v_bullet_square = v_bullet * v_bullet;
    float numerator_part1 = v_bullet_square;
    float numerator_part2_square = v_bullet_square * v_bullet_square - g * (g * horizontal_distance * horizontal_distance + 2 * vertical_height * v_bullet_square);
    if (numerator_part2_square < 0) {
        return result;  // 返回无效结果
    }
    float tan_angle1 = (numerator_part1 + sqrt(numerator_part2_square)) / denominator;
    float tan_angle2 = (numerator_part1 - sqrt(numerator_part2_square)) / denominator;
    
    // 计算两个可能的pitch角
    float pitch_rad1 = atan(tan_angle1);
    float pitch_rad2 = atan(tan_angle2);

    bool is_pitch1_smaller = pitch_rad1 <= pitch_rad2;
    result.target_pitch_result_smaller = is_pitch1_smaller ? pitch_rad1 : pitch_rad2;
    result.target_pitch_result_larger = (!is_pitch1_smaller) ? pitch_rad1 : pitch_rad2;
    result.valid = true;

    return result;
}

BallisticSolver::SimulateTrajectoryInfo BallisticSolver::simulateTrajectory(
    double v_bullet, double pitch_rad, double horizontal_distance,
    double max_flight_time, double dt, double min_height) {
    
    double g = 9.8;
    uint32_t time_step = 0;

    SimulateTrajectoryInfo result;
    result.valid = false;

    // 计算弹丸截面积
    double bullet_radius = ballisticParams.bullet_diameter / 2.0;
    double cross_section_area = M_PI * bullet_radius * bullet_radius;

    double pos_x = 0;
    double pos_y = 0;
    double vel_x = v_bullet * std::cos(pitch_rad);
    double vel_y = v_bullet * std::sin(pitch_rad);
    while (time_step * dt <= max_flight_time && pos_y >= min_height) {
        // 更新位置
        pos_x += vel_x * dt;
        pos_y += vel_y * dt;

        if (vel_x < 0.1) {
            break;
        }
        if (pos_x >= horizontal_distance) {
            result.hit_height = pos_y - (pos_x - horizontal_distance) * (vel_y / vel_x);
            result.valid = true;
            break;
        }

        // 计算当前速度大小
        double vel = std::sqrt(vel_x * vel_x + vel_y * vel_y);
        // 计算空气阻力（与速度平方成正比，方向与速度相反）
        double drag_force = 0.5 * ballisticParams.drag_coeff * ballisticParams.air_density * 
                           cross_section_area * vel * vel;
        // 计算阻力加速度分量
        double drag_accel_x = drag_force * vel_x / (ballisticParams.bullet_mass * vel);
        double drag_accel_y = drag_force * vel_y / (ballisticParams.bullet_mass * vel);
        // 更新速度（考虑空气阻力和重力）
        vel_x -= drag_accel_x * dt;
        vel_y -= (g + drag_accel_y) * dt;

        time_step += 1;
    }

    return result;
}

cv::Point3d BallisticSolver::calcNearestPointWithAirResistance(cv::Point3d target_pos, cv::Point3d self_pos, cv::Point2d aim_yaw_pitch, float v_bullet) {
    double max_flight_time = 5.0f;
    double dt = 1e-3;
    double min_height = -100.0f;

    double g = 9.8;
    uint32_t time_step = 0;

    cv::Point3d nearest_point = self_pos;
    double min_target_dist = cv::norm(target_pos - self_pos);

    // 计算弹丸截面积
    double bullet_radius = ballisticParams.bullet_diameter / 2.0;
    double cross_section_area = M_PI * bullet_radius * bullet_radius;

    cv::Point3d bullet_pos = self_pos;
    cv::Point3d bullet_vel;
    bullet_vel.x = v_bullet * std::cos(aim_yaw_pitch.y) * (-std::sin(aim_yaw_pitch.x));
    bullet_vel.y = v_bullet * std::cos(aim_yaw_pitch.y) * std::cos(aim_yaw_pitch.x);
    bullet_vel.z = v_bullet * std::sin(aim_yaw_pitch.y);
    while (time_step * dt <= max_flight_time && bullet_pos.z >= min_height) {
        // 更新位置
        bullet_pos += bullet_vel * dt;

        double target_dist = cv::norm(target_pos - bullet_pos);
        if (target_dist < min_target_dist) {
            nearest_point = bullet_pos;
            min_target_dist = target_dist;
        }

        // 计算当前速度大小
        double vel = cv::norm(bullet_vel);
        if (vel < 0.1) {
            break;
        }

        // 计算空气阻力（与速度平方成正比，方向与速度相反）
        double drag_force = 0.5 * ballisticParams.drag_coeff * ballisticParams.air_density * 
                           cross_section_area * vel * vel;
        // 计算阻力加速度分量
        double drag_accel = drag_force / ballisticParams.bullet_mass;
        cv::Point3d drag_accel_xyz = drag_accel * bullet_vel / vel;
        // 更新速度（考虑空气阻力和重力）
        bullet_vel -= drag_accel_xyz * dt;
        bullet_vel.z -= g * dt;

        time_step += 1;
    }

    return nearest_point;
}

// 使用该方法可能无法算出较高的弹道
BallisticSolver::CalcPitchInfo BallisticSolver::calcTargetPitchWithAirResistance(
    float horizontal_distance, float vertical_height, float v_bullet) {
    
    CalcPitchInfo result;
    result.valid = false;

    float min_pitch = -60 * M_PI / 180;
    float mid_pitch = 80 * M_PI / 180;
    float max_pitch = 89 * M_PI / 180;
    int start_check_n_low = 48;
    int start_check_n_high = 16;
    int max_refine_times = 10;
    float tolerance = 1e-3;

    double max_flight_time = 5.0f;
    double dt = 1e-3;
    double min_height = -100.0f;

    // 初始计算n_low+n_high个点的[pitch-目标距离处y]对应关系
    // 角度较小时（min_pitch ~ mid_pitch）使用角度均匀分布
    // 角度较大时（mid_pitch ~ max_pitch）使用正切值均匀分布
    float mid_aimed_h_at_1 = std::tan(mid_pitch);
    float max_aimed_h_at_1 = std::tan(max_pitch);
    std::vector<TrajectoryInfo> start_check_results(start_check_n_low + start_check_n_high);
    for (int start_check_index = 0; start_check_index < start_check_n_low + start_check_n_high; start_check_index += 1) {
        float pitch_rad = 0.0;
        if (start_check_index < start_check_n_low) {
            pitch_rad = min_pitch + (mid_pitch - min_pitch) * (static_cast<float>(start_check_index) / static_cast<float>(start_check_n_low));
        } else {
            float aimed_h_at_1 = mid_aimed_h_at_1 + 
                                (max_aimed_h_at_1 - mid_aimed_h_at_1) * 
                                (static_cast<float>(start_check_index - start_check_n_low) / 
                                (static_cast<float>(start_check_n_high) - 1.0));
            pitch_rad = std::atan(aimed_h_at_1);
        }
        SimulateTrajectoryInfo simulate_result = simulateTrajectory(v_bullet, pitch_rad, horizontal_distance,
                                                                    max_flight_time, dt, min_height);
        start_check_results[start_check_index].pitch = pitch_rad;
        if (simulate_result.valid) {
            start_check_results[start_check_index].hit_height = simulate_result.hit_height;
        } else {
            if (start_check_index == 0) {
                start_check_results[start_check_index].hit_height = min_height;
            } else {
                start_check_results[start_check_index].hit_height = start_check_results[start_check_index - 1].hit_height;
            }
        }
    }
    /* for (TrajectoryInfo& trajectory_info : start_check_results) {
        RCLCPP_INFO(node->get_logger(), "%f ,%f", trajectory_info.pitch, trajectory_info.hit_height);
    } */
    // 查找所有与目标高度差距符号转变的位置
    std::vector<RefineInfo> refine_infos;
    for (int start_check_index = 1; start_check_index < start_check_n_low + start_check_n_high; start_check_index += 1) {
        TrajectoryInfo trajectory_info1 = start_check_results[start_check_index-1];
        TrajectoryInfo trajectory_info2 = start_check_results[start_check_index];
        if ((trajectory_info1.hit_height < vertical_height) && (trajectory_info2.hit_height >= vertical_height)) {
            RefineInfo refine_info;
            refine_info.lower_pitch_trajectory = trajectory_info1;
            refine_info.upper_pitch_trajectory = trajectory_info2;
            refine_info.rising = true;
            refine_info.valid = true;
            refine_infos.push_back(refine_info);
        } else if ((trajectory_info1.hit_height >= vertical_height) && (trajectory_info2.hit_height < vertical_height)) {
            RefineInfo refine_info;
            refine_info.lower_pitch_trajectory = trajectory_info1;
            refine_info.upper_pitch_trajectory = trajectory_info2;
            refine_info.rising = false;
            refine_info.valid = true;
            refine_infos.push_back(refine_info);
        }
    }
    if (refine_infos.size() == 0) {
        return result;
    }
    // 二分查找符号转变点的准确pitch值
    std::vector<float> optional_pitchs;
    for (const RefineInfo& refine_info : refine_infos) {
        int refine_step = 0;
        float pitch_lower = refine_info.lower_pitch_trajectory.pitch;
        float pitch_upper = refine_info.upper_pitch_trajectory.pitch;
        float pitch_mid;
        while (refine_step < max_refine_times) {
            pitch_mid = (pitch_lower + pitch_upper) / 2;
            SimulateTrajectoryInfo simulate_result = simulateTrajectory(v_bullet, pitch_mid, horizontal_distance,
                                                                        max_flight_time, dt, min_height);
            if (!simulate_result.valid) {
                break;
            }
            float mid_pitch_hit_height = simulate_result.hit_height;
            if (std::abs(vertical_height - mid_pitch_hit_height) <= tolerance) {
                break;
            }
            bool in_first_half = (mid_pitch_hit_height < vertical_height) ^ refine_info.rising;
            if (in_first_half) {
                pitch_upper = pitch_mid;
            } else {
                pitch_lower = pitch_mid;
            }
            refine_step += 1;
        }
        optional_pitchs.push_back(pitch_mid);
    }
    result.target_pitch_result_smaller = *std::min_element(optional_pitchs.begin(), optional_pitchs.end());
    result.target_pitch_result_larger = *std::max_element(optional_pitchs.begin(), optional_pitchs.end());
    result.valid = true;
    return result;
}

BallisticInfo BallisticSolver::calcBallisticAngle(float x_camera, float y_camera, float z_camera, float deltax_camera, float deltay_camera, float deltaz_camera, 
                                  float v_bullet, float cur_pitch, float cur_yaw) {
    BallisticInfo result;
    result.valid = false;
    
    // 转换单位：mm到m
    x_camera = (x_camera + deltax_camera) / 1000.0f; // 向右
    y_camera = (y_camera + deltay_camera) / 1000.0f; // 向下
    z_camera = (z_camera + deltaz_camera) / 1000.0f; // 向前

    // 转换为水平坐标系
    float x_standard = x_camera;                                               // 向右
    float y_standard = z_camera*sin(cur_pitch) - y_camera*cos(cur_pitch);       // 向上
    float z_standard = z_camera*cos(cur_pitch) + y_camera*sin(cur_pitch);       // 向前
    float r_standard = sqrt(x_standard*x_standard + z_standard*z_standard);

    // 1. 计算目标yaw弧度
    float target_delta_yaw = -atan2(x_standard, z_standard) * 1.0;  // 速度补偿 TODO
    float target_yaw = normalizeRad(target_delta_yaw + cur_yaw);  // 标准化到[-M_PI, M_PI]
    
    // 4. 求解弹道方程
    //CalcPitchInfo pitch_info = calcTargetPitch(r_standard, y_standard, v_bullet);
    CalcPitchInfo pitch_info = calcTargetPitchWithAirResistance(r_standard, y_standard, v_bullet);
    if (!pitch_info.valid) {
        return result;
    }
    float final_pitch_rad = pitch_info.target_pitch_result_smaller;

    //RCLCPP_INFO(node->get_logger(), "pitch_origin:%.5f, %.5f", pitch_info.target_pitch_result_smaller, pitch_info.target_pitch_result_larger);

    //CalcPitchInfo pitch_info_with_air_resistance = calcTargetPitchWithAirResistance(r_standard, y_standard, v_bullet);
    //RCLCPP_INFO(node->get_logger(), "pitch_new:%.5f, %.5f", pitch_info_with_air_resistance.target_pitch_result_smaller, pitch_info_with_air_resistance.target_pitch_result_larger);
    
    //final_pitch_rad += 1.0 * M_PI / 180.0f;  // 角度补偿 TODO

    // 5. 计算需要转动的角度
    result.delta_pitch_rad = final_pitch_rad - cur_pitch;
    result.target_yaw_rad = target_yaw;
    
    result.valid = true;
    return result;
}
