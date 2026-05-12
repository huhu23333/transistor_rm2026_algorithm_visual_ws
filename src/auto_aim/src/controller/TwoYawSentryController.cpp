#include "controller/TwoYawSentryController.h"


TwoYawSentryController::TwoYawSentryController(std::shared_ptr<YAML::Node> config_file_ptr) {
    reset_behavior_infos.v_yaw = M_PI / 180.0 * (*config_file_ptr)["reset_vyaw"].as<float>();
    reset_behavior_infos.v_pitch = M_PI / 180.0 * (*config_file_ptr)["reset_vpitch"].as<float>();
    reset_behavior_infos.min_pitch = M_PI / 180.0 * (*config_file_ptr)["reset_min_pitch"].as<float>();
    reset_behavior_infos.max_pitch = M_PI / 180.0 * (*config_file_ptr)["reset_max_pitch"].as<float>();
    reset_behavior_infos.reset_max_big_yaw_gap = M_PI / 180.0 * (*config_file_ptr)["reset_max_big_yaw_gap"].as<float>();

    reset_ceasefire_ms = (*config_file_ptr)["reset_ceasefire_ms"].as<float>();

    last_step_time = std::chrono::steady_clock::now();

    big_yaw_smooth_factor = (*config_file_ptr)["big_yaw_smooth_factor"].as<float>();
    small_yaw_to_big_boundary = M_PI / 180.0 * (*config_file_ptr)["small_yaw_to_big_boundary"].as<float>();

    smallyaw_error_d_filter = std::make_shared<SimpleDataFilter>(3);
}

void TwoYawSentryController::update_gimbal_infos(float real_pitch_, float real_small_yaw_, float real_big_yaw_) {
    real_pitch = real_pitch_;
    real_small_yaw = real_small_yaw_;
    real_big_yaw = real_big_yaw_;
}


TwoYawGimbalControll_t TwoYawSentryController::step(bool reset, float pitch_target, float yaw_target, bool fire_flag) {
    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
    float dt = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_step_time).count()) / 1e6;

    TwoYawGimbalControll_t result;
    result.fire_flag = false;
    float smallyaw_error = 0.0;
    if (reset) {
        if (!last_reset_state) {
            last_pitch_target = real_pitch;
            last_big_yaw_target = real_big_yaw;
            smallyaw_error_d_filter -> clearHistory();
        }

        result.target_pitch = last_pitch_target + reset_behavior_infos.v_pitch * dt * reset_behavior_infos.v_pitch_direction;
        result.target_yaw_small = 0.0;
        result.target_yaw_big = last_big_yaw_target + reset_behavior_infos.v_yaw * dt;

        if (result.target_pitch > reset_behavior_infos.max_pitch) {
            result.target_pitch = reset_behavior_infos.max_pitch;
            reset_behavior_infos.v_pitch_direction = -1.0;
        } else if (result.target_pitch < reset_behavior_infos.min_pitch) {
            result.target_pitch = reset_behavior_infos.min_pitch;
            reset_behavior_infos.v_pitch_direction = 1.0;
        }

        float big_yaw_gap = result.target_yaw_big - real_big_yaw;
        big_yaw_gap = std::atan2(std::sin(big_yaw_gap), std::cos(big_yaw_gap));
        if (fabs(big_yaw_gap) > reset_behavior_infos.reset_max_big_yaw_gap) {
            if (big_yaw_gap > 0.0) {
                result.target_yaw_big = real_big_yaw + reset_behavior_infos.reset_max_big_yaw_gap;
            } else {
                result.target_yaw_big = real_big_yaw - reset_behavior_infos.reset_max_big_yaw_gap;
            }
        }
        last_reset_time = current_time;

    } else {
        if (last_reset_state) {
            last_big_yaw_target = yaw_target;
        }

        result.target_pitch = pitch_target;
        result.target_yaw_big = big_yaw_smooth_factor * yaw_target + (1.0 - big_yaw_smooth_factor) * last_big_yaw_target;

        float target_yaw_small = yaw_target - real_big_yaw;
        smallyaw_error = target_yaw_small - real_small_yaw;
        float smallyaw_error_d = (smallyaw_error - last_smallyaw_error) / std::max(dt, 1e-3f);
        smallyaw_error_d_filter -> addPoint(smallyaw_error_d);
        float filtered_smallyaw_error_d = smallyaw_error_d_filter -> meanFilter(3);
        result.target_yaw_small = real_small_yaw + smallyaw_error * 1.0; // + filtered_smallyaw_error_d * 0.1;

        if (fabs(result.target_yaw_small) > small_yaw_to_big_boundary) {
            result.target_yaw_big = yaw_target;
        }

        if (static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_reset_time).count()) / 1e3 > reset_ceasefire_ms) {
            result.fire_flag = fire_flag;
        }
    }


    last_pitch_target = result.target_pitch;
    last_small_yaw_target = result.target_yaw_small;
    last_big_yaw_target = result.target_yaw_big;
    last_reset_state = reset;
    last_step_time = current_time;

    last_smallyaw_error = smallyaw_error;

    return result;
}