#include "controller/TwoYawSentryController.h"


TwoYawSentryController::TwoYawSentryController(std::shared_ptr<YAML::Node> config_file_ptr) {
    reset_behavior_infos.v_yaw = M_PI / 180.0 * (*config_file_ptr)["reset_vyaw"].as<float>();
    reset_behavior_infos.v_pitch = M_PI / 180.0 * (*config_file_ptr)["reset_vpitch"].as<float>();
    reset_behavior_infos.min_pitch = M_PI / 180.0 * (*config_file_ptr)["reset_min_pitch"].as<float>();
    reset_behavior_infos.max_pitch = M_PI / 180.0 * (*config_file_ptr)["reset_max_pitch"].as<float>();
    reset_behavior_infos.last_time = std::chrono::steady_clock::now();
}

void TwoYawSentryController::update_gimbal_infos(float real_pitch, float real_small_yaw, float real_big_yaw) {
    
}


TwoYawGimbalControll_t TwoYawSentryController::step(bool reset, float pitch_target, float yaw_target) {

}