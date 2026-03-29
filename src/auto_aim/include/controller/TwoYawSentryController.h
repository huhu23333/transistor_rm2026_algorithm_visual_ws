// TwoYawSentryController.h
#ifndef TWO_YAW_SENTRY_CONTROLLER_H
#define TWO_YAW_SENTRY_CONTROLLER_H

#include <memory>
#include <vector>
#include <algorithm>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include "utils/SimpleDataFilter.h"
#include "macro/AutoAimMacro.h"

struct TwoYawGimbalControll_t {
    float target_pitch;
    float target_yaw_small;
    float target_yaw_big;
    float fire_flag;
};

class TwoYawSentryController {
private:
    float real_pitch = 0.0;
    float real_small_yaw = 0.0;
    float real_big_yaw = 0.0;

    struct {
        float v_yaw;
        float v_pitch;
        float max_pitch;
        float min_pitch;
        float reset_max_big_yaw_gap;

        float v_pitch_direction = 1.0f;
    } reset_behavior_infos;

    std::chrono::steady_clock::time_point last_step_time;

    float big_yaw_smooth_factor;
    float small_yaw_to_big_boundary;

    float last_pitch_target = 0.0;
    float last_big_yaw_target = 0.0;
    float last_small_yaw_target = 0.0;

    float last_smallyaw_error = 0.0;
    std::shared_ptr<SimpleDataFilter> smallyaw_error_d_filter;

    bool last_reset_state = true;
    std::chrono::steady_clock::time_point last_reset_time;
    float reset_ceasefire_ms;

public:
    TwoYawSentryController(std::shared_ptr<YAML::Node> config_file_ptr);

    void update_gimbal_infos(float real_pitch_, float real_small_yaw_, float real_big_yaw_);

    TwoYawGimbalControll_t step(bool reset, float pitch_target, float yaw_target, bool fire_flag);
};


#endif



