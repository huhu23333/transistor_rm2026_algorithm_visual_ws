// TwoYawSentryController.h
#ifndef TWO_YAW_SENTRY_CONTROLLER_H
#define TWO_YAW_SENTRY_CONTROLLER_H

#include <memory>
#include <vector>
#include <algorithm>
#include <chrono>
#include <yaml-cpp/yaml.h>
#include "macro/AutoAimMacro.h"

struct TwoYawGimbalControll_t {
    float target_pitch;
    float target_yaw_small;
    float target_yaw_big;
};

class TwoYawSentryController {
private:

    struct {
        float v_yaw;
        float v_pitch;
        float max_pitch;
        float min_pitch;

        std::chrono::steady_clock::time_point last_time;
    } reset_behavior_infos;


public:
    TwoYawSentryController(std::shared_ptr<YAML::Node> config_file_ptr);

    void update_gimbal_infos(float real_pitch, float real_small_yaw, float real_big_yaw);

    TwoYawGimbalControll_t step(bool reset, float pitch_target, float yaw_target);
};


#endif



