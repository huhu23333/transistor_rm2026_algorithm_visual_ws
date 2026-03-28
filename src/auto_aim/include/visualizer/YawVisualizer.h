// YawVisualizer.h
#ifndef YAW_VISUALIZER_H
#define YAW_VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
#include <numeric>
#include "macro/AutoAimMacro.h"
#include "visualizer/DataVisualizer.h"

class YawVisualizer {
private:
    std::shared_ptr<Oscilloscope> yaw_oscilloscope;

    float last_current_yaw = 0.0;
    float last_target_yaw = 0.0;

    int current_yaw_circle = 0;
    int target_yaw_circle = 0;

    cv::Mat display;

    std::vector<float> total_current_yaw_history;
    std::vector<float> total_target_yaw_history;

    std::chrono::steady_clock::time_point last_total_target_mid_time;
    float last_total_target_mid_value;
    int max_delay = 1000;
    std::vector<float> delay_history;

    bool raise_direction = false;

    void adjustScaleOffset();


    std::deque<float> total_current_yaw_history_for_adjust_scale_offset;
    std::deque<float> total_target_yaw_history_for_adjust_scale_offset;

public:
    // 构造函数
    YawVisualizer();

    // 添加数据点
    void update(float current_yaw, float target_yaw);

    void show();
    
    cv::Mat getDisplay();
};


#endif
