// YawVisualizer.h
#ifndef YAW_VISUALIZER_H
#define YAW_VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
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

public:
    // 构造函数
    YawVisualizer();

    // 添加数据点
    void update(float current_yaw, float target_yaw);

    void show();
};


#endif
