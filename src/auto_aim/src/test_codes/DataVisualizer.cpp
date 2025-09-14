// DataVisualizer.cpp
#include "test_codes/DataVisualizer.h"

// 构造函数
Oscilloscope::Oscilloscope(int w, int h, 
                           const std::string& name,
                           cv::Scalar bg_color,
                           cv::Scalar wf_color)
    : width(w), height(h), scale(1.0f), offset(0.0f), 
      window_name(name), background_color(bg_color), waveform_color(wf_color) {
    // 初始化显示图像
    display = cv::Mat::zeros(height, width, CV_8UC3);
    display.setTo(background_color);
}

// 添加数据点
void Oscilloscope::addDataPoint(float value) {
    std::lock_guard<std::mutex> lock(data_mutex);
    data.push_back(value);
    
    // 如果数据点超过窗口宽度，删除最旧的数据
    if (data.size() > width) {
        data.pop_front();
    }
}

// 更新显示
void Oscilloscope::update() {
    std::lock_guard<std::mutex> lock(data_mutex);
    
    // 将显示图像向左滚动一个像素
    cv::Mat rolled = display(cv::Rect(1, 0, width - 1, height));
    rolled.copyTo(display(cv::Rect(0, 0, width - 1, height)));
    
    // 清除最右侧的列
    display.col(width - 1).setTo(background_color);
    
    // 如果有数据点，绘制最新的数据点
    if (!data.empty()) {
        float latest_value = data.back();
        
        // 计算归一化坐标 (0到1范围)
        float normalized = (latest_value * scale + offset + 1.0f) / 2.0f;
        
        // 计算像素坐标
        int y = height - static_cast<int>(normalized * height);
        y = std::max(0, std::min(height - 1, y));
        
        // 绘制最新的数据点
        cv::circle(display, cv::Point(width - 1, y), 1, waveform_color, -1);
        
        // 如果数据点足够多，绘制连线
        if (data.size() > 1) {
            float prev_value = data[data.size() - 2];
            float prev_normalized = (prev_value * scale + offset + 1.0f) / 2.0f;
            int prev_y = height - static_cast<int>(prev_normalized * height);
            prev_y = std::max(0, std::min(height - 1, prev_y));
            
            cv::line(display, cv::Point(width - 2, prev_y), 
                     cv::Point(width - 1, y), waveform_color, 1);
        }
    }
}

// 显示窗口
void Oscilloscope::show() {
    cv::imshow(window_name, display);
    cv::waitKey(1);
}

// 设置垂直缩放
void Oscilloscope::setScale(float s) {
    scale = s;
}

// 设置垂直偏移
void Oscilloscope::setOffset(float o) {
    offset = o;
}

// 清除所有数据
void Oscilloscope::clear() {
    std::lock_guard<std::mutex> lock(data_mutex);
    data.clear();
    display.setTo(background_color);
}

// 获取当前数据点数量
size_t Oscilloscope::getDataSize() {
    std::lock_guard<std::mutex> lock(data_mutex);
    return data.size();
}