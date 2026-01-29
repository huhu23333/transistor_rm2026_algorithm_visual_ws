// DataVisualizer.cpp
#include "visualizer/DataVisualizer.h"

// 构造函数
Oscilloscope::Oscilloscope(int w, int h, 
                           const std::string& name,
                           size_t init_layer_num,
                           cv::Scalar bg_color,
                           cv::Scalar wf_color)
    : width(w), height(h), scale(1.0f), offset(0.0f), 
      window_name(name), background_color(bg_color),
      layer_num(init_layer_num)  // 修正：初始化layer_num为init_layer_num
{
    // 初始化显示图像
    data_displays.resize(layer_num);
    datas.resize(layer_num);
    waveform_colors.resize(layer_num);  // 初始化waveform_colors的大小

    for (size_t layer_index = 0; layer_index < layer_num; layer_index += 1) {
        cv::Mat& data_display = data_displays[layer_index];
        data_display = cv::Mat::zeros(height, width, CV_8UC3);
        data_display.setTo(cv::Scalar(0, 0, 0));  // 每个图层的背景为黑色

        waveform_colors[layer_index] = wf_color;  // 初始化每个图层的颜色为传入的wf_color
    }

    // 初始化display为背景色
    display = cv::Mat::zeros(height, width, CV_8UC3);
    display.setTo(background_color);
}

// 添加数据点
void Oscilloscope::addDataPoint(float value, size_t layer_index) {
    std::deque<float>& data = datas[layer_index];
    data.push_back(value);
    
    // 如果数据点超过窗口宽度，删除最旧的数据
    if (data.size() > width) {
        data.pop_front();
    }
}

// 更新显示
void Oscilloscope::update() {
    display.setTo(background_color);
    for (size_t layer_index = 0; layer_index < layer_num; layer_index += 1) {
        cv::Mat& data_display = data_displays[layer_index];
        std::deque<float>& data = datas[layer_index];
        cv::Scalar& waveform_color = waveform_colors[layer_index];

        // 将显示图像向左滚动n个像素
        cv::Mat rolled = data_display(cv::Rect(rolling_speed, 0, width - rolling_speed, height));
        rolled.copyTo(data_display(cv::Rect(0, 0, width - rolling_speed, height)));
        
        // 清除最右侧的列
        for (uint32_t i = 0; i < rolling_speed; i++) {
            data_display.col(width - 1 - i).setTo(cv::Scalar(0, 0, 0));
        }
        
        // 如果有数据点，绘制最新的数据点
        if (!data.empty()) {
            float latest_value = data.back();
            
            // 计算归一化坐标 (0到1范围)
            float normalized = (latest_value * scale + offset + 1.0f) / 2.0f;
            
            // 计算像素坐标
            int y = height - static_cast<int>(normalized * height);
            y = std::max(0, std::min(height - 1, y));
            
            // 绘制最新的数据点
            cv::circle(data_display, cv::Point(width - 1, y), 1, waveform_color, -1);
            
            // 如果数据点足够多，绘制连线
            if (data.size() > 1) {
                float prev_value = data[data.size() - 2];
                float prev_normalized = (prev_value * scale + offset + 1.0f) / 2.0f;
                int prev_y = height - static_cast<int>(prev_normalized * height);
                prev_y = std::max(0, std::min(height - 1, prev_y));
                
                cv::line(data_display, cv::Point(width - 1 - rolling_speed, prev_y), 
                        cv::Point(width - 1, y), waveform_color, 1);
            }
        }
        cv::add(display, data_display, display);
    }
}

// 显示窗口
void Oscilloscope::show() {
#ifdef SHOW_WINDOWS
    cv::imshow(window_name, display);
    cv::waitKey(1);
#endif
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
void Oscilloscope::clear_all() {
    for (size_t layer_index = 0; layer_index < layer_num; layer_index += 1) {
        cv::Mat& data_display = data_displays[layer_index];
        std::deque<float>& data = datas[layer_index];

        data.clear();
        data_display.setTo(cv::Scalar(0, 0, 0));
    }
}

void Oscilloscope::putText(
    const std::string& text,
    cv::Point org,
    cv::Scalar color,
    double fontScale,
    int thickness,
    int fontFace,
    int lineType,
    bool bottomLeftOrigin
) {
    cv::putText(display, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin);
}

cv::Mat Oscilloscope::getDisplay() {
    return display;
}

void Oscilloscope::setLayerColor(size_t layer_index, cv::Scalar color) {
    waveform_colors[layer_index] = color;
}

void Oscilloscope::setRollingSpeed(uint32_t speed) {
    rolling_speed = speed;
}
