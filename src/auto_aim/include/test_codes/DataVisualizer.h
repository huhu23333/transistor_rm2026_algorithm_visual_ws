// DataVisualizer.h
#ifndef DATAVISUALIZER_H
#define DATAVISUALIZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <mutex>
#include <string>
#include <algorithm>

class Oscilloscope {
private:
    cv::Mat data_display;          // 数据图像
    cv::Mat display;          // 显示图像
    std::deque<float> data;   // 存储数据点的双端队列
    int width;                // 显示窗口宽度
    int height;               // 显示窗口高度
    float scale;              // 垂直缩放因子
    float offset;             // 垂直偏移
    std::mutex data_mutex;    // 数据访问互斥锁
    std::string window_name;  // 窗口名称
    cv::Scalar background_color; // 背景颜色
    cv::Scalar waveform_color;   // 波形颜色

public:
    // 构造函数
    Oscilloscope(int w = 800, int h = 400, 
                 const std::string& name = "Oscilloscope",
                 cv::Scalar bg_color = cv::Scalar(0, 0, 0),
                 cv::Scalar wf_color = cv::Scalar(0, 255, 0));

    // 添加数据点
    void addDataPoint(float value);
    
    // 更新显示
    void update();
    
    // 显示窗口
    void show();
    
    // 设置垂直缩放
    void setScale(float s);
    
    // 设置垂直偏移
    void setOffset(float o);
    
    // 清除所有数据
    void clear();
    
    // 获取当前数据点数量
    size_t getDataSize();

    // 绘制文字
    void putText(
        const std::string& text,
        cv::Point org,
        cv::Scalar color,
        double fontScale = 1.0,
        int thickness = 1,
        int fontFace = cv::FONT_HERSHEY_COMPLEX,
        int lineType = 8,
        bool bottomLeftOrigin = false
    );
};


#endif // DATAVISUALIZER_H