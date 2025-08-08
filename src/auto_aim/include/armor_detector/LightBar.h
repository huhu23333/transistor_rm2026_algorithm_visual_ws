// LightBar.h
#ifndef LIGHTBAR_H
#define LIGHTBAR_H

#include <opencv2/opencv.hpp>

/**
 * @brief 灯条类，存储和处理单个灯条的信息
 */
class Light {
public:
    cv::RotatedRect el;      // 灯条的旋转矩形，包含中心点、大小和角度
    float length;            // 灯条的长度（较长边）
    float width;             // 灯条的宽度（较短边）
    float angle;             // 灯条的倾斜角度，范围[-90,90]
    cv::Point2f top;         // 灯条顶部点坐标
    cv::Point2f bottom;      // 灯条底部点坐标

    /**
     * @brief 构造函数，通过旋转矩形初始化灯条
     * @param rect 输入的旋转矩形
     */
    explicit Light(const cv::RotatedRect& rect);

    /**
     * @brief 计算灯条的所有几何参数
     * 包括长度、宽度、角度、顶部点和底部点
     */
    void calculateDimensions();

    /**
     * @brief 获取灯条的旋转矩形
     * @return 返回灯条的旋转矩形
     */
    cv::RotatedRect getRect() const { return el; }
};

#endif // LIGHTBAR_H
