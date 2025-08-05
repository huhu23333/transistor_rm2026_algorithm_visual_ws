// ArmorDetector.h
#ifndef ARMOR_DETECTOR_H
#define ARMOR_DETECTOR_H
#pragma once
#include <opencv2/opencv.hpp>
#include "LightBarDetector.h"
#include <vector>

// 物理尺寸常量
namespace ArmorConstants {
    constexpr float LIGHT_HEIGHT = 55.0f;       // 灯条高度
    // 小装甲板
    constexpr float SMALL_ARMOR_HEIGHT = 125.0f;  
    constexpr float SMALL_ARMOR_WIDTH = 135.0f;
    // 大装甲板
    constexpr float LARGE_ARMOR_HEIGHT = 127.0f;
    constexpr float LARGE_ARMOR_WIDTH = 230.0f;
    // 高度比例约为2.27和2.31
    constexpr float SMALL_HEIGHT_RATIO = SMALL_ARMOR_HEIGHT / LIGHT_HEIGHT;
    constexpr float LARGE_HEIGHT_RATIO = LARGE_ARMOR_HEIGHT / LIGHT_HEIGHT;
}


struct AimResult {
    cv::Point3f position;  // 装甲板中心在相机坐标系下的位置
    double distance;       // 距离
    bool valid;           // 解算是否有效
};


struct Armor {
    cv::RotatedRect leftLight;    // 左灯条
    cv::RotatedRect rightLight;   // 右灯条
    cv::Rect roi;                 // ROI区域
    float confidence;             // 置信度
    std::vector<cv::Point2f> corners;  // 四个角点坐标

    // 默认构造函数
    Armor() : confidence(0.0f) {}
    
    // 带参数的构造函数
    Armor(const cv::RotatedRect& left, const cv::RotatedRect& right) 
        : leftLight(left), rightLight(right), confidence(0.0f) {
        calculateROI();
    }
    
    // ROI计算函数
    void calculateROI() {
        // 获取左右灯条的顶点
        cv::Point2f left_vertices[4], right_vertices[4];
        leftLight.points(left_vertices);
        rightLight.points(right_vertices);
        
        // 找到左右灯条的中心点
        cv::Point2f left_center = leftLight.center;
        cv::Point2f right_center = rightLight.center;
        
        // 计算灯条的平均高度
        float avg_light_height = (leftLight.size.height + rightLight.size.height) / 2.0f;
        
        // 计算装甲板高度（使用小装甲板的高度比例作为默认值）
        // 在ROI阶段我们还不知道具体是大装甲板还是小装甲板，使用小装甲板比例可以确保ROI不会太大
        float armor_height = avg_light_height * ArmorConstants::SMALL_HEIGHT_RATIO;
        
        // 计算装甲板中心点
        cv::Point2f armor_center = (left_center + right_center) / 2.0f;
        
        // 找到两个灯条的内侧边界
        float left_right_x = -1280, right_left_x = 1280;
        for (int i = 0; i < 4; i++) {
            left_right_x = std::max(left_right_x, left_vertices[i].x);
            right_left_x = std::min(right_left_x, right_vertices[i].x);
        }
        
        // 计算装甲板的四个角点
        float half_height = armor_height / 2.0f;
        float y_top = armor_center.y - half_height;
        float y_bottom = armor_center.y + half_height;
        
        // 更新corners向量
        corners.clear();
        corners = {
            cv::Point2f(left_right_x, y_top),      // 左上
            cv::Point2f(right_left_x, y_top),      // 右上
            cv::Point2f(right_left_x, y_bottom),   // 右下
            cv::Point2f(left_right_x, y_bottom)    // 左下
        };
        
        // 计算ROI
        roi = cv::Rect(
            cv::Point(std::max(0, int(left_right_x)),
                     std::max(0, int(y_top))),
            cv::Size(
                std::min(1280 - int(left_right_x), int(right_left_x - left_right_x)),
                std::min(1024 - int(y_top), int(y_bottom - y_top)))
        );
    }
};

class ArmorDetector {
public:
    ArmorDetector() {
        max_angle_diff = 15.0f;        // 最大角度差
        max_height_diff_ratio = 0.5f;  // 最大高度差比例
        min_light_distance = 5.0f;     // 最小灯条距离
        max_light_distance = 50.0f;    // 最大灯条距离
        min_armor_confidence = 0.3f;   // 最小置信度阈值

        
        // 初始化相机参数
        initCameraMatrix();
        initArmorPoints();
    }
    // 新增3D到像素坐标投影函数
    cv::Point2f project3DToPixel(const cv::Point3f& world_point) const;
    std::vector<Armor> detectArmors(const std::vector<Light>& lights);
    AimResult solveArmor(const Armor& armor, int number) const; // 增加number参数
private:
    float max_angle_diff;
    float max_height_diff_ratio;
    float min_light_distance;
    float max_light_distance;
    float min_armor_confidence;
    // 相机参数
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    // 装甲板3D点(单位：mm)
    std::vector<cv::Point3f> armor_points_3d;
    bool isArmorPair(const Light& l1, const Light& l2);
    float getArmorConfidence(const Light& l1, const Light& l2);
    
    void initCameraMatrix();
    void initArmorPoints();
};

#endif // ARMOR_DETECTOR_H