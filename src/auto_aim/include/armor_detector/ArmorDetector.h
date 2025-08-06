// ArmorDetector.h
#ifndef ARMOR_DETECTOR_H
#define ARMOR_DETECTOR_H
#pragma once
#include <opencv2/opencv.hpp>
#include "LightBarDetector.h"
#include <vector>
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>

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
    rclcpp::Node* node;

    // 默认构造函数
    Armor() : confidence(0.0f) {}
    
    // 带参数的构造函数
    Armor(const cv::RotatedRect& left, const cv::RotatedRect& right, rclcpp::Node* node) 
        : leftLight(left), rightLight(right), confidence(0.0f), node(node) {
        calculateROI();
    }
    
    // ROI计算函数
    void calculateROI() {
        // 更新corners向量
        corners.clear();
        calculateCorners();
        RCLCPP_DEBUG(node->get_logger(), "----------Armor Debug Flag----------");

        // 计算ROI
        roi = cv::boundingRect(corners);
    }

    cv::Point2f vecToPoint(const cv::Vec2f& vec) {
        return cv::Point2f(vec[0], vec[1]);
    }

    cv::Vec2f pointToVec(const cv::Point2f& point) {
        return cv::Vec2f(point.x, point.y);
    }
    
    void calculateCorners() {
        // 获取左右灯条的顶点
        cv::Point2f left_vertices[4], right_vertices[4];
        leftLight.points(left_vertices);
        rightLight.points(right_vertices);
        
        // 找到左右灯条的中心点
        cv::Point2f left_center = leftLight.center;
        cv::Point2f right_center = rightLight.center;

        // 获取左右灯条的顶点转换为相对中心点的坐标
        cv::Vec2f relative_left_vertices[4], relative_right_vertices[4];
        for (int i = 0; i < 4; i+=1) {
            relative_left_vertices[i] = pointToVec(left_vertices[i] - left_center);
            relative_right_vertices[i] = pointToVec(right_vertices[i] - right_center);
        }

        // 获取左灯条中心点指向右灯条中心点的向量及垂直其向上的向量，并单位化
        cv::Vec2f d_center_vector = right_center - left_center;
        cv::Vec2f vertical_d_center_vector = cv::Vec2f(-d_center_vector[1], d_center_vector[0]);
        d_center_vector = cv::normalize(d_center_vector);
        vertical_d_center_vector = cv::normalize(vertical_d_center_vector);

        // 获取左右灯条长边及短边的单位方向向量，并调整方向为向右或向上
        float rad_left = -leftLight.angle * M_PI / 180.0;
        float rad_right = -rightLight.angle * M_PI / 180.0;
        cv::Vec2f left_length_direction = cv::Vec2f(std::sin(rad_left), std::cos(rad_left));
        cv::Vec2f left_width_direction = cv::Vec2f(-left_length_direction[1], left_length_direction[0]);
        cv::Vec2f right_length_direction = cv::Vec2f(std::sin(rad_right), std::cos(rad_right));
        cv::Vec2f right_width_direction = cv::Vec2f(-right_length_direction[1], right_length_direction[0]);
        if (left_length_direction.dot(vertical_d_center_vector) < 0) left_length_direction = -left_length_direction;
        if (left_width_direction.dot(d_center_vector) < 0) left_width_direction = -left_width_direction;
        if (right_length_direction.dot(vertical_d_center_vector) < 0) right_length_direction = -right_length_direction;
        if (right_width_direction.dot(d_center_vector) < 0) right_width_direction = -right_width_direction;

        // 将顶点相对坐标拆分为沿长边和短边两部分
        cv::Vec2f horizontal_relative_left_vertices[4], horizontal_relative_right_vertices[4],
                  vertical_relative_left_vertices[4], vertical_relative_right_vertices[4];
        for (int i = 0; i < 4; i+=1) {
            horizontal_relative_left_vertices[i] = left_width_direction * relative_left_vertices[i].dot(left_width_direction);
            horizontal_relative_right_vertices[i] = right_width_direction * relative_right_vertices[i].dot(right_width_direction);
            vertical_relative_left_vertices[i] = left_length_direction * relative_left_vertices[i].dot(left_length_direction);
            vertical_relative_right_vertices[i] = right_length_direction * relative_right_vertices[i].dot(right_length_direction);
        }

        // 获取灯条顶点坐标中靠近装甲板中心的四个点的相对中心点坐标的沿长边和短边两部分
        cv::Vec2f left_up_horizontal, left_up_vertical, left_down_horizontal, left_down_vertical, 
                  right_up_horizontal, right_up_vertical, right_down_horizontal, right_down_vertical; 
        for (int i = 0; i < 4; i+=1) {
            if (horizontal_relative_left_vertices[i].dot(left_width_direction) >= 0)
            {
                if (vertical_relative_left_vertices[i].dot(left_length_direction) >= 0)
                {
                    left_up_horizontal = horizontal_relative_left_vertices[i];
                    left_up_vertical = vertical_relative_left_vertices[i];
                }
                else
                {
                    left_down_horizontal = horizontal_relative_left_vertices[i];
                    left_down_vertical = vertical_relative_left_vertices[i];
                }
            }
            if (horizontal_relative_right_vertices[i].dot(right_width_direction) <= 0)
            {
                if (vertical_relative_right_vertices[i].dot(right_length_direction) >= 0)
                {
                    right_up_horizontal = horizontal_relative_right_vertices[i];
                    right_up_vertical = vertical_relative_right_vertices[i];
                }
                else
                {
                    right_down_horizontal = horizontal_relative_right_vertices[i];
                    right_down_vertical = vertical_relative_right_vertices[i];
                }
            }
        }

        // 沿长边部分使用小装甲板比例获得装甲板高度相对坐标
        left_up_vertical *= ArmorConstants::SMALL_HEIGHT_RATIO;
        left_down_vertical *= ArmorConstants::SMALL_HEIGHT_RATIO;
        right_up_vertical *= ArmorConstants::SMALL_HEIGHT_RATIO;
        right_down_vertical *= ArmorConstants::SMALL_HEIGHT_RATIO;

        // 按从左上角开始逆时针排序输出
        corners.push_back(left_center + vecToPoint(left_up_horizontal + left_up_vertical));
        corners.push_back(left_center + vecToPoint(left_down_horizontal + left_down_vertical));
        corners.push_back(right_center + vecToPoint(right_down_horizontal + right_down_vertical));
        corners.push_back(right_center + vecToPoint(right_up_horizontal + right_up_vertical));
    }
};

class ArmorDetector {
public:
    ArmorDetector(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node)
    : node(node) {
        max_angle_diff = (*config_file_ptr)["max_angle_diff"].as<float>();
        max_height_diff_ratio = (*config_file_ptr)["max_height_diff_ratio"].as<float>();
        min_light_distance = (*config_file_ptr)["min_light_distance"].as<float>();
        max_light_distance = (*config_file_ptr)["max_light_distance"].as<float>();
        min_armor_confidence = (*config_file_ptr)["min_armor_confidence"].as<float>();
        max_expected_small_distance_mismatch_ratio = (*config_file_ptr)["max_expected_small_distance_mismatch_ratio"].as<float>();
        max_expected_large_distance_mismatch_ratio = (*config_file_ptr)["max_expected_large_distance_mismatch_ratio"].as<float>();
        
        // 初始化相机参数
        initCameraMatrix(config_file_ptr, node);
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
    float max_expected_small_distance_mismatch_ratio;
    float max_expected_large_distance_mismatch_ratio;
    // 相机参数
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    // 装甲板3D点(单位：mm)
    std::vector<cv::Point3f> armor_points_3d;
    bool isArmorPair(const Light& l1, const Light& l2);
    float getArmorConfidence(const Light& l1, const Light& l2);
    
    void initCameraMatrix(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node);
    void initArmorPoints();
    rclcpp::Node* node;
};

#endif // ARMOR_DETECTOR_H