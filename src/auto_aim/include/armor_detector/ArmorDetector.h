// ArmorDetector.h
#ifndef ARMOR_DETECTOR_H
#define ARMOR_DETECTOR_H
#pragma once
#include <opencv2/opencv.hpp>
#include "LightBar.h"
#include "Armor.h"
#include <vector>
#include <yaml-cpp/yaml.h>

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
        corners_expand_ratio = (*config_file_ptr)["corners_expand_ratio"].as<float>();
        
        // 初始化相机参数
        initCameraMatrix(config_file_ptr, node);
        initArmorPoints();
    }
    // 新增3D到像素坐标投影函数
    cv::Point2f project3DToPixel(const cv::Point3f& world_point) const;
    std::vector<Armor> detectArmors(const std::vector<Light>& lights);
    AimResult solveArmor(const ArmorResult& armor_result) const; // 增加number参数
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
    float corners_expand_ratio;
    rclcpp::Node* node;
};

#endif // ARMOR_DETECTOR_H