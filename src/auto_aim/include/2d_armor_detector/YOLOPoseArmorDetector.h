// YOLOPoseArmorDetector.h
#ifndef YOLO_POSE_ARMOR_CLASSIFIER_H
#define YOLO_POSE_ARMOR_CLASSIFIER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "2d_armor_detector/Armor.h"
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#include "communication/SharedMemoryYOLOPose.h"
#define _USE_MATH_DEFINES // 启用数学常量
#include <cmath>

class YOLOPoseArmorDetector {
public:
    YOLOPoseArmorDetector(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node)
    : node(node), config_file_ptr(config_file_ptr) {
        shm_python_yolo_pose = std::make_shared<SharedMemoryYOLOPose>(config_file_ptr, node);
    }
    std::vector<Armor> detectArmors(const cv::Mat& frame, bool block);
private:
    std::shared_ptr<YAML::Node> config_file_ptr;
    rclcpp::Node* node;

    std::shared_ptr<SharedMemoryYOLOPose> shm_python_yolo_pose;
};


#endif // YOLO_POSE_ARMOR_CLASSIFIER_H
