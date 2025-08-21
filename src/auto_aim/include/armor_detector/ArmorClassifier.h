// ArmorClassifier.h
#ifndef ARMOR_CLASSIFIER_H
#define ARMOR_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <map>
//#include "model.h"
#include "armor_detector/Armor.h"
#include <filesystem>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include "utils/UnwarpUtils.h"
//#include "test_codes/model_rm2026.h"
#include "test_codes/PositionPredictor.h"
#include <iostream>
#include <sstream>
#include <string>
#include "utils/SharedMemoryTorch.h"
#include <algorithm>
#include <execution>
#include <thread>
#include <atomic>
#include "armor_detector/ArmorTracker.h"


class ArmorClassifier {
public:
    ArmorClassifier(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node);
    std::vector<std::vector<ArmorResult>> classify(const cv::Mat& img, const std::vector<Armor>& armors, const cv::Point2f& ground_stable_point);

private:
    rclcpp::Node* node;                  // 用于打印的节点
    std::shared_ptr<SharedMemoryTorch> shm_pytorch_processor;
    std::shared_ptr<ArmorTracker> armor_tracker;
    
    int MAX_ROI_SAVE_COUNT;  // 最大保存数量
    std::atomic<int> roi_save_count = 0;

    float IS_ARMOR_THRESHOLD;
    float IS_LARGE_THRESHOLD;
    float NOT_SCREEN_THRESHOLD;
    float NOT_SLANT_THRESHOLD;
    float CLASSIFY_THRESHOLD;
    int INPUT_HEIGHT;
    int INPUT_WIDTH;
    
    cv::Mat preprocessROI(const cv::Mat& img, const Armor& roi);
};

#endif // ARMOR_CLASSIFIER_H

