// ArmorClassifier.h
#ifndef ARMOR_CLASSIFIER_H
#define ARMOR_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <map>
//#include "model.h"
#include "2d_armor_detector/Armor.h"
#include <filesystem>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include "2d_armor_detector/UnwarpUtils.h"
//#include "test_codes/model_rm2026.h"
#include "predictor/PositionPredictor2D.h"
#include <iostream>
#include <sstream>
#include <string>
#include "communication/SharedMemoryClassifier.h"
#include <algorithm>
#include <execution>
#include <thread>
#include <atomic>
#include "2d_armor_detector/ArmorTracker.h"
#include "macro/AutoAimMacro.h"

namespace fs = std::filesystem;

class ArmorClassifier {
public:
    ArmorClassifier(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node, fs::path ws_dir_path);
    std::vector<ArmorResult> classify(const cv::Mat& img, const std::vector<Armor>& armors, const cv::Point2f& ground_stable_point);

private:
    rclcpp::Node* node;                  // 用于打印的节点
    std::shared_ptr<SharedMemoryClassifier> shm_python_classifier;
    std::shared_ptr<ArmorTracker> armor_tracker;
    
    int MAX_ROI_SAVE_COUNT;  // 最大保存数量
    std::atomic<int> roi_save_count = 0;

    float IS_ARMOR_THRESHOLD;
    float IS_LARGE_THRESHOLD;
    float CLASSIFY_THRESHOLD;
    int INPUT_HEIGHT;
    int INPUT_WIDTH;

    fs::path ws_dir_path; // 用于保存图片路径
    
    cv::Mat preprocessROI(const cv::Mat& img, const Armor& roi);
};

#endif // ARMOR_CLASSIFIER_H

