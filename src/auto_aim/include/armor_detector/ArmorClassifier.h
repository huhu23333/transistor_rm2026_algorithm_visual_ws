// ArmorClassifier.h
#ifndef ARMOR_CLASSIFIER_H
#define ARMOR_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <chrono>
#include <map>
//#include "model.h"
#include "Armor.h"
#include <filesystem>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include "test_codes/UnwarpUtils.h"
//#include "test_codes/model_rm2026.h"
#include "test_codes/PositionPredictor.h"
#include <iostream>
#include <sstream>
#include <string>
#include "test_codes/SharedMemoryTorch.h"
#include <algorithm>
#include <execution>
#include <thread>
#include <atomic>



class ArmorClassifier {
public:
    ArmorClassifier(std::shared_ptr<YAML::Node> config_file_ptr, bool use_cuda, rclcpp::Node* node);
    std::vector<std::vector<ArmorResult>> classify(const cv::Mat& img, const std::vector<Armor>& armors, const cv::Point2f ground_stable_point);

private:
    struct TrackedArmor {
        int number;
        int tracking_count;
        std::chrono::steady_clock::time_point last_seen;
        cv::Point2f center_last_seen;
        bool is_steady_tracked;
        bool is_tracked_now;
        Armor armor_last_seen;
        float confidence;
        bool is_large;
        bool not_slant;
        PositionPredictor2D predictor; 
        std::vector<cv::Point2f> predictions;
        cv::Point2f center_predicted;
        int prediction_index;
        cv::Point2f last_ground_stable_point;

        TrackedArmor(int number, std::chrono::steady_clock::time_point seen_time, cv::Point2f center, 
            Armor armor, float confidence, bool is_large, bool not_slant, int fit_step, cv::Point2f ground_stable_point) : 
        number(number), tracking_count(1), last_seen(seen_time), center_last_seen(center), is_steady_tracked(false),
        is_tracked_now(true), armor_last_seen(armor), confidence(confidence), is_large(is_large), not_slant(not_slant),
        predictor(fit_step), center_predicted(center), prediction_index(0), last_ground_stable_point(ground_stable_point) {
            predictor.addPoint(center);
        }
    };

    std::vector<TrackedArmor> tracked_armors;
    std::vector<TrackedArmor> classified_latest_tracked_armors;
    rclcpp::Node* node;                  // 用于打印的节点
    std::shared_ptr<SharedMemoryTorch> shm_pytorch_processor;
    
    int MAX_ROI_SAVE_COUNT;  // 最大保存数量
    std::atomic<int> roi_save_count = 0;
    
    int classify_classes;

    float IS_ARMOR_THRESHOLD;
    float IS_LARGE_THRESHOLD;
    float NOT_SCREEN_THRESHOLD;
    float NOT_SLANT_THRESHOLD;
    float CLASSIFY_THRESHOLD;
    int INPUT_HEIGHT;
    int INPUT_WIDTH;
    int MAX_TRACKING_AGE_MS;
    int MIN_TRACKING_COUNT;
    float IS_NEAR_MAX_DIST_RATIO;
    int fit_step;
    int predict_step;

    int fourier_fit_step;
    int fourier_fit_order;
    int fourier_predict_step;
    int MAX_FOURIER_TRACKING_AGE_MS;
    
    cv::Mat preprocessROI(const cv::Mat& img, const Armor& roi);
    bool isNearPreviousCenter(const Armor& current_armor, 
                                           const cv::Point2f& ground_stable_point,
                                           const TrackedArmor& previous_tracked_armor, 
                                           float max_dist_ratio = -1.0);
};

#endif // ARMOR_CLASSIFIER_H

