// ArmorClassifier.h
#ifndef ARMOR_CLASSIFIER_H
#define ARMOR_CLASSIFIER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <map>
#include "model.h"
#include "Armor.h"
#include <filesystem>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include "test_codes/UnwarpUtils.h"
#include "test_codes/model_rm2026.h"
#include "test_codes/PositionPredictor.h"


class ArmorClassifier {
public:
    ArmorClassifier(std::shared_ptr<YAML::Node> config_file_ptr, bool use_cuda, rclcpp::Node* node);
    std::vector<ArmorResult> classify(const cv::Mat& img, const std::vector<Armor>& armors);

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

        TrackedArmor(int number, std::chrono::steady_clock::time_point seen_time, cv::Point2f center, 
            Armor armor, float confidence, bool is_large, bool not_slant, int fit_step) : 
        number(number), tracking_count(1), last_seen(seen_time), center_last_seen(center), is_steady_tracked(false),
        is_tracked_now(true), armor_last_seen(armor), confidence(confidence), is_large(is_large), not_slant(not_slant),
        predictor(fit_step), prediction_index(0) {
            predictor.addPoint(center);
        }
    };

    std::shared_ptr<TransistorRM2026Net> model;
    torch::Device device;
    std::vector<TrackedArmor> tracked_armors;
    rclcpp::Node* node;                  // 用于打印的节点
    
    float IS_ARMOR_THRESHOLD;
    float IS_LARGE_THRESHOLD;
    float NOT_SCREEN_THRESHOLD;
    float NOT_SLANT_THRESHOLD;
    float CLASSIFY_THRESHOLD;
    int INPUT_HEIGHT;
    int INPUT_WIDTH;
    int MAX_TRACKING_AGE_MS;
    int MIN_TRACKING_COUNT;
    float IS_NEAR_MAX_DIST;
    int fit_step;
    int predict_step;
    
    cv::Mat preprocessROI(const cv::Mat& img, const Armor& roi);
    bool isNearPreviousCenter(const cv::Point2f& current, const cv::Point2f& previous, float max_dist = -1.0);
};

#endif // ARMOR_CLASSIFIER_H

