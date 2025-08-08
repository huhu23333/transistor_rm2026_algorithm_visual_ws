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
#include "test_codes/UnwarpUtils.h"
#include "test_codes/model_rm2026.h"


class ArmorClassifier {
public:
    ArmorClassifier(const std::string& model_path, bool use_cuda = false);
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

        TrackedArmor(int number, std::chrono::steady_clock::time_point seen_time, cv::Point2f center, 
            Armor armor, float confidence, bool is_large, bool not_slant) : 
        number(number), tracking_count(1), last_seen(seen_time), center_last_seen(center), is_steady_tracked(false),
        is_tracked_now(false), armor_last_seen(armor), confidence(confidence), is_large(is_large), not_slant(not_slant) {}
    };

    std::shared_ptr<TransistorRM2026Net> model;
    torch::Device device;
    std::vector<TrackedArmor> tracked_armors;
    
    static constexpr float IS_ARMOR_THRESHOLD = 0.5f;
    static constexpr float IS_LARGE_THRESHOLD = 0.5f;
    static constexpr float NOT_SCREEN_THRESHOLD = 0.5f;
    static constexpr float NOT_SLANT_THRESHOLD = 0.5f;
    static constexpr float CLASSIFY_THRESHOLD = 0.5f;
    static constexpr int INPUT_HEIGHT = 48;
    static constexpr int INPUT_WIDTH = 64;
    static constexpr int MAX_TRACKING_AGE_MS = 5000;  // 最大跟踪时间100ms // DEBUG // 100
    static constexpr int MIN_TRACKING_COUNT = 50;     // 最小连续跟踪次数           // 2
    
    cv::Mat preprocessROI(const cv::Mat& img, const Armor& roi);
    bool isNearPreviousCenter(const cv::Point2f& current, const cv::Point2f& previous, float max_dist = 50.0f);
};

#endif // ARMOR_CLASSIFIER_H

