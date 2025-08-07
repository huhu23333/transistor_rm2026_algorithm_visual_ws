// ArmorClassifier.h
#ifndef ARMOR_CLASSIFIER_H
#define ARMOR_CLASSIFIER_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <map>
#include "model.h"
#include "ArmorDetector.h"

#include "test_codes/model_rm2026.h" //debug_newModel

struct ArmorResult {
    Armor armor;              
    int number;              
    float confidence;        
    std::vector<cv::Point2f> corners;  

    ArmorResult(const Armor& a, int n, float conf) 
        : armor(a), number(n), confidence(conf), corners(a.corners) {}
};

class ArmorClassifier {
public:
    ArmorClassifier(const std::string& model_path, bool use_cuda = false);
    std::vector<ArmorResult> classify(const cv::Mat& img, const std::vector<Armor>& armors);

private:
    struct TrackedArmor {
        int number;
        float confidence;
        std::chrono::steady_clock::time_point last_seen;
        int tracking_count;
        cv::Point2f center;

        TrackedArmor() : number(0), confidence(0), tracking_count(0) {}
    };

    // std::shared_ptr<NumberNet> model;
    std::shared_ptr<TransistorRM2026Net> model; //debug_newModel
    torch::Device device;
    std::map<int, TrackedArmor> tracked_armors;
    
    static constexpr float CONFIDENCE_THRESHOLD = 0.5f;
    static constexpr int INPUT_HEIGHT = 48;
    static constexpr int INPUT_WIDTH = 64;
    static constexpr int MAX_TRACKING_AGE_MS = 100;  // 最大跟踪时间100ms
    static constexpr int MIN_TRACKING_COUNT = 2;     // 最小连续跟踪次数
    
    cv::Mat preprocessROI(const cv::Mat& img, const cv::Rect& roi);
    bool isNearPreviousCenter(const cv::Point2f& current, const cv::Point2f& previous, float max_dist = 50.0f);
};

#endif // ARMOR_CLASSIFIER_H

