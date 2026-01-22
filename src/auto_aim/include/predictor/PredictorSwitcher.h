#ifndef PREDICTOR_SWITCHER_H
#define PREDICTOR_SWITCHER_H
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <deque>

#include "utils/DataProcessFuncs.h"
#include "3d_processing/RestFrame.h"
#include "predictor/RotationJudge.h"

namespace PredictorType {
    enum PredictorType {
        None = 0,   // 直接瞄准装甲板
        EKF,
        FirePredictor,
        RotationMotionModel,
        AutoSwitch
    };

    extern std::vector<std::string> PredictorTypeStrings;
}

class PredictorSwitcher {
public:
    PredictorSwitcher(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node, int check_frames, std::shared_ptr<RestFrame> rest_frame_) 
    : config_file_ptr(config_file_ptr), node(node), check_frames(check_frames), rest_frame_(rest_frame_) {

        period_check_frames = 90;
        min_check_frames = 3;
        
        rotation_judge = std::make_shared<RotationJudge>(config_file_ptr, node, period_check_frames, rest_frame_);
    }

    PredictorType::PredictorType step(bool is_seen, cv::Point3f real_point, 
        cv::Point3f None_result, cv::Point3f EKF_result, cv::Point3f P3D_result, cv::Point3f RMM_result,
        float P3D_period, float RMM_period, cv::Point3f linear_predict_position);
    void clearHistory();
    
private:
    std::shared_ptr<YAML::Node> config_file_ptr; 
    rclcpp::Node* node;
    int check_frames;
    int period_check_frames;
    int min_check_frames;

    struct PredictorsResult {
        cv::Point3f None_result;
        cv::Point3f EKF_result;
        cv::Point3f P3D_result;
        cv::Point3f RMM_result;

        PredictorsResult(cv::Point3f None_result, cv::Point3f EKF_result, cv::Point3f P3D_result, cv::Point3f RMM_result)
        : None_result(None_result), EKF_result(EKF_result), P3D_result(P3D_result), RMM_result(RMM_result) {}
    };

    struct RealPoint {
        bool is_seen;
        cv::Point3f pos;

        RealPoint(bool is_seen, cv::Point3f pos) : is_seen(is_seen), pos(pos) {}
    };

    std::deque<PredictorsResult> predictors_results;
    std::deque<RealPoint> real_points;
    std::deque<float> P3D_periods;
    std::deque<float> RMM_periods;

    std::shared_ptr<RestFrame> rest_frame_;
    std::shared_ptr<RotationJudge> rotation_judge;
};

#endif