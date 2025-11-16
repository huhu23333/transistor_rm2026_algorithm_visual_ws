#ifndef ROTATION_JUDGE_H
#define ROTATION_JUDGE_H
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <algorithm>
#include <deque>

#include "predictor/PeriodicDataPredictor.h"
#include "3d_processing/RestFrame.h"
#include "utils/DataProcessFuncs.h"
#include "visualizer/DataVisualizer.h"
#include "macro/AutoAimMacro.h"
#include "utils/SimpleDataFilter.h"

class RotationJudge {
public:
    RotationJudge(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node, int check_frames, std::shared_ptr<RestFrame> rest_frame_) 
    : config_file_ptr(config_file_ptr), node(node), check_frames(check_frames), rest_frame_(rest_frame_) {
        min_rotation_frames = check_frames;
        min_seen_frames = check_frames * 0.5;
        periodic_data_fitter = std::make_shared<PeriodicDataPredictor>(check_frames, 10);
        max_period_divergence = 3.0;

        min_right_shift_fit_R_squared = 0.7;
        min_right_shift_variance = 2500.0;

        max_period = check_frames * 0.5;

        oscilloscope_ = std::make_shared<Oscilloscope>(640, 120, "RotationJudge Oscilloscope");
        oscilloscope_ -> setScale(1.0);
        oscilloscope_ -> setOffset(0.0);

        // right_shift_data_smoother = std::make_shared<SimpleDataFilter>(1);
        // right_shift_data_smoother -> setExponentialAlpha(0.3);
        // right_shift_data_smoother -> addPoint(0.0);
    }

    void addPoint(bool is_seen, const cv::Point3f& armor_position, const cv::Point3f& linear_predict_position);
    void clearHistory();
    bool is_rotation(float p3d_period, float rmm_period);
    
private:
    std::shared_ptr<YAML::Node> config_file_ptr; 
    rclcpp::Node* node;
    int check_frames;
    int min_rotation_frames;
    int min_seen_frames;
    float max_period_divergence;
    int max_period;

    float min_right_shift_fit_R_squared;
    float min_right_shift_variance;

    std::shared_ptr<RestFrame> rest_frame_;
    std::shared_ptr<PeriodicDataPredictor> periodic_data_fitter;

    struct armorPositionInfo {
        bool is_seen;
        cv::Point3f armor_position;
        cv::Point3f linear_predict_position;

        armorPositionInfo(bool is_seen, cv::Point3f armor_position, cv::Point3f linear_predict_position)
        : is_seen(is_seen), armor_position(armor_position), linear_predict_position(linear_predict_position) {}
    };

    std::deque<armorPositionInfo> armor_position_infos;
    std::deque<float> right_shifts;

    int seen_frame_count = 0;

    float get_right_shift(cv::Point3f armor_position, cv::Point3f linear_predict_position);

    std::shared_ptr<Oscilloscope> oscilloscope_;
    // std::shared_ptr<SimpleDataFilter> right_shift_data_smoother;
};

#endif