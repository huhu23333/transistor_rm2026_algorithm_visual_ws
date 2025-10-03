#pragma once
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <memory>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>

#include <3d_processing/BallisticSolver.h>
#include "3d_processing/ArmorSolver.h"
#include "EKF/Tracker.h"
#include "2d_armor_detector/Armor.h"
#include "utils/FrameRateCounter.h"
#include "3d_processing/RestFrame.h"
#include "visualizer/DataVisualizer.h"
#include "predictor/PositionPredictor3D.h"
#include "predictor/PeriodicDataPredictor.h"
#include "utils/SimpleDataFilter.h"
#include "predictor/RotationMotionModel.h"

struct PredictorResult {
    bool reset = true;
    float command_pitch = 0.0;
    float command_yaw = 0.0;
    float fire_flag = false;
};

namespace UsingPredictorType {
    enum UsingPredictorType {
        None,   // 直接瞄准装甲板
        EKF,
        FirePredictor,
        RotationMotionModel
    };
}

class AllPredictor {
public:
    AllPredictor(std::shared_ptr<YAML::Node> config_file_ptr, rclcpp::Node* node, 
        std::chrono::time_point<std::chrono::steady_clock> node_start_time, 
        std::shared_ptr<Tracker> tracker_, std::shared_ptr<ArmorSolver> armor_solver_,
        std::shared_ptr<BallisticSolver> ballistic_solver_,
        std::shared_ptr<RestFrame> rest_frame_, std::shared_ptr<FrameRateCounter> fps_counter
    ) : config_file_ptr(config_file_ptr), node(node), node_start_time(node_start_time), 
    tracker_(tracker_), armor_solver_(armor_solver_), ballistic_solver_(ballistic_solver_),
    rest_frame_(rest_frame_), fps_counter(fps_counter) {
        // 初始化参数
        bullet_velocity_ = (*config_file_ptr)["bullet_velocity_"].as<float>();

        delta_x_ = (*config_file_ptr)["delta_x_"].as<float>();
        delta_y_ = (*config_file_ptr)["delta_y_"].as<float>();
        delta_z_ = (*config_file_ptr)["delta_z_"].as<float>();
        
        yaw_rad_to_x_pixel_ratio = (*config_file_ptr)["yaw_rad_to_x_pixel_ratio"].as<float>(); 
        pitch_rad_to_y_pixel_ratio = (*config_file_ptr)["pitch_rad_to_y_pixel_ratio"].as<float>(); 

        RESET_DISTANCE_THRESHOLD = (*config_file_ptr)["RESET_DISTANCE_THRESHOLD"].as<float>(); 
        MAX_LOST_TIME = (*config_file_ptr)["MAX_LOST_TIME"].as<float>(); 

        reset_com_time = (*config_file_ptr)["reset_com_time"].as<float>(); 

        predictor3d_fit_step = (*config_file_ptr)["predictor3d_fit_step"].as<int>();
        predictor3d_predict_step = (*config_file_ptr)["predictor3d_predict_step"].as<int>();
        predictor3d_fourier_fit_order = (*config_file_ptr)["predictor3d_fourier_fit_order"].as<int>();

        fire_distance = (*config_file_ptr)["fire_distance"].as<float>();
        fire_data_predictor_fit_step = (*config_file_ptr)["fire_data_predictor_fit_step"].as<int>();

        last_com_time = std::chrono::steady_clock::now();
        
        predictor3d = std::make_shared<PositionPredictor3D>(predictor3d_fit_step);
        predictor3dArmorPredictions.push_back(cv::Point3f(0,0,0));
        predictor3dCenterPredictions.push_back(cv::Point3f(0,0,0));


        oscilloscope_fire_ = std::make_shared<Oscilloscope>(640, 120, "Fire Data Oscilloscope");
        oscilloscope_fire_ -> setScale(1.0);
        oscilloscope_fire_ -> setOffset(-0.5);

        oscilloscope_common_ = std::make_shared<Oscilloscope>(640, 480, "Common Debug Oscilloscope", 30);
        oscilloscope_common_ -> setScale(2.0);
        oscilloscope_common_ -> setOffset(-1.0);

        fire_data_predictor_ = std::make_shared<PeriodicDataPredictor>(fire_data_predictor_fit_step);
        fire_data_predictor_ -> setPeriod(1);
        pred_fire_data_filter_ = std::make_shared<SimpleDataFilter>(1);
        pred_fire_data_filter_ -> setExponentialAlpha((*config_file_ptr)["pred_fire_data_smooth_factor"].as<float>());
        pred_fire_data_filter_ -> addPoint(0.0);

        armor_distance_filter_ = std::make_shared<SimpleDataFilter>(1);
        armor_distance_filter_ -> setExponentialAlpha((*config_file_ptr)["armor_distance_smooth_factor"].as<float>());
    }

    PredictorResult step(std::vector<ArmorResult>& classifyResults, cv::Mat& frame);

    void update_serial_info(float bullet_velocity, float last_pitch_rad_delayed, float last_yaw_rad_delayed, float total_yaw_rad_delayed);

private:
    UsingPredictorType::UsingPredictorType using_predictor_type = UsingPredictorType::RotationMotionModel;
    std::shared_ptr<YAML::Node> config_file_ptr; 
    rclcpp::Node* node;

    int current_target_id_ = -1;      // 当前跟踪目标ID - EKF

    float RESET_DISTANCE_THRESHOLD; // 单位：mm
    float MAX_LOST_TIME;              // 单位：秒

    float last_total_delay_ = 0.0;
    
    std::shared_ptr<ArmorSolver> armor_solver_;
    std::shared_ptr<BallisticSolver> ballistic_solver_;

    std::shared_ptr<RestFrame> rest_frame_;
    std::shared_ptr<Oscilloscope> oscilloscope_fire_;
    std::shared_ptr<PeriodicDataPredictor> fire_data_predictor_;
    std::shared_ptr<SimpleDataFilter> pred_fire_data_filter_;
    
    std::shared_ptr<Oscilloscope> oscilloscope_common_;
    std::shared_ptr<SimpleDataFilter> armor_distance_filter_;

    std::shared_ptr<RotationMotionModel> rotation_motion_model_;

    std::chrono::time_point<std::chrono::steady_clock> node_start_time;
    
    float bullet_velocity_;
    float delta_x_;
    float delta_y_;
    float delta_z_;
    float last_pitch_rad_delayed_ = 0;
    float last_yaw_rad_delayed_ = 0;
    float total_yaw_rad_delayed_ = 0;

    // EKF/Tracker 相关新增成员
    std::shared_ptr<Tracker> tracker_;

    // 帧率计算器
    std::shared_ptr<FrameRateCounter> fps_counter;

    float pitch_integration = 0.0;
    float yaw_integration = 0.0;
    float yaw_rad_to_x_pixel_ratio;
    float pitch_rad_to_y_pixel_ratio;
    float reset_com_time;
    std::chrono::steady_clock::time_point last_com_time;
    std::shared_ptr<PositionPredictor3D> predictor3d;
    int predictor3dPrediction_nowIndex = 0;
    std::vector<cv::Point3f> predictor3dArmorPredictions;
    std::vector<cv::Point3f> predictor3dCenterPredictions;
    int predictor3d_fit_step;
    int predictor3d_predict_step;
    int predictor3d_fourier_fit_order;
    float fire_distance;
    cv::Point2f last_aim_yaw_pitch_;
    cv::Point2f last_aim_yaw_pitch_pixel_;
    float last_command_pitch_;
    float last_command_yaw_;
    int fire_data_predictor_fit_step;
};